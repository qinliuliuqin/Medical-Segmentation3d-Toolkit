import importlib
import numpy as np
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from segmentation3d.dataloader.dataset import SegmentationDataset
from segmentation3d.dataloader.sampler import EpochConcateSampler
from segmentation3d.loss.focal_loss import FocalLoss
from segmentation3d.loss.multi_dice_loss import MultiDiceLoss
from segmentation3d.utils.file_io import load_config, setup_logger
from segmentation3d.utils.image_tools import save_intermediate_results
from segmentation3d.utils.model_io import load_checkpoint, save_checkpoint


def train(train_config_file):
    """ Medical image segmentation training engine
    :param train_config_file: the input configuration file
    :return: None
    """
    assert os.path.isfile(train_config_file), 'Config not found: {}'.format(train_config_file)

    # load config file
    train_cfg = load_config(train_config_file)

    # clean the existing folder if training from scratch
    model_folder = os.path.join(train_cfg.general.save_dir, train_cfg.general.model_scale)
    if os.path.isdir(model_folder):
        if train_cfg.general.resume_epoch < 0:
            shutil.rmtree(model_folder)
            os.makedirs(model_folder)
    else:
        os.makedirs(model_folder)

    # copy training and inference config files to the model folder
    shutil.copy(train_config_file, os.path.join(model_folder, 'train_config.py'))
    infer_config_file = os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'infer_config.py'))
    shutil.copy(infer_config_file, os.path.join(train_cfg.general.save_dir, 'infer_config.py'))

    # enable logging
    log_file = os.path.join(train_cfg.general.save_dir, 'train_log.txt')
    logger = setup_logger(log_file, 'seg3d')

    # control randomness during training
    np.random.seed(train_cfg.general.seed)
    torch.manual_seed(train_cfg.general.seed)
    if train_cfg.general.num_gpus > 0:
        torch.cuda.manual_seed(train_cfg.general.seed)

    # dataset
    dataset = SegmentationDataset(
                imlist_file=train_cfg.general.imseg_list,
                num_classes=train_cfg.dataset.num_classes,
                spacing=train_cfg.dataset.spacing,
                crop_size=train_cfg.dataset.crop_size,
                sampling_method=train_cfg.dataset.sampling_method,
                random_translation=train_cfg.dataset.random_translation,
                interpolation=train_cfg.dataset.interpolation,
                crop_normalizers=train_cfg.dataset.crop_normalizers)

    sampler = EpochConcateSampler(dataset, train_cfg.train.epochs)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=train_cfg.train.batchsize,
                             num_workers=train_cfg.train.num_threads, pin_memory=True)

    net_module = importlib.import_module('segmentation3d.network.' + train_cfg.net.name)
    net = net_module.SegmentationNet(dataset.num_modality(), train_cfg.dataset.num_classes)
    max_stride = net.max_stride()
    net_module.parameters_kaiming_init(net)
    if train_cfg.general.num_gpus > 0:
        net = nn.parallel.DataParallel(net, device_ids=list(range(train_cfg.general.num_gpus)))
        net = net.cuda()

    assert np.all(np.array(train_cfg.dataset.crop_size) % max_stride == 0), 'crop size not divisible by max stride'

    # training optimizer
    opt = optim.Adam(net.parameters(), lr=train_cfg.train.lr, betas=train_cfg.train.betas)

    # load checkpoint if resume epoch > 0
    if train_cfg.general.resume_epoch >= 0:
        last_save_epoch, batch_start = load_checkpoint(train_cfg.general.resume_epoch, net, opt, model_folder)
    else:
        last_save_epoch, batch_start = 0, 0

    if train_cfg.loss.name == 'Focal':
        # reuse focal loss if exists
        loss_func = FocalLoss(class_num=train_cfg.dataset.num_classes, alpha=train_cfg.loss.obj_weight, gamma=train_cfg.loss.focal_gamma,
                              use_gpu=train_cfg.general.num_gpus > 0)
    elif train_cfg.loss.name == 'Dice':
        loss_func = MultiDiceLoss(weights=train_cfg.loss.obj_weight, num_class=train_cfg.dataset.num_classes,
                                  use_gpu=train_cfg.general.num_gpus > 0)
    else:
        raise ValueError('Unknown loss function')

    writer = SummaryWriter(os.path.join(model_folder, 'tensorboard'))

    batch_idx = batch_start
    data_iter = iter(data_loader)

    # loop over batches
    for i in range(len(data_loader)):
        begin_t = time.time()

        crops, masks, frames, filenames = data_iter.next()

        if train_cfg.general.num_gpus > 0:
            crops, masks = crops.cuda(), masks.cuda()

        # clear previous gradients
        opt.zero_grad()

        # network forward and backward
        outputs = net(crops)
        train_loss = loss_func(outputs, masks)
        train_loss.backward()

        # update weights
        opt.step()

        # save training crops for visualization
        if train_cfg.debug.save_inputs:
            batch_size = crops.size(0)
            save_intermediate_results(list(range(batch_size)), crops, masks, outputs, frames, filenames,
                                      os.path.join(model_folder, 'batch_{}'.format(i)))

        epoch_idx = batch_idx * train_cfg.train.batchsize // len(dataset)
        batch_idx += 1
        batch_duration = time.time() - begin_t
        sample_duration = batch_duration * 1.0 / train_cfg.train.batchsize

        # print training loss per batch
        msg = 'epoch: {}, batch: {}, train_loss: {:.4f}, time: {:.4f} s/vol'
        msg = msg.format(epoch_idx, batch_idx, train_loss.item(), sample_duration)
        logger.info(msg)

        # save checkpoint
        if epoch_idx != 0 and (epoch_idx % train_cfg.train.save_epochs == 0):
            if last_save_epoch != epoch_idx:
                save_checkpoint(net, opt, epoch_idx, batch_idx, train_cfg, max_stride, dataset.num_modality())
                last_save_epoch = epoch_idx

        writer.add_scalar('Train/Loss', train_loss.item(), batch_idx)

    writer.close()