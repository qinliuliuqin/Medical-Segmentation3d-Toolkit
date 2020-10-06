import importlib
import numpy as np
import os
import shutil
import time
import torch
import torch.distributions.beta as beta
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from segmentation3d.core.seg_infer import segmentation
from segmentation3d.core.seg_eval import evaluation
from segmentation3d.dataloader.dataset import SegmentationDataset
from segmentation3d.dataloader.sampler import EpochConcateSampler
from segmentation3d.loss.focal_loss import FocalLoss
from segmentation3d.loss.multi_dice_loss import MultiDiceLoss
from segmentation3d.loss.cross_entropy_loss import CrossEntropyLoss
from segmentation3d.loss.entropy_minization import EntropyMinimizationLoss
from segmentation3d.utils.file_io import load_config, setup_logger
from segmentation3d.utils.image_tools import save_intermediate_results
from segmentation3d.utils.model_io import load_checkpoint, save_checkpoint, delete_checkpoint


def get_data_loader(train_cfg):
    """
    Get data loader
    """
    dataset = SegmentationDataset(
                imlist_file=train_cfg.general.imseg_list_train,
                num_classes=train_cfg.dataset.num_classes,
                spacing=train_cfg.dataset.spacing,
                crop_size=train_cfg.dataset.crop_size,
                sampling_method=train_cfg.dataset.sampling_method,
                random_translation=train_cfg.dataset.random_translation,
                random_scale=train_cfg.dataset.random_scale,
                interpolation=train_cfg.dataset.interpolation,
                crop_normalizers=train_cfg.dataset.crop_normalizers)

    sampler = EpochConcateSampler(dataset, train_cfg.train.epochs)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=train_cfg.train.batchsize,
                             num_workers=train_cfg.train.num_threads, pin_memory=True)

    dataset_m = SegmentationDataset(
            imlist_file=train_cfg.general.imseg_list_train_ul,
            num_classes=train_cfg.dataset.num_classes,
            spacing=train_cfg.dataset.spacing,
            crop_size=train_cfg.dataset.crop_size,
            sampling_method=train_cfg.dataset.sampling_method,
            random_translation=train_cfg.dataset.random_translation,
            random_scale=train_cfg.dataset.random_scale,
            interpolation=train_cfg.dataset.interpolation,
            crop_normalizers=train_cfg.dataset.crop_normalizers)

    sampler_m = EpochConcateSampler(dataset_m, train_cfg.train.epochs)
    data_loader_m = DataLoader(dataset_m, sampler=sampler_m, batch_size=train_cfg.train.batchsize,
                               num_workers=train_cfg.train.num_threads, pin_memory=True)

    return data_loader, data_loader_m


def train_one_epoch(net, data_loader, data_loader_m, loss_funces, opt, logger, epoch_idx, use_gpu=True, use_mixup=False,
                    mixup_alpha=-1, use_ul=False, debug=False, model_folder=''):
    """
    """
    data_iter = iter(data_loader)
    if use_ul: data_iter_m = iter(data_loader_m)

    for batch_idx in range(len(data_loader.dataset)):
        begin_t = time.time()

        crops, masks, frames, filenames = data_iter.next()
        if use_gpu: crops, masks = crops.cuda(), masks.cuda()

        if use_ul:
            crops_m, _, _, _ = data_iter_m.next()
            if torch.randn(1) > 0:
                crops_mn = crops_m + 0.3 * torch.rand_like(crops_m)
            else:
                crops_mn = crops_m - 0.3 * torch.rand_like(crops_m)

            if use_gpu:
                crops_m = crops_m.cuda()
                crops_mn = crops_mn.cuda()

        # if use_mixup:
        #     beta_func = beta.Beta(mixup_alpha, mixup_alpha)
        #     crops_perm, masks_perm, _, _ = data_iter_m.next()
        #     weight_lambda = beta_func.sample()
        #     if use_gpu: weight_lambda = weight_lambda.cuda()
        #     crops = weight_lambda * crops + (1 - weight_lambda) * crops_perm

        # clear previous gradients
        opt.zero_grad()

        # network forward and backward
        outputs = net(crops)
        train_loss = sum([loss_func(outputs, masks) for loss_func in loss_funces])

        if use_ul:
            outputs_m = net(crops_m)
            outputs_mn = net(crops_mn)

            # masks_m is the pseudo-label, vals_mn is the prediction
            _, masks_m = outputs_m.max(dim=1)
            vals_mn, _ = outputs_mn.max(dim=1)

            valid_index = vals_mn > 0.8
            masks_m_valid = masks_m[:, valid_index[0, :]]
            outputs_mn_valid = outputs_mn[:, :, valid_index[0, :]]
            if masks_m_valid.nelement() == 0:
               train_loss_m = 0
            else:
               train_loss_m = sum([loss_func(outputs_mn_valid, masks_m_valid) for loss_func in loss_funces])

            if epoch_idx > 1000:
                train_loss = train_loss + min(1, (epoch_idx - 1000) / 1000) * train_loss_m

            # add consistency regularization
            # train_loss_mn = EntropyMinimizationLoss()(outputs_m, outputs_mn)
            # train_loss += train_loss_mn

        # if use_mixup:
        #     if use_gpu: masks_perm = masks_perm.cuda()
        #     train_loss_perm = loss_func(outputs, masks_perm)
        #     train_loss = weight_lambda * train_loss + (1 - weight_lambda) * train_loss_perm

        train_loss.backward()

        # update weights
        opt.step()

        # save training crops for visualization
        if debug:
            batch_size = crops.size(0)
            save_intermediate_results(list(range(batch_size)), crops, masks, outputs, frames, filenames,
                                      os.path.join(model_folder, 'batch_{}'.format(batch_idx)))

        batch_duration = time.time() - begin_t

        # print training loss per batch
        if use_ul:
            msg = 'epoch: {}, batch: {}, train_loss: {:.4f}, {:.4f}, time: {:.4f} s/vol'
            msg = msg.format(epoch_idx, batch_idx, train_loss.item() - train_loss_m.item(),
                             train_loss_m.item(), batch_duration)

        else:
            msg = 'epoch: {}, batch: {}, train_loss: {:.4f}, time: {:.4f} s/vol'
            msg = msg.format(epoch_idx, batch_idx, train_loss.item(), batch_duration)

        logger.info(msg)


def train(train_config_file, infer_config_file, infer_gpu_id):
    """ Medical image segmentation training engine
    :param train_config_file: the input configuration file
    :return: None
    """
    assert os.path.isfile(train_config_file), 'Config not found: {}'.format(train_config_file)

    # load config file
    train_cfg = load_config(train_config_file)
    infer_cfg = load_config(infer_config_file)

    # get training parameters
    use_ul = train_cfg.train.use_ul
    use_debug = train_cfg.debug.save_inputs
    use_gpu = train_cfg.general.num_gpus > 0
    use_mixup = train_cfg.dataset.mixup_alpha >= 0
    mixup_alpha = train_cfg.dataset.mixup_alpha

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
    shutil.copy(infer_config_file, os.path.join(train_cfg.general.save_dir, 'infer_config.py'))

    # enable logging
    log_file = os.path.join(model_folder, 'train_log.txt')
    logger = setup_logger(log_file, 'seg3d')

    # control randomness during training
    np.random.seed(train_cfg.general.seed)
    torch.manual_seed(train_cfg.general.seed)
    if use_gpu: torch.cuda.manual_seed(train_cfg.general.seed)

    # dataset
    data_loader, data_loader_m = get_data_loader(train_cfg)
    net_module = importlib.import_module('segmentation3d.network.' + train_cfg.net.name)
    net = net_module.SegmentationNet(1, train_cfg.dataset.num_classes)
    max_stride = net.max_stride()
    net_module.parameters_kaiming_init(net)
    if use_gpu: net = nn.parallel.DataParallel(net, device_ids=list(range(train_cfg.general.num_gpus))).cuda()
    assert np.all(np.array(train_cfg.dataset.crop_size) % max_stride == 0), 'crop size not divisible by max stride'

    # training optimizer
    opt = optim.Adam(net.parameters(), lr=train_cfg.train.lr, betas=train_cfg.train.betas)
    scheduler = StepLR(opt, step_size=train_cfg.train.step_size, gamma=0.5)

    # load checkpoint if resume epoch > 0
    if train_cfg.general.resume_epoch >= 0:
        last_save_epoch, batch_start = load_checkpoint(train_cfg.general.resume_epoch, net, opt, model_folder)
    else:
        last_save_epoch, batch_start = 0, 0

    focal_loss_func = FocalLoss(class_num=train_cfg.dataset.num_classes, alpha=train_cfg.loss.obj_weight,
                              gamma=train_cfg.loss.focal_gamma, use_gpu=use_gpu)
    dice_loss_func = MultiDiceLoss(weights=train_cfg.loss.obj_weight, num_class=train_cfg.dataset.num_classes,
                                  use_gpu=use_gpu)
    ce_loss_func = CrossEntropyLoss()

    loss_funces = []
    for loss_name in train_cfg.loss.name:
        if loss_name == 'Focal':
            loss_funces.append(focal_loss_func)
        elif loss_name == 'Dice':
            loss_funces.append(dice_loss_func)
        elif loss_name == 'CE':
            loss_funces.append(ce_loss_func)
        else:
            raise ValueError('Unsupported loss name.')

    writer = SummaryWriter(os.path.join(model_folder, 'tensorboard'))

    # loop over batches
    best_epoch, best_dsc_mean, best_dsc_std = last_save_epoch, 0.0, 0
    for epoch_idx in range(last_save_epoch + 1, train_cfg.train.epochs + 1):
        train_one_epoch(net, data_loader, data_loader_m, loss_funces, opt, logger, epoch_idx, use_gpu,
                        use_mixup, mixup_alpha, use_ul, use_debug, train_cfg.general.save_dir)

        scheduler.step()

        # inference
        if epoch_idx % train_cfg.train.save_epochs == 0:
            save_checkpoint(net, opt, epoch_idx, 0, train_cfg, max_stride, 1)
            seg_folder = os.path.join(train_cfg.general.save_dir, 'results')
            segmentation(train_cfg.general.imseg_list_val, train_cfg.general.save_dir, seg_folder, 'seg.nii.gz',
                         infer_gpu_id, False, True, False, False)

            # test DSC
            labels = [idx for idx in range(1, train_cfg.dataset.num_classes)]
            mean, std = evaluation(train_cfg.general.imseg_list_val, seg_folder, 'seg.nii.gz', labels, 10, None)

            if mean <= best_dsc_mean: delete_checkpoint(epoch_idx, train_cfg)
            else:
                best_dsc_mean = mean
                best_dsc_std = std
                best_epoch = epoch_idx

            msg = 'Best epoch {}, mean (std) = {:.4f} ({:.4f}), current epoch {} mean (std) = {:.4f} ({:.4f})'.format(
                best_epoch, best_dsc_mean, best_dsc_std, epoch_idx, mean, std)
            logger.info(msg)

    writer.close()
