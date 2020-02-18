import argparse
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
from segmentation3d.dataloader.image_tools import save_intermediate_results
from segmentation3d.dataloader.sampler import EpochConcateSampler
from segmentation3d.loss.focal_loss import FocalLoss
from segmentation3d.loss.multi_dice_loss import MultiDiceLoss
from segmentation3d.utils.file_io import load_config, setup_logger
from segmentation3d.utils.model_io import load_checkpoint, save_checkpoint
from segmentation3d.utils.voxel_rend_helper import get_uncertain_voxel_coords_with_randomness, calculate_uncertainty, \
    voxel_sample_features, voxel_sample


def train(config_file):
    """ Medical image segmentation training engine
    :param config_file: the input configuration file
    :return: None
    """
    assert os.path.isfile(config_file), 'Config not found: {}'.format(config_file)

    # load config file
    cfg = load_config(config_file)

    # clean the existing folder if training from scratch
    if cfg.general.resume_epoch < 0 and os.path.isdir(cfg.general.save_dir):
        shutil.rmtree(cfg.general.save_dir)

    # enable logging
    log_file = os.path.join(cfg.general.save_dir, 'train_log.txt')
    logger = setup_logger(log_file, 'seg3d')

    # control randomness during training
    np.random.seed(cfg.general.seed)
    torch.manual_seed(cfg.general.seed)
    if cfg.general.num_gpus > 0:
        torch.cuda.manual_seed(cfg.general.seed)

    # dataset
    dataset = SegmentationDataset(
                imlist_file=cfg.general.imseg_list,
                num_classes=cfg.dataset.num_classes,
                spacing=cfg.dataset.spacing,
                crop_size=cfg.dataset.crop_size,
                sampling_method=cfg.dataset.sampling_method,
                random_translation=cfg.dataset.random_translation,
                interpolation=cfg.dataset.interpolation,
                crop_normalizers=cfg.dataset.crop_normalizers)

    sampler = EpochConcateSampler(dataset, cfg.train.epochs)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=cfg.train.batchsize,
                             num_workers=cfg.train.num_threads, pin_memory=True)

    net_module = importlib.import_module('segmentation3d.network.' + cfg.net.name)
    net = net_module.SegmentationNet(dataset.num_modality(), cfg.dataset.num_classes, cfg.net.dropout_turn_on)
    max_stride = net.max_stride()
    net_module.parameters_kaiming_init(net)

    voxel_head_fine_channels = net.num_multi_layer_features()
    voxel_head_coarse_channels = cfg.dataset.num_classes
    voxel_net = net_module.VoxelHead(
        voxel_head_fine_channels, voxel_head_coarse_channels, cfg.dataset.num_classes, cfg.voxel_head.num_fc
    )
    net_module.parameters_kaiming_init(voxel_net)

    if cfg.general.num_gpus > 0:
        net = nn.parallel.DataParallel(net, device_ids=list(range(cfg.general.num_gpus)))
        net = net.cuda()

    assert np.all(np.array(cfg.dataset.crop_size) % max_stride == 0), 'crop size not divisible by max stride'

    # training optimizer
    opt = optim.Adam(net.parameters(), lr=cfg.train.lr, betas=cfg.train.betas)

    # load checkpoint if resume epoch > 0
    if cfg.general.resume_epoch >= 0:
        last_save_epoch, batch_start = load_checkpoint(cfg.general.resume_epoch, net, opt, cfg.general.save_dir, -1)
    else:
        last_save_epoch, batch_start = 0, 0

    # loss function for mask head
    if cfg.loss.name == 'Focal':
        mask_head_loss_func = FocalLoss(class_num=cfg.dataset.num_classes, alpha=cfg.loss.obj_weight,
                                        gamma=cfg.loss.focal_gamma, use_gpu=cfg.general.num_gpus > 0)
    elif cfg.loss.name == 'Dice':
        mask_head_loss_func = MultiDiceLoss(weights=cfg.loss.obj_weight, num_class=cfg.dataset.num_classes,
                                  use_gpu=cfg.general.num_gpus > 0)
    else:
        raise ValueError('Unknown loss function')

    # loss function for voxel head
    if cfg.voxel_head.loss_name == 'Focal':
        voxel_head_loss_func = FocalLoss(class_num=cfg.dataset.num_classes, alpha=cfg.voxel_head.loss_obj_weight,
                                         gamma=cfg.voxel_head.loss_focal_gamma, use_gpu=cfg.general.num_gpus > 0)
    elif cfg.loss.name == 'Dice':
        voxel_head_loss_func = MultiDiceLoss(weights=cfg.voxel_head.loss_obj_weight, num_class=cfg.dataset.num_classes,
                                  use_gpu=cfg.general.num_gpus > 0)
    else:
        raise ValueError('Unknown loss function')

    writer = SummaryWriter(os.path.join(cfg.general.save_dir, 'tensorboard'))

    batch_idx = batch_start
    data_iter = iter(data_loader)

    # loop over batches
    for i in range(len(data_loader)):
        begin_t = time.time()

        crops, masks_fine, masks_coarse, frames, filenames = data_iter.next()

        if cfg.general.num_gpus > 0:
            crops, masks_fine, masks_coarse = crops.cuda(), masks_fine.cuda(), masks_coarse.cuda()

        # clear previous gradients
        opt.zero_grad()

        # network forward and backward
        mask_preds_coarse, mask_fine_features = net(crops)

        # up-sample the coarse predictions
        up_sampler = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        mask_preds_fine = up_sampler(mask_preds_coarse)
        mask_head_loss = mask_head_loss_func(mask_preds_fine, masks_fine)

        # sample points according to the fine predictions
        num_voxels, oversample_ratio, importance_sample_ratio = \
            cfg.voxel_head.num_voxels, cfg.voxel_head.oversample_ratio, cfg.voxel_head.importance_sample_ratio
        voxel_coords = get_uncertain_voxel_coords_with_randomness(
            mask_preds_fine, calculate_uncertainty, num_voxels, oversample_ratio, importance_sample_ratio
        )

        # compute features for each selected sample. There are two types of features, namely the fine-grained features
        # and the coarse features. the fine-grained features is extracted from the multi-layer feature maps, while the
        # coarse features are extracted from the fine predictions which are up-sampled from the coarse predictions.
        voxel_fine_features = voxel_sample_features(mask_fine_features, voxel_coords)
        voxel_coarse_features = voxel_sample(mask_preds_fine, voxel_coords)
        voxel_labels = voxel_sample(masks_fine, voxel_coords, mode='nearest')
        assert voxel_fine_features.dim() == voxel_coarse_features.dim() == voxel_labels.dim()
        assert voxel_coarse_features.shape[1] == cfg.dataset.num_classes

        # train a fully connected layer for classification
        voxel_preds = voxel_net(voxel_fine_features, voxel_coarse_features)
        voxel_head_loss = voxel_head_loss_func(voxel_preds, voxel_labels)

        train_loss = (1 - cfg.voxel_head.loss_weight) * mask_head_loss + cfg.voxel_head.loss_weight * voxel_head_loss
        train_loss.backward()

        # update weights
        opt.step()

        # save training crops for visualization
        if cfg.debug.save_inputs:
            batch_size = crops.size(0)
            save_intermediate_results(list(range(batch_size)), crops, masks_fine, mask_preds_fine, frames, filenames,
                                      os.path.join(cfg.general.save_dir, 'batch_{}'.format(i)))

        epoch_idx = batch_idx * cfg.train.batchsize // len(dataset)
        batch_idx += 1
        batch_duration = time.time() - begin_t
        sample_duration = batch_duration * 1.0 / cfg.train.batchsize

        # print training loss per batch
        msg = 'epoch: {}, batch: {}, train_loss: {:.4f}, mask_loss: {:.4f}, voxel_loss: {:.4f}, time: {:.4f} s/vol'
        msg = msg.format(
            epoch_idx, batch_idx, train_loss.item(), mask_head_loss.item(), voxel_head_loss.item(), sample_duration
        )
        logger.info(msg)

        # save checkpoint
        if epoch_idx != 0 and (epoch_idx % cfg.train.save_epochs == 0):
            if last_save_epoch != epoch_idx:
                save_checkpoint(net, opt, epoch_idx, batch_idx, cfg, config_file, max_stride, dataset.num_modality())
                last_save_epoch = epoch_idx

        writer.add_scalar('Train/Loss', train_loss.item(), batch_idx)

    writer.close()


def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,5,6,7'

    long_description = "Training engine for 3d medical image segmentation"
    parser = argparse.ArgumentParser(description=long_description)

    parser.add_argument('-i', '--input',
                        default='./config/train_config.py',
                        help='configure file for medical image segmentation training.')
    args = parser.parse_args()

    train(args.input)


if __name__ == '__main__':
    main()
