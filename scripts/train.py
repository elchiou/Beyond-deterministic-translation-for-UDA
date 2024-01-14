import sys

import argparse
import os
import os.path as osp
import pprint
import random
import warnings
import datetime
import numpy as np
import yaml
import torch
from torch.utils import data

from model.deeplabv2 import get_deeplab_v2
from dataset.gta5 import GTA5DataSet
from dataset.synthia import SYNTHIADataSet
from dataset.cityscapes import CityscapesDataSet
from dataset.cityscapes_sl import CityscapesDataSetSL
from domain_adaptation.config import cfg, cfg_from_file
from domain_adaptation.train_UDA import train_domain_adaptation

from munit.trainer import MUNIT_Trainer
from munit.utils import get_config
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")

def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) training")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    parser.add_argument('--cfg_munit', type=str, default='',
                        help='munit config file', )
    parser.add_argument("--random-train", action="store_true",
                        help="not fixing random seed.")
    parser.add_argument("--tensorboard", action="store_true",
                        help="visualize training loss with tensorboardX.")
    parser.add_argument("--viz-every-iter", type=int, default=None,
                        help="frequency of results visualization.")
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    parser.add_argument("--data-root", type=str, default=None,
                        help="path to the data")
    parser.add_argument("--project-root", type=str, default=None,
                        help="path to the project")
    parser.add_argument('--source-domain', type=str, default='gta', 
                        help='source domain dataset')
    parser.add_argument('--num-classes', type=int, default=19, 
                        help='number of classes')
    parser.add_argument('--train-restore-from', type=str,
                        default='../../pretrained_models/DeepLab_resnet_pretrained_imagenet.pth',
                        help='path to the pretrained deeplab')
    parser.add_argument('--use-synth-s2t', type=bool, default=False,
                        help='True when we train a target network')
    parser.add_argument('--transl-net', type=str, default='munit', 
                        help='translation network: munit')
    parser.add_argument('--load-dir-transl', type=str, default='.', 
                        help='pretrained munit model')
    parser.add_argument('--load-iter-transl', type=int, default='',
                        help='which iteration to load? if load_iter > 0,' 
                        +'the code will load checkpoint iter_[load_iter]')
    parser.add_argument('--batch-size', type=int, default=1, 
                        help='batch size')
    parser.add_argument('--gpu-ids', type=str, default='0', 
                        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--subnorm_type', type=str, default='batch', 
                        help='batch or sync_batch')
    parser.add_argument('--var', type=int, default=1, 
                        help='variance of the style code')
    parser.add_argument('--use-synth-t2s', type=bool, default=False, 
                        help='True when we train the source domain network')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--pseudo_labels_train', type=bool, default=False,
                         help='use pseudolabels during training : True/False')
    parser.add_argument('--round', type=int, default=0, help='round of pseudolabeling')
    parser.add_argument('--start-iter', type=int, default=0, help='start iteration')

    return parser.parse_args()


def main():
    # load args
    args = get_arguments()
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = [int(str_id) for str_id in str_ids if int(str_id) >= 0]
    print('Called with args:', args)
    assert args.cfg is not None, 'Missing cfg file'
    cfg_from_file(args.cfg, args)
    
    # auto-generate exp name if not specified
    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}'

    if args.exp_suffix:
        cfg.EXP_NAME += f'_{args.exp_suffix}'
    
    # auto-generate snapshot path if not specified
    if cfg.TRAIN.SNAPSHOT_DIR == '':
        cfg.TRAIN.SNAPSHOT_DIR = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)
    # tensorboard
    if args.tensorboard:
        if cfg.TRAIN.TENSORBOARD_LOGDIR == '':
            cfg.TRAIN.TENSORBOARD_LOGDIR = osp.join(cfg.EXP_ROOT_LOGS, 'tensorboard', cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.TENSORBOARD_LOGDIR, exist_ok=True)
        if args.viz_every_iter is not None:
            cfg.TRAIN.TENSORBOARD_VIZRATE = args.viz_every_iter
    else:
        cfg.TRAIN.TENSORBOARD_LOGDIR = ''
    pprint.pprint(cfg)

    # initialization
    _init_fn = None
    if not args.random_train:
        torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
        torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
        np.random.seed(cfg.TRAIN.RANDOM_SEED)
        random.seed(cfg.TRAIN.RANDOM_SEED)

        def _init_fn(worker_id):
            np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)

    # load segmentation network
    assert osp.exists(cfg.TRAIN.RESTORE_FROM), f'Missing init model {cfg.TRAIN.RESTORE_FROM}'
    if cfg.TRAIN.MODEL == 'DeepLabv2':
        model = get_deeplab_v2(num_classes=cfg.NUM_CLASSES, multi_level=cfg.TRAIN.MULTI_LEVEL, subnorm_type=args.subnorm_type)
        saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM)
        if 'DeepLab_resnet_pretrained_imagenet' in cfg.TRAIN.RESTORE_FROM:
            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            model.load_state_dict(new_params)
        else:
            model.load_state_dict(saved_state_dict)
    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")
    
    # load image-to-image translation network
    cfg_transl = get_config(args.cfg_munit)
    model_transl = MUNIT_Trainer(cfg_transl, args)
    load_suffix = 'gen_%08d.pt' % (args.load_iter_transl)
    load_path = os.path.join(args.load_dir_transl, load_suffix)
    state_dict = torch.load(load_path)
    model_transl.gen_a.load_state_dict(state_dict['a'])
    model_transl.gen_b.load_state_dict(state_dict['b'])
    model_transl.eval()

    # dataloaders
    if args.source_domain == 'gta':
        source_dataset = GTA5DataSet(root=cfg.DATA_DIRECTORY_SOURCE,
                                     list_path=cfg.DATA_LIST_SOURCE,
                                     set=cfg.TRAIN.SET_SOURCE,
                                     max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_SOURCE,
                                     crop_size=cfg.TRAIN.INPUT_SIZE_SOURCE,
                                     mean=cfg.TRAIN.IMG_MEAN)
    else:
        source_dataset = SYNTHIADataSet(root=cfg.DATA_DIRECTORY_SOURCE,
                                     list_path=cfg.DATA_LIST_SOURCE,
                                     set=cfg.TRAIN.SET_SOURCE,
                                     max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_SOURCE,
                                     crop_size=cfg.TRAIN.INPUT_SIZE_SOURCE,
                                     mean=cfg.TRAIN.IMG_MEAN, num_classes=args.num_classes)

    source_loader = data.DataLoader(source_dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE_SOURCE,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=True,
                                    pin_memory=True,
                                    worker_init_fn=_init_fn)
    # use different dataloader when using pseudolabels
    if args.use_ps:
        target_dataset = CityscapesDataSet(root=cfg.DATA_DIRECTORY_TARGET,
                                        list_path=cfg.DATA_LIST_TARGET,
                                        set=cfg.TRAIN.SET_TARGET,
                                        info_path=cfg.TRAIN.INFO_TARGET,
                                        max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_TARGET,
                                        crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
                                        mean=cfg.TRAIN.IMG_MEAN)
    else:
        target_dataset = CityscapesDataSet(root=cfg.DATA_DIRECTORY_TARGET,
                                        list_path=cfg.DATA_LIST_TARGET,
                                        set=cfg.TRAIN.SET_TARGET,
                                        info_path=cfg.TRAIN.INFO_TARGET,
                                        max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_TARGET,
                                        crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
                                        mean=cfg.TRAIN.IMG_MEAN)
    target_loader = data.DataLoader(target_dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE_TARGET,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=True,
                                    pin_memory=True,
                                    worker_init_fn=_init_fn)

    with open(osp.join(cfg.TRAIN.SNAPSHOT_DIR, 'train_cfg.yml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    # UDA training
    train_domain_adaptation(model, model_transl, source_loader, target_loader, args, cfg, cfg_transl)


if __name__ == '__main__':
    main()
