# --------------------------------------------------------
# AdvEnt training
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import argparse
import os
import os.path as osp
import pprint
import warnings
import sys
from torch.utils import data
sys.path.append("../..") # Adds higher directory to python modules path.
sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append('/cluster/project9/echiou_domainAdaptation/CVPR_21/ADVENT_CVPR_21/')
from advent.model.deeplabv2 import get_deeplab_v2
from advent.dataset.cityscapes import CityscapesDataSet
from advent.domain_adaptation.config import cfg, cfg_from_file
from advent.domain_adaptation.eval_UDA import evaluate_domain_adaptation
from munit.trainer import MUNIT_Trainer
from munit.utils import get_config
import torch
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")
import numpy as np
import random 

def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for evaluation")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    parser.add_argument("--data-root", type=str, default=None,
                        help="path to the data")
    parser.add_argument("--project-root", type=str, default=None,
                        help="path to the project")
    parser.add_argument('--source-domain', type=str, default='gta', help='source domain dataset')
    parser.add_argument('--num-classes', type=int, default=19, help='number of classes')
    parser.add_argument('--train-restore-from', type=str,
                        default='../../pretrained_models/DeepLab_resnet_pretrained_imagenet.pth',
                        help='path to the pretrained deeplab')
    parser.add_argument('--test-restore-from', type=str, default=' ', help='path to the pretrained deeplab')
    parser.add_argument('--test-snapshot', type=str, default=' ', help='path to the pretrained deeplab')
    parser.add_argument('--test-mode', type=str, default='single', help='test mode: single/best')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--use-synth-t2s', type=bool, default=False, help='use synthetic target: True/False')
    parser.add_argument('--transl-net', type=str, default='munit', help='translation network: cyclegan/munit')
    parser.add_argument('--load-dir-transl', type=str, default='.', help='pretrained cyclegan model')
    parser.add_argument('--load-iter-transl', type=int, default='50000', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]')
    parser.add_argument('--cfg_munit', type=str, default='/home/echiou/Dropbox/PhD/code/iccv/ADVENT_2/munit/gta2cityscapes_folder.yaml', help='optional config file', )
    parser.add_argument('--use-synth-t2t', type=bool, default=False, help='use synthetic target: True/False')
    parser.add_argument('--num-fake-source', type=int, default=1, help='how many fake samples we should use to generate the labels')
    parser.add_argument('--num-fake-target', type=int, default=1, help='how many fake target samples we should use')
    parser.add_argument('--new', type=bool, default=False, help='new experiment')
    parser.add_argument('--ignore-adv', type=bool, default=False, help='ignore adversarial loss')
    parser.add_argument('--ignore-source', type=bool, default=False, help='ignore source data')
    return parser.parse_args()


def main(config_file, exp_suffix, args):
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)    # LOAD ARGS
    assert config_file is not None, 'Missing cfg file'
    cfg_from_file(config_file, args)
    # auto-generate exp name if not specified
    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}'
    if exp_suffix:
        cfg.EXP_NAME += f'_{exp_suffix}'
    # auto-generate snapshot path if not specified
    if cfg.TEST.SNAPSHOT_DIR[0] == '':
        cfg.TEST.SNAPSHOT_DIR[0] = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
        os.makedirs(cfg.TEST.SNAPSHOT_DIR[0], exist_ok=True)

    print('Using config:')
    pprint.pprint(cfg)
    # INIT
    _init_fn = None
    if True:
        torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
        torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
        np.random.seed(cfg.TRAIN.RANDOM_SEED)
        random.seed(cfg.TRAIN.RANDOM_SEED)

        def _init_fn(worker_id):
            np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)

    # load models
    models = []
    n_models = len(cfg.TEST.MODEL)
    if cfg.TEST.MODE == 'best':
        assert n_models == 1, 'Not yet supported'
    for i in range(n_models):
        if cfg.TEST.MODEL[i] == 'DeepLabv2':
            model = get_deeplab_v2(num_classes=cfg.NUM_CLASSES,
                                   multi_level=cfg.TEST.MULTI_LEVEL[i])
        else:
            raise NotImplementedError(f"Not yet supported {cfg.TEST.MODEL[i]}")
        models.append(model)
    if args.transl_net == 'cyclegan' and (args.use_synth_t2s or args.use_synth_t2t):
        cfg_transl = None
        args.isTrain = False
        args.no_dropout = True
        model_transl = CycleGANModel(args)
        model_transl.setup(args)
        model_transl.eval()
    elif args.transl_net == 'munit'  and (args.use_synth_t2s or args.use_synth_t2t):
        #args.gpu_ids = [cfg.GPU_ID]
        cfg_transl = get_config(args.cfg_munit)
        model_transl = MUNIT_Trainer(cfg_transl, args)
        load_suffix = 'gen_%08d.pt' % (args.load_iter_transl)
        load_path = os.path.join(args.load_dir_transl, load_suffix)
        state_dict = torch.load(load_path)
        model_transl.gen_a.load_state_dict(state_dict['a'])
        model_transl.gen_b.load_state_dict(state_dict['b'])
        model_transl.eval()
    else:
        model_transl = None
        cfg_transl = None
    if os.environ.get('ADVENT_DRY_RUN', '0') == '1':
        return
    # dataloaders
    test_dataset = CityscapesDataSet(root=cfg.DATA_DIRECTORY_TARGET,
                                     list_path=cfg.DATA_LIST_TARGET,
                                     set=cfg.TEST.SET_TARGET,
                                     info_path=cfg.TEST.INFO_TARGET,
                                     crop_size=cfg.TEST.INPUT_SIZE_TARGET,
                                     mean=cfg.TEST.IMG_MEAN,
                                     labels_size=cfg.TEST.OUTPUT_SIZE_TARGET)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=cfg.TEST.BATCH_SIZE_TARGET,
                                  num_workers=cfg.NUM_WORKERS,
                                  shuffle=False,
                                  pin_memory=True)
    # eval
    evaluate_domain_adaptation(models, test_loader, cfg, args, model_transl, cfg_transl)


if __name__ == '__main__':
    args = get_arguments()
    print('Called with args:')
    print(args)
    main(args.cfg, args.exp_suffix, args)
