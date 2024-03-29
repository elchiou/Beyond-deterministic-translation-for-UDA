import os.path as osp

import numpy as np
from easydict import EasyDict

from utils import project_root
from utils.serialization import yaml_load


cfg = EasyDict()

## common configs ##
# source domain
cfg.SOURCE = 'GTA'
# target domain
cfg.TARGET = 'Cityscapes'
# Number of workers for dataloading
cfg.NUM_WORKERS = 4
# list of training images
cfg.DATA_LIST_SOURCE = str(project_root / 'data/gta5_list/{}.txt')
cfg.DATA_LIST_TARGET = str(project_root / 'data/cityscapes_list/{}.txt')
# directories
cfg.DATA_DIRECTORY_SOURCE = str(project_root / 'data/GTA5')
cfg.DATA_DIRECTORY_TARGET = str(project_root / 'data/Cityscapes')
# number of object classes
cfg.NUM_CLASSES = 19
# exp dirs
cfg.EXP_NAME = ''
cfg.EXP_ROOT = project_root / ''
cfg.EXP_ROOT_SNAPSHOT = osp.join(cfg.EXP_ROOT, 'snapshots')
cfg.PS_ROOT_SNAPSHOT = osp.join(project_root, '')
cfg.EXP_ROOT_LOGS = osp.join(cfg.EXP_ROOT, 'logs')
# CUDA
cfg.GPU_ID = 0

## train configs ##
cfg.TRAIN = EasyDict()
cfg.TRAIN.SET_SOURCE = 'train'
cfg.TRAIN.SET_TARGET = 'train'
cfg.TRAIN.BATCH_SIZE_SOURCE = 1
cfg.TRAIN.BATCH_SIZE_TARGET = 1
cfg.TRAIN.IGNORE_LABEL = 255
cfg.TRAIN.INPUT_SIZE_SOURCE = (1280, 720)
cfg.TRAIN.INPUT_SIZE_TARGET = (1024, 512)
# class info
cfg.TRAIN.INFO_SOURCE = ''
cfg.TRAIN.INFO_TARGET = str(project_root / 'data/cityscapes_list/info.json')
# segmentation network params
cfg.TRAIN.MODEL = 'DeepLabv2'
cfg.TRAIN.MULTI_LEVEL = True
cfg.TRAIN.RESTORE_FROM = ''
cfg.TRAIN.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TRAIN.LEARNING_RATE = 2.5e-4
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 0.0005
cfg.TRAIN.POWER = 0.9
cfg.TRAIN.LAMBDA_SEG_MAIN = 1.0
cfg.TRAIN.LAMBDA_SEG_AUX = 0.1  # weight of conv4 prediction. Used in multi-level setting.
# domain adaptation
cfg.TRAIN.DA_METHOD = 'AdvEnt'
# adversarial training params
cfg.TRAIN.LEARNING_RATE_D = 1e-4
cfg.TRAIN.LAMBDA_ADV_MAIN = 0.0
cfg.TRAIN.LAMBDA_ADV_AUX = 0.0
# other params
cfg.TRAIN.MAX_ITERS = 250000
cfg.TRAIN.EARLY_STOP = 250000
cfg.TRAIN.SAVE_PRED_EVERY = 1
cfg.TRAIN.SNAPSHOT_DIR = ''
cfg.TRAIN.RANDOM_SEED = 1234
cfg.TRAIN.TENSORBOARD_LOGDIR = ''
cfg.TRAIN.TENSORBOARD_VIZRATE = 100

## test configs ##
cfg.TEST = EasyDict()
# model
cfg.TEST.MODEL = ('DeepLabv2','DeepLabv2', 'DeepLabv2') # set it to ('DeepLabv2',) to evaluate a single model
cfg.TEST.MODEL_WEIGHT = (1.0/3, 1.0/3, 1.0/3) # set it to (1,) to evaluate a single model
cfg.TEST.MULTI_LEVEL = (True, True, True) # set it to (True,) to evaluate a single model
cfg.TEST.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TEST.STORE_PS_LABELS = ''
cfg.TEST.SNAPSHOT_DIR = ('',) 
# test sets
cfg.TEST.SET_TARGET = 'val' # set to train to generate pseudolabels for the target domain
cfg.TEST.BATCH_SIZE_TARGET = 1
cfg.TEST.INPUT_SIZE_TARGET = (1024, 512)
cfg.TEST.OUTPUT_SIZE_TARGET = (2048, 1024)
cfg.TEST.INFO_TARGET = str(project_root / 'data/cityscapes_list/info.json')
cfg.TEST.WAIT_MODEL = False

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not EasyDict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        # if not b.has_key(k):
        if k not in b:
            raise KeyError(f'{k} is not a valid config key')

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(f'Type mismatch ({type(b[k])} vs. {type(v)}) '
                                 f'for config key: {k}')

        # recursively merge dicts
        if type(v) is EasyDict:
            try:
                _merge_a_into_b(a[k], b[k])
            except Exception:
                print(f'Error under config key: {k}')
                raise
        else:
            b[k] = v


def cfg_from_file(filename, args):
    """Load a config file and merge it into the default options.
    """
    yaml_cfg = EasyDict(yaml_load(filename))
    # directories
    if args.source_domain == 'gta':
        cfg.DATA_LIST_SOURCE = str(args.project_root + '/data/gta5_list/{}.txt')
        cfg.DATA_DIRECTORY_SOURCE = str(args.data_root + '/gta')
    else:
        cfg.SOURCE = 'SYNTHIA'
        yaml_cfg.SOURCE = 'SYNTHIA'
        cfg.DATA_LIST_SOURCE = str(args.project_root + '/data/synthia_list/{}.txt')
        cfg.DATA_DIRECTORY_SOURCE = str(args.data_root + '/rand_cityscapes')
        cfg.TRAIN.INPUT_SIZE_SOURCE = (1280, 760)
        cfg.NUM_CLASSES = args.num_classes
        cfg.TRAIN.MAX_ITERS = 90000
        cfg.TRAIN.EARLY_STOP = 90000
        cfg.TEST.SNAPSHOT_MAXITER = 90000

    cfg.DATA_DIRECTORY_TARGET = str(args.data_root + '/cityscapes')
    cfg.TRAIN.RESTORE_FROM = args.train_restore_from
    
    cfg.TEST.STORE_PS_LABELS = args.pseudo_labels_path
    cfg.TEST.SNAPSHOT_DIR = args.test_snapshot
    if args.num_classes == 19:
        cfg.TRAIN.INFO_TARGET = str(args.project_root + '/data/cityscapes_list/info.json')
        cfg.TEST.INFO_TARGET = str(args.project_root + '/data/cityscapes_list/info.json')
    else:
        cfg.TRAIN.INFO_TARGET = str(args.project_root + '/data/cityscapes_list/info16class.json')
        cfg.TEST.INFO_TARGET = str(args.project_root + '/data/cityscapes_list/info16class.json')
    cfg.TRAIN.BATCH_SIZE_SOURCE = args.batch_size
    cfg.TRAIN.BATCH_SIZE_TARGET = args.batch_size
    cfg.TRAIN.RANDOM_SEED = args.seed

    _merge_a_into_b(yaml_cfg, cfg)
