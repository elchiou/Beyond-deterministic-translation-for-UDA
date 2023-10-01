import os.path as osp
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from advent.utils.func import per_class_iu, fast_hist
from advent.utils.serialization import pickle_dump, pickle_load
import xlsxwriter
from openpyxl import load_workbook
import pickle
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from munit.utils import vgg_preprocess
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image

plt.ioff()
import os

def evaluate_domain_adaptation( models, test_loader, cfg, args, model_transl, cfg_transl,
                                fixed_test_size=True,
                                verbose=True):
    device = torch.device('cuda:{}'.format(cfg.GPU_ID)) # get device name: CPU or GPU
    interp = None
    if fixed_test_size:
        interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]), mode='bilinear', align_corners=True)
    # eval
    if cfg.TEST.MODE == 'single':
        eval_single(cfg, models,
                    device, test_loader, interp, fixed_test_size,
                    verbose, args, model_transl, cfg_transl)
    elif cfg.TEST.MODE == 'best':
        eval_best(cfg, models,
                  device, test_loader, interp, fixed_test_size,
                  verbose, args, model_transl, cfg_transl)
    else:
        raise NotImplementedError(f"Not yet supported test mode {cfg.TEST.MODE}")

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)
def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def eval_single(cfg, models,
                device, test_loader, interp,
                fixed_test_size, verbose, args, model_transl, cfg_transl):
    img_mean = torch.from_numpy(cfg.TRAIN.IMG_MEAN).to(device)
    assert len(cfg.TEST.SNAPSHOT_DIR) == len(models), 'Number of models are not matched'
    for checkpoint, model in zip(cfg.TEST.SNAPSHOT_DIR, models):
        load_checkpoint_for_evaluation(model, checkpoint, device)
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]
    list_all = []
    list_mIoU = []
    list_mIoU_16 = []
    list_mIoU_13 = []
    results_dir = osp.join(cfg.TEST.RESTORE_FROM[0], 'results_single')
    os.makedirs(results_dir, exist_ok=True)
   
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    inters_over_union_classes_im = []
    names_all = []
    for index, batch in tqdm(enumerate(test_loader)):
        image, _, label, _, name = batch
        name = name[0].split('/')[1].split('.')[0]
        if not fixed_test_size:
            interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
        with torch.no_grad():
            output = None
            for model, model_weight in zip(models, cfg.TEST.MODEL_WEIGHT):
                pred_main = model(image.to(device))[1]
                output_ = interp(pred_main).cpu().data[0].numpy()
                if output is None:
                    output = model_weight * output_
                else:
                    output += model_weight * output_
            assert output is not None, 'Output is None'
            output = output.transpose(1, 2, 0)
            output = np.argmax(output, axis=2)
        label = label.numpy()[0]
    
        #  mask: numpy array of the mask
        output_nomask = np.asarray(utput.copy(), dtype=np.uint8)
        output_col = colorize_mask(output_nomask)
        #output_nomask.save('%s/%s' % (args.save, name))
        output_col.save('%s/%s.png' % (results_dir, name))
        hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
       
    inters_over_union_classes = per_class_iu(hist)
    print(f'mIoU = \t{round(np.nanmean(inters_over_union_classes) * 100, 2)}')
    if verbose:
        #display_stats(cfg, test_loader.dataset.class_names, inters_over_union_classes)
        list_all, list_mIoU, list_mIoU_16, list_mIoU_13 = display_stats(cfg, test_loader.dataset.class_names,
                                                                        inters_over_union_classes, list_all, list_mIoU,
                                                                        list_mIoU_16, list_mIoU_13, i_iter='single')

        open_file = open(results_dir + '/res_all.pkl', "wb")
        pickle.dump(list_all, open_file)
        open_file.close()

        open_file = open(results_dir + '/res_mIoU.pkl', "wb")
        pickle.dump(list_mIoU, open_file)
        open_file.close()

        open_file = open(results_dir + '/res_mIoU_16.pkl', "wb")
        pickle.dump(list_mIoU_16, open_file)
        open_file.close()

        open_file = open(results_dir + '/res_mIoU_13.pkl', "wb")
        pickle.dump(list_mIoU_13, open_file)
        open_file.close()

def eval_best(cfg, models,
              device, test_loader, interp,
              fixed_test_size, verbose, args, model_transl, cfg_transl):
    img_mean = torch.from_numpy(cfg.TRAIN.IMG_MEAN).to(device)
    assert len(models) == 1, 'Not yet supported multi models in this mode'
    assert osp.exists(cfg.TEST.SNAPSHOT_DIR[0]), 'SNAPSHOT_DIR is not found'
    start_iter = cfg.TEST.START_ITER
    step = cfg.TEST.SNAPSHOT_STEP
    max_iter = cfg.TEST.SNAPSHOT_MAXITER   
    cur_best_miou = -1
    cur_best_model = ''
    list_all = []
    list_mIoU = []
    list_mIoU_16 = []
    list_mIoU_13 = []
    if args.use_synth_t2s:
        if args.new:
            results_dir = osp.join(cfg.TEST.SNAPSHOT_DIR[0], 'results_t2s_munit_conf_' + str(args.num_fake_source))
        else:
            results_dir = osp.join(cfg.TEST.SNAPSHOT_DIR[0], 'results_t2s_' + str(args.num_fake_source))
        cache_path = osp.join(results_dir, 'all_res.pkl')
        if osp.exists(cache_path):
            all_res = pickle_load(cache_path)
        else:
            all_res = {}
    elif args.use_synth_t2t:
        if args.new:
            results_dir = osp.join(cfg.TEST.SNAPSHOT_DIR[0], 'results_t2t_munit_conf_' + str(args.num_fake_source))
        else:
            results_dir = osp.join(cfg.TEST.SNAPSHOT_DIR[0], 'results_t2t_' + str(args.num_fake_source))        
        cache_path = osp.join(results_dir, 'all_res.pkl')
        if osp.exists(cache_path):
            all_res = pickle_load(cache_path)
        else:
            all_res = {}
    else:
        results_dir = osp.join(cfg.TEST.SNAPSHOT_DIR[0], 'results')
        cache_path = osp.join(results_dir, 'all_res.pkl')
        if osp.exists(cache_path):
            all_res = pickle_load(cache_path)
        else:
            all_res = {}
    os.makedirs(results_dir, exist_ok=True)
    for i_iter in range(start_iter, max_iter + 1, step):
        restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'model_{i_iter}.pth')
        if not osp.exists(restore_from):
            # continue
            if cfg.TEST.WAIT_MODEL:
                print('Waiting for model..!')
                while not osp.exists(restore_from):
                    time.sleep(5)
        print("Evaluating model", restore_from)
        if i_iter not in all_res.keys():
            load_checkpoint_for_evaluation(models[0], restore_from, device)
            # eval
            hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
            # for index, batch in enumerate(test_loader):
            #     image, _, _, name = batch
            test_iter = iter(test_loader)
            for index in tqdm(range(len(test_loader))):
                image, image_min_max, label, _, name = next(test_iter)
                if not fixed_test_size:
                    interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
                with torch.no_grad():
                    if args.use_synth_t2s:
                        output_sum = 0
                        c_b, _ = model_transl.gen_b.encode(image_min_max.to(device))
                        for _ in range(args.num_fake_source):
                            s_a = Variable(torch.randn(image_min_max.size(0), cfg_transl['gen']['style_dim'], 1, 1).to(device))
                            image = model_transl.gen_a.decode(c_b.detach(), s_a.detach()).detach()
                            image = vgg_preprocess(image, img_mean).detach()

                            pred_main = models[0](image.to(device))[1]
                            output_ = interp(pred_main.detach())
                            output_ = F.softmax(output_).cpu().data[0].numpy()
                            output_ = output_.transpose(1, 2, 0)
                            output_sum = output_sum + output_
                        output_sum = output_sum / args.num_fake_source
                        output = np.argmax(output_sum, axis=2)
                    else:
                        pred_main = models[0](image.to(device))[1]
                        output = interp(pred_main)
                        output = F.softmax(output).cpu().data[0].numpy()
                        output = output.transpose(1, 2, 0)
                        if args.use_synth_t2t:
                            c_b, _ = model_transl.gen_b.encode(image_min_max.to(device))
                            for _ in range(args.num_fake_target):
                                s_b = Variable(torch.randn(image_min_max.size(0), cfg_transl['gen']['style_dim'], 1, 1).to(device))
                                image = model_transl.gen_b.decode(c_b.detach(), s_b.detach()).detach()
                                image = vgg_preprocess(image, img_mean).detach()
                            
                                pred_main = models[0](image.to(device))[1]
                                output_ = interp(pred_main)
                                output_ = F.softmax(output_).cpu().data[0].numpy()
                                output_ = output_.transpose(1, 2, 0)
                                output = output + output_
                            output = output / (args.num_fake_target + 1)
                        output = np.argmax(output, axis=2)
                label = label.numpy()[0]
                hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
                if verbose and index > 0 and index % 100 == 0:
                    print('{:d} / {:d}: {:0.2f}'.format(
                        index, len(test_loader), 100 * np.nanmean(per_class_iu(hist))))
            inters_over_union_classes = per_class_iu(hist)
            all_res[i_iter] = inters_over_union_classes
            pickle_dump(all_res, cache_path)
        else:
            inters_over_union_classes = all_res[i_iter]
        computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
        if cur_best_miou < computed_miou:
            cur_best_miou = computed_miou
            cur_best_model = restore_from
        print('\tCurrent mIoU:', computed_miou)
        print('\tCurrent best model:', cur_best_model)
        print('\tCurrent best mIoU:', cur_best_miou)
        if verbose:
            list_all, list_mIoU, list_mIoU_16, list_mIoU_13 = display_stats(cfg, test_loader.dataset.class_names,
                                                                            inters_over_union_classes, list_all,
                                                                            list_mIoU,
                                                                            list_mIoU_16, list_mIoU_13, i_iter=i_iter)

            open_file = open(results_dir + '/res_all.pkl', "wb")
            pickle.dump(list_all, open_file)
            open_file.close()

            open_file = open(results_dir + '/res_mIoU.pkl', "wb")
            pickle.dump(list_mIoU, open_file)
            open_file.close()

            open_file = open(results_dir + '/res_mIoU_16.pkl', "wb")
            pickle.dump(list_mIoU_16, open_file)
            open_file.close()

            open_file = open(results_dir + '/res_mIoU_13.pkl', "wb")
            pickle.dump(list_mIoU_13, open_file)
            open_file.close()


def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    saved_state_dict = {k.replace('module.', ''): v for k, v in saved_state_dict.items()}
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.to(device)


def display_stats(cfg, name_classes, inters_over_union_classes, list_all, list_mIoU, list_mIoU_16, list_mIoU_13, i_iter):
    rows = [i_iter]
    IoU_16 = []
    IoU_13 = []

    class_16 = ["road", "sidewalk", "building", "wall", "fence", "pole", "light", "sign", "vegetation", "sky", "person", "rider", "car", "bus", "motocycle", "bicycle"]
    class_13 = ["road", "sidewalk", "building", "light", "sign", "vegetation", "sky", "person", "rider", "car", "bus", "motocycle", "bicycle"]
    for ind_class in range(cfg.NUM_CLASSES):
        print(name_classes[ind_class]
              + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)))
        rows.append(str(round(inters_over_union_classes[ind_class] * 100, 2)))

        if name_classes[ind_class] in class_16:
            IoU_16.append(inters_over_union_classes[ind_class])
        if name_classes[ind_class] in class_13:
            IoU_13.append(inters_over_union_classes[ind_class])

    rows.append(round(np.nanmean(inters_over_union_classes) * 100, 2))
    rows.append(round(np.nanmean(IoU_16) * 100, 2))
    rows.append(round(np.nanmean(IoU_13) * 100, 2))

    list_all.append(rows)
    list_mIoU.append(rows[-3])
    list_mIoU_16.append(rows[-2])
    list_mIoU_13.append(rows[-1])
    

    return list_all, list_mIoU, list_mIoU_16, list_mIoU_13
