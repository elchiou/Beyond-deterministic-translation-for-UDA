import os.path as osp
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from utils.func import per_class_iu, fast_hist
from utils.serialization import pickle_dump, pickle_load
import xlsxwriter
from openpyxl import load_workbook
import pickle
import pandas as pd
from munit.utils import vgg_preprocess
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image

def evaluate_domain_adaptation(models, loader, cfg, args, model_transl, cfg_transl,
                                fixed_test_size=True):
    device = torch.device('cuda:{}'.format(cfg.GPU_ID)) # get device name: CPU or GPU
    interp = None
    if fixed_test_size:
        interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]), mode='bilinear', align_corners=True)
    # eval
    eval_single(cfg, models,
                    device, loader, interp, fixed_test_size,
                    args, model_transl, cfg_transl)


def eval_single(cfg, models,
                device, loader, interp,
                fixed_test_size, args, model_transl, cfg_transl):

    img_mean = torch.from_numpy(cfg.TRAIN.IMG_MEAN).to(device)
    assert len(cfg.TEST.SNAPSHOT_DIR) == len(models), 'Number of models are not matched'
    for checkpoint, model in zip(cfg.TEST.SNAPSHOT_DIR, models):
        load_checkpoint_for_evaluation(model, checkpoint, device)
  
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    inters_over_union_classes_im = []

    # check if generating pseudo-labels is enabled.
    # if true, initialize arrays to store predicted labels and probabilities for each pixel.
    if args.gen_pseudo_labels:
        predicted_label = np.zeros((len(loader), 1024, 2048), dtype=int)
        predicted_prob = np.zeros((len(loader), 1024, 2048))
        image_name = []
    for index, batch in tqdm(enumerate(loader)):
        image, image_min_max, label, _, name = batch
        if not fixed_test_size:
            interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
        with torch.no_grad():
            output = None
            # perform evaluation using the available models
            for mod_id, (model, model_weight) in enumerate(zip(models, cfg.TEST.MODEL_WEIGHT)):
                # translate from target to source when evaluation is performed on the source model
                if mod_id == 0 and args.use_synth_t2s:
                    c_b, _ = model_transl.gen_b.encode(image_min_max.to(device))
                    output_sum = 0
                    # get the average score across args.num_fake_source predictions on synthetic source images
                    for nf_id in range(args.num_fake_source):
                        s_a = Variable(torch.randn(image_min_max.size(0), cfg_transl['gen']['style_dim'], 1, 1).to(device))
                        image_t2s = model_transl.gen_a.decode(c_b.detach(), s_a.detach()).detach()
                        image_t2s = vgg_preprocess(image_t2s, img_mean).detach()

                        pred_main = model(image_t2s.to(device))[1]
                        output_ = interp(pred_main.detach())
                        output_ = F.softmax(output_).cpu().data[0].numpy()
                        output_sum = output_sum + output_
                       
                    output_ = output_sum / args.num_fake_source
                # perform inference on the original target image when evaluation is performed using the target model
                else:
                    pred_main = model(image.to(device))[1]
                    output_ = interp(pred_main.detach())
                    output_ = F.softmax(output_).cpu().data[0].numpy()
                # calculate the average score
                if output is None:
                    output = model_weight * output_
                else:
                    output += model_weight * output_
            assert output is not None, 'Output is None'
            output = output.transpose(1, 2, 0)
            pred_lab = np.argmax(output, axis=2)
            # copy the class prediction and the corresponding probability and image name when pseudolabels are generated 
            if args.gen_pseudo_labels:
                prob = np.max(output, axis=2)
                predicted_label[index] = pred_lab.copy()
                predicted_prob[index] = prob.copy()
                image_name.append(name[0])

        label = label.numpy()[0]
    
        hist += fast_hist(label.flatten(), pred_lab.flatten(), cfg.NUM_CLASSES)
       
    inters_over_union_classes = per_class_iu(hist)
    print(f'mIoU = \t{round(np.nanmean(inters_over_union_classes) * 100, 2)}')
    for ind_class in range(cfg.NUM_CLASSES):
        print(loader.dataset.class_names[ind_class]
              + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)))
    
    if args.gen_pseudo_labels:
        class_balanced_pseudo_labels(predicted_label, predicted_prob, loader, cfg)

def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    saved_state_dict = {k.replace('module.', ''): v for k, v in saved_state_dict.items()}
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.to(device)

def class_balanced_pseudo_labels(predicted_label, predicted_prob, loader, cfg):
    # Proportion of pseudo-labels to retain
    proportion_to_retain = 0.6
    
    # Calculate threshold for each class
    thresholds = []
    for class_index in range(cfg.NUM_CLASSES):
        class_probabilities = predicted_prob[predicted_label == class_index]
        if len(class_probabilities) == 0:
            thresholds.append(0)
            continue
        sorted_probs = np.sort(class_probabilities)
        thresholds.append(sorted_probs[int(np.round(len(sorted_probs) * proportion_to_retain))])

    thresholds = np.array(thresholds)
    thresholds[thresholds > 0.9] = 0.9

    for index in range(len(loader)):
        label = predicted_label[index]
        prob = predicted_prob[index]
        for class_index in range(cfg.NUM_CLASSES):
            label[(prob < thresholds[class_index]) * (label == class_index)] = 19

        label = np.asarray(label, dtype=np.uint8)
        label = Image.fromarray(label)
        name = name.split('/')[-1]

        save_path = cfg.TEST.STORE_PS_LABELS
        os.makedirs(save_path, exist_ok=True)
        label.save(os.path.join(save_path, name))