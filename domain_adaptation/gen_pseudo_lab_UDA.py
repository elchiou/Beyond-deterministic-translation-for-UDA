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
import pandas
import os
plt.ioff()
import os
from PIL import Image

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

def gen_pseudo_lab_ens( models, loader, cfg, args, model_transl, cfg_transl,
                                fixed_ps_size=True,
                                verbose=True):
    device = torch.device('cuda:{}'.format(cfg.GPU_ID)) # get device name: CPU or GPU
    interp = None
    if fixed_ps_size:
        interp = nn.Upsample(size=(cfg.PS.OUTPUT_SIZE_TARGET[1], cfg.PS.OUTPUT_SIZE_TARGET[0]), mode='bilinear', align_corners=True)
    # eval
    if cfg.PS.MODE == 'single':
        ps_single(cfg, models,
                    device, loader, interp, fixed_ps_size,
                    verbose, args, model_transl, cfg_transl)
    else:
        raise NotImplementedError(f"Not yet supported ps mode {cfg.PS.MODE}")


def ps_single(cfg, models,
                device, loader, interp,
                fixed_ps_size, verbose, args, model_transl, cfg_transl):
    img_mean = torch.from_numpy(cfg.PS.IMG_MEAN).to(device)
    assert len(cfg.PS.RESTORE_FROM) == len(models), 'Number of models are not matched'
    for checkpoint, model in zip(cfg.PS.RESTORE_FROM, models):
        load_checkpoint_for_evaluation(model, checkpoint, device)
 
    results_dir = osp.join(cfg.PS.SNAPSHOT_DIR[0], 'results_ps')
    os.makedirs(results_dir, exist_ok=True)

 
    predicted_label = np.zeros((len(loader), 1024, 2048), dtype=int)
    predicted_prob = np.zeros((len(loader), 1024, 2048))
    for index, batch in tqdm(enumerate(loader)):
        image, image_min_max, label, _, name = batch
        if not fixed_ps_size:
            interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
        with torch.no_grad():
            output = None
            for mod_id, (model, model_weight) in enumerate(zip(models, cfg.PS.MODEL_WEIGHT)):
                if mod_id == 0 and args.use_synth_t2s:
                    c_b, _ = model_transl.gen_b.encode(image_min_max.to(device))
                    output_sum = 0
                    for nf_id in range(args.num_fake_source):
                        s_a = Variable(torch.randn(image_min_max.size(0), cfg_transl['gen']['style_dim'], 1, 1).to(device))
                        image_t2s = model_transl.gen_a.decode(c_b.detach(), s_a.detach()).detach()
                        image_t2s = vgg_preprocess(image_t2s, img_mean).detach()

                        pred_main = model(image_t2s.to(device))[1]
                        output_ = interp(pred_main.detach())
                        output_ = F.softmax(output_).cpu().data[0].numpy()
                        output_sum = output_sum + output_
                       
                    output = output_sum / args.num_fake_source
                else:
                    pred_main = model(image.to(device))[1]
                    output_ = interp(pred_main.detach())
                    output_ = F.softmax(output_).cpu().data[0].numpy()
                              
                if output is None:
                    output = model_weight * output_
                else:
                    output += model_weight * output_
            assert output is not None, 'Output is None'
            output = output.transpose(1, 2, 0)
            pred_lab = np.argmax(output, axis=2)
            prob = np.max(output, axis=2)
            predicted_label[index] = pred_lab.copy()
            predicted_prob[index] = prob.copy()
            image_name.append(name[0])

        save_path_col = osp.join(cfg.PS_ROOT_SNAPSHOT, 'aver_col_pseudo_labels_im_' + str(args.n_of_train_im) + '_trains',
                                 cfg.PS.SET_TARGET + '_prop_' + str(0) + '_' + args.source_domain)
        if not os.path.isdir(save_path_col):
            os.makedirs(save_path_col)
        output_nomask = np.asarray(pred_lab, dtype=np.uint8)
        output_col = colorize_mask(output_nomask)
        #output_nomask.save('%s/%s' % (args.save, name))
        output_col.save('%s/%s' % (save_path_col, name[0].split('/')[-1]))

        label = label.numpy()[0]

       
    
  checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.to(device)


