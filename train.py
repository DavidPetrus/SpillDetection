import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as F_vis
import glob
import datetime
import random

from dataloader import CustomDataGen
from spill_model import SpillDetector
import clip

import wandb

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('exp','test','')
flags.DEFINE_integer('num_workers',16,'')
flags.DEFINE_integer('batch_size',8,'')
flags.DEFINE_integer('epochs',20,'')
flags.DEFINE_integer('val_batch',10,'')
flags.DEFINE_float('lr',0.01,'')
flags.DEFINE_float('temperature',0.06,'')

flags.DEFINE_string('clip_model','ViT-B/16','')
flags.DEFINE_integer('num_prototypes',30,'')
flags.DEFINE_integer('top_k_spill',1,'')
flags.DEFINE_integer('top_k_vids',1,'')
flags.DEFINE_float('margin',0.025,'')
flags.DEFINE_float('puddle_coeff',0.3,'')
flags.DEFINE_float('vid_coeff',0.5,'')
flags.DEFINE_string('scale','xlarge','')

# Color Augmentations
flags.DEFINE_bool('hue_pos',False,'')
flags.DEFINE_bool('hue_neg',False,'')
flags.DEFINE_bool('hue_full',False,'')
flags.DEFINE_bool('gamma_dark',False,'')
flags.DEFINE_bool('gamma_light',False,'')
flags.DEFINE_bool('invert',False,'')
flags.DEFINE_bool('posterize',False,'')

# Superimpose
flags.DEFINE_float('min_alpha',140,'')
flags.DEFINE_float('max_alpha',220,'')
flags.DEFINE_float('min_spill_frac',0.4,'')
flags.DEFINE_float('max_spill_frac',1.,'')
flags.DEFINE_float('superimpose_frac',0.65,'')

def main(argv):

    wandb.init(project="SpillDetection",name=FLAGS.exp)
    wandb.config.update(flags.FLAGS)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    TRAIN_IMAGES_PATH = "./images/train"
    VAL_IMAGES_PATH = "./images/val"
    NUMBER_OF_TRAINING_IMAGES = len(glob.glob(TRAIN_IMAGES_PATH+"/spills/*"))
    NUMBER_OF_VALIDATION_IMAGES = len(glob.glob(VAL_IMAGES_PATH+"/spills/*"))
    print("Num train:",NUMBER_OF_TRAINING_IMAGES)
    print("Num val:",NUMBER_OF_VALIDATION_IMAGES)
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs

    clip_model, preprocess = clip.load(FLAGS.clip_model,device='cuda')

    color_distorts = [None]
    if FLAGS.hue_pos:
        color_distorts.append(lambda x: F_vis.adjust_hue(x,0.25))
    if FLAGS.hue_neg:
        color_distorts.append(lambda x: F_vis.adjust_hue(x,-0.25))
    if FLAGS.hue_full:
        color_distorts.append(lambda x: F_vis.adjust_hue(x,0.5))
    if FLAGS.invert:
        color_distorts.append(lambda x: F_vis.invert(x))
    if FLAGS.gamma_light:
        color_distorts.append(lambda x: F_vis.adjust_gamma(x,0.5))
    if FLAGS.gamma_dark:
        color_distorts.append(lambda x: F_vis.adjust_gamma(x,2))
    if FLAGS.posterize:
        color_distorts.append(lambda x: F_vis.posterize(x,2))
        

    training_set = CustomDataGen(TRAIN_IMAGES_PATH, batch_size, preprocess, train=True, color_distorts=color_distorts)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=None, shuffle=False, num_workers=FLAGS.num_workers, pin_memory=True)
    validation_set = CustomDataGen(VAL_IMAGES_PATH, batch_size, preprocess, train=False, color_distorts=color_distorts)
    validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=None, shuffle=False, num_workers=2, pin_memory=True)

    spill_det = SpillDetector(clip_model)
    spill_det.to('cuda')

    #spill_det.load_state_dict(torch.load('weights/20Sep3.pt',map_location=torch.device('cuda')))

    optimizer = torch.optim.Adam(params=[spill_det.prototypes], lr=FLAGS.lr)

    color_aug = torchvision.transforms.ColorJitter(0.6,0.6,0.6,0.1)

    num_distorts = len(color_distorts)

    if FLAGS.scale == 'all':
        num_vid_patches = 30*num_distorts
        num_spill_patches = 10*num_distorts
        num_puddle_patches = 10*num_distorts
    elif FLAGS.scale == 'small':
        num_vid_patches = 30*num_distorts
        num_spill_patches = 10*num_distorts
        num_puddle_patches = 10*num_distorts
    elif FLAGS.scale == 'large':
        num_vid_patches = 8*num_distorts
        num_spill_patches = 1*num_distorts
        num_puddle_patches = 3*num_distorts
    elif FLAGS.scale == 'xlarge':
        num_vid_patches = 2*num_distorts
        num_spill_patches = 1*num_distorts
        num_puddle_patches = 1*num_distorts
    elif FLAGS.scale == 'med':
        num_vid_patches = 3*num_distorts
        num_spill_patches = 2*num_distorts
        num_puddle_patches = 2*num_distorts

    lab = torch.zeros(60,dtype=torch.int64).to('cuda')
    vid_mask = torch.zeros(4,num_vid_patches,FLAGS.num_prototypes).to('cuda')
    spill_mask = torch.zeros(8,num_spill_patches,FLAGS.num_prototypes).to('cuda')

    min_acc = 0.
    total_loss = 0.
    step_loss = 0.
    train_iter = 0
    val_iter = 0
    for epoch in range(epochs):
        optimizer.zero_grad()
        for data in training_generator:
            img_patches,_ = data

            img_patches = color_aug(img_patches.to('cuda'))
            #img_patches = img_patches.to('cuda')

            sims = spill_det(img_patches)
            pos_sims_vids = sims[:4*num_vid_patches].reshape(4,num_vid_patches,FLAGS.num_prototypes)
            pos_sims_spills = sims[4*num_vid_patches:4*num_vid_patches + 8*num_spill_patches].reshape(8,num_spill_patches,FLAGS.num_prototypes)
            pos_sims_puddles = sims[4*num_vid_patches + 8*num_spill_patches:4*num_vid_patches + 8*num_spill_patches + 4*num_puddle_patches].reshape(4,num_puddle_patches,FLAGS.num_prototypes)
            neg_sims = sims[4*num_vid_patches + 8*num_spill_patches + 4*num_puddle_patches:].max(dim=1)[0].reshape(1,-1).tile(4,1)

            # Compute vid loss
            logits = []
            top_mask = vid_mask.clone()
            for k in range(FLAGS.top_k_vids):
                p_sim = (pos_sims_vids+top_mask).max(dim=2,keepdim=True)[0].max(dim=1,keepdim=True)[0]
                logits.append(torch.cat([p_sim[:,:,0]-FLAGS.margin,neg_sims],dim=1)/FLAGS.temperature)
                top_mask = top_mask.clone()
                top_mask[pos_sims_vids == p_sim] = -10.

            logits = torch.cat(logits,dim=0)
            vid_loss = F.cross_entropy(logits,lab[:4*FLAGS.top_k_vids])
            vid_acc = (torch.argmax(logits,dim=1)==0).float().mean()

            # Compute puddle loss
            logits = []
            p_sim = pos_sims_puddles.max(dim=2,keepdim=True)[0].max(dim=1,keepdim=True)[0]
            logits.append(torch.cat([p_sim[:,:,0]-FLAGS.margin,neg_sims],dim=1)/FLAGS.temperature)

            logits = torch.cat(logits,dim=0)
            puddle_loss = F.cross_entropy(logits,lab[:4])
            puddle_acc = (torch.argmax(logits,dim=1)==0).float().mean()

            # Compute spill loss
            neg_sims = neg_sims.tile(2,1)
            logits = []
            top_mask = spill_mask.clone()
            for k in range(FLAGS.top_k_spill):
                p_sim = (pos_sims_spills+top_mask).max(dim=2,keepdim=True)[0].max(dim=1,keepdim=True)[0]
                logits.append(torch.cat([p_sim[:,:,0]-FLAGS.margin,neg_sims],dim=1)/FLAGS.temperature)
                top_mask = top_mask.clone()
                top_mask[pos_sims_spills == p_sim] = -10.

            logits = torch.cat(logits,dim=0)
            spill_loss = F.cross_entropy(logits,lab[:8*FLAGS.top_k_spill])
            spill_acc = (torch.argmax(logits,dim=1)==0).float().mean()

            final_loss = spill_loss + FLAGS.vid_coeff*vid_loss + FLAGS.puddle_coeff*puddle_loss

            train_iter += 1
            log_dict = {"Epoch":epoch, "Train Iteration":train_iter, "Final Loss": final_loss, \
                        "Video Loss": vid_loss, "Spill Loss":spill_loss, "Puddle Loss":puddle_loss, \
                        "Video Accuracy": vid_acc, "Spill Accuracy": spill_acc, "Puddle Accuracy": puddle_acc}

            final_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if train_iter % 10 == 0:
                print(log_dict)

            wandb.log(log_dict)

            '''if train_iter == 300:
                for g in optimizer.param_groups:
                    g['lr'] = 0.01'''

            if train_iter == 500:
                for g in optimizer.param_groups:
                    g['lr'] = 0.001

        val_count = 0
        loss_2x3 = loss_3x5 = loss_4x7 = acc_clear_2x3 = acc_clear_3x5 = acc_clear_4x7 = acc_dark_2x3 = acc_dark_3x5 = acc_dark_4x7 = \
        acc_opaque_2x3 = acc_opaque_3x5 = acc_opaque_4x7 = 0.
        for data in validation_generator:
            with torch.no_grad():
                img_patches = data

                sims = spill_det(img_patches.to('cuda'))
                pos_sims = sims[:(10+3+5)*3*num_distorts].reshape(10+3+5,3,FLAGS.num_prototypes*num_distorts).max(dim=2)[0]
                neg_sims = sims[(10+3+5)*3*num_distorts:].reshape(FLAGS.val_batch,8+23+46,FLAGS.num_prototypes*num_distorts).max(dim=2)[0]

                sims_2x3 = torch.cat([pos_sims[:,0:1],neg_sims[:,:8].reshape(1,FLAGS.val_batch*8).tile(10+3+5,1)],dim=1)
                sims_3x5 = torch.cat([pos_sims[:,1:2],neg_sims[:,8:8+23].reshape(1,FLAGS.val_batch*23).tile(10+3+5,1)],dim=1)
                sims_4x7 = torch.cat([pos_sims[:,2:3],neg_sims[:,8+23:8+23+46].reshape(1,FLAGS.val_batch*46).tile(10+3+5,1)],dim=1)

                loss_2x3 += F.cross_entropy(sims_2x3/FLAGS.temperature,lab[:10+3+5])
                loss_3x5 += F.cross_entropy(sims_3x5/FLAGS.temperature,lab[:10+3+5])
                loss_4x7 += F.cross_entropy(sims_4x7/FLAGS.temperature,lab[:10+3+5])

                acc_clear_2x3 += (torch.argmax(sims_2x3[:10],dim=1)==0).float().mean()
                acc_clear_3x5 += (torch.argmax(sims_3x5[:10],dim=1)==0).float().mean()
                acc_clear_4x7 += (torch.argmax(sims_4x7[:10],dim=1)==0).float().mean()

                acc_dark_2x3 += (torch.argmax(sims_2x3[10:10+3],dim=1)==0).float().mean()
                acc_dark_3x5 += (torch.argmax(sims_3x5[10:10+3],dim=1)==0).float().mean()
                acc_dark_4x7 += (torch.argmax(sims_4x7[10:10+3],dim=1)==0).float().mean()

                acc_opaque_2x3 += (torch.argmax(sims_2x3[10+3:10+3+5],dim=1)==0).float().mean()
                acc_opaque_3x5 += (torch.argmax(sims_3x5[10+3:10+3+5],dim=1)==0).float().mean()
                acc_opaque_4x7 += (torch.argmax(sims_4x7[10+3:10+3+5],dim=1)==0).float().mean()

                val_count += 1

        log_dict = {"Epoch":epoch}
        log_dict["Val_loss_2x3"] = loss_2x3/val_count
        log_dict["Val_loss_3x5"] = loss_3x5/val_count
        log_dict["Val_loss_4x7"] = loss_4x7/val_count
        log_dict["Val_acc_clear_2x3"] = acc_clear_2x3/val_count
        log_dict["Val_acc_clear_3x5"] = acc_clear_3x5/val_count
        log_dict["Val_acc_clear_4x7"] = acc_clear_4x7/val_count
        log_dict["Val_acc_dark_2x3"] = acc_dark_2x3/val_count
        log_dict["Val_acc_dark_3x5"] = acc_dark_3x5/val_count
        log_dict["Val_acc_dark_4x7"] = acc_dark_4x7/val_count
        log_dict["Val_acc_opaque_2x3"] = acc_opaque_2x3/val_count
        log_dict["Val_acc_opaque_3x5"] = acc_opaque_3x5/val_count
        log_dict["Val_acc_opaque_4x7"] = acc_opaque_4x7/val_count

        wandb.log(log_dict)

        val_accs = [acc_clear_2x3,acc_clear_3x5,acc_dark_2x3,acc_dark_3x5,acc_opaque_2x3,acc_opaque_3x5]

        if sum(val_accs) > min_acc:
            torch.save({'prototypes': spill_det.prototypes},'weights/{}.pt'.format(FLAGS.exp))
            min_acc = sum(val_accs)


if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force=True)
    app.run(main)