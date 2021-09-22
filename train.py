import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision
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
flags.DEFINE_integer('epochs',100,'')
flags.DEFINE_float('lr',0.01,'')
flags.DEFINE_float('temperature',0.06,'')

flags.DEFINE_string('clip_model','ViT-B/16','')
flags.DEFINE_integer('num_prototypes',30,'')
flags.DEFINE_integer('top_k_spill',1,'')
flags.DEFINE_integer('top_k_vids',1,'')
flags.DEFINE_float('margin',0.025,'')
flags.DEFINE_float('puddle_coeff',0.3,'')
flags.DEFINE_string('scale','all','')

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

    training_set = CustomDataGen(TRAIN_IMAGES_PATH, batch_size, preprocess, train=True)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=None, shuffle=False, num_workers=FLAGS.num_workers, pin_memory=True)
    validation_set = CustomDataGen(VAL_IMAGES_PATH, batch_size, preprocess, train=False)
    validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=None, shuffle=False, num_workers=FLAGS.num_workers, pin_memory=True)

    spill_det = SpillDetector(clip_model)
    spill_det.to('cuda')

    #spill_det.load_state_dict(torch.load('weights/20Sep3.pt',map_location=torch.device('cuda')))

    optimizer = torch.optim.Adam(params=[spill_det.prototypes], lr=FLAGS.lr)

    color_aug = torchvision.transforms.ColorJitter(0.6,0.6,0.6,0.15)

    val_data = ['pool','large_water','small_water','large_other','small_other','spill']
    num_vals = [60,30,15,30,15,20]
    val_patches = [5,20,40,20,40,10]

    if FLAGS.scale == 'all':
        num_vid_patches = 30
        num_spill_patches = 10
        num_puddle_patches = 10
    elif FLAGS.scale == 'small':
        num_vid_patches = 30
        num_spill_patches = 10
        num_puddle_patches = 10
    elif FLAGS.scale == 'large':
        num_vid_patches = 8
        num_spill_patches = 1
        num_puddle_patches = 3
    elif FLAGS.scale == 'xlarge':
        num_vid_patches = 2
        num_spill_patches = 1
        num_puddle_patches = 1
    elif FLAGS.scale == 'med':
        num_vid_patches = 3
        num_spill_patches = 2
        num_puddle_patches = 2

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

            final_loss = vid_loss + spill_loss + FLAGS.puddle_coeff*puddle_loss

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

        val_accs = []
        val_losses = []
        val_count = 0
        for data in validation_generator:
            with torch.no_grad():
                img_patches,_ = data

                sims = spill_det(img_patches.to('cuda'))
                if val_count == 0:
                    pos_sims = sims[:300].reshape(num_vals[val_count],val_patches[val_count],FLAGS.num_prototypes)
                    neg_sims = sims[300:].max(dim=1)[0].reshape(1,-1).tile(num_vals[val_count],1)
                elif val_count < 5:
                    pos_sims = sims[:600].reshape(num_vals[val_count],val_patches[val_count],FLAGS.num_prototypes)
                    neg_sims = sims[600:].max(dim=1)[0].reshape(1,-1).tile(num_vals[val_count],1)
                else:
                    pos_sims = sims[:200].reshape(num_vals[val_count],val_patches[val_count],FLAGS.num_prototypes)
                    neg_sims = sims[200:].max(dim=1)[0].reshape(1,-1).tile(num_vals[val_count],1)

                p_sim = pos_sims.max(dim=2,keepdim=True)[0].max(dim=1,keepdim=True)[0]
                logits = torch.cat([p_sim[:,:,0],neg_sims],dim=1)/FLAGS.temperature

                val_loss = F.cross_entropy(logits,lab[:num_vals[val_count]])
                val_acc = (torch.argmax(logits,dim=1)==0).float().mean()

                val_losses.append(val_loss)
                val_accs.append(val_acc)
                val_count += 1

        log_dict = {"Epoch":epoch,"Val Loss": sum(val_losses)/len(val_losses), "Val Acc": sum(val_accs)/len(val_accs)}
        for v_ix in range(len(val_accs)):
            log_dict["Val loss "+val_data[v_ix]] = val_losses[v_ix]
            log_dict["Val acc "+val_data[v_ix]] = val_accs[v_ix]

        wandb.log(log_dict)

        if sum(val_accs) > min_acc:
            torch.save({'prototypes': spill_det.prototypes},'weights/{}.pt'.format(FLAGS.exp))
            min_acc = sum(val_accs)


if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force=True)
    app.run(main)