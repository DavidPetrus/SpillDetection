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
flags.DEFINE_integer('epochs',1000,'')
flags.DEFINE_float('lr',0.01,'')
flags.DEFINE_float('temperature',0.1,'')

flags.DEFINE_string('clip_model','ViT-B/32','')
flags.DEFINE_integer('proj_head',0,'')
flags.DEFINE_integer('num_prototypes',20,'')
flags.DEFINE_integer('top_k_spill',3,'')
flags.DEFINE_integer('top_k_vids',6,'')

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

    optimizer = torch.optim.Adam(params=[spill_det.prototypes], lr=FLAGS.lr)

    color_aug = torchvision.transforms.ColorJitter(0.4,0.4,0.4,0.1)

    lab = torch.zeros(8*FLAGS.top_k_spill,dtype=torch.int64).to('cuda')
    vid_mask = torch.ones(4,30,FLAGS.num_prototypes).to('cuda')
    spill_mask = torch.ones(8,10,FLAGS.num_prototypes).to('cuda')

    min_loss = 100.
    total_loss = 0.
    step_loss = 0.
    train_iter = 0
    val_iter = 0
    for epoch in range(10000):
        optimizer.zero_grad()
        for data in training_generator:
            img_patches,_ = data

            img_patches = color_aug(img_patches.to('cuda'))
            #img_patches = img_patches.to('cuda')

            sims = spill_det(img_patches)
            pos_sims_vids = sims[:120].reshape(4,30,FLAGS.num_prototypes)
            pos_sims_spills = sims[120:200].reshape(8,10,FLAGS.num_prototypes)
            neg_sims = sims[200:].max(dim=1)[0].reshape(1,-1).tile(4,1)

            logits = []
            top_mask = vid_mask.clone()
            for k in range(FLAGS.top_k_vids):
                p_sim = (pos_sims_vids*top_mask).max(dim=2,keepdim=True)[0].max(dim=1,keepdim=True)[0]
                logits.append(torch.cat([p_sim[:,:,0],neg_sims],dim=1)/FLAGS.temperature)
                top_mask = top_mask.clone()
                top_mask[pos_sims_vids == p_sim] = 0.

            logits = torch.cat(logits,dim=0)
            vid_loss = F.cross_entropy(logits,lab[:4*FLAGS.top_k_vids])
            vid_acc = (torch.argmax(logits,dim=1)==0).float().mean()

            neg_sims = neg_sims.tile(2,1)
            logits = []
            top_mask = spill_mask.clone()
            for k in range(FLAGS.top_k_spill):
                p_sim = (pos_sims_spills*top_mask).max(dim=2,keepdim=True)[0].max(dim=1,keepdim=True)[0]
                logits.append(torch.cat([p_sim[:,:,0],neg_sims],dim=1)/FLAGS.temperature)
                top_mask = top_mask.clone()
                top_mask[pos_sims_spills == p_sim] = 0.

            logits = torch.cat(logits,dim=0)
            spill_loss = F.cross_entropy(logits,lab)
            spill_acc = (torch.argmax(logits,dim=1)==0).float().mean()

            final_loss = vid_loss + spill_loss

            train_iter += 1
            log_dict = {"Epoch":epoch,"Train Iteration":train_iter, "Final Loss": final_loss, "Video Loss": vid_loss, "Spill Loss":spill_loss, \
                        "Video Accuracy": vid_acc, "Spill Accuracy": spill_acc}

            final_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if train_iter % 10 == 0:
                print(log_dict)

            wandb.log(log_dict)

            if train_iter == 300:
                for g in optimizer.param_groups:
                    g['lr'] = 0.001

        val_video_loss = 0.
        val_count = 0
        for data in validation_generator:
            with torch.no_grad():
                img_patches,_ = data

                sims = spill_det(img_patches.to('cuda'))
                pos_sims_vids = sims[:120].reshape(4,30,FLAGS.num_prototypes)
                pos_sims_spills = sims[120:160].reshape(4,10,FLAGS.num_prototypes)
                neg_sims = sims[160:].max(dim=1)[0].reshape(1,-1).tile(4,1)

                logits = []
                top_mask = vid_mask.clone()
                for k in range(4):
                    p_sim = (pos_sims_vids*top_mask).max(dim=2,keepdim=True)[0].max(dim=1,keepdim=True)[0]
                    logits.append(torch.cat([p_sim[:,:,0],neg_sims],dim=1)/FLAGS.temperature)
                    top_mask[pos_sims_vids == p_sim] = 0.

                logits = torch.cat(logits,dim=0)
                vid_loss = F.cross_entropy(logits,lab[:4*4])
                vid_acc = (torch.argmax(logits,dim=1)==0).float().mean()

                #neg_sims = neg_sims.tile(2,1)
                logits = []
                top_mask = spill_mask[:4].clone()
                for k in range(FLAGS.top_k_spill):
                    p_sim = (pos_sims_spills*top_mask).max(dim=2,keepdim=True)[0].max(dim=1,keepdim=True)[0]
                    logits.append(torch.cat([p_sim[:,:,0],neg_sims],dim=1)/FLAGS.temperature)
                    top_mask[pos_sims_spills == p_sim] = 0.

                logits = torch.cat(logits,dim=0)
                spill_loss = F.cross_entropy(logits,lab[:4*FLAGS.top_k_spill])
                spill_acc = (torch.argmax(logits,dim=1)==0).float().mean()

                final_loss = vid_loss + spill_loss

                val_video_loss += vid_loss
                val_count += 1

                val_iter += 1
                log_dict = {"Epoch":epoch,"Val_iter":val_iter,"Val Loss": final_loss, "Val Video Loss": vid_loss, "Val Spill Loss":spill_loss, \
                            "Val Video Accuracy": vid_acc, "Val Spill Accuracy": spill_acc}

                wandb.log(log_dict)

        val_video_loss = val_video_loss/val_count
        print("Val Video Loss",val_video_loss)

        if val_video_loss < min_loss:
            torch.save(spill_det.state_dict(),'weights/{}.pt'.format(FLAGS.exp))


if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force=True)
    app.run(main)