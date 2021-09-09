import numpy as np
import cv2
import torch
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
flags.DEFINE_integer('num_workers',4,'')
flags.DEFINE_integer('batch_size',8,'')
flags.DEFINE_integer('epochs',1000,'')
flags.DEFINE_float('lr',3*10**-4,'')
flags.DEFINE_float('temperature',0.1,'')

flags.DEFINE_string('clip_model','ViT-B/32','')
flags.DEFINE_integer('num_prototypes',20,'')
flags.DEFINE_integer('top_k',4,'')

def main(argv):

    wandb.init(project="SpillDetection",name=FLAGS.exp)
    wandb.config.update(flags.FLAGS)

    TRAIN_IMAGES_PATH = "./images/train"
    VAL_IMAGES_PATH = "./images/val"
    NUMBER_OF_TRAINING_IMAGES = len(glob.glob(TRAIN_IMAGES_PATH+"/spills/*"))
    NUMBER_OF_VALIDATION_IMAGES = len(glob.glob(VAL_IMAGES_PATH+"/spills/*"))
    print("Num train:",NUMBER_OF_TRAINING_IMAGES)
    print("Num val:",NUMBER_OF_VALIDATION_IMAGES)
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs

    device = 'cuda'
    clip_model, preprocess = clip.load(FLAGS.clip_model,device=device)

    training_set = CustomDataGen(TRAIN_IMAGES_PATH, batch_size, preprocess, train=True)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=None, shuffle=False, num_workers=FLAGS.num_workers)
    validation_set = CustomDataGen(VAL_IMAGES_PATH, batch_size, preprocess, train=False)
    validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=None, shuffle=False, num_workers=FLAGS.num_workers)

    spill_det = SpillDetector(clip_model)
    spill_det.to('cuda')

    optimizer = torch.optim.Adam(params=model.parameters(), lr=FLAGS.lr)

    lab = torch.zeros(8*FLAGS.top_k,dtype=torch.int64).to('cuda')
    vid_mask = torch.ones(4,30,FLAGS.num_prototypes).to('cuda')
    spill_mask = torch.ones(8,10,FLAGS.num_prototypes).to('cuda')

    min_loss = 100.
    total_loss = 0.
    step_loss = 0.
    train_iter = 0
    for epoch in range(10000):
        model.train()
        model.val = False
        optimizer.zero_grad()
        for data in training_generator:
            img_patches,_ = data

            sims = spill_det(img_patches)
            pos_sims_vids = sims[:120].reshape(4,30,FLAGS.num_prototypes)
            pos_sims_spills = sims[120:200].reshape(8,10,FLAGS.num_prototypes)
            neg_sims = sims[200:].reshape(1,-1).tile(4,1)

            logits = []
            top_mask = vid_mask.clone()
            for k in range(FLAGS.top_k):
                p_sim = (pos_sims_vids*top_mask).max(dim=2,keepdim=True)[0].max(dim=1,keepdim=True)[0]
                logits.append(torch.cat([p_sim[:,:,0],neg_sims],dim=1)/FLAGS.temperature)
                top_mask[pos_sims_vids == p_sim] = 0.
                print(k,p_sim)

            vid_loss = F.cross_entropy(torch.cat(logits,dim=0),lab[:4*FLAGS.top_k])

            neg_sims = neg_sims.tile(2,1)
            logits = []
            top_mask = spill_mask.clone()
            for k in range(FLAGS.top_k):
                p_sim = (pos_sims_spills*top_mask).max(dim=2,keepdim=True)[0].max(dim=1,keepdim=True)[0]
                logits.append(torch.cat([p_sim[:,:,0],neg_sims],dim=1)/FLAGS.temperature)
                top_mask[pos_sims_vids == p_sim] = 0.
                print(k,p_sim)

            spill_loss = F.cross_entropy(torch.cat(logits,dim=0),lab)

            final_loss = vid_loss + spill_loss

            train_iter += 1
            log_dict = {"Epoch":epoch,"Train Iteration":train_iter, "Final Loss": final_loss, "Video Loss": vid_loss, "Spill Loss":spill_loss}

            final_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if train_iter % 100 == 0:
                print(log_dict)

            wandb.log(log_dict)


if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force=True)
    app.run(main)