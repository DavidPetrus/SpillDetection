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
flags.DEFINE_integer('num_workers',8,'')
flags.DEFINE_integer('batch_size',8,'')
flags.DEFINE_integer('epochs',20,'')
flags.DEFINE_integer('val_batch',10,'')
flags.DEFINE_float('lr',0.01,'')
flags.DEFINE_float('temperature',0.06,'')

flags.DEFINE_string('clip_model','ViT-B/16','')
flags.DEFINE_integer('proj_head',0,'')
flags.DEFINE_integer('num_prototypes',30,'')
flags.DEFINE_integer('num_proto_sets',1,'')
flags.DEFINE_integer('top_k_spill',1,'')
flags.DEFINE_integer('top_k_vids',1,'')
flags.DEFINE_float('margin',0.025,'')
flags.DEFINE_float('puddle_coeff',0.0,'')
flags.DEFINE_float('vid_coeff',0.5,'')
flags.DEFINE_string('scale','xlarge','')

flags.DEFINE_bool('autocontrast',True,'')

# Multiple Augmentations
flags.DEFINE_integer('num_augs',20,'')
flags.DEFINE_integer('max_compose',3,'')
flags.DEFINE_float('aug_temp',0.05,'')

# Learned Augmentation ranges
flags.DEFINE_float('augnet_coeff',1.,'')
flags.DEFINE_float('saturation_range',3.,'')
flags.DEFINE_float('hue_range',0.45,'')
flags.DEFINE_float('gamma_range',2,'')

# Color Augmentations
flags.DEFINE_bool('inc_saturation',False,'')
flags.DEFINE_bool('grayscale',False,'')
flags.DEFINE_bool('hue_pos',False,'')
flags.DEFINE_bool('hue_neg',False,'')
flags.DEFINE_bool('hue_full',False,'')
flags.DEFINE_bool('gamma_dark',False,'')
flags.DEFINE_bool('gamma_light',False,'')
flags.DEFINE_bool('invert',False,'')
flags.DEFINE_bool('posterize',False,'')

# Superimpose
flags.DEFINE_float('min_alpha',140,'')
flags.DEFINE_float('max_alpha',180,'')
flags.DEFINE_float('min_spill_frac',0.4,'')
flags.DEFINE_float('max_spill_frac',1.,'')
flags.DEFINE_float('superimpose_frac',0.,'')

batch_img_nums = [2+4,4,0]

def main(argv):
    global batch_img_nums

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
    if FLAGS.inc_saturation:
        color_distorts.append(lambda x: F_vis.adjust_saturation(x,2.5))
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
    if FLAGS.grayscale:
        color_distorts.append(lambda x: F_vis.rgb_to_grayscale(x,3))
        

    training_set = CustomDataGen(TRAIN_IMAGES_PATH, batch_size, preprocess, train=True, color_distorts=color_distorts, batch_nums=batch_img_nums)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=None, shuffle=False, num_workers=FLAGS.num_workers, pin_memory=True)
    validation_set = CustomDataGen(VAL_IMAGES_PATH, batch_size, preprocess, train=False, color_distorts=color_distorts)
    validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=None, shuffle=False, num_workers=4, pin_memory=True)

    spill_det = SpillDetector(clip_model)
    spill_det.to('cuda')

    #spill_det.load_state_dict(torch.load('weights/20Sep3.pt',map_location=torch.device('cuda')))
    if FLAGS.proj_head > 0:
        optimizer = torch.optim.Adam([{'params':[spill_det.prototypes],'lr':FLAGS.lr},{'params':list(spill_det.proj_head.parameters())+list(spill_det.aug_net.parameters()),'lr':0.001}], lr=FLAGS.lr)
    else:
        optimizer = torch.optim.Adam([{'params':[spill_det.prototypes],'lr':FLAGS.lr},{'params':list(spill_det.aug_net.parameters()),'lr':0.001}], lr=FLAGS.lr)

    color_aug = torchvision.transforms.ColorJitter(0.8,0.3,2,0.4)

    normalize = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    inv_normalize = torchvision.transforms.Normalize((-0.48145466/0.26862954, -0.4578275/0.26130258, -0.40821073/0.27577711), (1/0.26862954, 1/0.26130258, 1/0.27577711))

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
        num_vid_patches = 1*num_distorts
        num_spill_patches = 1*num_distorts
        num_puddle_patches = 1*num_distorts
    elif FLAGS.scale == 'med':
        num_vid_patches = 3*num_distorts
        num_spill_patches = 2*num_distorts
        num_puddle_patches = 2*num_distorts

    num_patches = [num_vid_patches,num_spill_patches,num_puddle_patches]

    lab = torch.zeros(60,dtype=torch.int64).to('cuda')

    aug_loss_avg = np.zeros(FLAGS.num_augs)

    min_acc = 0.
    total_loss = 0.
    step_loss = 0.
    train_iter = 0
    val_iter = 0
    for epoch in range(epochs):
        optimizer.zero_grad()
        #with torch.autograd.detect_anomaly():
        for data in training_generator:
            img_patches,_ = data

            if FLAGS.autocontrast:
                img_patches = normalize(F_vis.autocontrast(color_aug(inv_normalize(img_patches.to('cuda')).clamp(0,1))))
            else:
                img_patches = normalize(color_aug(inv_normalize(img_patches.to('cuda')).clamp(0,1)))
            #img_patches = img_patches.to('cuda')

            pos_patches = img_patches[:sum(batch_img_nums)]
            neg_patches = img_patches[sum(batch_img_nums):]

            pos_patches, pos_pred = spill_det.aug_pred(pos_patches, apply_all=True)
            with torch.no_grad():
                neg_patches, neg_pred = spill_det.aug_pred(neg_patches)

            sims = spill_det(torch.cat([pos_patches, neg_patches], dim=0))
            pos_sims = sims[:FLAGS.num_augs*sum(batch_img_nums)].reshape(FLAGS.num_augs, sum(batch_img_nums)).movedim(0,1)
            neg_sims = sims[FLAGS.num_augs*sum(batch_img_nums):]

            augnet_loss = -torch.sum(F.log_softmax(pos_pred, dim=1) * F.softmax(-pos_sims/FLAGS.aug_temp, dim=1).detach(), dim=1).mean()

            pos_sims = torch.gather(pos_sims, 1, torch.multinomial(F.softmax(pos_pred, dim=1), 1))
            losses, accs = loss_func(torch.cat([pos_sims,neg_sims], dim=0), num_patches, lab)

            spill_loss,puddle_loss,vid_loss = losses
            spill_acc,puddle_acc,vid_acc = accs

            final_loss = spill_loss + FLAGS.vid_coeff*vid_loss + FLAGS.puddle_coeff*puddle_loss + FLAGS.augnet_coeff*augnet_loss

            train_iter += 1
            log_dict = {"Epoch":epoch, "Train Iteration":train_iter, "Final Loss": final_loss, \
                        "AugNet Loss": augnet_loss, \
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

            if train_iter == 1500:
                for g in optimizer.param_groups:
                    g['lr'] = 0.001

        val_count = 0
        num_crop_dims = 1
        org_loss_2x3 = org_loss_3x5 = org_loss_4x7 = loss_2x3 = loss_3x5 = loss_4x7 = acc_clear_2x3 = acc_clear_3x5 = acc_clear_4x7 = \
        acc_dark_2x3 = acc_dark_3x5 = acc_dark_4x7 = acc_opaque_2x3 = acc_opaque_3x5 = acc_opaque_4x7 = 0.

        probs = 0.1 + (aug_loss_avg-aug_loss_avg.min())/(aug_loss_avg.max()-aug_loss_avg.min()+0.01)
        probs /= probs.sum()
        aug_loss_sums = np.array([0. for i in range(FLAGS.num_augs)])
        aug_loss_count = np.array([0 for i in range(FLAGS.num_augs)])

        for data in validation_generator:
            with torch.no_grad():
                img_patches = data.to('cuda')
                if FLAGS.autocontrast:
                    img_patches = normalize(F_vis.autocontrast(inv_normalize(img_patches).clamp(0,1)))

                pos_patches = img_patches[:(10+3+4)*num_crop_dims]
                neg_patches = img_patches[(10+3+4)*num_crop_dims:]

                aug_indices = np.random.choice(FLAGS.num_augs, size=4, replace=False, p=probs)

                pos_patches, pos_pred = spill_det.aug_pred(pos_patches, apply_all=True, val=True, five_crop=False, aug_indices=aug_indices)
                neg_patches, neg_pred = spill_det.aug_pred(neg_patches, apply_all=True, val=True, aug_indices=aug_indices)

                sims = spill_det(torch.cat([pos_patches, neg_patches], dim=0))
                pos_sims,m_idxs = sims[:4*1*(10+3+4)*num_crop_dims].reshape(4,1,(10+3+4),num_crop_dims).max(dim=1)[0].max(dim=0)
                #print(m_idxs)
                neg_sims = sims[4*1*(10+3+4)*num_crop_dims*num_distorts:].reshape(4,(10+3+4),3*12,1*num_distorts).max(dim=-1)[0].max(dim=0)[0]

                #img_patches, aug_pred = spill_det.aug_pred(img_patches, val=True)

                # Original patches
                #sims = spill_det(img_patches)
                #pos_sims = sims[:(10+3+5)*2*num_distorts].reshape(10+3+5,2,1*num_distorts).max(dim=2)[0]
                #neg_sims = sims[(10+3+5)*2*num_distorts:].reshape(FLAGS.val_batch,8+23,1*num_distorts).max(dim=2)[0]

                sims_2x3 = torch.cat([pos_sims[:,0:1],neg_sims],dim=1)
                #sims_3x5 = torch.cat([pos_sims[:,1:2],neg_sims[:,8:8+23].reshape(1,FLAGS.val_batch*23).tile(10+3+5,1)],dim=1)
                #sims_4x7 = torch.cat([pos_sims[:,2:3],neg_sims[:,8+23:8+23+46].reshape(1,FLAGS.val_batch*46).tile(10+3+5,1)],dim=1)

                batch_loss = F.cross_entropy(sims_2x3/FLAGS.temperature,lab[:10+3+4])
                org_loss_2x3 += batch_loss
                #org_loss_3x5 += F.cross_entropy(sims_3x5/FLAGS.temperature,lab[:10+3+5])
                #org_loss_4x7 += F.cross_entropy(sims_4x7/FLAGS.temperature,lab[:10+3+5])

                aug_loss_sums[aug_indices] += 10-batch_loss.cpu().numpy()
                aug_loss_count[aug_indices] += 1

                '''# Augmented patches
                sims = spill_det(aug_patches)
                pos_sims = sims[:(10+3+5)*2*num_distorts].reshape(10+3+5,2,1*num_distorts).max(dim=2)[0]
                neg_sims = sims[(10+3+5)*2*num_distorts:].reshape(FLAGS.val_batch,8+23,1*num_distorts).max(dim=2)[0]

                sims_2x3 = torch.cat([pos_sims[:,0:1],neg_sims[:,:8].reshape(1,FLAGS.val_batch*8).tile(10+3+5,1)],dim=1)
                sims_3x5 = torch.cat([pos_sims[:,1:2],neg_sims[:,8:8+23].reshape(1,FLAGS.val_batch*23).tile(10+3+5,1)],dim=1)
                #sims_4x7 = torch.cat([pos_sims[:,2:3],neg_sims[:,8+23:8+23+46].reshape(1,FLAGS.val_batch*46).tile(10+3+5,1)],dim=1)

                loss_2x3 += F.cross_entropy(sims_2x3/FLAGS.temperature,lab[:10+3+5])
                loss_3x5 += F.cross_entropy(sims_3x5/FLAGS.temperature,lab[:10+3+5])
                #loss_4x7 += F.cross_entropy(sims_4x7/FLAGS.temperature,lab[:10+3+5])'''

                acc_clear_2x3 += (torch.argmax(sims_2x3[:10],dim=1)==0).float().mean()
                #acc_clear_3x5 += (torch.argmax(sims_3x5[:10],dim=1)==0).float().mean()
                #acc_clear_4x7 += (torch.argmax(sims_4x7[:10],dim=1)==0).float().mean()

                acc_dark_2x3 += (torch.argmax(sims_2x3[10:10+3],dim=1)==0).float().mean()
                #acc_dark_3x5 += (torch.argmax(sims_3x5[10:10+3],dim=1)==0).float().mean()
                #acc_dark_4x7 += (torch.argmax(sims_4x7[10:10+3],dim=1)==0).float().mean()

                acc_opaque_2x3 += (torch.argmax(sims_2x3[10+3:10+3+4],dim=1)==0).float().mean()
                #acc_opaque_3x5 += (torch.argmax(sims_3x5[10+3:10+3+5],dim=1)==0).float().mean()
                #acc_opaque_4x7 += (torch.argmax(sims_4x7[10+3:10+3+5],dim=1)==0).float().mean()

                val_count += 1

        aug_loss_avg = aug_loss_sums/aug_loss_count

        log_dict = {"Epoch":epoch}
        log_dict["Org_val_loss_2x3"] = org_loss_2x3/val_count
        log_dict["Org_val_loss_3x5"] = org_loss_3x5/val_count
        log_dict["Org_val_loss_4x7"] = org_loss_4x7/val_count
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

        val_accs = [acc_clear_2x3,acc_dark_2x3,acc_opaque_2x3]

        if sum(val_accs) > min_acc:
            torch.save({'prototypes': spill_det.prototypes},'weights/{}.pt'.format(FLAGS.exp))
            min_acc = sum(val_accs)

def loss_func(sims, num_patches, lab):
    global batch_img_nums

    num_vid_patches,num_spill_patches,num_puddle_patches = num_patches

    pos_sims_vids = sims[:batch_img_nums[0]*num_vid_patches].reshape(batch_img_nums[0],num_vid_patches,1)
    pos_sims_spills = sims[batch_img_nums[0]*num_vid_patches:batch_img_nums[0]*num_vid_patches + batch_img_nums[1]*num_spill_patches].reshape(batch_img_nums[1],num_spill_patches,1)
    pos_sims_puddles = sims[batch_img_nums[0]*num_vid_patches + batch_img_nums[1]*num_spill_patches:batch_img_nums[0]*num_vid_patches + \
                        batch_img_nums[1]*num_spill_patches + batch_img_nums[2]*num_puddle_patches].reshape(batch_img_nums[2],num_puddle_patches,1)
    neg_sims = sims[batch_img_nums[0]*num_vid_patches + batch_img_nums[1]*num_spill_patches + batch_img_nums[2]*num_puddle_patches:].max(dim=1)[0].reshape(1,-1).tile(max(batch_img_nums),1)

    # Compute vid loss
    p_sim,vid_ix = pos_sims_vids.max(dim=2,keepdim=True)[0].max(dim=1,keepdim=True)
    logits = torch.cat([p_sim[:,:,0]-FLAGS.margin,neg_sims[:batch_img_nums[0]]],dim=1)/FLAGS.temperature
    vid_loss = F.cross_entropy(logits,lab[:batch_img_nums[0]],reduction='mean')
    vid_acc = (torch.argmax(logits,dim=1)==0).float().mean()

    # Compute spill loss
    p_sim,spill_ix = pos_sims_spills.max(dim=2,keepdim=True)[0].max(dim=1,keepdim=True)
    logits = torch.cat([p_sim[:,:,0]-FLAGS.margin,neg_sims[:batch_img_nums[1]]],dim=1)/FLAGS.temperature
    spill_loss = F.cross_entropy(logits,lab[:batch_img_nums[1]],reduction='mean')
    spill_acc = (torch.argmax(logits,dim=1)==0).float().mean()

    # Compute puddle loss
    if batch_img_nums[2] > 0:
        p_sim,puddle_ix = pos_sims_puddles.max(dim=2,keepdim=True)[0].max(dim=1,keepdim=True)
        logits = torch.cat([p_sim[:,:,0]-FLAGS.margin,neg_sims[:batch_img_nums[2]]],dim=1)/FLAGS.temperature
        puddle_loss = F.cross_entropy(logits,lab[:batch_img_nums[2]],reduction='mean')
        puddle_acc = (torch.argmax(logits,dim=1)==0).float().mean()
    else:
        puddle_loss = puddle_acc = 0.

    return [spill_loss,puddle_loss,vid_loss], [spill_acc,puddle_acc,vid_acc]#, [vid_ix.tile(1,1,3),spill_ix.tile(1,1,3),puddle_ix.tile(1,1,3)]


if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force=True)
    app.run(main)