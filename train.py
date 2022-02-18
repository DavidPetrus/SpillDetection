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
flags.DEFINE_integer('epochs',70,'')
flags.DEFINE_integer('val_batch',10,'')
flags.DEFINE_float('lr',0.01,'')
flags.DEFINE_float('temperature',0.06,'')
flags.DEFINE_float('aux_coeff',0.,'')

flags.DEFINE_string('clip_model','ViT-B/16','')
flags.DEFINE_integer('proj_hidden',512,'')
flags.DEFINE_integer('proj_head',256,'')
flags.DEFINE_integer('num_prototypes',30,'')
flags.DEFINE_integer('num_proto_sets',1,'')
flags.DEFINE_integer('top_k_spill',1,'')
flags.DEFINE_integer('top_k_vids',1,'')
flags.DEFINE_float('margin',0.07,'')
flags.DEFINE_integer('num_puddle',2,'')
flags.DEFINE_float('puddle_coeff',0.5,'')
flags.DEFINE_float('vid_coeff',2.,'')
flags.DEFINE_float('spill_coeff',0.5,'')
flags.DEFINE_string('scale','large','full,large,small')

flags.DEFINE_bool('autocontrast',True,'')
flags.DEFINE_integer('num_distorts',1,'')
flags.DEFINE_integer('num_crop_dims',1,'')

# Multiple Augmentations
flags.DEFINE_integer('num_augs',20,'')
flags.DEFINE_integer('max_compose',3,'')
flags.DEFINE_float('aug_temp',0.05,'')

# Learned Augmentation ranges
flags.DEFINE_float('augnet_coeff',1.,'')
flags.DEFINE_float('saturation_range',0.6,'')
flags.DEFINE_float('hue_range',0.2,'')
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
flags.DEFINE_float('min_spill_frac',0.3,'')
flags.DEFINE_float('max_spill_frac',1.,'')
flags.DEFINE_float('superimpose_frac',0.,'')

batch_img_nums = [2+10,4,0]

def main(argv):
    global batch_img_nums

    batch_img_nums[2] = FLAGS.num_puddle

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
        
    FLAGS.num_distorts = len(color_distorts)

    training_set = CustomDataGen(TRAIN_IMAGES_PATH, batch_size, preprocess, train=True, color_distorts=color_distorts, batch_nums=batch_img_nums)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=None, shuffle=False, num_workers=FLAGS.num_workers, pin_memory=True)
    validation_set = CustomDataGen(VAL_IMAGES_PATH, batch_size, preprocess, train=False, color_distorts=color_distorts)
    validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=None, shuffle=False, num_workers=4, pin_memory=True)

    calibration_videos = {'not_spills': glob.glob("images/calibration/not_spills/*"), 'difficult_spills': glob.glob("images/calibration/difficult_spills/*"), \
                          'obvious_spills': glob.glob("images/calibration/obvious_spills/*")}

    spill_det = SpillDetector(clip_model, preprocess)
    spill_det.to('cuda')

    #spill_det.load_state_dict(torch.load('weights/3Feb2.pt',map_location=torch.device('cuda')))

    if FLAGS.proj_head > 0:
        optimizer = torch.optim.Adam([{'params':[spill_det.prototypes],'lr':FLAGS.lr},{'params':list(spill_det.proj_head.parameters()),'lr':0.001}], lr=FLAGS.lr)
    else:
        optimizer = torch.optim.Adam([{'params':[spill_det.prototypes],'lr':FLAGS.lr}], lr=FLAGS.lr)

    color_aug = torchvision.transforms.ColorJitter(0.3,0.3,FLAGS.saturation_range,FLAGS.hue_range)

    normalize = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    inv_normalize = torchvision.transforms.Normalize((-0.48145466/0.26862954, -0.4578275/0.26130258, -0.40821073/0.27577711), (1/0.26862954, 1/0.26130258, 1/0.27577711))

    num_distorts = len(color_distorts)

    lab = torch.zeros(60,dtype=torch.int64).to('cuda')

    aug_loss_avg = np.zeros(FLAGS.num_augs)

    min_low = 0.
    min_diff = 0.
    min_all = 0.

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

            '''patches_show = inv_normalize(img_patches).clamp(0,1).movedim(1,3).cpu().numpy()
            for p_ix,patch in enumerate(patches_show[:batch_img_nums[0]]):
                cv2.imshow("Vid"+str(p_ix),patch[:,:,::-1])

            for p_ix,patch in enumerate(patches_show[batch_img_nums[0]:batch_img_nums[0]+batch_img_nums[1]]):
                cv2.imshow("Spill"+str(p_ix),patch[:,:,::-1])

            for p_ix,patch in enumerate(patches_show[batch_img_nums[0]+batch_img_nums[1]:batch_img_nums[0]+batch_img_nums[1]+batch_img_nums[2]]):
                cv2.imshow("Puddle"+str(p_ix),patch[:,:,::-1])

            for p_ix,patch in enumerate(patches_show[sum(batch_img_nums):]):
                if p_ix % 3 > 0 or p_ix > 50: continue
                cv2.imshow("Vid"+str(p_ix),patch[:,:,::-1])

            for p_ix,patch in enumerate(patches_show[-32:]):
                cv2.imshow("Aux"+str(p_ix),patch[:,:,::-1])

            key = cv2.waitKey(0)
            if key==27:
                cv2.destroyAllWindows()
                exit()'''

            if FLAGS.aux_coeff > 0.:
                sims = spill_det(img_patches[:-32])

                feats = spill_det(img_patches[-32:], ret_embd=True)
                feats = F.normalize(feats).reshape(8,4,-1)
                aux_sims = feats @ feats.transpose(1,2)
                aux_loss = (1-aux_sims).sum()
            else:
                sims = spill_det(img_patches)
                aux_loss = 0.

            losses, accs = loss_func(sims, lab)

            spill_loss,puddle_loss,vid_loss = losses
            spill_acc,puddle_acc,vid_acc = accs

            final_loss = FLAGS.spill_coeff*spill_loss + FLAGS.vid_coeff*vid_loss + FLAGS.puddle_coeff*puddle_loss + FLAGS.aux_coeff*aux_loss

            train_iter += 1
            log_dict = {"Epoch":epoch, "Train Iteration":train_iter, "Final Loss": final_loss, \
                        "Video Loss": vid_loss, "Spill Loss":spill_loss, "Puddle Loss":puddle_loss, "Auxiliary Loss": aux_loss, \
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

            if train_iter == 2000:
                for g in optimizer.param_groups:
                    g['lr'] /= 3
            elif train_iter == 4000:
                for g in optimizer.param_groups:
                    g['lr'] /= 3
            elif train_iter == 6000:
                for g in optimizer.param_groups:
                    g['lr'] /= 3

        val_count = 0
        total_val_conf = 0.
        total_neg_conf = 0.
        num_crop_dims = FLAGS.num_crop_dims
        if FLAGS.num_crop_dims == 1:
            num_neg = 12
        else:
            num_neg = 45

        org_loss_2x3 = org_loss_3x5 = org_loss_4x7 = loss_2x3 = loss_3x5 = loss_4x7 = acc_clear_2x3 = acc_clear_3x5 = acc_clear_4x7 = \
        acc_dark_2x3 = acc_dark_3x5 = acc_dark_4x7 = acc_opaque_2x3 = acc_opaque_3x5 = acc_opaque_4x7 = 0.

        '''probs = 0.1 + (aug_loss_avg-aug_loss_avg.min())/(aug_loss_avg.max()-aug_loss_avg.min()+0.01)
        probs /= probs.sum()
        aug_loss_sums = np.array([0. for i in range(FLAGS.num_augs)])
        aug_loss_count = np.array([0 for i in range(FLAGS.num_augs)])'''

        '''for data in validation_generator:
            with torch.no_grad():
                img_patches = data.to('cuda')
                if FLAGS.autocontrast:
                    img_patches = normalize(F_vis.autocontrast(inv_normalize(img_patches).clamp(0,1)))

                sims = spill_det(img_patches)
                pos_sims = sims[:(20)*FLAGS.num_distorts*num_crop_dims].reshape((20),FLAGS.num_distorts,num_crop_dims).max(dim=1)[0]
                neg_sims = sims[(20)*FLAGS.num_distorts*num_crop_dims:].reshape((20),3,FLAGS.num_distorts,num_neg).max(dim=2)[0]

                #img_patches, aug_pred = spill_det.aug_pred(img_patches, val=True)

                # Original patches
                #sims = spill_det(img_patches)
                #pos_sims = sims[:(10+3+5)*2*num_distorts].reshape(10+3+5,2,1*num_distorts).max(dim=2)[0]
                #neg_sims = sims[(10+3+5)*2*num_distorts:].reshape(FLAGS.val_batch,8+23,1*num_distorts).max(dim=2)[0]

                sims_2x3 = torch.cat([pos_sims[:,0:1],neg_sims[:,:,:12].reshape(-1,3*12)],dim=1)
                if FLAGS.num_crop_dims > 1:
                    sims_3x5 = torch.cat([pos_sims[:,1:2],neg_sims[:,:,12:12+45].reshape(-1,3*45)],dim=1)
                #sims_4x7 = torch.cat([pos_sims[:,2:3],neg_sims[:,8+23:8+23+46].reshape(1,FLAGS.val_batch*46).tile(10+3+5,1)],dim=1)

                org_loss_2x3 += F.cross_entropy(sims_2x3/FLAGS.temperature,lab[:20])
                if FLAGS.num_crop_dims > 1:
                    org_loss_3x5 += F.cross_entropy(sims_3x5/FLAGS.temperature,lab[:20])
                #org_loss_4x7 += F.cross_entropy(sims_4x7/FLAGS.temperature,lab[:10+3+5])

                #aug_loss_sums[aug_indices] += 10-batch_loss.cpu().numpy()
                #aug_loss_count[aug_indices] += 1

                acc_clear_2x3 += (torch.argmax(sims_2x3[:20],dim=1)==0).float().mean()
                if FLAGS.num_crop_dims > 1:
                    acc_clear_3x5 += (torch.argmax(sims_3x5[:20],dim=1)==0).float().mean()
                #acc_clear_4x7 += (torch.argmax(sims_4x7[:10],dim=1)==0).float().mean()

                total_val_conf += pos_sims.mean()
                total_neg_conf += neg_sims.mean()
                val_count += 1


        #aug_loss_avg = aug_loss_sums/aug_loss_count

        log_dict = {"Epoch":epoch}
        log_dict["Average_Val_Conf"] = total_val_conf/val_count
        log_dict["Average_Neg_Conf"] = total_neg_conf/val_count
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
        log_dict["Val_acc_opaque_4x7"] = acc_opaque_4x7/val_count'''

        print("Start Val: ",datetime.datetime.now())
        all_sims = {}
        for vid_class, vids in calibration_videos.items():
            all_sims[vid_class] = []
            for vid in vids:
                cap = cv2.VideoCapture(vid)
                all_sims[vid_class].append([])

                count = 0
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    count += 1
                    if count % 15 > 0:
                        continue

                    max_sim = spill_det.video_forward(frame)

                    all_sims[vid_class][-1].append(max_sim)

        for k,sims in all_sims.items():
            gr_sims = []
            for vid_sims in sims:
                vid_freqs,bins = np.histogram(np.array(vid_sims),bins=80,range=(-1.,1.))
                gr_sims.append(vid_freqs)

            gr_sims = np.array(gr_sims)
            gr_sims = gr_sims/gr_sims.sum(axis=1, keepdims=True)
            gr_sims = np.cumsum(gr_sims, axis=1)
            gr_sims = (1-gr_sims).mean(axis=0)

            if k == 'not_spills':
                not_hist = gr_sims.copy()
            elif k == 'difficult_spills':
                diff_hist = gr_sims.copy()
            elif k == 'obvious_spills':
                obv_hist = gr_sims.copy()

        low_criteria = (obv_hist*(1-not_hist)).max()
        low_thresh_ix = (obv_hist*(1-not_hist)).argmax()
        low_thresh = [i/40 for i in range(-39,41)][low_thresh_ix]
        low_not_perc = (1-not_hist)[low_thresh_ix]
        low_diff_perc = diff_hist[low_thresh_ix]
        low_obv_perc = obv_hist[low_thresh_ix]

        diff_criteria = (diff_hist*(1-not_hist)).max()
        diff_thresh_ix = (diff_hist*(1-not_hist)).argmax()
        diff_thresh = [i/40 for i in range(-39,41)][diff_thresh_ix]
        diff_not_perc = (1-not_hist)[diff_thresh_ix]
        diff_diff_perc = diff_hist[diff_thresh_ix]
        diff_obv_perc = obv_hist[diff_thresh_ix]

        all_criteria = (diff_hist*obv_hist*(1-not_hist)**2).max()
        all_thresh_ix = (diff_hist*obv_hist*(1-not_hist)**2).argmax()
        all_thresh = [i/40 for i in range(-39,41)][all_thresh_ix]
        all_not_perc = (1-not_hist)[all_thresh_ix]
        all_diff_perc = diff_hist[all_thresh_ix]
        all_obv_perc = obv_hist[all_thresh_ix]
            

        print("End Val: ",datetime.datetime.now())

        log_dict['low_criteria'] = low_criteria
        log_dict['low_thresh'] = low_thresh
        log_dict['low_not_perc'] = low_not_perc
        log_dict['low_diff_perc'] = low_diff_perc
        log_dict['low_obv_perc'] = low_obv_perc

        log_dict['diff_criteria'] = diff_criteria
        log_dict['diff_thresh'] = diff_thresh
        log_dict['diff_not_perc'] = diff_not_perc
        log_dict['diff_diff_perc'] = diff_diff_perc
        log_dict['diff_obv_perc'] = diff_obv_perc

        log_dict['all_criteria'] = all_criteria
        log_dict['all_thresh'] = all_thresh
        log_dict['all_not_perc'] = all_not_perc
        log_dict['all_diff_perc'] = all_diff_perc
        log_dict['all_obv_perc'] = all_obv_perc

        wandb.log(log_dict)

        #val_accs = [acc_clear_2x3,acc_dark_2x3,acc_opaque_2x3]

        if low_criteria > min_low:
            torch.save({'prototypes': spill_det.prototypes, 'proj_head': spill_det.proj_head, 'low_sensitivity_threshold': low_thresh, 'high_sensitivity_threshold': all_thresh},'weights/{}_low.pt'.format(FLAGS.exp))

            min_low = low_criteria

        if diff_criteria > min_diff:
            torch.save({'prototypes': spill_det.prototypes, 'proj_head': spill_det.proj_head, 'low_sensitivity_threshold': low_thresh, 'high_sensitivity_threshold': diff_thresh},'weights/{}_diff.pt'.format(FLAGS.exp))

            min_diff = diff_criteria

        if all_criteria > min_all:
            torch.save({'prototypes': spill_det.prototypes, 'proj_head': spill_det.proj_head, 'low_sensitivity_threshold': low_thresh, 'high_sensitivity_threshold': all_thresh},'weights/{}_all.pt'.format(FLAGS.exp))

            min_all = all_criteria

def loss_func(sims, lab):
    global batch_img_nums

    pos_sims_vids = sims[:batch_img_nums[0]*FLAGS.num_distorts].reshape(batch_img_nums[0],FLAGS.num_distorts,1)
    pos_sims_spills = sims[batch_img_nums[0]*FLAGS.num_distorts:(batch_img_nums[0] + batch_img_nums[1])*FLAGS.num_distorts].reshape(batch_img_nums[1],FLAGS.num_distorts,1)
    pos_sims_puddles = sims[(batch_img_nums[0] + batch_img_nums[1])*FLAGS.num_distorts:(batch_img_nums[0] + \
                        batch_img_nums[1] + batch_img_nums[2])*FLAGS.num_distorts].reshape(batch_img_nums[2],FLAGS.num_distorts,1)
    neg_sims = sims[(batch_img_nums[0] + batch_img_nums[1] + batch_img_nums[2])*FLAGS.num_distorts:].max(dim=1)[0].reshape(1,-1).tile(max(batch_img_nums),1)

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