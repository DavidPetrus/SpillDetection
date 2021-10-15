import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as F_vis
import cv2

from absl import flags

FLAGS = flags.FLAGS

class SpillDetector(nn.Module):

    def __init__(self,clip_model,num_prototypes=None,device='cuda'):
        super(SpillDetector, self).__init__()

        if num_prototypes:
            self.num_prototypes = num_prototypes
        else:
            self.num_prototypes = FLAGS.num_prototypes

        self.clip_model = clip_model
        self.dtype = torch.half if device=='cuda' else torch.float32

        '''self.color_transforms = [lambda x,y: F_vis.adjust_saturation(x,y*FLAGS.saturation_range), \
                                 lambda x,y: F_vis.adjust_hue(x,y*2*FLAGS.hue_range - FLAGS.hue_range), \
                                 lambda x,y: F_vis.adjust_gamma(x,y*FLAGS.gamma_range + 1/FLAGS.gamma_range)]'''

        if FLAGS.proj_head > 0:
            self.proj_head = nn.Linear(512,FLAGS.proj_head,bias=False)
            self.prototypes = nn.Parameter(torch.randn(FLAGS.num_proto_sets,self.num_prototypes, FLAGS.proj_head).to(device), requires_grad=True)
        else:
            self.prototypes = nn.Parameter(torch.randn(FLAGS.num_proto_sets,self.num_prototypes, 512).to(device), requires_grad=True)

        self.build_aug_net()

        self.normalize = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.inv_normalize = torchvision.transforms.Normalize((-0.48145466/0.26862954, -0.4578275/0.26130258, -0.40821073/0.27577711), (1/0.26862954, 1/0.26130258, 1/0.27577711))

        self.all_augs = [lambda x: F_vis.adjust_saturation(x,2.5), \
                         lambda x: F_vis.adjust_saturation(x,0.2), \
                         lambda x: F_vis.adjust_hue(x,0.25), \
                         lambda x: F_vis.adjust_hue(x,-0.4), \
                         lambda x: F_vis.invert(x), \
                         lambda x: F_vis.adjust_gamma(x,0.7), \
                         lambda x: F_vis.adjust_gamma(x,2)]

        self.generate_augmentations()

        self.five_crop = torchvision.transforms.FiveCrop(134)
        self.resize = torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)

    def forward(self, x):
        with torch.no_grad():
            img_features = self.clip_model.encode_image(x).to(torch.float32)

        if FLAGS.proj_head > 0:
            img_features = self.proj_head(img_features)

        img_features = F.normalize(img_features).unsqueeze(0)

        prototypes = F.normalize(self.prototypes,dim=2)

        sims = img_features @ prototypes.transpose(1,2)
        sims = sims.max(dim=2,keepdim=True)[0].mean(dim=0)

        return sims

    def build_aug_net(self):
        self.aug_net = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2), nn.ReLU(inplace=True), nn.Conv2d(16, 32, 3, stride=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True), \
                                     nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(inplace=True), nn.Conv2d(64, 128, 3, stride=2), nn.BatchNorm2d(128), \
                                     nn.MaxPool2d(13), nn.Flatten(), nn.Linear(128, FLAGS.num_augs))

    def augs_compose(self, x, augs):
        aug_patches = []
        for patch,patch_aug in zip(x, augs):
            new_patch = self.inv_normalize(patch).clamp(0,1)
            sampled_augs = self.aug_set[torch.multinomial(patch_aug,1)[0]]
            for t in sampled_augs:
                new_patch = t(new_patch)

            if FLAGS.autocontrast:
                aug_patches.append(self.normalize(F_vis.autocontrast(new_patch)))
            else:
                aug_patches.append(self.normalize(new_patch))

        return torch.stack(aug_patches)

    def aug_pred(self, x, apply_all=False, val=False, five_crop=False, aug_indices=None):
        temp = 100 if val else 1.
        if five_crop:
            crop_ls = []
            crops = self.five_crop(x) + (x,)
            for cr in crops:
                x_trans = self.resize(cr)
                crop_ls.append(x_trans)
            x = torch.cat(crop_ls, dim=0)

        pred = self.aug_net(x)
        #pred = logits[:FLAGS.num_augs]
        #crop_pred = logits[FLAGS.num_augs:-1]
        with torch.no_grad():
            if apply_all:
                aug_imgs = []
                if aug_indices is not None:
                    aug_set = [self.aug_set[i] for i in aug_indices]
                else:
                    aug_set = self.aug_set

                for augs in aug_set:
                    x_trans = self.inv_normalize(x).clamp(0,1)
                    for t in augs:
                        x_trans = t(x_trans)

                    if FLAGS.autocontrast:
                        x_trans = self.normalize(F_vis.autocontrast(x_trans))
                    else:
                        x_trans = self.normalize(x_trans)
                    aug_imgs.append(x_trans)
                aug_imgs = torch.cat(aug_imgs, dim=0)
            else:
                aug_imgs = self.augs_compose(x, F.softmax(pred/temp, dim=1))

        return aug_imgs, pred

    def generate_augmentations(self):
        self.aug_set = [[]]
        for i in range(FLAGS.num_augs-1):
            #self.aug_set.append([self.all_augs[i]])
            self.aug_set.append([])
            num_compose = np.random.randint(1,FLAGS.max_compose+1)
            for c in range(num_compose):
                self.aug_set[-1].append(random.choice(self.all_augs))

        print(self.all_augs)
        print(self.aug_set)