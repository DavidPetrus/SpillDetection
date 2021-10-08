import numpy as np
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

        self.normalize = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        self.color_transforms = [lambda x,y: F_vis.adjust_saturation(x,y*FLAGS.saturation_range), \
                                 lambda x,y: F_vis.adjust_hue(x,y*2*FLAGS.hue_range - FLAGS.hue_range), \
                                 lambda x,y: F_vis.adjust_gamma(x,y*FLAGS.gamma_range + 1/FLAGS.gamma_range)]

        if FLAGS.proj_head > 0:
            self.proj_head = nn.Linear(512,FLAGS.proj_head,bias=False)
            self.prototypes = nn.Parameter(torch.randn(FLAGS.num_proto_sets,self.num_prototypes, FLAGS.proj_head).to(device), requires_grad=True)
        else:
            self.prototypes = nn.Parameter(torch.randn(FLAGS.num_proto_sets,self.num_prototypes, 512).to(device), requires_grad=True)

        self.build_aug_net()

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
        self.aug_net = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2),nn.ReLU(inplace=True),nn.Conv2d(16, 32, 3, stride=2),nn.ReLU(inplace=True), \
                                     nn.Conv2d(32, 64, 3, stride=2),nn.ReLU(inplace=True),nn.Conv2d(64, 128, 3, stride=2),nn.ReLU(inplace=True), \
                                     nn.AvgPool2d(13), nn.Linear(128,3))

    def augs_compose(self, x, augs):
        aug_patches = []
        for patch,patch_aug in zip(x, augs):
            new_patch = patch.clone()
            for t,t_mag in zip(self.color_transforms,patch_aug):
                new_patch = t(new_patch,t_mag)

            aug_patches.append(self.normalize(new_patch))

        return torch.stack(aug_patches)

    def aug_pred(self, x):
        print("-------",x.shape)
        pred = self.aug_net(x).sigmoid()
        print(pred.shape)
        aug_imgs = self.augs_compose(x,pred)

        return aug_imgs