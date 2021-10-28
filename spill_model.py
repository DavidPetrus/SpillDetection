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

        if FLAGS.proj_head > 0:
            if FLAGS.proj_hidden > 0:
                self.proj_head = nn.Sequential(nn.Linear(512,FLAGS.proj_hidden), nn.ReLU(inplace=True), nn.Linear(FLAGS.proj_hidden,FLAGS.proj_head))
            else:
                self.proj_head = nn.Linear(512,FLAGS.proj_head,bias=False)

            self.prototypes = nn.Parameter(torch.randn(FLAGS.num_proto_sets,self.num_prototypes, FLAGS.proj_head).to(device), requires_grad=True)
        else:
            self.prototypes = nn.Parameter(torch.randn(FLAGS.num_proto_sets,self.num_prototypes, 512).to(device), requires_grad=True)

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