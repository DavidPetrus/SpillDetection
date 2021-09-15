import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from absl import flags

FLAGS = flags.FLAGS

class SpillDetector(nn.Module):

    def __init__(self,clip_model,device='cuda'):
        super(SpillDetector, self).__init__()

        self.num_prototypes = FLAGS.num_prototypes

        self.clip_model = clip_model
        self.dtype = torch.half if device=='cuda' else torch.float32

        if FLAGS.proj_head > 0:
            self.proj_head = nn.Linear(512,FLAGS.proj_head,bias=False,dtype=self.dtype)
            self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, FLAGS.proj_head).to(device), requires_grad=True)
        else:
            self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, 512).to(device), requires_grad=True)

    def forward(self, x):
        with torch.no_grad():
            img_features = self.clip_model.encode_image(x)
        
        if FLAGS.proj_head > 0:
            img_features = self.proj_head(img_features)

        img_features = F.normalize(img_features)

        prototypes = F.normalize(self.prototypes).to(self.dtype)

        sims = img_features @ prototypes.T

        return sims