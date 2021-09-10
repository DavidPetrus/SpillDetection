import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from absl import flags

FLAGS = flags.FLAGS

class SpillDetector(nn.Module):

    def __init__(self,clip_model):
        super(SpillDetector, self).__init__()

        self.num_prototypes = FLAGS.num_prototypes

        self.clip_model = clip_model
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, 512).to('cuda'), requires_grad=True)

    def forward(self, x):
        with torch.no_grad():
            img_features = self.clip_model.encode_image(x)
            img_features = F.normalize(img_features)

        prototypes = F.normalize(self.prototypes).to(torch.half)

        sims = img_features @ prototypes.T

        return sims