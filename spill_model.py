import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms.functional as F_vis
import cv2
from PIL import Image

from absl import flags

FLAGS = flags.FLAGS

class SpillDetector(nn.Module):

    def __init__(self,clip_model, preprocess,num_prototypes=None,device='cuda'):
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


        self.zoom_index = 6
        self.preprocess = preprocess
        self.autocontrast = lambda x: tv.transforms.functional.autocontrast(x)
        self.x_crop_fractions = [(0.,0.281),(0.242,0.523),(0.477,0.758),(0.719,1.)]
        self.y_crop_fractions = [(0.,0.5),(0.25,0.75),(0.5,1.)]

    def forward(self, x, ret_embd=False):
        with torch.no_grad():
            img_features = self.clip_model.encode_image(x).to(torch.float32)

        if FLAGS.proj_head > 0:
            img_features = self.proj_head(img_features)

        if ret_embd:
            return img_features

        img_features = F.normalize(img_features).unsqueeze(0)

        prototypes = F.normalize(self.prototypes,dim=2)

        sims = img_features @ prototypes.transpose(1,2)
        sims = sims.max(dim=2,keepdim=True)[0].mean(dim=0)

        return sims

    def video_forward(self, frame):
        frame = self.cv2_to_pil_image(frame)
        img_w, img_h = frame.size

        prediction_class = 'no_spill'
        bbox = None

        with torch.no_grad():
            img_crops, bboxes = self.frame_to_crops(frame, img_w, img_h, self.zoom_index)
            similarities = self(img_crops)

            max_similarity, max_index = similarities.max(dim=0)
            max_similarity = max_similarity.cpu().numpy().item()
            max_index = max_index.cpu().numpy()

        if max_index < similarities.shape[0]-5:
            self.zoom_index = max_index

        return max_similarity

    def cv2_to_pil_image(self, frame):
        '''
        Args:
            frame ():
        Returns:
        '''

        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(rgb_img)

        return im_pil

    def frame_to_crops(self, frame, img_w, img_h, zoom_index):

        img_patches = []
        bboxes = []
        zoomed_patches = []
        zoomed_bboxes = []
        p_count = 0
        for y_dim in self.y_crop_fractions:
            for x_dim in self.x_crop_fractions:
                patch_bbox = (int(x_dim[0]*img_w),int(y_dim[0]*img_h),int(x_dim[1]*img_w),int(y_dim[1]*img_h))

                patch = self.autocontrast(frame.crop(patch_bbox))
                img_patches.append(self.preprocess(patch))
                bboxes.append(patch_bbox)

                if p_count == zoom_index:
                    p_size = min(patch.size)

                    patch_bboxes = [(int(x_dim[0]*img_w - 0.2*p_size),int(y_dim[0]*img_h - 0.2*p_size),int(x_dim[1]*img_w - 0.5*p_size),int(y_dim[1]*img_h - 0.5*p_size)), \
                                    (int(x_dim[0]*img_w + 0.5*p_size),int(y_dim[0]*img_h - 0.2*p_size),int(x_dim[1]*img_w + 0.2*p_size),int(y_dim[1]*img_h - 0.5*p_size)), \
                                    (int(x_dim[0]*img_w - 0.2*p_size),int(y_dim[0]*img_h + 0.5*p_size),int(x_dim[1]*img_w - 0.5*p_size),int(y_dim[1]*img_h + 0.2*p_size)), \
                                    (int(x_dim[0]*img_w + 0.5*p_size),int(y_dim[0]*img_h + 0.5*p_size),int(x_dim[1]*img_w + 0.2*p_size),int(y_dim[1]*img_h + 0.2*p_size)), \
                                    (int(x_dim[0]*img_w + 0.15*p_size),int(y_dim[0]*img_h + 0.15*p_size),int(x_dim[1]*img_w - 0.15*p_size),int(y_dim[1]*img_h - 0.15*p_size))]
                    
                    for patch_bbox in patch_bboxes:
                        patch = self.autocontrast(frame.crop(patch_bbox))
                        zoomed_patches.append(self.preprocess(patch))
                        zoomed_bboxes.append(patch_bbox)

                p_count += 1

        img_patches = torch.stack(img_patches+zoomed_patches).to('cuda')

        return img_patches, bboxes+zoomed_bboxes