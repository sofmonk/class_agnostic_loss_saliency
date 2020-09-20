import torch
import random

import os
import sys

import config
from config import ALPHA, NUM_INPUT_CHANNELS, NUM_CLASSES, NUM_OUTPUT_CHANNELS, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS

eps = 0.00003
class lss(torch.nn.Module):
   def __init__(self):
           super(lss, self).__init__()

   def forward(self, gt, softmaxed_tensor):           
           gt_comp = 1-gt

           gt = gt+eps
           gt_comp = gt_comp + eps
           gt_comp = gt_comp.clone().requires_grad_(True) #.cuda()
           
           multiply = softmaxed_tensor*gt    ##BxCxWxH
           summ = torch.sum(multiply, dim = (2, 3))  #BxC
           num_pixels = torch.sum(gt, dim = (2, 3))  #BxC
           
           val = torch.div(summ, num_pixels)

           multiply_comp = softmaxed_tensor*gt_comp
           summ_comp = torch.sum(multiply_comp, dim = (2, 3))  #BxC
           num_pixels_comp = torch.sum(gt_comp, dim = (2, 3))  #BxC
          
           val_comp = torch.div(summ_comp, num_pixels_comp)


           term_2 = torch.sum((val - val_comp)**2, dim = 1)
           u_a = (multiply-(gt*val[:, :, None, None]))**2
           s = torch.sum(u_a, dim = (2, 3))
           n = torch.sum(gt, dim = (2, 3))
           n = n + eps
           
           term_1 = torch.sum(torch.div(s, n), dim =1)

           loss_batch = ALPHA*torch.sum(term_1) - (1-ALPHA)*torch.sum(term_2)
           return loss_batch/BATCH_SIZE
