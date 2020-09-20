from __future__ import print_function
import argparse
import torchvision.models as models

import scipy.io as sio
from collections import OrderedDict
from data_test import saliencydata
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import time
import sys
plt.switch_backend('agg')
plt.axis('off')
import matplotlib

NUM_INPUT_CHANNELS = 3

NUM_CLASSES = 2
NUM_OUTPUT_CHANNELS = NUM_CLASSES

BATCH_SIZE = 1
torch.manual_seed(192)


# Arguments
parser = argparse.ArgumentParser(description='Validate a SegNet model')
parser.add_argument('--model_name', required = True, default = "1")
parser.add_argument('--dataset', required = True, default = "1")

args = parser.parse_args()
model_name = args.model_name
dataset = args.dataset
eps  =0.003


def validate():
    model.eval()
    t_start = time.time()
    x = len(val_dataloader)
    k = 0


        
    for batch in val_dataloader:
        input_tensor = batch['image']#.cuda()
        target_tensor = batch['mask']#.cuda()


        prob = model(input_tensor)
        probabilities = prob["out"]#.cuda().clone().requires_grad_(True)
        softmaxed_tensor = torch.nn.functional.softmax(probabilities, dim=1)
        softmaxed_tensor = softmaxed_tensor#.cuda().clone().requires_grad_(True)
        delta = time.time() - t_start

        target_mask = target_tensor[0].cpu()
        input_image = input_tensor[0].cpu()

        predicted_mask = softmaxed_tensor.detach().cpu().numpy()


        if k<20:
            fig = plt.figure()
            plt.axis('off')

            plt.imshow(input_image.transpose(0, 2))
            plt.savefig(os.path.join(OUTPUT_DIR_image, "image_{}.png".format(k)), bbox_inches='tight')

            plt.close(fig)

        fig = plt.figure()
        plt.axis('off')

        predicted = predicted_mask



        target_mx = target_mask.numpy()
        #sio.savemat(OUTPUT_DIR_target, "target_{}.mat".format(k), {'vect':target_mx})

        target_mx[target_mx>0.5] = 1
        target_mx[target_mx<=0.5] = 0
        target_mx = target_mx.astype(int)
        plt.imshow(target_mx, cmap = 'gray')


        plt.savefig(os.path.join(OUTPUT_DIR_target, "target_{}.png".format(k)), bbox_inches='tight')


        plt.close(fig)
        fig = plt.figure()
   

        p  = predicted[0, 0, :, :]
        threshold = 2*np.mean(p)
        #sio.savemat(OUTPUT_DIR_pred_1, "prediction_{}.mat".format(k), {'vect':p})

        
        p[p>threshold] = 1
        p[p<=threshold] = 0

   
        prec_1 += prec(target_mx, p)
        rec_1 += rec(target_mx, p)
        f_1 += f_beta(prec(target_mx, p), rec(target_mx, p))
        mae_1 += mae(target_mx, p)


        plt.axis('off')
        plt.imshow(p, cmap = 'gray')
     
        plt.savefig(os.path.join(OUTPUT_DIR_pred_1, "prediction_{}.png".format(k)), bbox_inches='tight')
 
        plt.close(fig)
   
        fig = plt.figure()
   

        pp  = predicted[0, 1, :, :]
        threshold = 2*np.mean(pp)
        #sio.savemat(OUTPUT_DIR_pred_2, "prediction_{}.mat".format(k), {'vect':pp})

        
        pp[pp>threshold] = 1
        pp[pp<=threshold] = 0


        plt.axis('off')
        plt.imshow(pp, cmap = 'gray')
        plt.savefig(os.path.join(OUTPUT_DIR_pred_2, "prediction_{}.png".format(k)), bbox_inches='tight')
 
        plt.close(fig)
      
   
        k=k+1

    return 





    


if __name__ == "__main__":

    save_dir = '/../../mnt/data4/sofmonk/evaluate_sup_mat/deep_cas/'
    os.mkdir('{}'.format(save_dir + dataset))
    os.makedirs('{}/{}'.format(save_dir+dataset, 'gt'))
    os.makedirs('{}/{}'.format(save_dir+dataset, 'pred_1'))
    os.makedirs('{}/{}'.format(save_dir+dataset, 'pred_2'))

    os.makedirs('{}/{}'.format(save_dir+dataset, 'image'))   
    
    SAVED_MODEL_PATH = model_name + ".pth"
    OUTPUT_DIR_target = '{}/{}'.format(save_dir+dataset, 'gt')
    OUTPUT_DIR_pred_1 = '{}/{}'.format(save_dir+dataset, 'pred_1')
    OUTPUT_DIR_pred_2 = '{}/{}'.format(save_dir+dataset, 'pred_2')

    OUTPUT_DIR_image = '{}/{}'.format(save_dir+dataset, 'image')



    if dataset == 'M':

        data_root = '/home/sofmonk/datasets/saliency/MSRA_10k/MSRA10K_Imgs_GT/MSRA10K_Imgs_GT/Imgs/'
        val_path = data_root + 'testing.txt'
        img_dir =  data_root
        mask_dir = data_root

    if dataset == 'D':

        data_root = '/../../mnt/data4/sofmonk/DUTS/DUTS-TE/'
        val_path = data_root + 'test.txt'
        img_dir =  data_root + 'DUTS-TE-Image/'
        mask_dir = data_root + 'DUTS-TE-Mask/'

    if dataset == 'E':

        data_root = '/../../mnt/data4/sofmonk/ECSSD/'
        val_path = data_root + 'images/all.txt'
        
        img_dir =  data_root + 'images/'
        mask_dir = data_root + 'ground_truth_mask/'

    if dataset == 'P':

        data_root = '/../../mnt/data4/sofmonk/salObj/datasets/'
        val_path = data_root + 'imgs/pascal/all.txt'
        
        img_dir =  data_root + 'imgs/pascal/'
        mask_dir = data_root + 'masks/pascal/'


    if dataset == 'H':

        data_root = '/../../mnt/data4/sofmonk/HKU-IS/'
        val_path = data_root + 'test.txt'
        
        img_dir =  data_root + 'imgs/'
        mask_dir = data_root + 'gt/'

    if dataset == 'O':

        data_root = '/../../mnt/data4/sofmonk/DUT-OMRON/'
        val_path = data_root + 'DUT-OMRON-image/DUT-OMRON-image/all.txt'
 
        img_dir =  data_root + 'DUT-OMRON-image/DUT-OMRON-image/'
        mask_dir = data_root + 'DUT-OMRON-gt-pixelwise.zip/pixelwiseGT-new-PNG/'

    if dataset == 'B':

        data_root = '/../../mnt/data4/sofmonk/MSRA-B/'
        val_path = data_root + 'test.txt'
        
        img_dir =  data_root 
        mask_dir = data_root 

    if dataset == 'T':

        data_root = '/../../mnt/data4/sofmonk/THUR15000/all/'
        val_path = data_root + 'all_test.txt'
        
        img_dir =  data_root 
        mask_dir = data_root     



    if dataset == 'K':

        data_root = 'KAUST_IMAGES'
        val_path = data_root + '/all_test.txt'
        
        img_dir =  data_root 
        mask_dir = data_root     + '/mask'




    f = open(dataset+'_' + model_name+'.txt', 'w')
    sys.stdout = f


    print("validating from dataset--"+ dataset, flush = True)
    #dtype = torch.cuda.FloatTensor
    val_dataset = saliencydata(list_file = val_path, img_dir = img_dir, mask_dir = mask_dir)
    val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False)
    
    
    model = models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes = 2)#.cuda()

    print("validating from best model saved--" + model_name+ ".pth")

    checkpoint = torch.load(model_name + ".pth")
    model.load_state_dict(checkpoint['model_state_dict'])


    validate()
    f.close()
