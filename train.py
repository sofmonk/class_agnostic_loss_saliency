import torchvision.models as models
from dataset import saliencydata
import argparse
import os
import torch
from torch.utils.data import DataLoader
import time
import numpy as np
import collections
from cas import lss
import sys


import config
from config import ALPHA, NUM_INPUT_CHANNELS, NUM_CLASSES, NUM_OUTPUT_CHANNELS, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS#, plott


import datetime
now = datetime.datetime.now()

torch.set_printoptions(threshold=500000)
# Arguments
parser = argparse.ArgumentParser(description = 'Train a model')
parser.add_argument('--model_name', default = "casl")
parser.add_argument('--loss_v', default = "casl")
#parser.add_argument('--train_dataset', default = 'train_10.txt' )

args = parser.parse_args()
cwd = os.getcwd()

model_name = args.model_name
loss_v = args.loss_v
#train_dataset = args.train_dataset

def train():
    is_better = True
    prev_loss = float('inf')
    
    model.train()
    #pl = 0
    
    print("+++++++++++++++++++++SUPERVISED TRAINING +++++++++++++++++++++++")

    for epoch in range(NUM_EPOCHS):

            total_loss = 0.0
            t_start = time.time()
        
            for batch in train_dataloader:
                input_tensor = batch['image'].cuda().clone().requires_grad_(True) #.cuda()
                target_tensor = batch['mask'].cuda().clone() #.cuda()
                

                prob = model(input_tensor)

                probabilities = prob["out"].cuda().clone().requires_grad_(True)
                softmaxed_tensor = torch.nn.functional.softmax(probabilities, dim=1)
                softmaxed_tensor = softmaxed_tensor.cuda().clone().requires_grad_(True)

                optimizer.zero_grad()
                tt = torch.zeros(BATCH_SIZE, NUM_CLASSES, 256, 256).cuda().clone()
                for i in range(BATCH_SIZE):
                    #print(i, tt.shape, target_tensor.shape)
                    tt[i, :, :, :] = target_tensor[i, :, :].cuda().clone()
                tt = tt.cuda().clone().requires_grad_(True)
                #losss = criterion(probabilities, target_tensor).cuda().clone().requires_grad_(True)

                losss = loss(tt, softmaxed_tensor).cuda().clone().requires_grad_(True)
        

                losss.backward()

                optimizer.step()
                total_loss += losss.item()


            delta = time.time() - t_start

            is_better = total_loss < prev_loss


            if is_better:
                prev_loss = total_loss
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            },  model_name + "_best.pth")
                print("-------------------BEST checkpoint saved!--------------------", flush = True)


            if epoch%1000==0:
                #prev_loss = total_loss
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            },  model_name + "epoch_{}.pth".format(epoch))
                print("-------------------EPOCH checkpoint saved!--------------------", flush = True)



 
            print("Epoch #{}\tLoss: {:.8f}\t Time: {:2f}s".format(epoch, total_loss, delta), flush = True)



if __name__ == "__main__":

    f = open(model_name  + '.txt', 'w')
    sys.stdout = f

    print("Time and Date", now.isoformat())
    pid = os.getpid()
    print("pid", pid)


    data_root = '/../../mnt/data4/sofmonk/DUTS/DUTS-TR/'
    train_path = data_root + 'train.txt'
    img_dir =  data_root + 'DUTS-TR-Image/'
    mask_dir = data_root + 'DUTS-TR-Mask/'





    print("loading from dataset~~~" + train_path, flush = True)
    print("model will be saved with name~~~" + model_name + ".pth", flush = True)
    print("using loss_v~~~" + loss_v, flush = True)
    print("ALPHA = " + str(ALPHA), flush = True)
    print("BATCH_SIZE = " + str(BATCH_SIZE), flush = True)
    print("NUM_CLASSES = " + str(NUM_CLASSES), flush = True)
    print("NUM_EPOCHS = " + str(NUM_EPOCHS), flush = True)
    

    dtype = torch.cuda.FloatTensor
    
    train_dataset = saliencydata(list_file = train_path, img_dir = img_dir, mask_dir = mask_dir)
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, pin_memory = True)
    
    model = models.segmentation.fcn_resnet101(pretrained=False, num_classes = 2).cuda()
    print(model)
    #print(summary(model, (3, 224, 224), BATCH_SIZE))
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    

    # pre_dict = model_pre.state_dict()
    # model_dict = model.state_dict()
    # pre_dict = {k: v for k, v in pre_dict.items() if   (k in model_dict) and (model_dict[k].shape == pre_dict[k].shape)}
    # model_dict.update(pre_dict)  
    # model.load_state_dict(model_dict)
    # checkpoint = torch.load("ce_best.pth")
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

    #criterion = torch.nn.CrossEntropyLoss().cuda()
    loss = lss()
    train()
    f.close()