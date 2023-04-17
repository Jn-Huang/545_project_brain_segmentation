#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
#import import_ipynb
#import dataset.ipynb
import torch
from transformers import Swinv2Model
from transformers import SwinModel
import torchvision as vision
import torch.nn.functional as F
import torch.optim as optim
#from torchmetrics.functional import Dice


# In[2]:

#%run dataset.ipynb
#!ln -s ./545_project_brain_segmentation/dataset.py dataset.py
from dataset import data_loaders
from utils import DiceLoss, log_loss_summary, dsc_per_volume, log_scalar_summary
from utils import postprocess_per_volume, dsc_distribution, plot_dsc, gray2rgb, outline


# In[3]:


path = './kaggle_3m' + '/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_10_mask.tif'
exam = Image.open(path)
im = np.asarray(exam)
im.shape


# In[4]:


#Use dataloaders 
batch_size = 16
workers = 1 #CHANGE to 1 for next round...
image_size = 256 #Default is also this below
train_loaders, valid_loaders = data_loaders(batch_size, workers, image_size,None,None)


# In[21]:


#Below is exploratory analysis of the sizes/outputs of 'hidden state' (feature maps) that pretrained SwinT outputs
#Once we figure out the output format, we can use this when we actually train later

model = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
for i, data in enumerate(train_loaders, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data
    #output = model(inputs,return_dict=False)
    output = model(inputs, output_hidden_states = True)
    print(output.last_hidden_state.size())
    print(len(output.hidden_states))
    #print(output.hidden_states[0].size()) #This is (16,4096,96)
    #print(output.hidden_states[1].size()) #(16,1024,192)
    #print(output.hidden_states[2].size()) #(16,256,384)
    #print(output.hidden_states[3].size()) #(16,64,768)
    #print(output.hidden_states[4].size()) #(16,64,768) just embedding rep
    
    #Now try resizing final output state to be (batch,8,8,768)
    out = output.last_hidden_state.detach().numpy()
    out = np.reshape(np.transpose(out,axes = (0,2,1)),(16,-1,8,8))
    out = torch.from_numpy(out)
    #print(out.size())


# In[14]:



#Initial attempt was to create simple upsampling network (not convolution) to see if dimensions worked.
#But it is better to use trainable parameters!
'''
class Net(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.l0 = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
    #WHEN this works, use deconv upsampling like in U-Net
    #Even more advanced is MMSegment
    #self.l1 = torch.nn.UpSample((32,32)) #Go back up to regular size in simple way...same way you came down in SwinT
    self.l1 = torch.nn.Upsample((64,64))
    self.l2 = torch.nn.Upsample((256,256))
    
  def forward(self, x):
    x = self.l0(x)
    print(x)
    torch.stack(x).to(device)
    x = self.l1(x)
    x = self.l2(x)
    #x= self.l3(x)
    #print(x.shape)
    return x
'''

#Create network that performs 'deconvolution', or convolution upsampling 
class Net(torch.nn.Module):
  def __init__(self):
    super().__init__()
    '''OLD attempts when trying to train swinT AND conv layers (vs freezing swint as here)
    #self.l0 = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
    #self.l0 = self.l0(return_dict=False) #Don't return Model type; want pytorch tensors
    #self.l0 = self.l0.train() #Set to training mode not eval mode... OR eval if don't want to train...
    '''
    #Ideally we would like to also train swint but it is the next stage of complexity in model building/tuning
    #Even more advanced is MMSegment
    in_channels = 768
    self.l1 = torch.nn.ConvTranspose2d(in_channels,384,kernel_size=9,stride=1)#To get to (batch,384,16,16)
    self.l2 = torch.nn.ConvTranspose2d(384,96,kernel_size = 4,stride=4) #To get to (batch,96,64,64)
    self.l3 = torch.nn.ConvTranspose2d(96,1,kernel_size=4,stride=4) #To get to (batch,1,256,256) #Maybe add intermediate layer...
    #Try simple with no activations at first
    self.l4 = torch.nn.ReLU() #So we get probability of pixel being of either background or mask(2 classes here)
        
  def forward(self, x):
    x = self.l1(x)
    x = self.l2(x)
    x= self.l3(x)
    #print(x.shape) #To confirm
    x = self.l4(x)
    return x


# In[15]:


#Train the custom upsampling layers 
def train_validate(epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
    SwinM = Net()
    model.to(device)
    SwinM.to(device)
    
    #criterion = torch.nn.MSELoss(reduction='sum') #Use Mean Square Error between Images
    criterion = DiceLoss()
    #optimizer = optim.SGD(SwinM.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(SwinM.parameters(),lr=0.0001)
    loaders = {"train": train_loaders, "valid": valid_loaders}

    loss_train = []
    loss_valid = []
    
    for ep in range(epochs):
        for phase in ['train','valid']:
            validation_pred = []
            validation_true = []

            for i, data in enumerate(loaders[phase], 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                #print(labels)
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"): #Only enable grad calcs if train phase
                    output = model(inputs, output_hidden_states = True)
                    out = output.last_hidden_state.detach().numpy()
                    out = np.reshape(np.transpose(out,axes = (0,2,1)),(16,-1,8,8)) #Make shape be same as input image
                    out = torch.from_numpy(out) #Convert to pytorch tensor
                    out = SwinM(out)

                    loss = criterion(out, labels)

                    if phase == "valid":
                        loss_valid.append(loss.item())
                        y_pred_np = out.detach().cpu().numpy()
                        validation_pred.extend(
                            [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                        )
                        y_true_np = labels.detach().cpu().numpy()
                        validation_true.extend(
                            [y_true_np[s] for s in range(y_true_np.shape[0])]
                        )
                        if i % 10 == 0: # print and run validation every 30 mini-batches! LOOK at my pytorch class nb
                            print(f'[{ep + 1}, {i + 1:5d}] validation loss: {(sum(loss_valid)/ 30):.3f}')

                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()
                        if i % 10 == 0: # print and run validation every 30 mini-batches! LOOK at my pytorch class nb
                            print(f'[{ep + 1}, {i + 1:5d}] training loss: {(sum(loss_train) / 30):.3f}')


# In[1]:

train_validate(3)

#Ideally should define class and function first, and run dataloader creation + train/test in order in main in .py file




