import os
import csv
from PIL import Image
from PIL import ImageFilter
import pandas as pd
import numpy as np
import random
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy import ndimage
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import scipy.io
from scipy.misc import toimage
from torchsample.transforms import *
from sklearn.metrics import confusion_matrix
import glob
from torchvision.datasets.folder import IMG_EXTENSIONS
import time

#matlab
import pytorch_ssim
import matlab
import matlab.engine



IMG_EXTENSIONS.append('tif')

# Hyper Parameterss
EPOCH = 1500             # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 1
BATCH_SIZE_D = 20
rootdir_t = './train1/'
rootdir_t2 = './train2/'
rootdir_t3 = './train3/'
rootdir_v = './val_14mci/'
STEP = 225

MSE_T = np.zeros(EPOCH)
MSE_V = np.zeros(EPOCH)

dir_name = "MIC_GAN_EDSR_new_36_rand"
dir_name1 = "MIC_GAN_EDSR_new_36_rand"

pre_path = "./pretrain_params"


output_path = "./fig_gen"
path = "./params/" +  dir_name

pre_path1 = "./params/" + dir_name1
v_MSE_name = "./MSE/" + dir_name + ".txt"


crop_size = 96#96
sigma = 0.000001
rep = 43# 22 edsr: 420



outsave=1
psave=1


ch = 64
in_ch = 4
ksize = 3
pad = 1
res_rate = 0.1
class D_crop2(nn.Module):
	def __init__(self):
        	super(D_crop2, self).__init__()
		self.input = nn.Sequential(         
            		nn.Conv2d(
              	  		in_channels=1,              # input height
                		out_channels=64,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                  
            			padding=pad,
				),

        		nn.BatchNorm2d(64),
			nn.Dropout2d(0.2),

 	    		#nn.LeakyReLU(inplace=True),
			nn.LeakyReLU(0.2, inplace=True)
		)
		self.layer1 = nn.Sequential(         
            		nn.Conv2d(
              	  		in_channels=64,              # input height
                		out_channels=64,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,            			
			),
	    		
 	    		#nn.LeakyReLU(inplace=True),
			#nn.Dropout2d(0.25),
        		nn.BatchNorm2d(64),
			nn.Dropout(0.2),
			nn.LeakyReLU(0.2, inplace=True),

			#nn.BatchNorm2d(64, 0.8),
			#nn.MaxPool2d(3, stride=2),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=ksize, stride=2, padding=1),
        		nn.BatchNorm2d(64),
			nn.Dropout(0.2),			
			nn.LeakyReLU(0.2, inplace=True),
			#nn.Dropout2d(0.25),
			#nn.BatchNorm2d(64, 0.8),

		)
		self.layer2 = nn.Sequential(  
			nn.Conv2d(
              	  		in_channels=64,              # input height
                		out_channels=64,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,             			
			),

        		#nn.BatchNorm2d(64),
			#nn.Dropout2d(0.25),
			#nn.LeakyReLU(inplace=True),
        		nn.BatchNorm2d(64),
			nn.Dropout(0.2),
			nn.LeakyReLU(0.2, inplace=True),	

			#nn.BatchNorm2d(64, 0.8),

			#nn.MaxPool2d(3, stride=2),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=ksize, stride=2, padding=1),
        		nn.BatchNorm2d(64),
			nn.Dropout(0.2),
			nn.LeakyReLU(0.2, inplace=True),
			#nn.Dropout2d(0.25),
			#nn.BatchNorm2d(64, 0.8),

		)

		self.layer3 = nn.Sequential(  
			
			nn.Conv2d(
              	  		in_channels=64,              # input height
                		out_channels=96,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,             			
			),

        		#nn.BatchNorm2d(96),
			#nn.Dropout2d(0.25),
			#nn.LeakyReLU(inplace=True),
        		nn.BatchNorm2d(96),
			nn.Dropout(0.2),
			nn.LeakyReLU(0.2, inplace=True),

			#nn.BatchNorm2d(96, 0.8),

			#nn.MaxPool2d(3, stride=2),
			nn.Conv2d(in_channels=96, out_channels=96, kernel_size=ksize, stride=2, padding=1),
        		nn.BatchNorm2d(96),
			nn.Dropout(0.2),
			nn.LeakyReLU(0.2, inplace=True),
			#nn.Dropout2d(0.25),
			#nn.BatchNorm2d(96, 0.8),

		)
		self.layer4 = nn.Sequential(  	
			nn.Conv2d(
              	  		in_channels=96,              # input height
                		out_channels=128,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,             			
			),

        		#nn.BatchNorm2d(160),
			#nn.Dropout2d(0.25),
			#nn.LeakyReLU(inplace=True),
        		nn.BatchNorm2d(128),
			nn.Dropout(0.2),
			nn.LeakyReLU(0.2, inplace=True),

			#nn.BatchNorm2d(128, 0.8),

			#nn.MaxPool2d(3, stride=2),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=ksize, stride=2, padding=1),
        		nn.BatchNorm2d(128),
			nn.Dropout(0.2),
			nn.LeakyReLU(0.2, inplace=True),
			#nn.Dropout2d(0.25),
			#nn.BatchNorm2d(128, 0.8),

		)
		
		self.output = nn.Sequential( 
			#nn.Linear(1728, 320),
			nn.Linear(128*36, 360),
			nn.Dropout2d(0.05),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(360, 100),
			nn.Dropout2d(0.05),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(100, 1),


		)
		
		for m in self.modules():
            		if isinstance(m, nn.Conv2d):
                		n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                		m.weight.data.normal_(0, math.sqrt(2. / n))
	

    	def I_weights(self, m):
        	#print(m)
        	if type(m) == nn.Conv2d:
            		w = 1
            	if m.in_channels == m.out_channels and m.kernel_size > 2:
               		w = 3
            	alpha = math.pow(m.kernel_size[0]*1.0/m.stride[0]/w, 2.0)
            	scale = math.sqrt(6.0/(m.in_channels+m.out_channels*alpha))
            	#print scale
            	m.weight.data.uniform_(-scale,scale)
            	#print(m.weight) 
   
    	"""
   	def inititalize_W(self):
        #self.I_weights(self.conv1[0])
        #self.I_weights(self.conv2[0])
        #self.I_weights(self.conv3[0])
    	"""


    	def forward(self, x):

		x = self.input(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)


		#x = x.view(-1,1728)
		#print x.size()
		x = x.view(x.shape[0],-1)
		out = self.output(x)
	
		return out


	def train_(self):

		for m in self.modules():
            		if isinstance(m, nn.Conv2d):
				m.weight.requires_grad = True
			elif isinstance(m, nn.Linear):
				m.weight.requires_grad = True

	def not_train_(self):

		for m in self.modules():
            		if isinstance(m, nn.Conv2d):
				m.weight.requires_grad = False
			elif isinstance(m, nn.Linear):
				m.weight.requires_grad = False

    	def save_p(self, epoch):
		modelparam = path + "/D_input_Epoch_" + str(epoch) +".pkl"
		torch.save(self.input.state_dict(), modelparam) 
		modelparam = path + "/D_layer1_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer1.state_dict(), modelparam)
		modelparam = path + "/D_layer2_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer2.state_dict(), modelparam)
		modelparam = path + "/D_layer3_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer3.state_dict(), modelparam)
		modelparam = path + "/D_layer4_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer4.state_dict(), modelparam)
		#modelparam = path + "/D_layer5_Epoch_" + str(epoch) +".pkl"
	        #torch.save(self.layer5.state_dict(), modelparam)
		modelparam = path + "/D_output_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.output.state_dict(), modelparam)
		print "Epoch ", epoch, ":  Model saved!!"

	def restore_netparam(self, epoch):	
		modelparam = pre_path1 + "/D_input_Epoch_" + str(epoch) +".pkl"
		self.input.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_layer1_Epoch_" + str(epoch) +".pkl"
	        self.layer1.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_layer2_Epoch_" + str(epoch) +".pkl"
	        self.layer2.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_layer3_Epoch_" + str(epoch) +".pkl"
	        self.layer3.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_layer4_Epoch_" + str(epoch) +".pkl"
	        self.layer4.load_state_dict(torch.load(modelparam))
		#modelparam = pre_path1 + "/D_layer5_Epoch_" + str(epoch) +".pkl"
	        #self.layer5.load_state_dict(torch.load(modelparam))
	        modelparam = pre_path1 + "/D_output_Epoch_" + str(epoch) +".pkl"
	        self.output.load_state_dict(torch.load(modelparam))
		print "Epoch ", epoch, ":  Model restored!!"


    	def save_p_lr(self, epoch):
		modelparam = path + "/D_lr_input_Epoch_" + str(epoch) +".pkl"
		torch.save(self.input.state_dict(), modelparam) 
		modelparam = path + "/D_lr_layer1_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer1.state_dict(), modelparam)
		modelparam = path + "/D_lr_layer2_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer2.state_dict(), modelparam)
		modelparam = path + "/D_lr_layer3_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer3.state_dict(), modelparam)
		modelparam = path + "/D_lr_layer4_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer4.state_dict(), modelparam)
		#modelparam = path + "/D_lr_layer5_Epoch_" + str(epoch) +".pkl"
	        #torch.save(self.layer5.state_dict(), modelparam)
		modelparam = path + "/D_lr_output_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.output.state_dict(), modelparam)
		print "Epoch ", epoch, ":  Model saved!!"

	def restore_netparam_lr(self, epoch):	
		modelparam = pre_path1 + "/D_lr_input_Epoch_" + str(epoch) +".pkl"
		self.input.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_lr_layer1_Epoch_" + str(epoch) +".pkl"
	        self.layer1.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_lr_layer2_Epoch_" + str(epoch) +".pkl"
	        self.layer2.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_lr_layer3_Epoch_" + str(epoch) +".pkl"
	        self.layer3.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_lr_layer4_Epoch_" + str(epoch) +".pkl"
	        self.layer4.load_state_dict(torch.load(modelparam))
		#modelparam = pre_path1 + "/D_lr_layer5_Epoch_" + str(epoch) +".pkl"
	        #self.layer5.load_state_dict(torch.load(modelparam))
	        modelparam = pre_path1 + "/D_lr_output_Epoch_" + str(epoch) +".pkl"
	        self.output.load_state_dict(torch.load(modelparam))
		print "Epoch ", epoch, ":  Model restored!!"

class D_crop(nn.Module):
	def __init__(self):
        	super(D_crop, self).__init__()
		self.input = nn.Sequential(         
            		nn.Conv2d(
              	  		in_channels=1,              # input height
                		out_channels=64,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                  
            			padding=pad,
				),

        		#nn.BatchNorm2d(64),
			#nn.Dropout2d(0.5),

 	    		#nn.LeakyReLU(inplace=True),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
		)
		self.layer1 = nn.Sequential(         
            		#nn.Conv2d(
              	  	#	in_channels=64,              # input height
                	#	out_channels=64,            # n_filters
                	#	kernel_size=ksize,              # filter size
                	#	stride=1,                   # filter movement/step                 
			#	padding=pad,            			
			#),
	    		
 	    		#nn.LeakyReLU(inplace=True),
			#nn.Dropout2d(0.25),
			#nn.LeakyReLU(negative_slope=0.3333, inplace=False),
			#nn.MaxPool2d(3, stride=2),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=ksize, stride=2, padding=0),
                	nn.LeakyReLU(inplace=False),
		)
		self.layer2 = nn.Sequential(  
			#nn.Conv2d(
              	  	#	in_channels=64,              # input height
                	#	out_channels=64,            # n_filters
                	#	kernel_size=ksize,              # filter size
                	#	stride=1,                   # filter movement/step                 
			#	padding=pad,             			
			#),

        		#nn.BatchNorm2d(64),
			#nn.Dropout2d(0.25),
			#nn.LeakyReLU(inplace=True),
			#nn.LeakyReLU(negative_slope=0.3333, inplace=False),			

			#nn.MaxPool2d(3, stride=2),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=ksize, stride=2, padding=0),
                	nn.LeakyReLU(inplace=False),

		)

		self.layer3 = nn.Sequential(  
			
			nn.Conv2d(
              	  		in_channels=64,              # input height
                		out_channels=96,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,             			
			),

        		#nn.BatchNorm2d(96),
			#nn.Dropout2d(0.25),
			#nn.LeakyReLU(inplace=True),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),

			#nn.MaxPool2d(3, stride=2),
			nn.Conv2d(in_channels=96, out_channels=96, kernel_size=ksize, stride=2, padding=0),
                	nn.LeakyReLU(inplace=False),

		)
		self.layer4 = nn.Sequential(  	
			nn.Conv2d(
              	  		in_channels=96,              # input height
                		out_channels=128,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,             			
			),

        		#nn.BatchNorm2d(160),
			#nn.Dropout2d(0.25),
			#nn.LeakyReLU(inplace=True),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),

			#nn.MaxPool2d(3, stride=2),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=ksize, stride=2, padding=0),
                	nn.LeakyReLU(inplace=False),

		)
		
		self.output = nn.Sequential( 
			#nn.Linear(1728, 320),
			nn.Linear(64*64, 2000),
			#nn.Dropout(0.30),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
			nn.Linear(2000, 1000),
			#nn.Dropout(0.30),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
			nn.Linear(1000, 500),
			#nn.Dropout(0.30),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
			nn.Linear(500, 250),
			#nn.Dropout(0.30),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
			nn.Linear(250, 100),
			#nn.Dropout(0.30),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
			nn.Linear(100, 1),
			#nn.Dropout(0.20),
			#nn.Sigmoid(),
			#nn.LeakyReLU(negative_slope=0.3333, inplace=False),


		)
		
		for m in self.modules():
            		if isinstance(m, nn.Conv2d):
                		n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                		m.weight.data.normal_(0, math.sqrt(2. / n))
	

    	def I_weights(self, m):
        	#print(m)
        	if type(m) == nn.Conv2d:
            		w = 1
            	if m.in_channels == m.out_channels and m.kernel_size > 2:
               		w = 3
            	alpha = math.pow(m.kernel_size[0]*1.0/m.stride[0]/w, 2.0)
            	scale = math.sqrt(6.0/(m.in_channels+m.out_channels*alpha))
            	#print scale
            	m.weight.data.uniform_(-scale,scale)
            	#print(m.weight) 
   
    	"""
   	def inititalize_W(self):
        #self.I_weights(self.conv1[0])
        #self.I_weights(self.conv2[0])
        #self.I_weights(self.conv3[0])
    	"""


    	def forward(self, x):

		#x = self.input(x)
		#x = self.layer1(x)
		#x = self.layer2(x)
		#x = self.layer3(x)
		#x = self.layer4(x)
		#x = self.layer5(x)

		#x = x.view(-1,1728)
		#print x.size()
		x = x.view(x.shape[0],-1)
		out = self.output(x)
	
		return out


	def train_(self):

		for m in self.modules():
            		if isinstance(m, nn.Conv2d):
				m.weight.requires_grad = True
			elif isinstance(m, nn.Linear):
				m.weight.requires_grad = True

	def not_train_(self):

		for m in self.modules():
            		if isinstance(m, nn.Conv2d):
				m.weight.requires_grad = False
			elif isinstance(m, nn.Linear):
				m.weight.requires_grad = False

    	def save_p(self, epoch):
		modelparam = path + "/D_input_Epoch_" + str(epoch) +".pkl"
		torch.save(self.input.state_dict(), modelparam) 
		modelparam = path + "/D_layer1_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer1.state_dict(), modelparam)
		modelparam = path + "/D_layer2_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer2.state_dict(), modelparam)
		modelparam = path + "/D_layer3_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer3.state_dict(), modelparam)
		modelparam = path + "/D_layer4_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer4.state_dict(), modelparam)
		#modelparam = path + "/D_layer5_Epoch_" + str(epoch) +".pkl"
	        #torch.save(self.layer5.state_dict(), modelparam)
		modelparam = path + "/D_output_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.output.state_dict(), modelparam)
		print "Epoch ", epoch, ":  Model saved!!"

	def restore_netparam(self, epoch):	
		modelparam = pre_path1 + "/D_input_Epoch_" + str(epoch) +".pkl"
		self.input.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_layer1_Epoch_" + str(epoch) +".pkl"
	        self.layer1.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_layer2_Epoch_" + str(epoch) +".pkl"
	        self.layer2.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_layer3_Epoch_" + str(epoch) +".pkl"
	        self.layer3.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_layer4_Epoch_" + str(epoch) +".pkl"
	        self.layer4.load_state_dict(torch.load(modelparam))
		#modelparam = pre_path1 + "/D_layer5_Epoch_" + str(epoch) +".pkl"
	        #self.layer5.load_state_dict(torch.load(modelparam))
	        modelparam = pre_path1 + "/D_output_Epoch_" + str(epoch) +".pkl"
	        self.output.load_state_dict(torch.load(modelparam))
		print "Epoch ", epoch, ":  Model restored!!"


    	def save_p_lr(self, epoch):
		modelparam = path + "/D_lr_input_Epoch_" + str(epoch) +".pkl"
		torch.save(self.input.state_dict(), modelparam) 
		modelparam = path + "/D_lr_layer1_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer1.state_dict(), modelparam)
		modelparam = path + "/D_lr_layer2_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer2.state_dict(), modelparam)
		modelparam = path + "/D_lr_layer3_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer3.state_dict(), modelparam)
		modelparam = path + "/D_lr_layer4_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer4.state_dict(), modelparam)
		#modelparam = path + "/D_lr_layer5_Epoch_" + str(epoch) +".pkl"
	        #torch.save(self.layer5.state_dict(), modelparam)
		modelparam = path + "/D_lr_output_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.output.state_dict(), modelparam)
		print "Epoch ", epoch, ":  Model saved!!"

	def restore_netparam_lr(self, epoch):	
		modelparam = pre_path1 + "/D_lr_input_Epoch_" + str(epoch) +".pkl"
		self.input.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_lr_layer1_Epoch_" + str(epoch) +".pkl"
	        self.layer1.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_lr_layer2_Epoch_" + str(epoch) +".pkl"
	        self.layer2.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_lr_layer3_Epoch_" + str(epoch) +".pkl"
	        self.layer3.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_lr_layer4_Epoch_" + str(epoch) +".pkl"
	        self.layer4.load_state_dict(torch.load(modelparam))
		#modelparam = pre_path1 + "/D_lr_layer5_Epoch_" + str(epoch) +".pkl"
	        #self.layer5.load_state_dict(torch.load(modelparam))
	        modelparam = pre_path1 + "/D_lr_output_Epoch_" + str(epoch) +".pkl"
	        self.output.load_state_dict(torch.load(modelparam))
		print "Epoch ", epoch, ":  Model restored!!"

class D(nn.Module):
	def __init__(self):
        	super(D, self).__init__()
		self.input = nn.Sequential(         
            		nn.Conv2d(
              	  		in_channels=1,              # input height
                		out_channels=64,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                  
            			padding=pad,
				),

        		#nn.BatchNorm2d(64),
			#nn.Dropout2d(0.5),

 	    		#nn.LeakyReLU(inplace=True),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
		)
		self.layer1 = nn.Sequential(         
            		nn.Conv2d(
              	  		in_channels=64,              # input height
                		out_channels=64,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,            			
			),
	    		
 	    		#nn.LeakyReLU(inplace=True),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
			nn.MaxPool2d(3, stride=2),
		)
		self.layer2 = nn.Sequential(  
			nn.Conv2d(
              	  		in_channels=64,              # input height
                		out_channels=64,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,             			
			),

        		#nn.BatchNorm2d(64),
			#nn.Dropout2d(0.5),
			#nn.LeakyReLU(inplace=True),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
			nn.Conv2d(
              	  		in_channels=64,              # input height
                		out_channels=64,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,             			
			),
			#nn.LeakyReLU(inplace=True),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),			

			nn.MaxPool2d(3, stride=2),

		)

		self.layer3 = nn.Sequential(  
			
			nn.Conv2d(
              	  		in_channels=64,              # input height
                		out_channels=96,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,             			
			),

        		#nn.BatchNorm2d(96),
			#nn.Dropout2d(0.5),
			#nn.LeakyReLU(inplace=True),'
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),

			nn.Conv2d(
              	  		in_channels=96,              # input height
                		out_channels=96,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,             			
			),
			#nn.LeakyReLU(inplace=True),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
			nn.MaxPool2d(3, stride=2),
			nn.Conv2d(
              	  		in_channels=96,              # input height
                		out_channels=128,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,              			
			),
			#nn.LeakyReLU(inplace=True),

			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
			nn.Conv2d(
              	  		in_channels=128,              # input height
                		out_channels=128,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,             			
			),
			#nn.LeakyReLU(inplace=True),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),

			nn.MaxPool2d(3, stride=2),

		)
		self.layer4 = nn.Sequential(  	
			nn.Conv2d(
              	  		in_channels=128,              # input height
                		out_channels=160,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,             			
			),

        		#nn.BatchNorm2d(160),
			#nn.Dropout2d(0.5),
			#nn.LeakyReLU(inplace=True),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
			nn.Conv2d(
              	  		in_channels=160,              # input height
                		out_channels=160,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,             			
			),
			#nn.LeakyReLU(inplace=True),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),

			nn.MaxPool2d(3, stride=2),

		)

		self.layer5 = nn.Sequential(  
			nn.Conv2d(
              	  		in_channels=160,              # input height
                		out_channels=192,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,             			
			),

        		#nn.BatchNorm2d(192),
			#nn.Dropout2d(0.5),
			#nn.LeakyReLU(inplace=True),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
			nn.Conv2d(
              	  		in_channels=192,              # input height
                		out_channels=192,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,             			
			),
			#nn.LeakyReLU(inplace=True),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
			nn.MaxPool2d(3, stride=2),
			
		)

		
		self.output = nn.Sequential( 
			#nn.Linear(1728, 320),
			nn.Linear(768, 360),
			#nn.LeakyReLU(inplace=True),
			#nn.LeakyReLU(negative_slope=0.3333, inplace=False),
			#nn.Dropout(0.5),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
			nn.Linear(360, 300),
			#nn.Dropout(0.5),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
			nn.Linear(300, 1),

			#nn.LeakyReLU(negative_slope=0.3333, inplace=False),


		)
		
		for m in self.modules():
            		if isinstance(m, nn.Conv2d):
                		n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                		m.weight.data.normal_(0, math.sqrt(2. / n))
	

    	def I_weights(self, m):
        	#print(m)
        	if type(m) == nn.Conv2d:
            		w = 1
            	if m.in_channels == m.out_channels and m.kernel_size > 2:
               		w = 3
            	alpha = math.pow(m.kernel_size[0]*1.0/m.stride[0]/w, 2.0)
            	scale = math.sqrt(6.0/(m.in_channels+m.out_channels*alpha))
            	#print scale
            	m.weight.data.uniform_(-scale,scale)
            	#print(m.weight) 
   
    	"""
   	def inititalize_W(self):
        #self.I_weights(self.conv1[0])
        #self.I_weights(self.conv2[0])
        #self.I_weights(self.conv3[0])
    	"""


    	def forward(self, x):

		x = self.input(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)

		#x = x.view(-1,1728)
		x = x.view(-1,768)
		out = self.output(x)
	
		return out

    	def save_p(self, epoch):
		modelparam = path + "/D_input_Epoch_" + str(epoch) +".pkl"
		torch.save(self.input.state_dict(), modelparam) 
		modelparam = path + "/D_layer1_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer1.state_dict(), modelparam)
		modelparam = path + "/D_layer2_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer2.state_dict(), modelparam)
		modelparam = path + "/D_layer3_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer3.state_dict(), modelparam)
		modelparam = path + "/D_layer4_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer4.state_dict(), modelparam)
		modelparam = path + "/D_layer5_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer5.state_dict(), modelparam)
		modelparam = path + "/D_output_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.output.state_dict(), modelparam)
		print "Epoch ", epoch, ":  Model saved!!"

	def restore_netparam(self, epoch):	
		modelparam = pre_path1 + "/D_input_Epoch_" + str(epoch) +".pkl"
		self.input.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_layer1_Epoch_" + str(epoch) +".pkl"
	        self.layer1.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_layer2_Epoch_" + str(epoch) +".pkl"
	        self.layer2.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_layer3_Epoch_" + str(epoch) +".pkl"
	        self.layer3.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_layer4_Epoch_" + str(epoch) +".pkl"
	        self.layer4.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_layer5_Epoch_" + str(epoch) +".pkl"
	        self.layer5.load_state_dict(torch.load(modelparam))
	        modelparam = pre_path1 + "/D_output_Epoch_" + str(epoch) +".pkl"
	        self.output.load_state_dict(torch.load(modelparam))
		print "Epoch ", epoch, ":  Model restored!!"


    	def save_p_lr(self, epoch):
		modelparam = path + "/D_lr_input_Epoch_" + str(epoch) +".pkl"
		torch.save(self.input.state_dict(), modelparam) 
		modelparam = path + "/D_lr_layer1_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer1.state_dict(), modelparam)
		modelparam = path + "/D_lr_layer2_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer2.state_dict(), modelparam)
		modelparam = path + "/D_lr_layer3_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer3.state_dict(), modelparam)
		modelparam = path + "/D_lr_layer4_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer4.state_dict(), modelparam)
		modelparam = path + "/D_lr_layer5_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer5.state_dict(), modelparam)
		modelparam = path + "/D_lr_output_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.output.state_dict(), modelparam)
		print "Epoch ", epoch, ":  Model saved!!"

	def restore_netparam_lr(self, epoch):	
		modelparam = pre_path1 + "/D_lr_input_Epoch_" + str(epoch) +".pkl"
		self.input.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_lr_layer1_Epoch_" + str(epoch) +".pkl"
	        self.layer1.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_lr_layer2_Epoch_" + str(epoch) +".pkl"
	        self.layer2.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_lr_layer3_Epoch_" + str(epoch) +".pkl"
	        self.layer3.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_lr_layer4_Epoch_" + str(epoch) +".pkl"
	        self.layer4.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_lr_layer5_Epoch_" + str(epoch) +".pkl"
	        self.layer5.load_state_dict(torch.load(modelparam))
	        modelparam = pre_path1 + "/D_lr_output_Epoch_" + str(epoch) +".pkl"
	        self.output.load_state_dict(torch.load(modelparam))
		print "Epoch ", epoch, ":  Model restored!!"

class D2(nn.Module):
	def __init__(self):
        	super(D2, self).__init__()
		self.input = nn.Sequential(         
            		nn.Conv2d(
              	  		in_channels=1,              # input height
                		out_channels=64,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                  
            			padding=pad,
				),

 	    		#nn.LeakyReLU(inplace=True),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
		)
		self.layer1 = nn.Sequential(         
            		nn.Conv2d(
              	  		in_channels=64,              # input height
                		out_channels=64,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,            			
			),
	    		
 	    		#nn.LeakyReLU(inplace=True),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
			nn.MaxPool2d(3, stride=2),
		)
		self.layer2 = nn.Sequential(  
			nn.Conv2d(
              	  		in_channels=64,              # input height
                		out_channels=64,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,             			
			),
			#nn.LeakyReLU(inplace=True),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
			nn.Conv2d(
              	  		in_channels=64,              # input height
                		out_channels=64,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,             			
			),
			#nn.LeakyReLU(inplace=True),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),			

			nn.MaxPool2d(3, stride=2),

		)

		self.layer3 = nn.Sequential(  
			
			nn.Conv2d(
              	  		in_channels=64,              # input height
                		out_channels=96,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,             			
			),
			#nn.LeakyReLU(inplace=True),'
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),

			nn.Conv2d(
              	  		in_channels=96,              # input height
                		out_channels=96,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,             			
			),
			#nn.LeakyReLU(inplace=True),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
			nn.MaxPool2d(3, stride=2),
			nn.Conv2d(
              	  		in_channels=96,              # input height
                		out_channels=128,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,              			
			),
			#nn.LeakyReLU(inplace=True),

			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
			nn.Conv2d(
              	  		in_channels=128,              # input height
                		out_channels=128,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,             			
			),
			#nn.LeakyReLU(inplace=True),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),

			nn.MaxPool2d(3, stride=2),

		)
		self.layer4 = nn.Sequential(  	
			nn.Conv2d(
              	  		in_channels=128,              # input height
                		out_channels=160,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,             			
			),
			#nn.LeakyReLU(inplace=True),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
			nn.Conv2d(
              	  		in_channels=160,              # input height
                		out_channels=160,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,             			
			),
			#nn.LeakyReLU(inplace=True),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),

			nn.MaxPool2d(3, stride=2),

		)

		self.layer5 = nn.Sequential(  
			nn.Conv2d(
              	  		in_channels=160,              # input height
                		out_channels=192,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,             			
			),
			#nn.LeakyReLU(inplace=True),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
			nn.Conv2d(
              	  		in_channels=192,              # input height
                		out_channels=192,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step                 
				padding=pad,             			
			),
			#nn.LeakyReLU(inplace=True),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
			nn.MaxPool2d(3, stride=2),
			
		)



		
		self.output = nn.Sequential( 
			#nn.Linear(1728, 320),
			nn.Linear(768, 360),
			#nn.Dropout(0.5),
			#nn.LeakyReLU(inplace=True),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
			#nn.Sigmoid(),
			nn.Linear(360, 300),
			#nn.Dropout(0.5),
			nn.LeakyReLU(negative_slope=0.3333, inplace=False),
			nn.Linear(300, 1),
			#nn.LeakyReLU(negative_slope=0.3333, inplace=False),


		)
		
		for m in self.modules():
            		if isinstance(m, nn.Conv2d):
                		n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                		m.weight.data.normal_(0, math.sqrt(2. / n))
	

    	def I_weights(self, m):
        	#print(m)
        	if type(m) == nn.Conv2d:
            		w = 1
            	if m.in_channels == m.out_channels and m.kernel_size > 2:
               		w = 3
            	alpha = math.pow(m.kernel_size[0]*1.0/m.stride[0]/w, 2.0)
            	scale = math.sqrt(6.0/(m.in_channels+m.out_channels*alpha))
            	#print scale
            	m.weight.data.uniform_(-scale,scale)
            	#print(m.weight) 
   
    	"""
   	def inititalize_W(self):
        #self.I_weights(self.conv1[0])
        #self.I_weights(self.conv2[0])
        #self.I_weights(self.conv3[0])
    	"""


    	def forward(self, x):

		x = self.input(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		#x = x.view(-1,1728)
		x = x.view(-1,768)
		out = self.output(x)
	
		return out

    	def save_p(self, epoch):
		modelparam = path + "/D_input_Epoch_" + str(epoch) +".pkl"
		torch.save(self.input.state_dict(), modelparam) 
		modelparam = path + "/D_layer1_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer1.state_dict(), modelparam)
		modelparam = path + "/D_layer2_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer2.state_dict(), modelparam)
		modelparam = path + "/D_layer3_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer3.state_dict(), modelparam)
		modelparam = path + "/D_layer4_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer4.state_dict(), modelparam)
		modelparam = path + "/D_layer5_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer5.state_dict(), modelparam)
		modelparam = path + "/D_output_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.output.state_dict(), modelparam)
		print "Epoch ", epoch, ":  Model saved!!"

	def restore_netparam(self, epoch):	
		modelparam = pre_path1 + "/D_input_Epoch_" + str(epoch) +".pkl"
		self.input.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_layer1_Epoch_" + str(epoch) +".pkl"
	        self.layer1.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_layer2_Epoch_" + str(epoch) +".pkl"
	        self.layer2.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_layer3_Epoch_" + str(epoch) +".pkl"
	        self.layer3.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_layer4_Epoch_" + str(epoch) +".pkl"
	        self.layer4.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_layer5_Epoch_" + str(epoch) +".pkl"
	        self.layer5.load_state_dict(torch.load(modelparam))
	        modelparam = pre_path1 + "/D_output_Epoch_" + str(epoch) +".pkl"
	        self.output.load_state_dict(torch.load(modelparam))
		print "Epoch ", epoch, ":  Model restored!!"


    	def save_p_lr(self, epoch):
		modelparam = path + "/D_lr_input_Epoch_" + str(epoch) +".pkl"
		torch.save(self.input.state_dict(), modelparam) 
		modelparam = path + "/D_lr_layer1_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer1.state_dict(), modelparam)
		modelparam = path + "/D_lr_layer2_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer2.state_dict(), modelparam)
		modelparam = path + "/D_lr_layer3_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer3.state_dict(), modelparam)
		modelparam = path + "/D_lr_layer4_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer4.state_dict(), modelparam)
		modelparam = path + "/D_lr_layer5_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.layer5.state_dict(), modelparam)
		modelparam = path + "/D_lr_output_Epoch_" + str(epoch) +".pkl"
	        torch.save(self.output.state_dict(), modelparam)
		print "Epoch ", epoch, ":  Model saved!!"

	def restore_netparam_lr(self, epoch):	
		modelparam = pre_path1 + "/D_lr_input_Epoch_" + str(epoch) +".pkl"
		self.input.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_lr_layer1_Epoch_" + str(epoch) +".pkl"
	        self.layer1.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_lr_layer2_Epoch_" + str(epoch) +".pkl"
	        self.layer2.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_lr_layer3_Epoch_" + str(epoch) +".pkl"
	        self.layer3.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_lr_layer4_Epoch_" + str(epoch) +".pkl"
	        self.layer4.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/D_lr_layer5_Epoch_" + str(epoch) +".pkl"
	        self.layer5.load_state_dict(torch.load(modelparam))
	        modelparam = pre_path1 + "/D_lr_output_Epoch_" + str(epoch) +".pkl"
	        self.output.load_state_dict(torch.load(modelparam))
		print "Epoch ", epoch, ":  Model restored!!"


#SR model
class VDSR(nn.Module):
	def __init__(self):
		super(VDSR, self).__init__()
		self.input = nn.Sequential(         
            		nn.Conv2d(
              	  		in_channels=in_ch,              # input height
                		out_channels=ch,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step
                		padding=pad,                  
            			),
	    		
 	    		nn.LeakyReLU(inplace=False),
            
			)
		self.layers = nn.Sequential(
		
                 	nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=1, padding=pad),
			
                	nn.LeakyReLU(inplace=False),

                	nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=1, padding=pad),
			
                	nn.LeakyReLU(inplace=False),

                	nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=1, padding=pad),
			
                	nn.LeakyReLU(inplace=False),

                	nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=1, padding=pad),
			
                	nn.LeakyReLU(inplace=False),

                	nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=1, padding=pad),
			
                	nn.LeakyReLU(inplace=False),

                	nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=1, padding=pad),
			
                	nn.LeakyReLU(inplace=False),

                	nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=1, padding=pad),
			
                	nn.LeakyReLU(inplace=False),

                	nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=1, padding=pad),
			
                	nn.LeakyReLU(inplace=False),

                	nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=1, padding=pad),
			
                	nn.LeakyReLU(inplace=False),

                	nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=1, padding=pad),
			
                	nn.LeakyReLU(inplace=False),

                	nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=1, padding=pad),
			
                	nn.LeakyReLU(inplace=False),

                	nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=1, padding=pad),
			
                	nn.LeakyReLU(inplace=False),

                	nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=1, padding=pad),
			
                	nn.LeakyReLU(inplace=False),

                	nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=1, padding=pad),
			
                	nn.LeakyReLU(inplace=False),

                	nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=1, padding=pad),
			
                	nn.LeakyReLU(inplace=False),

                	nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=1, padding=pad),
			
                	nn.LeakyReLU(inplace=False),

                	nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=1, padding=pad),
			
                	nn.LeakyReLU(inplace=False),

                	nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=1, padding=pad),
			
                	nn.LeakyReLU(inplace=False),



		)

		self.output = nn.Sequential(
            		nn.Conv2d(
                		in_channels=ch,              # input height
                		out_channels=1,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step
                		padding=pad,
            		),
            		
        
        	)

            

		for m in self.modules():
            		if isinstance(m, nn.Conv2d):
                		n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                		m.weight.data.normal_(0, math.sqrt(2. / n))
	

    	def I_weights(self, m):
        	#print(m)
        	if type(m) == nn.Conv2d:
            		w = 1
            	if m.in_channels == m.out_channels and m.kernel_size > 2:
               		w = 3
            	alpha = math.pow(m.kernel_size[0]*1.0/m.stride[0]/w, 2.0)
            	scale = math.sqrt(6.0/(m.in_channels+m.out_channels*alpha))
            	#print scale
            	m.weight.data.uniform_(-scale,scale)
            	#print(m.weight) 
   
    	"""
   	def inititalize_W(self):
        #self.I_weights(self.conv1[0])
        #self.I_weights(self.conv2[0])
        #self.I_weights(self.conv3[0])
    	"""


    	def forward(self, x):
		LR = x[:,:1,:,:]
		x1 = self.input(x)
		x2 = self.layers(x1)
		x3 = self.output(x2)
		
		out = torch.add(x3, LR)
		
		return x2

    	def save_p(self, epoch):
		modelparam = path + "/inputEpoch" + str(epoch) +".pkl"
		torch.save(self.input.state_dict(), modelparam)    
		modelparam = path + "/layersEpoch" + str(epoch) +".pkl"
	        torch.save(self.layers.state_dict(), modelparam)
		modelparam = path + "/outputEpoch" + str(epoch) +".pkl"
	        torch.save(self.output.state_dict(), modelparam)
		print "Epoch ", epoch, ":  Model saved!!"

	def restore_netparam(self, epoch):	
		modelparam = pre_path + "/inputEpoch" + str(epoch) +".pkl"
		self.input.load_state_dict(torch.load(modelparam))
		modelparam = pre_path + "/layersEpoch" + str(epoch) +".pkl"
	        self.layers.load_state_dict(torch.load(modelparam))
	        modelparam = pre_path + "/outputEpoch" + str(epoch) +".pkl"
	        self.output.load_state_dict(torch.load(modelparam))
		print "Epoch ", epoch, ":  Model restored!!"





ch2 = 36
ch3 = 128
pad2 = 1
ksize2 = 3
drate = 0.50

class G(nn.Module):
	def __init__(self):
		super(G, self).__init__()


		self.input = nn.Sequential(         
            		nn.Conv2d(
              	  		in_channels=4,              # input height
                		out_channels=ch3,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step
                		padding=pad,                  
            			),
			nn.Dropout(0.05),
			nn.LeakyReLU(0.2, inplace=True),
			#nn.Dropout2d(0.10),
			#nn.Dropout(0.05),
		)


		self.input2 = nn.Sequential(         
            		nn.Conv2d(
              	  		in_channels=64,              # input height
                		out_channels=ch3,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step
                		padding=pad,                  
            			),
			nn.Dropout(0.05),
			nn.LeakyReLU(0.2, inplace=True),

			#nn.Dropout2d(0.10),
			#nn.Dropout(0.05),
		)


		self.layer1 = nn.Sequential(
		
			nn.Conv2d(in_channels=ch3*2, out_channels=ch2, kernel_size=ksize, stride=1, padding=pad),
			nn.Dropout(drate),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(in_channels=ch2, out_channels=ch2, kernel_size=ksize, stride=1, padding=pad),
			nn.Dropout(drate),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(in_channels=ch2, out_channels=ch2, kernel_size=ksize, stride=1, padding=pad),
			nn.Dropout(drate),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(in_channels=ch2, out_channels=ch2, kernel_size=ksize, stride=1, padding=pad),
			nn.Dropout(drate),
			nn.LeakyReLU(0.2, inplace=True),


		)

		self.output = nn.Sequential(

			nn.Conv2d(in_channels=ch2, out_channels=ch2, kernel_size=ksize, stride=1, padding=pad),
			nn.Dropout(drate),
			nn.LeakyReLU(0.2, inplace=True),
			#nn.Tanh(),
            		nn.Conv2d(
                		in_channels=ch2,              # input height
                		out_channels=1,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step
                		padding=pad,
            		),
			#nn.Tanh(),
			#nn.Dropout2d(0.025),
			#nn.Dropout(0.05),
        
        	)



		for m in self.modules():
            		if isinstance(m, nn.Conv2d):
                		n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                		m.weight.data.normal_(0, math.sqrt(2. / n))
	

    	def I_weights(self, m):
        	#print(m)
        	if type(m) == nn.Conv2d:
            		w = 1
            	if m.in_channels == m.out_channels and m.kernel_size > 2:
               		w = 3
            	alpha = math.pow(m.kernel_size[0]*1.0/m.stride[0]/w, 2.0)
            	scale = math.sqrt(6.0/(m.in_channels+m.out_channels*alpha))
            	#print scale
            	m.weight.data.uniform_(-scale,scale)
            	#print(m.weight) 
   
    	"""
   	def inititalize_W(self):
        #self.I_weights(self.conv1[0])
        #self.I_weights(self.conv2[0])
        #self.I_weights(self.conv3[0])
    	"""


    	def forward(self, x, sr):
		LR = x[:,:1,:,:]

		
		x1 = self.input(x)

		r2 = self.input2(sr)

		x1 = torch.cat((r2,x1), 1)		

		x2 = self.layer1(x1) # fusion

		x13 = self.output(x2)
	
		
		out = torch.add(x13, LR)
		
		return out

    	def save_p(self, epoch):
		modelparam = path + "/G_EDSR_inputEpoch" + str(epoch) +".pkl"
		torch.save(self.input.state_dict(), modelparam)
		modelparam = path + "/G_EDSR_input2Epoch" + str(epoch) +".pkl"
		torch.save(self.input2.state_dict(), modelparam)     
		modelparam = path + "/G_EDSR_layer1Epoch" + str(epoch) +".pkl"
	        torch.save(self.layer1.state_dict(), modelparam)
		modelparam = path + "/G_EDSR_outputEpoch" + str(epoch) +".pkl"
	        torch.save(self.output.state_dict(), modelparam)


		print "Epoch ", epoch, ":  Model saved!!"

	def restore_netparam(self, epoch):	
		modelparam = pre_path1 + "/G_EDSR_inputEpoch" + str(epoch) +".pkl"
		self.input.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/G_EDSR_input2Epoch" + str(epoch) +".pkl"
		self.input2.load_state_dict(torch.load(modelparam))
		modelparam = pre_path1 + "/G_EDSR_layer1Epoch" + str(epoch) +".pkl"
	        self.layer1.load_state_dict(torch.load(modelparam))
	        modelparam = pre_path1 + "/G_EDSR_outputEpoch" + str(epoch) +".pkl"
	        self.output.load_state_dict(torch.load(modelparam))
		print "Epoch ", epoch, ":  Model restored!!"


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False, padding=4)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter



class G_r(nn.Module):
	def __init__(self):
		super(G_r, self).__init__()


		self.g6 = get_gaussian_kernel(kernel_size=9, sigma=5.5/2.355, channels=1)


		self.input = nn.Sequential(         
            		nn.Conv2d(
              	  		in_channels=3,              # input height
                		out_channels=ch,            # n_filters
                		kernel_size=ksize,              # filter size
                		stride=1,                   # filter movement/step
                		padding=pad,                  
            			),
	    		
 	    		nn.LeakyReLU(inplace=False),
            
			)
		self.layers = nn.Sequential(


                	nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=1, padding=pad),
			
                	nn.LeakyReLU(inplace=False),

                	nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=1, padding=pad),
			
                	nn.LeakyReLU(inplace=False),

                	nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=1, padding=pad),
			
                	nn.LeakyReLU(inplace=False),


			#added for pretrain
                	nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=1, padding=pad),
			
                	nn.LeakyReLU(inplace=False),

                	nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=1, padding=pad),
			
                	nn.LeakyReLU(inplace=False),

                	nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=1, padding=pad),
			
                	nn.LeakyReLU(inplace=False),




		)

		self.output = nn.Sequential(


            		nn.Conv2d(
                		in_channels=ch,              # input height
                		out_channels=1,            # n_filters
                		kernel_size=1,              # filter size
                		stride=1,                   # filter movement/step
                		padding=0,
            		),
            		
        
        	)

            

		for m in self.modules():
            		if isinstance(m, nn.Conv2d):
				m.weight.requires_grad = False


    	def I_weights(self, m):
        	#print(m)
        	if type(m) == nn.Conv2d:
            		w = 1
            	if m.in_channels == m.out_channels and m.kernel_size > 2:
               		w = 3
            	alpha = math.pow(m.kernel_size[0]*1.0/m.stride[0]/w, 2.0)
            	scale = math.sqrt(6.0/(m.in_channels+m.out_channels*alpha))
            	#print scale
            	m.weight.data.uniform_(-scale,scale)
            	#print(m.weight) 
   
    	"""
   	def inititalize_W(self):
        #self.I_weights(self.conv1[0])
        #self.I_weights(self.conv2[0])
        #self.I_weights(self.conv3[0])
    	"""


    	def forward(self, x, y):
		LR = x[:,:1,:,:]
		sp = y[:,2:,:,:]
		x_new = torch.cat((x, sp), 1)
		x1 = self.input(x_new)
		x2 = self.layers(x1)
		x3 = self.output(x2)
		
		out = torch.add(x3, LR)
		

		#out = self.g6(x)

		return out

    	def save_p(self, epoch):
		modelparam = path + "/G2_inputEpoch" + str(epoch) +".pkl"
		torch.save(self.input.state_dict(), modelparam)    
		modelparam = path + "/G2_layersEpoch" + str(epoch) +".pkl"
	        torch.save(self.layers.state_dict(), modelparam)
		modelparam = path + "/G2_outputEpoch" + str(epoch) +".pkl"
	        torch.save(self.output.state_dict(), modelparam)
		print "Epoch ", epoch, ":  Model saved!!"

	def restore_netparam(self, epoch):	
		modelparam = pre_path + "/G2_inputEpoch" + str(epoch) +".pkl"
		self.input.load_state_dict(torch.load(modelparam))
		modelparam = pre_path + "/G2_layersEpoch" + str(epoch) +".pkl"
	        self.layers.load_state_dict(torch.load(modelparam))
	        modelparam = pre_path + "/G2_outputEpoch" + str(epoch) +".pkl"
	        self.output.load_state_dict(torch.load(modelparam))
		print "Epoch ", epoch, ":  Model restored!!"





def PSNR(imgs, img_y):
    
    	imgs = imgs.clamp(0, 1)

    	out = imgs - img_y
    
    	MSE = np.mean(out.numpy() ** 2)
    	#print "MSE : ", MSE
    	if MSE < 0:
        
		print "something wrong!!"
		quit()

    	#return 10 * math.log10(1.0 / MSE), MSE


    	return MSE


def rand_crop_loader(path):
	

	#print path
	data = scipy.io.loadmat(path)
	#print(path)




	data = scipy.io.loadmat(path)
	dir_path = os.path.basename(os.path.dirname(path))

	if dir_path == "sr":
		label = 1
		img = np.squeeze(data['img'])
		p_max = data['pmax']
		imgs = np.zeros((192,192,5))
		imgs[:,:,4] = img[:,:] / p_max[0]
		p = p_max[0]
	else: 
		label = 0


		imgs = np.squeeze(data['mat'])
		p_max = data['pmax']
		m_max = data['mmax']
		Rrange = 179.6051
		Rmin = 0.7071
		#print im.size
        	#new_imgs = np.zeros((256,256,3))

		if in_ch != 3:
			temp = imgs[:,:,2] / p_max[0]
			imgs[:,:,0] = imgs[:,:,0] / p_max[0]
			imgs[:,:,1] = imgs[:,:,1] / m_max[0]
			imgs[:,:,2] = (imgs[:,:,3] - Rmin) / Rrange 
			imgs[:,:,3] = (imgs[:,:,4] + 127.5) / 255.0
			imgs[:,:,4] = temp

		else:
			temp = imgs[:,:,2] / p_max[0]
			imgs[:,:,0] = imgs[:,:,0] / p_max[0]
			imgs[:,:,1] = imgs[:,:,1] / m_max[0]
			imgs[:,:,2] = (imgs[:,:,3] - Rmin) / Rrange 
			imgs[:,:,3] = (imgs[:,:,4] + 127.5) / 255.0
			imgs[:,:,4] = temp

		p = p_max[0];

	name = os.path.basename(path)


	

	#img1 = toimage(imgs[:,:,2])

	#img1.show()	
	#quit()
	x = random.randint(40, 152)#126
	y = random.randint(40, 152)#126
	#return imgs[random.randint(0,127), random.randint(0,127), :], p_max[0]
	return imgs[x:x+crop_size, y:y+crop_size,:], p,  label, name

def default_loader_G_crop(path):
	

	#print path
	data = scipy.io.loadmat(path)
	#print(path)




	data = scipy.io.loadmat(path)
	dir_path = os.path.basename(os.path.dirname(path))

	if dir_path == "sr":
		label = [0, 1]
		img = np.squeeze(data['img'])
		p_max = data['pmax']
		imgs = np.zeros((192,192,5))
		imgs[:,:,4] = img[:,:] / p_max[0]
		p = p_max[0]

	else: 
		label = [1, 0]


		imgs = np.squeeze(data['mat'])
		p_max = data['pmax']
		m_max = data['mmax']
		Rrange = 179.6051
		Rmin = 0.7071
		#print im.size
        	#new_imgs = np.zeros((256,256,3))
		noise = np.random.normal(0,sigma, size=(192,192))
		noise2 = np.random.normal(0,sigma, size=(192,192))
		#imgs[:,:,4] = np.rot90((img[:,:]+ noise) / p_max[0], r)
		if in_ch != 3:
			temp = imgs[:,:,2] / p_max[0]

			r = random.randint(0, 360) 
			imgs[:,:,0] = ndimage.rotate((imgs[:,:,0] / p_max[0]), r, reshape=False)
			imgs[:,:,1] = ndimage.rotate((imgs[:,:,1]  / m_max[0]), r, reshape=False)
			imgs[:,:,2] = ndimage.rotate(((imgs[:,:,3] - Rmin) / Rrange), r, reshape=False)
			imgs[:,:,3] = ndimage.rotate(((imgs[:,:,4] + 127.5) / 255.0), r, reshape=False)
			imgs[:,:,4] = ndimage.rotate((temp), r, reshape=False)

		else:
			temp = imgs[:,:,2] / p_max[0]
			imgs[:,:,0] = imgs[:,:,0] / p_max[0]
			imgs[:,:,1] = imgs[:,:,1] / m_max[0]
			imgs[:,:,2] = (imgs[:,:,3] - Rmin) / Rrange 
			imgs[:,:,3] = (imgs[:,:,4] + 127.5) / 255.0
			imgs[:,:,4] = temp

		p = p_max[0];


	name = os.path.basename(path)
	

	x = random.randint(30, 66)#126
	y = random.randint(30, 66)#126

	return imgs[x:x+crop_size, y:y+crop_size,:], p,  label, name



def default_loader_G(path):
	

	#print path
	data = scipy.io.loadmat(path)
	#print(path)




	data = scipy.io.loadmat(path)
	dir_path = os.path.basename(os.path.dirname(path))

	if dir_path == "sr":
		label = [0, 1]
		img = np.squeeze(data['img'])
		p_max = data['pmax']
		imgs = np.zeros((192,192,5))
		imgs[:,:,4] = img[:,:] / p_max[0]
		p = p_max[0]

	else: 
		label = [1, 0]


		imgs = np.squeeze(data['mat'])
		p_max = data['pmax']
		m_max = data['mmax']
		Rrange = 179.6051
		Rmin = 0.7071
		#print im.size
        	#new_imgs = np.zeros((256,256,3))

		if in_ch != 3:
			temp = imgs[:,:,2] / p_max[0]


			r = random.randint(1, 4) 
			imgs[:,:,0] = np.rot90(imgs[:,:,0] / p_max[0], r)
			imgs[:,:,1] = np.rot90(imgs[:,:,1] / m_max[0], r)
			imgs[:,:,2] =  np.rot90((imgs[:,:,3] - Rmin) / Rrange, r)
			imgs[:,:,3] =  np.rot90((imgs[:,:,4] + 127.5) / 255.0, r)
			imgs[:,:,4] =  np.rot90(temp,r)

		else:
			temp = imgs[:,:,2] / p_max[0]
			imgs[:,:,0] = imgs[:,:,0] / p_max[0]
			imgs[:,:,1] = imgs[:,:,1] / m_max[0]
			imgs[:,:,2] = (imgs[:,:,3] - Rmin) / Rrange 
			imgs[:,:,3] = (imgs[:,:,4] + 127.5) / 255.0
			imgs[:,:,4] = temp

		p = p_max[0];

	name = os.path.basename(path)
	
	return imgs, p, label, name

def default_loader(path):
	

	#print path
	data = scipy.io.loadmat(path)
	#print(path)




	data = scipy.io.loadmat(path)
	dir_path = os.path.basename(os.path.dirname(path))

	if dir_path == "sr":
		label = [0, 1]
		img = np.squeeze(data['img'])
		p_max = data['pmax']
		imgs = np.zeros((192,192,5))
		imgs[:,:,4] = img[:,:] / p_max[0]

	else: 
		label = [1, 0]


		imgs = np.squeeze(data['mat'])
		p_max = data['pmax']
		m_max = data['mmax']
		Rrange = 179.6051
		Rmin = 0.7071
		#print im.size
        	#new_imgs = np.zeros((256,256,3))

		if in_ch != 3:
			temp = imgs[:,:,2] / p_max[0]
			imgs[:,:,0] = imgs[:,:,0] / p_max[0]
			imgs[:,:,1] = imgs[:,:,1] / m_max[0]
			imgs[:,:,2] = (imgs[:,:,3] - Rmin) / Rrange 
			imgs[:,:,3] = (imgs[:,:,4] + 127.5) / 255.0
			imgs[:,:,4] = temp

		else:
			temp = imgs[:,:,2] / p_max[0]
			imgs[:,:,0] = imgs[:,:,0] / p_max[0]
			imgs[:,:,1] = imgs[:,:,1] / m_max[0]
			imgs[:,:,2] = (imgs[:,:,3] - Rmin) / Rrange 
			imgs[:,:,3] = (imgs[:,:,4] + 127.5) / 255.0
			imgs[:,:,4] = temp

		p = p_max[0];

	name = os.path.basename(path)
	
	return imgs, p, label, name



def default_loader_noise(path):
	

	#print path
	data = scipy.io.loadmat(path)
	#print(path)




	data = scipy.io.loadmat(path)
	dir_path = os.path.basename(os.path.dirname(path))

	if dir_path == "sr":
		label = [0, 1]
		img = np.squeeze(data['img'])
		p_max = data['pmax']
		imgs = np.zeros((192,192,5))
		r = random.randint(0, 360) 
		noise = np.random.normal(0,sigma, size=(192,192))
		imgs[:,:,0] =  ndimage.rotate((imgs[:,:,0] / p_max[0]), r, reshape=False)
		p = p_max[0]

	else: 
		label = [1, 0]


		imgs = np.squeeze(data['mat'])
		p_max = data['pmax']
		m_max = data['mmax']
		Rrange = 179.6051
		Rmin = 0.7071
		#print im.size
        	#new_imgs = np.zeros((256,256,3))

		if in_ch != 3:
			noise = np.random.normal(0,sigma, size=(192,192))
			temp = imgs[:,:,2] / p_max[0]
			r = random.randint(0, 360) 
			imgs[:,:,0] = ndimage.rotate((imgs[:,:,0] / p_max[0]), r, reshape=False)
			imgs[:,:,1] = ndimage.rotate((imgs[:,:,1]  / m_max[0]), r, reshape=False)
			imgs[:,:,2] = ndimage.rotate(((imgs[:,:,3] - Rmin) / Rrange), r, reshape=False)
			imgs[:,:,3] = ndimage.rotate(((imgs[:,:,4] + 127.5) / 255.0), r, reshape=False)
			imgs[:,:,4] = ndimage.rotate((temp), r, reshape=False)
		else:
			temp = imgs[:,:,2] / p_max[0]
			imgs[:,:,0] = imgs[:,:,0] / p_max[0]
			imgs[:,:,1] = imgs[:,:,1] / m_max[0]
			imgs[:,:,2] = (imgs[:,:,3] - Rmin) / Rrange 
			imgs[:,:,3] = (imgs[:,:,4] + 127.5) / 255.0
			imgs[:,:,4] = temp

		p = p_max[0];

	name = os.path.basename(path)
	
	#return imgs, p, label, name

	x = random.randint(30, 66)#30 66
	y = random.randint(30, 66)#30 66

	return imgs[x:x+crop_size, y:y+crop_size,:], p,  label, name
	#return imgs, p, label, name


def default_loader_lr(path):
	

	#print path
	data = scipy.io.loadmat(path)
	#print(path)




	data = scipy.io.loadmat(path)
	dir_path = os.path.basename(os.path.dirname(path))

	if dir_path == "sr":
		label = [0, 1]
		img = np.squeeze(data['img'])
		p_max = data['pmax']
		imgs = np.zeros((192,192,5))
		r = random.randint(0, 360) 
		noise = np.random.normal(0,sigma, size=(192,192))
		imgs[:,:,0] =  ndimage.rotate((imgs[:,:,0] / p_max[0]), r, reshape=False)
		p = p_max[0]

	else: 
		label = [1, 0]


		imgs = np.squeeze(data['mat'])
		p_max = data['pmax']
		m_max = data['mmax']
		Rrange = 179.6051
		Rmin = 0.7071
		#print im.size
        	#new_imgs = np.zeros((256,256,3))

		if in_ch != 3:
			noise = np.random.normal(0,sigma, size=(192,192))
			temp = imgs[:,:,2] / p_max[0]
			r = random.randint(0, 360) 
			imgs[:,:,0] = ndimage.rotate((imgs[:,:,0] / p_max[0]), r, reshape=False)
			imgs[:,:,1] = ndimage.rotate((imgs[:,:,1]  / m_max[0]), r, reshape=False)
			imgs[:,:,2] = ndimage.rotate(((imgs[:,:,3] - Rmin) / Rrange), r, reshape=False)
			imgs[:,:,3] = ndimage.rotate(((imgs[:,:,4] + 127.5) / 255.0), r, reshape=False)
			imgs[:,:,4] = ndimage.rotate((temp), r, reshape=False)

		else:
			temp = imgs[:,:,2] / p_max[0]
			imgs[:,:,0] = imgs[:,:,0] / p_max[0]
			imgs[:,:,1] = imgs[:,:,1] / m_max[0]
			imgs[:,:,2] = (imgs[:,:,3] - Rmin) / Rrange 
			imgs[:,:,3] = (imgs[:,:,4] + 127.5) / 255.0
			imgs[:,:,4] = temp

		p = p_max[0];

	name = os.path.basename(path)
	
	#return imgs, p, label, name
	x = random.randint(30, 66)#126
	y = random.randint(30, 66)#126

	return imgs[x:x+crop_size, y:y+crop_size,:], p,  label, name


class myImageFloder(Data.Dataset):
    	def __init__(self, root, label, transform = None, target_transform=None, loader=default_loader, mode = 0):
        
		self.root = root
		if mode == 0:
        		self.imgs = glob.glob(root + "*/*.mat")
		elif mode == 1:
			self.imgs = glob.glob(root + "*/p13_z67.mat")
		elif mode == 2:
			self.imgs = glob.glob(root + "train/*.mat")
		else:
			self.imgs = glob.glob(root + "sr/*.mat")
        	#self.classes = class_names
        	self.transform = transform
        	self.target_transform = target_transform
        	self.loader = loader

    	def __getitem__(self, index):
        	img_path = self.imgs[index]
        	img1, pmax, label, name = self.loader(img_path)

		img1 = torch.FloatTensor(torch.from_numpy(img1.transpose((2, 0, 1))).float())
		label= torch.FloatTensor(label)

        	if self.transform is not None:

       			img1[:4,:,:]= self.transform(img1[:4,:,:])
		


		#img = torch.from_numpy(img.transpose((2, 0, 1))).float()

		
        
			
		return img1, pmax, label, name

    	def __len__(self):
        	return len(self.imgs)
    
    	def getName(self):
        	return self.classes


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight
 
    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
 
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class newTVLoss(nn.Module):
    def __init__(self,newTVLoss_weight=1):
        super(newTVLoss,self).__init__()
        self.newTVLoss_weight = newTVLoss_weight
 
    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv_1 = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv_1 = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()

	h_tv_2 = torch.pow((x[:,:,2:,:]-torch.mul(x[:,:,1:h_x-1,:],2)- x[:,:,:h_x-2,:]),2).sum()
	w_tv_2 = torch.pow((x[:,:,2:,:]-torch.mul(x[:,:,1:w_x-1,:],2)- x[:,:,:w_x-2,:]),2).sum()

        return (self.newTVLoss_weight*1*(h_tv_1/count_h+w_tv_1/count_w)+2.0*(h_tv_2/count_h+w_tv_2/count_w))/batch_size
 
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class newTVLossab(nn.Module):
    def __init__(self,newTVLoss_weight=1):
        super(newTVLossab,self).__init__()
        self.newTVLoss_weight = newTVLoss_weight
 
    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        #h_tv_1 = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]).sum(),1)
        #w_tv_1 = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]).sum(),1)
        h_tv_1 = torch.abs((x[:,:,1:,:]-x[:,:,:h_x-1,:])).sum()
        w_tv_1 = torch.abs((x[:,:,:,1:]-x[:,:,:,:w_x-1])).sum()


	h_tv_2 = torch.abs((x[:,:,2:,:]-torch.mul(x[:,:,1:h_x-1,:],2.0)- x[:,:,:h_x-2,:])).sum()
	w_tv_2 = torch.abs((x[:,:,2:,:]-torch.mul(x[:,:,1:w_x-1,:],2.0)- x[:,:,:w_x-2,:])).sum()




        return (self.newTVLoss_weight*2*1*torch.pow((h_tv_1/count_h+w_tv_1/count_w),2)+2*1*torch.pow((h_tv_2/count_h+w_tv_2/count_w),2))/batch_size
 
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]




transform2=torchvision.transforms.Compose([
	#RandomAffine(translation_range=(0.0078125,0.0078125)),

])

transform1=torchvision.transforms.Compose([
	#RandomAffine(translation_range=(0.0078125,0.0078125)),
	RandomAffine(translation_range=(0.00390625, 0.00390625), shear_range=1, rotation_range=1),
	#RandomAffine(translation_range=(0.0078125,0.0078125), shear_range=2, rotation_range=2),
	#Shear(15),
	#Rotate(180),
	#Translate(0.01)
])


train_loader_G = torch.utils.data.DataLoader(
        myImageFloder(root = rootdir_t, label = None, loader=default_loader_G_crop), 
        batch_size= BATCH_SIZE,
	shuffle= True, 
	num_workers= 4, 
	drop_last = False)


train_loader = torch.utils.data.DataLoader(
        myImageFloder(root = rootdir_t, label = None), 
        batch_size= 1,
	shuffle= True, 
	num_workers= 0, 
	drop_last = False)


train_loader2_real = torch.utils.data.DataLoader(
        myImageFloder(root = rootdir_t2, label = None, mode = 2, loader=default_loader_noise), 
        batch_size= BATCH_SIZE_D,
	shuffle= True, 
	num_workers= 4, 
	drop_last = False)

train_loader2_fake = torch.utils.data.DataLoader(
        myImageFloder(root = rootdir_t2, label = None, mode = 3, loader=default_loader_noise), 
        batch_size= BATCH_SIZE_D,
	shuffle= True, 
	num_workers= 4, 
	drop_last = False)



val_1 = torch.utils.data.DataLoader(
       	myImageFloder(root = rootdir_v, label = None, transform = None, mode = 1),
	batch_size= 1, 
	shuffle= False, 
	num_workers=0, 
	drop_last = False)



val_loader = torch.utils.data.DataLoader(
       	myImageFloder(root = rootdir_v, label = None, transform = None, mode = 2 ),
	batch_size= 1, 
	shuffle= False, 
	num_workers=0, 
	drop_last = False)



train_loader_lr_real = torch.utils.data.DataLoader(
        myImageFloder(root = rootdir_t3, label = None, loader=default_loader_lr, mode = 2), 
        batch_size= BATCH_SIZE_D,
	shuffle= True, 
	num_workers= 4, 
	drop_last = False)

train_loader_lr_fake = torch.utils.data.DataLoader(
        myImageFloder(root = rootdir_t3, label = None, loader=default_loader_lr, mode = 3), 
        batch_size= BATCH_SIZE_D,
	shuffle= True, 
	num_workers= 4, 
	drop_last = False)





#D1 = D()
#D_lr = D2()
D1 = D_crop2()
D_lr = D_crop2()
G = G()
G2 = G_r()
SR = VDSR()
#SR2 = VDSR()

SR.restore_netparam(100)#55 117   100 blurred hrrt to hrrt
#SR2.restore_netparam(117)
G.restore_netparam(716)
G2.restore_netparam(250)
#D1.restore_netparam(rep)
#D_lr.restore_netparam_lr(rep)
SR.cuda()	
D1.cuda()
D_lr.cuda()
G.cuda()
G2.cuda()
print(SR)  # net architecture
print(D)
print(D_lr)
print(G)
print(G2)





#loss_func_D = nn.CrossEntropyLoss()                       # the target label is not one-hotted
loss_func_D = nn.BCELoss()
#loss_func_D = nn.BCEWithLogitsLoss
loss_func_GV = nn.MSELoss()
loss_func_G1= nn.L1Loss()
loss_func_G = nn.MSELoss()
#loss_func_G= nn.L1Loss()
#loss_func_TV = TVLoss()
loss_func_TV = newTVLoss()
print "USR 4ch" 
print "Epoch : ", EPOCH, " Batch Size : ", BATCH_SIZE
print "Input channels : ", in_ch
print "rootdir_t : ", rootdir_t
print "rootdir_t2 : ", rootdir_t2
print "rootdir_v : ", rootdir_v
print "Path : ", path




engine = matlab.engine.start_matlab()

epnum = 1

#optimizer = torch.optim.SGD(cnn.parameters(), lr=0.00001)
for epoch in range(1):

	G.eval()
	temp = 0.0
	t_ssim = 0.0
	print "------------------------------"
	#validation

	#in_ch = 1
	for step, (x, pmax, label, name)in enumerate(val_loader):           
		
		


		with torch.no_grad():

			imgs_x = torch.FloatTensor(BATCH_SIZE, in_ch, x.size()[2], x.size()[3])
			imgs_y = torch.FloatTensor(BATCH_SIZE, 1, x.size()[2], x.size()[3])
			


			imgs_x[:,:,:,:] = x[:,:in_ch,:,:]
			imgs_y[:,0,:,:] = x[:,4,:,:]
			
			t_x = Variable(imgs_x.cuda())   # batch x
			t_y = Variable(imgs_y.cuda())

			output = SR(t_x)               # cnn output
				
		
		
			y_t = G(t_x, output)


                	loss = loss_func_G(y_t, t_y)

			temp += loss.data[0]
		


			res = np.squeeze(y_t.cpu().data.numpy()) * pmax
			n = ''.join(name)

			#loc = output_path + "/"+ n
			loc = "../Revision_imgs/SSSR2/" + n
			print loc
			t_ssim += pytorch_ssim.ssim(y_t, t_y)
			scipy.io.savemat(loc, {'img': res.numpy(), 'pmax': pmax.numpy()})


        
	#print "Val loss: ", temp, "    ssim: ", t_ssim/(90*14)


	print "------------------------------"


#np.savetxt(t_MSE_name, MSE_T, delimiter=',')

	

print "Finished"




