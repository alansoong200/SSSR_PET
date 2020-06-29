import os
import numpy as np
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from matplotlib import pyplot as plt
import scipy.io
import glob

rootdir_v = './val_14mci/'
dir_name1 = "MIC_GAN_EDSR_new_36_rand"
pre_path1 = "./params/" + dir_name1
pre_path = "./pretrain_params"


ksize = 3
pad = 1
ch = 64
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



    def forward(self, x, sr):
        LR = x[:,:1,:,:]


        x1 = self.input(x)

        r2 = self.input2(sr)

        x1 = torch.cat((r2,x1), 1)		

        x2 = self.layer1(x1) # fusion

        x13 = self.output(x2)


        out = torch.add(x13, LR)

        return out


    def restore_netparam(self, epoch):	
        modelparam = pre_path1 + "/G_EDSR_inputEpoch" + str(epoch) +".pkl"
        self.input.load_state_dict(torch.load(modelparam))
        modelparam = pre_path1 + "/G_EDSR_input2Epoch" + str(epoch) +".pkl"
        self.input2.load_state_dict(torch.load(modelparam))
        modelparam = pre_path1 + "/G_EDSR_layer1Epoch" + str(epoch) +".pkl"
        self.layer1.load_state_dict(torch.load(modelparam))
        modelparam = pre_path1 + "/G_EDSR_outputEpoch" + str(epoch) +".pkl"
        self.output.load_state_dict(torch.load(modelparam))
        print("The model is restored!!")

        
        
        
class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        self.input = nn.Sequential(         
                    nn.Conv2d(
                        in_channels=4,              # input height
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






    def forward(self, x):
        LR = x[:,:1,:,:]
        x1 = self.input(x)
        x2 = self.layers(x1)
        x3 = self.output(x2)

        out = torch.add(x3, LR)

        return x2


    def restore_netparam(self, epoch):	
        modelparam = pre_path + "/inputEpoch" + str(epoch) +".pkl"
        self.input.load_state_dict(torch.load(modelparam))
        modelparam = pre_path + "/layersEpoch" + str(epoch) +".pkl"
        self.layers.load_state_dict(torch.load(modelparam))
        modelparam = pre_path + "/outputEpoch" + str(epoch) +".pkl"
        self.output.load_state_dict(torch.load(modelparam))
        print("Sim VDST model is restored!!")

        
        
        

def default_loader(path):


    #print path
    data = scipy.io.loadmat(path)
    #print(path)
    imgs = np.squeeze(data['mat'])
    p_max = data['pmax']
    m_max = data['mmax']
    Rrange = 179.6051
    Rmin = 0.7071
    #print im.size
        #new_imgs = np.zeros((256,256,3))
    in_ch = 4
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
        #imgs[:,:,1] = imgs[:,:,1] / m_max[0]
        imgs[:,:,1] = (imgs[:,:,3] - Rmin) / Rrange 
        imgs[:,:,2] = (imgs[:,:,4] + 127.5) / 255.0
        imgs[:,:,4] = temp



    name = os.path.basename(path)

    return imgs, p_max[0], m_max[0], name




class myImageFloder(Data.Dataset):
    def __init__(self, root=rootdir_v, label=None, transform = None, target_transform=None, loader=default_loader, mode = 0):

        self.root = root
        self.imgs = glob.glob(root + "train/p13_z67.mat")
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img1, pmax, mmax, name = self.loader(img_path)

        img1 = torch.FloatTensor(torch.from_numpy(img1.transpose((2, 0, 1))).float())

        if self.transform is not None:

            img1[:4,:,:]= self.transform(img1[:4,:,:])

        return img1, pmax, mmax, name

    def __len__(self):
        return len(self.imgs)

    def getName(self):
        return self.classes