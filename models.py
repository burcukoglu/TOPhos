from mxnet import sym

import numpy as np
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn, rnn, loss
from mxnet.gluon.nn import HybridSequential, Sequential #LeakyReLU
import mxnet as mx
import math
import noise

USE_INT64_TENSOR_SIZE=1

w_dec, h_dec = 256,256
w, h         = 128,128

context=mx.gpu(0) #cpu()

class ConvLayer(gluon.HybridBlock):
    def __init__(self, n_input, n_output,  k_size=3, stride=1, padding=1): 
        super(ConvLayer, self).__init__()

        with self.name_scope():
            self.conv = nn.Conv2D(channels=n_output,  kernel_size=k_size, strides=stride, padding=padding, in_channels=n_input, use_bias=False)
            self.swish = nn.Swish() 

    def hybrid_forward(self, F, x):
        out = self.swish(self.conv(x))

        return out


class ResidualBlock(nn.HybridBlock):
    def __init__(self, n_channels, strides=1): 
        super(ResidualBlock, self).__init__()

        with self.name_scope():
            self.conv1 = nn.Conv2D(channels=n_channels, kernel_size=3, strides=1,padding=1, use_bias=True, in_channels=n_channels)
            self.swish = nn.Swish() 
            self.conv2 = nn.Conv2D(channels=n_channels, kernel_size=3, strides=1,padding=1, use_bias=True, in_channels=n_channels)
        
    def hybrid_forward(self, F, x):
        residual = x
        out = self.swish(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        out = self.swish(out)
    

        return out


class E2E_Encoder(nn.HybridBlock):
    """
    #Simple non-generic encoder class that receives 128x128 input and outputs 32x32 feature map as stimulation protocol
    """   
    def __init__(self, in_channels=1, out_channels=1 ,binary_stimulation=False):
        super(E2E_Encoder, self).__init__()
             
        self.binary_stimulation = binary_stimulation

        with self.name_scope():
            self.convlayer1 = ConvLayer(in_channels,8,3,1,1) 
            self.convlayer2 = ConvLayer(8,16,3,1,1)
            self.maxpool1 = nn.MaxPool2D(2)
            self.convlayer3 = ConvLayer(16,32,3,1,1)
            self.maxpool2 =nn.MaxPool2D(2)
            self.res1 = ResidualBlock(32)
            self.res2 = ResidualBlock(32)
            self.res3 = ResidualBlock(32)
            self.res4 = ResidualBlock(32)
            self.convlayer4 =ConvLayer(32,16,3,1,1)
            self.encconv1 = nn.Conv2D(channels=out_channels,kernel_size=3,strides=1,padding=1, use_bias=True, in_channels=16) #bias true
            self.tanh1 = nn.Activation('tanh')

    def hybrid_forward(self, F, x):
        x = self.convlayer1(x)
        x = self.maxpool1(self.convlayer2(x))
        x = self.maxpool2(self.convlayer3(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.convlayer4(x)
        x = self.tanh1(self.encconv1(x))        

        stimulation = .5 *(x+1) 

        return stimulation    


class MainModel(gluon.HybridBlock):
    def __init__(self, n_actions, context, **kwargs):
        super(MainModel, self).__init__(**kwargs)

        with self.name_scope():
            self.enc = E2E_Encoder()
            self.phosim = E2E_PhospheneSimulator(sigma=1.5, intensity=15, ctx=context)
            self.model = Model(n_actions)
            self.dec = E2E_Decoder()


    def hybrid_forward(self, F,pmask, pre_state0,pre_state1, pre_state2, pre_state3):
   
        pre_state=F.concat(pre_state0,pre_state1,pre_state2,pre_state3, dim=0)

        states_encoded=self.enc(F.reshape(pre_state,(-1,1,h,w)))
        states_phosphene=self.phosim(states_encoded, pmask)
        states_decoded=self.dec(states_phosphene)

        states_encoded=F.split(states_encoded, axis=0, num_outputs=4)
        states_encoded=F.concat(states_encoded[0], states_encoded[1], states_encoded[2], states_encoded[3], dim=1)

        states_phosphene=F.split(states_phosphene, axis=0, num_outputs=4)
        states_phosphene=F.concat(states_phosphene[0],states_phosphene[1],states_phosphene[2],states_phosphene[3] , dim=1)

        states_decoded=F.split(states_decoded, axis=0, num_outputs=4)
        states_decoded=F.concat(states_decoded[0],states_decoded[1],states_decoded[2],states_decoded[3] , dim=1)

        probs, values = self.model(states_phosphene)

        return probs, values, states_encoded, states_phosphene , states_decoded

class E2E_PhospheneSimulator(gluon.HybridBlock):

    """ Uses three steps to convert  the stimulation vectors to phosphene representation:
    1. Resizes the feature map (default: 32x32) to SVP template (256x256)
    2. Uses pMask to sample the phosphene locations from the SVP activation template
    2. Performs convolution with gaussian kernel for realistic phosphene simulations
    """
    def __init__(self,scale_factor=8, sigma=1.5,kernel_size=11, intensity=15, ctx=context):  
        super(E2E_PhospheneSimulator, self).__init__()
        
        self.ctx=ctx

        self.intensity = intensity 
        self.scale_factor=scale_factor

        self.gaussian_kernel = self.get_gaussian_layer(kernel_size=kernel_size, sigma=sigma, channels=1)
        
        self.gaussian_filter= nn.Conv2D(channels=1, kernel_size=kernel_size, groups=1, use_bias=False, in_channels=1)
        
        self.gaussian_filter.weight.initialize(ctx=self.ctx)
        self.gaussian_filter.weight.set_data(self.gaussian_kernel)
        self.gaussian_filter.weight.grad_req='null' 

    def get_gaussian_layer(self, kernel_size, sigma, channels):
        """non-trainable Gaussian filter layer for more realistic phosphene simulation"""

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = nd.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).reshape(kernel_size, kernel_size)
        y_grid = nd.transpose(x_grid) 
        xy_grid = nd.stack(x_grid, y_grid, axis=-1) #.float()

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                          nd.exp(
                              -nd.sum((xy_grid - mean)**2., axis=-1) /\
                              (2*variance)
                          )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / nd.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.reshape(1, 1, kernel_size, kernel_size)

        gaussian_kernel = nd.tile(gaussian_kernel, (channels, 1, 1, 1)) 

        return gaussian_kernel 

    def hybrid_forward(self, F, stimulation, pmask):
        
        phosphenes = F.UpSampling(stimulation ,scale=self.scale_factor, sample_type="nearest")

        phosphenes=F.broadcast_mul(phosphenes,F.reshape(pmask,(-1,1,h_dec,w_dec))) 

        phosphenes = self.gaussian_filter(F.pad(phosphenes, mode="constant", constant_value=0, pad_width=(0,0,0,0,5,5,5,5))) 

        return self.intensity*phosphenes 



class E2E_Decoder(gluon.HybridBlock):

    """
    Simple non-generic phosphene decoder.
    in: (256x256) SVP representation
    out: (128x128) Reconstruction
    """   
    def __init__(self, in_channels=1, out_channels=1, out_activation='sigmoid'):
        super(E2E_Decoder, self).__init__()
             
        # Activation of output layer
        self.out_activation = {'tanh': nn.Activation('tanh'),
                               'sigmoid': nn.Activation('sigmoid'),
                               'relu': nn.LeakyReLU(alpha=0.01)}[out_activation]
                              #  'softmax':nd.softmax(axis=1)}[out_activation]
                              #  use 'sigmoid' for grayscale reconstructions, 'softmax' for boundary segmentation task
        

        with self.name_scope():
            self.convlayer1=ConvLayer(in_channels,16,3,1,1)
            self.convlayer2=ConvLayer(16,32,3,1,1)
            self.convlayer3=ConvLayer(32,64,3,2,1)
            self.res1=ResidualBlock(64)
            self.res2=ResidualBlock(64)
            self.res3=ResidualBlock(64)
            self.res4=ResidualBlock(64)
            self.convlayer4=ConvLayer(64,32,3,1,1)
            self.decconv1=nn.Conv2D(channels=out_channels,kernel_size=3,strides=1,padding=1,in_channels=32)
            self.activ= self.out_activation


    def hybrid_forward(self, F, x):
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.convlayer3(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.convlayer4(x)
        x = self.decconv1(x)
        x = self.activ(x)
        

        return x


# RL agent
class Model(gluon.HybridBlock):
    def __init__(self, n_actions, **kwargs):
        super(Model, self).__init__(**kwargs)

        with self.name_scope():
            self.conv1 = nn.Conv2D(32,8, strides=(4,4))
            self.conv2 = nn.Conv2D(64,4, strides=(2,2))
            self.conv3 = nn.Conv2D(64,3, strides=(1,1))
            self.dense = nn.Dense(512)
            self.flat = nn.Flatten()
            self.action = nn.Dense(n_actions)
            self.value = nn.Dense(1)

    def hybrid_forward(self, F, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.dense(x))
        probs = self.action(x)
        values = self.value(x)
        return F.softmax(probs), values








