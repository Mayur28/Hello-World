import numpy as np
import torch
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
import random
from . import networks
import sys


class SingleModel(BaseModel):
    def name(self):
        return 'SingleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)# Just sets 4 minor stuff

        nb = opt.batchSize# Batch size (16)
        size = opt.fineSize # From my understanding, the size of the input images
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)#We are basically creating a tensor to store 16 low-light colour images with size fineSize x fineSize
        self.input_B = self.Tensor(nb, opt.output_nc, size, size) # Same as above but now for storing the normal-light images (NOT THE RESULT!)
        self.input_img = self.Tensor(nb, opt.input_nc, size, size) # What this actually mean?
        self.input_A_gray = self.Tensor(nb, 1, size, size) # this is for the attention maps
		
		#Track the above carefully

        if opt.vgg > 0: # We are using this!
            self.vgg_loss = networks.PerceptualLoss(opt)# We just create the instance that defines an instance norm layer. We still need to place this into the image.
            self.vgg_loss.cuda()#--> moves the variable to the GPU
            self.vgg = networks.load_vgg16("./model", self.gpu_ids) #Actually load the VGG model(THIS IS CRUCIAL!)... This is the weights that we had to manually add
            self.vgg.eval() # We call eval() when some layers within the self.vgg network behave differently during training and testing... This will not be trained (Its frozen!)!
			#The eval function is often used as a pair with the requires.grad or torch.no grad functions (which makse sense)
            for param in self.vgg.parameters():
                param.requires_grad = False# Verified! For all the weights in the VGG network, we do not want to be updating those weights, therefore, we save computation using the above!

		#G_A : Is our only generator
		#D_A : Is the Global Discriminator
		#D_P : Is the patch discriminator

       #ngf is the number of filters in the first conv layer in the generator. ndf is the number of filters used in the first conv layer in the disc.
        skip = True if opt.skip > 0 else False # We are using skip connections!
		
        self.netG_A = networks.define_G(opt.norm, self.gpu_ids,skip=skip,opt=opt)
        #It looks like they are handling each sub-network as an attribute of 'self'.
        if self.isTrain: #We have this
            #Below is the global discriminator
			#Below is correct that we are accepting 'output_nc' which represents a char. of the sample produced by the generator.
            self.netD_A = networks.define_D(opt.n_layers_D, opt.norm, self.gpu_ids)
			#3,64, no_norm_4,5,instance,false,0,false
            if self.opt.patchD:
                #This is the local 'patch' based discriminator( n_layers_patchD=4 ( one less than the global discriminator( this excludes the boundary layers)))
                self.netD_P = networks.define_D(opt.n_layers_patchD, opt.norm, self.gpu_ids)
                #3,64,no_norm_4,4,'instance',False,..... Last parameter specifies if its patch based)--> But on the other side, the patch is not accounted for...
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                if self.opt.patchD:
                    self.load_network(self.netD_P, 'D_P', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_B_pool = ImagePool(opt.pool_size)# This is just an initializer. sets the number of images in the pool to 50 and create a list for storing the images. This is where our results should be stored
            # define loss functions						# Use lsGAN!
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor) # Read the note on the LSGAN and MSE loss!
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))#Generator - should still reflect that its trained through the discriminator. Investigate
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))#Global Discriminator
            if self.opt.patchD:
                self.optimizer_D_P = torch.optim.Adam(self.netD_P.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))# Local Discriminator

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        if self.isTrain:
            networks.print_network(self.netD_A)
            if self.opt.patchD:
                networks.print_network(self.netD_P)
        if opt.isTrain:
            self.netG_A.train()# Make the generator trainable. The gradients are volatile! Its by default trainable but maybe this is just to make sure.
        else:
            self.netG_A.eval()
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']# We can do this because it is a dictionary!
        input_B = input['B' if AtoB else 'A']
        input_img = input['input_img']
        input_A_gray = input['A_gray'] # Remember that in our confituation, input_img=A
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_A_gray.resize_(input_A_gray.size()).copy_(input_A_gray)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_img.resize_(input_img.size()).copy_(input_img)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def predict(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_A_gray = Variable(self.input_A_gray, volatile=True)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A))/(torch.max(self.real_A) - torch.min(self.real_A))
        # print(np.transpose(self.real_A.item().cpu().float().numpy(),(1,2,0))[:2][:2][:])
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A, self.real_A_gray)
        else:
            self.fake_B = self.netG_A.forward(self.real_A, self.real_A_gray)

        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        A_gray = util.atten2im(self.real_A_gray.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake, use_ragan):
        pred_real = netD.forward(real) # Returns the prediction if the disc. thinks the sample is real or fake 
        pred_fake = netD.forward(fake.detach())#--> This is wary!
		
        if self.opt.use_ragan and use_ragan:#This is what I am doing! Below is similar to what we did for the generator where to used the average loss... But why do we subtract the mean? why are the subtraction values switched?
            loss_D = (self.criterionGAN(pred_real - torch.mean(pred_fake), True) +
            self.criterionGAN(pred_fake - torch.mean(pred_real), False)) / 2
        else:
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5 # most of the time, we are taking the average loss between the real and fake images
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)# What is this used for? It seems to just get overwritten immediately? #It just returns some images from fake_B. This seems deceiving...
        fake_B = self.fake_B
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, True)
        self.loss_D_A.backward()

    def backward_D_P(self):
        if self.opt.hybrid_loss:# We are using this
            loss_D_P = self.backward_D_basic(self.netD_P, self.real_patch, self.fake_patch, False) #The average disc. loss for the individual image
            if self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):# account for the patches
                    loss_D_P += self.backward_D_basic(self.netD_P, self.real_patch_1[i], self.fake_patch_1[i], False)
                self.loss_D_P = loss_D_P/float(self.opt.patchD_3 + 1) # Averaged out over the whole image and the patches
        self.loss_D_P.backward()

  
    def forward(self): #Produce the fake sample and set the individual patch variable and form the list of patches
        self.real_A = Variable(self.input_A) #Variable is basically a tensor (which represents a node in the comp. graph) and is part of the autograd package to easily compute gradients
		#Dont be confused my the 'real' appended to the start... real_A is the input_A low-light image
        self.real_B = Variable(self.input_B)#This contains the normal-light images
        self.real_A_gray = Variable(self.input_A_gray)# This is the attention map
        self.real_img = Variable(self.input_img)#In our configuation, input_img=input_A
		
        #print("Shape of real_A: %s " % self.real_A.size())
        #print("Shape of real_B: %s " % self.real_B.size())
        #print("Shape of real_A_gray: %s " % self.real_A_gray.size())
        #print("Shape of real_img: %s " % self.real_img.size())
        
        if self.opt.skip == 1: # This sort of makes sense, but where does the latent stuff fit in.
            self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_img, self.real_A_gray)# Ive got an idea of whats going on here. Fake_B stores our fake samples( our result). As seen a little later, this will be (one of) the input to the discriminator 
			
			#We feed the generator the low-light image and the attention map!

        if self.opt.patchD:
			#Here, we are finding the random patches
            w = self.real_A.size(3)
            h = self.real_A.size(2)
            w_offset = random.randint(0, max(0, w - self.opt.patchSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.patchSize - 1))
            self.fake_patch = self.fake_B[:,:, h_offset:h_offset + self.opt.patchSize,
                   w_offset:w_offset + self.opt.patchSize]# FAKE_B IS OUR PRODUCED SAMPLE
            self.real_patch = self.real_B[:,:, h_offset:h_offset + self.opt.patchSize,
                   w_offset:w_offset + self.opt.patchSize]# REAL_B IS OUR NORMAL_LIGHT REFERENCE IMAGE
            self.input_patch = self.real_A[:,:, h_offset:h_offset + self.opt.patchSize,
                   w_offset:w_offset + self.opt.patchSize]#REAL_A IS INPUT LOW-LIGHT IMAGE THAT WE ARE ENHANCING.
			
		
        if self.opt.patchD_3 > 0: # patchD_3=5, but what does it represent? A) Looks like its used for looping below.. But determine for what.
            self.fake_patch_1 = []
            self.real_patch_1 = []
            self.input_patch_1 = []
            w = self.real_A.size(3) # Remember, the tensors are represented as (N,C,W,H)
            h = self.real_A.size(2)
            for i in range(self.opt.patchD_3):# We are basically saying that for each image, we are taking 5 random patches of the fake, real and input image.
                w_offset_1 = random.randint(0, max(0, w - self.opt.patchSize - 1))# This is concrete
                h_offset_1 = random.randint(0, max(0, h - self.opt.patchSize - 1))
                self.fake_patch_1.append(self.fake_B[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])
                self.real_patch_1.append(self.real_B[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])
                self.input_patch_1.append(self.real_A[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])
				
				#By the looks of it, we need to set one internal variable 'xxx_patch' that stores a single patch and we also have xxx_patch_1 which stores a list of patches What do we do with the list of patches?
				#Verify if this is correct: Forward is called by the batch so is the forward function called once for a batch or is it call once for every element in the batch?


    def backward_G(self, epoch):
        pred_fake = self.netD_A.forward(self.fake_B)# Our predictions on the fake samples... Tracing is pretty useless as it just passes through to a defined Sequential model.

        if self.opt.use_ragan:
			#Leave our predictions of our fake samples aside and make predictions on the real samples first.
			# I see that they are testing the discriminator in stages
            pred_real = self.netD_A.forward(self.real_B) # Real B is the normal_light images from the data ( this is correct)
			#CONFIRM THE SWITCHING STORY!!!
            self.loss_G_A = (self.criterionGAN(pred_real - torch.mean(pred_fake), False) +
                                      self.criterionGAN(pred_fake - torch.mean(pred_real), True)) / 2
			
			#The generator's loss is taken to be the average of the discriminator's performance on the real and fake sets
			# The boolean is used to indicate whether the target is real or not.
			#criterionGAN is actually an instance of the GANloss class. Above, we are in fact calling the __call__ function which calls the 'get_target_tensor' which makes sense now. It is this function that accepts 'input' and a 'is_target_real' parameter.
        loss_G_A = 0
		
		# This distinction is understood
        if self.opt.patchD:# Predict the individual path (not the list)
            pred_fake_patch = self.netD_P.forward(self.fake_patch)# The single patch, not the list.
            if self.opt.hybrid_loss:# We use this, but what makes it hybrid?
                loss_G_A += self.criterionGAN(pred_fake_patch, True)# The True represents the 'target is real' label. I believe this is where we do the 'TRICK'.
		
		#This distinction is understood
        if self.opt.patchD_3 > 0:# Discriminate our list of patches
            for i in range(self.opt.patchD_3):
                pred_fake_patch_1 = self.netD_P.forward(self.fake_patch_1[i])# This indexing makes sense
                if self.opt.hybrid_loss:
                    loss_G_A += self.criterionGAN(pred_fake_patch_1, True)#JUST DOUBLE THIS SWITCHING STORY
					#Note: We are accumulating the loss_G_A for the entire images as well as for the patches.
            if not self.opt.D_P_times2:
                self.loss_G_A += loss_G_A/float(self.opt.patchD_3 + 1)# We have this and are basically saying that the official loss_G_A (the one that is global) is equal to the average of the locally computed loss_G_A from above. The +1 is for discriminating the individual patch (on line 245).



        if epoch < 0:
            vgg_w = 0
        else:
            vgg_w = 1 # The loss_vgg_b is very important because it is this variable that we add to the gan loss to compute the total loss of the generator.
        if self.opt.vgg > 0: #vgg_loss is actually an instance of the Perceptual loss class.
			#self.vgg is the actual vgg model that we loaded, not to be confused with a scalar variable
			
			#The line below if for computing the perceptual loss (the mean difference between the feature maps)
            self.loss_vgg_b = self.vgg_loss.compute_vgg_loss(self.vgg,
                    self.fake_B, self.real_A) * self.opt.vgg if self.opt.vgg > 0 else 0 # The if statement and the self.opt.vgg seem very redundant. It coult be to force the result to be a scalar as Turgay used to do.
			#The above statement calculates the vgg loss of the whole images
            if self.opt.patch_vgg: #True
                if not self.opt.IN_vgg:#True
                    loss_vgg_patch = self.vgg_loss.compute_vgg_loss(self.vgg,
                    self.fake_patch, self.input_patch) * self.opt.vgg# Calculates the vgg loss at the patch level.
                else:
                    loss_vgg_patch = self.vgg_patch_loss.compute_vgg_loss(self.vgg,
                    self.fake_patch, self.input_patch) * self.opt.vgg
                if self.opt.patchD_3 > 0: # True
                    for i in range(self.opt.patchD_3):
                        if not self.opt.IN_vgg: #Calculate the vgg loss of the patches (similar to what we did above for the GAN loss)
                            loss_vgg_patch += self.vgg_loss.compute_vgg_loss(self.vgg,
                                self.fake_patch_1[i], self.input_patch_1[i]) * self.opt.vgg
                        else:
                            loss_vgg_patch += self.vgg_patch_loss.compute_vgg_loss(self.vgg,
                                self.fake_patch_1[i], self.input_patch_1[i]) * self.opt.vgg
                    self.loss_vgg_b += loss_vgg_patch/float(self.opt.patchD_3 + 1)
                else:
                    self.loss_vgg_b += loss_vgg_patch
            self.loss_G = self.loss_G_A + self.loss_vgg_b*vgg_w # This is nice and compact! This is the sum of the GAN loss and the vgg loss for the whole image and the patches.
        self.loss_G.backward() # I see, we have the loss and we now want to backpropagate (this is the built-in function!)--> This just computes the gradients!

    def optimize_parameters(self, epoch): #Do the forward,backprop and update the weights... this is a very powerful and 'highly abstracted' function
        # forward
        self.forward()# This only does the generator--> It's fine. The discriminator's influence should be accounted for in the backward pass
		
		#In the forward function, we forward propagate the low-light image and start preparing the patch and the list of patches
        self.optimizer_G.zero_grad()#<-- This is extremely important and needs to be performed before we do anything related to updating the weights
        self.backward_G(epoch)# This is the function where we passed the syn. samples and real images (along with the samples to the disc. --> This also calculated the GANLoss and the VGG loss... INDIRECTLY, THIS REPRESENTS THE FULL PASS OF THE GRAND NETWORK USED TO UPDATE THE GENERATOR) Calculate the updated gradients. Accepting epoches is quite useless since we just check if epoch<0 to set 1 variable ( I dont see when would epoch ever be < 0???)
        self.optimizer_G.step()# Update the actual weights
		
		#We first get the operations of the generator out of the way before we work on the discriminator.
		
        # D_A
        self.optimizer_D_A.zero_grad()#--> This is crucial!
        self.backward_D_A()# Calculate the gradients for the global discriminator.
        #Perform the actual update for both discriminators
        self.optimizer_D_P.zero_grad()
        self.backward_D_P()#--> The local discriminator gradient calculation. There is a pattern to all of them (they all doing the same thing but the generator needs to account for the VGG loss as well.)
        self.optimizer_D_A.step()
        self.optimizer_D_P.step()


    def get_current_errors(self, epoch):
        D_A = self.loss_D_A.item()
        D_P = self.loss_D_P.item() if self.opt.patchD else 0
        G_A = self.loss_G_A.item()
        if self.opt.vgg > 0:
            vgg = self.loss_vgg_b.item()/self.opt.vgg if self.opt.vgg > 0 else 0
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ("vgg", vgg), ("D_P", D_P)])
        elif self.opt.fcn > 0:
            fcn = self.loss_fcn_b.item()/self.opt.fcn if self.opt.fcn > 0 else 0
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ("fcn", fcn), ("D_P", D_P)])


    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        if self.opt.skip > 0:
            latent_real_A = util.tensor2im(self.latent_real_A.data)
            latent_show = util.latent2im(self.latent_real_A.data)
            if self.opt.patchD:
                fake_patch = util.tensor2im(self.fake_patch.data)
                real_patch = util.tensor2im(self.real_patch.data)
                if self.opt.patch_vgg:
                    input_patch = util.tensor2im(self.input_patch.data)
                    self_attention = util.atten2im(self.real_A_gray.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B),('self_attention',self_attention)])
					
                    #ORIGINAL return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                #('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                #('fake_patch', fake_patch), ('input_patch', input_patch), ('self_attention', self_attention)])
     

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        if self.opt.patchD:
            self.save_network(self.netD_P, 'D_P', label, self.gpu_ids)

    def update_learning_rate(self):

        if self.opt.new_lr:
            lr = self.old_lr/2
        else:
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        if self.opt.patchD:
            for param_group in self.optimizer_D_P.param_groups:
                param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
