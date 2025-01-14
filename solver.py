# +
# Basic imports
import os
import csv
import numpy as np
import time
import datetime

# DL imports
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

# Reqd functions imports
from evaluation import *
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net

# Functional imports
import kornia
import hiddenlayer as hl


# -

class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):
        
        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.bce_loss = torch.nn.BCELoss()
        self.augmentation_prob = config.augmentation_prob

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.lamda = config.lamda
        
        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size
        self.save_model = config.save_model
        
        # Plots
        self.loss_history = hl.History()
        self.acc_history = hl.History()
        self.dc_history = hl.History()
        self.canvas = hl.Canvas()

        # Step size for plotting
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Paths
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode

        # Model training properties
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.t = config.t
        self.build_model()

    def build_model(self):
        # Load required model
        if self.model_type =='U_Net':
            self.unet = U_Net(img_ch=3,output_ch=1)
        elif self.model_type =='R2U_Net':
            self.unet = R2U_Net(img_ch=3,output_ch=1,t=self.t)
        elif self.model_type =='AttU_Net':
            self.unet = AttU_Net(img_ch=3,output_ch=1)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=3,output_ch=1,t=self.t)
        
        # Load optimizer
        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                      self.lr, [self.beta1, self.beta2])
        # Move model to device
        self.unet.to(self.device)

    def print_network(self, model, name):
        # Print out the network information
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))
        
    def dice_loss(self, pred, target):
        pred = pred.view(32, -1)
        target = target.view(32, -1)
        numerator = 2 * torch.sum(pred * target)
        denominator = torch.sum(pred + target)
        return 1 - (numerator + 1) / (denominator + 1)

    def train(self):
        
        # Debugging (Uncomment following lines)
        # a = torch.zeros((4, 3, 224, 224))
        # self.unet(a.to(self.device))
        
        unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,\
                                                                             self.lr,self.num_epochs_decay,\
                                                                             self.augmentation_prob))

        # U-Net Train
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
        else:
            # Train for Encoder
            lr = self.lr
            best_unet_score = 0.
            
            for epoch in range(self.num_epochs):

                self.unet.train(True)
                epoch_loss = 0
                
                acc = 0.    # Accuracy
                SE = 0.        # Sensitivity (Recall)
                SP = 0.        # Specificity
                PC = 0.     # Precision
                F1 = 0.        # F1 Score
                JS = 0.        # Jaccard Similarity
                DC = 0.        # Dice Coefficient
                length = 0

                for i, (images, GT) in enumerate(self.train_loader):
                    # GT : Ground Truth
                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    
                    # Zero grad
                    self.optimizer.zero_grad()

                    # SR : Segmentation Result
                    SR = self.unet(images)
                    SR_probs = torch.sigmoid(SR)
                    
                    # Convert to 1D tensor for loss calculation
                    SR_flat = SR_probs.view(SR_probs.size(0),-1)
                    GT_flat = GT.view(GT.size(0),-1)
                    
                    # Compute loss
                    loss = self.bce_loss(SR_flat,GT_flat) + self.lamda*self.dice_loss(SR_flat,GT_flat)
                    epoch_loss += loss.item()
                    
                    # Backprop
                    loss.backward()
                    self.optimizer.step()

                    # Get metrics
                    acc += get_accuracy(SR_probs,GT)
                    SE += get_sensitivity(SR_probs,GT)
                    SP += get_specificity(SR_probs,GT)
                    PC += get_precision(SR_probs,GT)
                    F1 += get_F1(SR_probs,GT)
                    JS += get_JS(SR_probs,GT)
                    DC += get_DC(SR_probs,GT)
                    length = i
                    
                length = (i + 1)
                acc = acc/length
                SE = SE/length
                SP = SP/length
                PC = PC/length
                F1 = F1/length
                JS = JS/length
                DC = DC/length
                
                train_dc = DC
                train_acc = acc
                train_loss = epoch_loss/length

#                 # Decay learning rate
#                 if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
#                     lr -= (self.lr / float(self.num_epochs_decay))
#                     for param_group in self.optimizer.param_groups:
#                         param_group['lr'] = lr
#                     print ('Decay learning rate to lr: {}.'.format(lr))
                
                
                # VALIDATION
                with torch.no_grad():
                    epoch_loss = 0
                    self.unet.train(False)
                    self.unet.eval()

                    acc = 0.    # Accuracy
                    SE = 0.        # Sensitivity (Recall)
                    SP = 0.        # Specificity
                    PC = 0.     # Precision
                    F1 = 0.        # F1 Score
                    JS = 0.        # Jaccard Similarity
                    DC = 0.        # Dice Coefficient
                    length=0
                    for i, (images, GT) in enumerate(self.valid_loader):

                        images = images.to(self.device)
                        GT = GT.to(self.device)
                        SR = torch.sigmoid(self.unet(images))
                        
                        # Convert to 1D tensor for loss calculation
                        SR_flat = SR.view(SR.size(0),-1)
                        GT_flat = GT.view(GT.size(0),-1)

                        # Compute loss
                        loss = self.bce_loss(SR_flat,GT_flat) + self.lamda*self.dice_loss(SR_flat,GT_flat)
                        epoch_loss += loss.item()
                        
                        acc += get_accuracy(SR,GT)
                        SE += get_sensitivity(SR,GT)
                        SP += get_specificity(SR,GT)
                        PC += get_precision(SR,GT)
                        F1 += get_F1(SR,GT)
                        JS += get_JS(SR,GT)
                        DC += get_DC(SR,GT)

                        length = i

                    length = (i + 1)
                    acc = acc/length
                    SE = SE/length
                    SP = SP/length
                    PC = PC/length
                    F1 = F1/length
                    JS = JS/length
                    DC = DC/length
                    unet_score = JS + DC
                    
                    valid_dc = DC
                    valid_acc = acc
                    valid_loss = epoch_loss/length

                    self.loss_history.log(epoch+1, train_loss=train_loss, valid_loss=valid_loss)
                    self.acc_history.log(epoch+1, train_acc=train_acc, valid_acc=valid_acc)
                    self.dc_history.log(epoch+1, train_dc=train_dc, valid_dc=valid_dc)
                    
                    with self.canvas:
                        self.canvas.draw_plot([self.loss_history['train_loss'], self.loss_history['valid_loss']],
                                              labels=['Train Loss', 'Valid loss'])
                        self.canvas.draw_plot([self.acc_history['train_acc'], self.acc_history['valid_acc']],
                                              labels=['Train Acc', 'Valid Acc'])
                        self.canvas.draw_plot([self.dc_history['train_dc'], self.dc_history['valid_dc']],
                                              labels=['Train Dice Coeff', 'Valid Dice Coeff'])
                    
                    grid_images = torch.cat([(images + 1)/2, torch.cat([SR, SR, SR], dim=1), torch.cat([GT, GT, GT], dim=1)])
                    grid = torchvision.utils.make_grid(grid_images, nrow=4)
                    torchvision.utils.save_image(grid, \
                                                  os.path.join(self.result_path,'%s_valid_%d_image.png'%\
                                                               (self.model_type,epoch+1)))
                    # Save Best U-Net model
                    if self.save_model:
                        if unet_score > best_unet_score:
                            best_unet_score = unet_score
                            best_epoch = epoch
                            best_unet = self.unet.state_dict()
                            print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
                            torch.save(best_unet,unet_path)
                                            
    def test(self):
        del self.unet
        del best_unet
        self.build_model()
        self.unet.load_state_dict(torch.load(unet_path))

        self.unet.train(False)
        self.unet.eval()

        acc = 0.    # Accuracy
        SE = 0.        # Sensitivity (Recall)
        SP = 0.        # Specificity
        PC = 0.     # Precision
        F1 = 0.        # F1 Score
        JS = 0.        # Jaccard Similarity
        DC = 0.        # Dice Coefficient
        length=0
        for i, (images, GT) in enumerate(self.valid_loader):

            images = images.to(self.device)
            GT = GT.to(self.device)

            SR = torch.sigmoid(self.unet(images))
            acc += get_accuracy(SR,GT)
            SE += get_sensitivity(SR,GT)
            SP += get_specificity(SR,GT)
            PC += get_precision(SR,GT)
            F1 += get_F1(SR,GT)
            JS += get_JS(SR,GT)
            DC += get_DC(SR,GT)

            length += images.size(0)

        acc = acc/length
        SE = SE/length
        SP = SP/length
        PC = PC/length
        F1 = F1/length
        JS = JS/length
        DC = DC/length
        unet_score = JS + DC


        f = open(os.path.join(self.result_path,'result.csv'), 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)
        wr.writerow([self.model_type,acc,SE,SP,PC,F1,JS,DC,self.lr,best_epoch,self.num_epochs,self.num_epochs_decay,self.augmentation_prob])
        f.close()


