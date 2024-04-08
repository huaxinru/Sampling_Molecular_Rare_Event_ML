import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Distribution, Uniform
from config import *
from utils import *

class Sequentialmodel_logp(nn.Module):
    """
    Neural network model for learning a potential function.
    
    The input is a 2x1 tensor, and the output is the gradient of the learned potential function with respect to x.
    The neural network has a 2D->1D architecture.
    """
    
    def __init__(self, layers):
        super().__init__()  # Call __init__ from parent class            
        self.activation = nn.Tanh()  # Activation function
        linears = []

        for i in range(len(layers)-1):
            linears.append(nn.Linear(layers[i], layers[i+1]))
        linears.append(nn.Linear(layers[-1], 1))
        
        self.linears = nn.Sequential(*linears)
        self.nn_depth = len(layers) # One neural network's depth, we have 3 copies of it
        self.r = 0.5      # The radius of the smooth Indicator function
        self.F_scale = 100   # The scale of the smooth Indicator function
        self.T_deadline = T_deadline
        self.energy_scale = 0.3
        
        self.layers = layers
        
        # Randomly initialize the weights
        for i in range(len(layers)): 
            nn.init.xavier_uniform_(self.linears[i].weight.data)
            nn.init.zeros_(self.linears[i].bias.data)
            
    def forward(self,x):    
        """
        Forward pass of the model.
        
        Outputs the energy function.
        """  
        output = torch.zeros(x.shape[0])

        feature1 = torch.exp(-torch.norm(x-A, dim=1)).unsqueeze(1)
        feature2 = torch.exp(-torch.norm(x-B, dim=1)).unsqueeze(1)
        a = torch.cat((x,feature1,feature2), dim=1)
        # a = x.clone()

        for i in range(len(self.layers)-1):
            a = self.linears[i](a)
            a = self.activation(a) 

        output = self.linears[-1](a).squeeze()
        return output
    
    def multiply_r_withrate(self, rate, F_rate):
        """
        Multiply the radius and scale wtih rate 
        """
        self.r = self.r*rate
        self.F_scale = self.F_scale*F_rate

    def change_energy_scale(self, rate):
        """
        Change the energy scale.
        """
        self.energy_scale = rate
        

    def dvdx_batch(self,x):
        """
        Calculate the gradient of the potential function with respect to x for a batch of samples.
        """
        # x.requires_grad = True 
        q = self.forward(x)
        u_x = torch.autograd.grad(torch.sum(q),x, retain_graph=True, create_graph=True, allow_unused=True)[0]

        if rugged_muller:
            return u_x+grad_ruggered_muller(x)*self.energy_scale
        else:
            return u_x+grad_wei_nd(x)*self.energy_scale
        
    def KL_loss(self, delta_prime_history, x_history):
        """
        Compute the KL loss between the generated gradient and the gradient of the potential function.
        """
        if rugged_muller:
            grad_energy = grad_ruggered_muller(x_history[1:]).detach()
        else:
            grad_energy = grad_wei_nd(x_history[1:]).detach()

        
        noise = (x_history[1:]-x_history[:-1]+grad_energy*dt)/np.sqrt(2*kbT*dt)
        delta = torch.norm(noise,dim=1)**2
        # noise_prime = (x_history[1:]-x_history[:-1]+grad_history*dt)/np.sqrt(2*kbT*dt)
        delta_prime = torch.norm(delta_prime_history, dim=1)**2

        return 0.5*torch.sum(delta-delta_prime), -0.5*torch.sum(delta_prime)


    def position_loss(self, x_trajectory):
        """
        Compute the position loss by using a smooth Indicator function.
        """
        if rugged_muller:
            return position_cost(x_trajectory[:2], self.r, self.F_scale)

        return position_cost(x_trajectory, self.r, self.F_scale)

class Combined_model(nn.Module):
    """
    The model that combines two bias potentials
    
    The input is a 2x1 tensor, and the output is the gradient of the learned potential function with respect to x.
    The neural network has a 2D->1D architecture.
    """
    
    def __init__(self, model1, model2):
        super().__init__()  # Call __init__ from parent class            
        self.model1 = model1
        self.model2 = model2
        self.energy_scale = 0.3
        self.a1 = 0.2755
        # self.a1 = 0.5
        self.a2 = 1-self.a1

    def forward(self,x):    
        """
        Forward pass of the model.
        
        Outputs the energy function.
        """  
        p1 = self.model1(x)
        p2 = self.model2(x)

        output = -torch.log(self.a1*torch.exp(-p1/(2*kbT))+self.a2*torch.exp(-p2/(2*kbT)))*2*kbT
        return output
        

    def dvdx_batch(self,x):
        """
        Calculate the gradient of the potential function with respect to x for a batch of samples.
        """
        # x.requires_grad = True 
        p1_grad = self.model1.dvdx_batch(x)
        p2_grad = self.model2.dvdx_batch(x)
        u1 = self.model1(x)
        u2 = self.model2(x)
        p1 = torch.exp(-u1/(2*kbT)).unsqueeze(1)
        p2 = torch.exp(-u2/(2*kbT)).unsqueeze(1)
        average_grad = self.a1*torch.mul(p1,p1_grad)+self.a2*torch.mul(p2,p2_grad)
        constant = (self.a1*p1+self.a2*p2)
        div = torch.div(average_grad, constant)
        return div+grad_wei_nd(x)*self.energy_scale

    def change_energy_scale(self, rate):
        """
        Change the energy scale.
        """
        self.energy_scale = rate
        
class Singularity_model(nn.Module):
    """
    Neural network model for learning a potential function.
    
    The input is a 2x1 tensor, and the output is the gradient of the learned potential function with respect to x.
    The neural network has a 2D->1D architecture.
    """
    
    def __init__(self, layers):
        super().__init__()  # Call __init__ from parent class            
        self.activation = nn.Tanh()  # Activation function
        self.relu_activation = nn.ReLU()
        linears = []
        for j in range(3):
            for i in range(len(layers)-1):
                linears.append(nn.Linear(layers[i], layers[i+1]))
            

            linears.append(nn.Linear(layers[-1], 1))
        
        self.linears = nn.Sequential(*linears)
        print(len(linears))
        self.nn_depth = len(layers) # One neural network's depth, we have 3 copies of it
        self.r = 0.5         # The radius of the smooth Indicator function
        self.F_scale = 100    # The scale of the smooth Indicator function
        self.T_deadline = T_deadline
        self.energy_scale = 1.0
        
        self.layers = layers
        
        # Randomly initialize the weights
        for i in range(len(layers)): 
            nn.init.xavier_uniform_(self.linears[i].weight.data)
            nn.init.zeros_(self.linears[i].bias.data)

    def singularity(self, x, a, b): 
        sum_s = torch.square(x[:,0] - a)+torch.square(x[:,1]-b)
        s = torch.log(sum_s)*1e-2
        return s

            
    def forward(self,x):    
        """
        Forward pass of the model.
        
        Outputs the energy function.
        """  
        output = torch.zeros(x.shape[0])

        layer_index = 0
        for j in range(3):
            a = x.clone()
            for ll in range(len(self.layers)-1):
                a = self.linears[layer_index](a)
                if ll!=len(self.layers)-2 or j!=2: a = self.relu_activation(a) 
                layer_index+=1
       
            if j==0: 
                a = self.linears[layer_index](a)
                layer_index +=1
                a = torch.multiply(self.singularity(x,A[0],A[1]),a.squeeze())

            if j==1: 
                a = self.linears[layer_index](a)
                layer_index +=1
                a = torch.multiply(self.singularity(x,B[0],B[1]),a.squeeze())
            if j==2:      
                a = self.activation(a)
                a = self.linears[layer_index](a)
            output +=a.squeeze()
        return output
    
    def multiply_r_withrate(self, rate, F_rate):
        """
        Multiply the radius and scale wtih rate 
        """
        self.r = self.r*rate
        self.F_scale = self.F_scale*F_rate

    def change_energy_scale(self, rate):
        """
        Change the energy scale.
        """
        self.energy_scale = rate
        

    def dvdx_batch(self,x):
        """
        Calculate the gradient of the potential function with respect to x for a batch of samples.
        """
        # x.requires_grad = True 
        q = self.forward(x)
        u_x = torch.autograd.grad(torch.sum(q),x, retain_graph=True, create_graph=True, allow_unused=True)[0]
        return u_x[:,:]+grad_wei_nd(x)*self.energy_scale
        
    def KL_loss(self, delta_prime_history, x_history):
        """
        Compute the KL loss between the generated gradient and the gradient of the potential function.
        """
        grad_energy = grad_wei_nd(x_history[1:]).detach()

        
        noise = (x_history[1:]-x_history[:-1]+grad_energy*dt)/np.sqrt(2*kbT*dt)
        delta = torch.norm(noise,dim=1)**2
        # noise_prime = (x_history[1:]-x_history[:-1]+grad_history*dt)/np.sqrt(2*kbT*dt)
        delta_prime = torch.norm(delta_prime_history, dim=1)**2

        return 0.5*torch.sum(delta-delta_prime), -0.5*torch.sum(delta_prime)


    def position_loss(self, x_trajectory):
        """
        Compute the position loss by using a smooth Indicator function.
        """
        return position_cost(x_trajectory, self.r, self.F_scale)
