import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torch import nn, optim
from tqdm.auto import tqdm
import random
import os
from utils import *
from config import *
import copy

objective_history = []
x_history = []
x_history_numpy = []
success_num=0
num_steps = int(T_deadline/dt)

A = A.numpy()
B = B.numpy()

x = x_A[:2]
x = np.expand_dims(x,0)

batch_size = 5000
sim_num = 20000

init_x = np.repeat(x,batch_size, axis=0)
up = 0
down = 0
fig, ax = plt.subplots()
unsuccess_path = 0
print("kbT=",kbT)
noise_pos_trajectory = []

for sim in tqdm(range(sim_num)):
    x = init_x.copy()
    mask = np.zeros(x.shape[0])
    x_history = np.zeros((num_steps+1,batch_size,2))
    delta_prime_history = np.zeros((num_steps,batch_size,2))
    x_history[0] = x.copy()

    for i in range(num_steps): 
        x, delta_prime = langevin_montecarlo_np(x, grad_wei_np)
        x_history[i+1] = x.copy()
        delta_prime_history[i]=delta_prime.copy()
        for ii in range(batch_size):
            if mask[ii]==0:
                if np.linalg.norm(x[ii]-B)<delta:
                    pass_index = np.argmin(np.abs(x_history[:i+1,ii,0]))
                    if x_history[pass_index,ii,1]>0:
                        up+=1
                    else:
                        down+=1
                    success_num +=1
                    mask[ii]=1
                    path = x_history[:(i+1),ii,:]
                    noise_pos_trajectory.append((delta_prime_history[:i,ii,:], x_history[:i+1,ii,:]))
                    plt.plot(np.array(path)[:,0], np.array(path)[:,1], linewidth=0.5,alpha=0.5)
                    print("Success reaching local minimum B, total_num: ", success_num)

                if np.linalg.norm(x[ii]-A)<delta:
                    mask[ii]=1
                    # if unsuccess_path<1000:
                    #     path = x_history[:(i+1),ii,:]
                    #     plt.plot(np.array(path)[:,0], np.array(path)[:,1], linewidth=0.5,alpha=0.5)
                    #     unsuccess_path +=1

    # if unsuccess_path<1000:
    #     for ii in range(batch_size):
    #         if mask[ii]==0: 
    #             path = x_history[:,ii,:]
    #             plt.plot(np.array(path)[:,0], np.array(path)[:,1], linewidth=1,alpha=0.5)
    #             unsuccess_path += 1

num_simulation = sim_num*batch_size
success_rate = success_num/num_simulation

plt.xlabel("x")
plt.ylabel("y")
plt.xlim([-1.5,1.5])
plt.ylim([-1.5,1.5])
plt.title("Trajectory of naive monte carlo")
plt.savefig("position_fun.png")

with open("Monte_carlo_testResults.txt", "a") as file:
    file.write(f"dt={dt}, T={temp}, tau={T_deadline}, num_simulation{num_simulation}.\n")
    file.write(f"numbers of going through two channels: {up},{down}.\n")
    file.write(f"Success num: {success_num}.\n")
    file.write(f"Success Rate: {success_rate}.\n")
    confidence_interval = np.sqrt(success_rate*(1-success_rate)/num_simulation)*1.96
    file.write(f"Confidence interval: {success_rate-confidence_interval, success_rate+confidence_interval}.\n")
np.save("noise_x_1", noise_pos_trajectory)