import numpy as np
import torch 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
# training parameters
N = 200 # number of batches in training
test_N = 20 # number of batches in test
batch_size = 512

# simulation parameters
dt = 0.01# discretized time step
sqrt_dt = np.sqrt(dt)     

temp = 1200 # temperature in K
kB = 8.6173303e-5  # Boltzmann constant in eV/K
kbT = kB * temp  # in eV
delta = 0.05 # the size of A and B balls

# final T of trajectories
# T_deadline = np.power(2*kbT, -0.5)
T_deadline = 4
num_steps = int(T_deadline/dt)
print("T deadline is",T_deadline)

lr_rate = 1e-3
KL_weight = 1

rugged_muller = True

A = torch.tensor([-1.11802943, -0.00285716],dtype=torch.float32).to(device)
B = torch.tensor([1.11802943, -0.00285716],dtype=torch.float32).to(device)
x_A = np.array([-1.0, -0.00285716])

if rugged_muller:
    A = np.hstack((np.array((-0.57274888, 1.42614653)), np.zeros(d - 2)))  # center of A
    B = np.hstack((np.array((0.56303756, 0.04406736)), np.zeros(d - 2)))  # center of B
    tmp = np.zeros(10)
    tmp[0] = 2.5 * np.random.rand(1) - 1.5
    tmp[1] = 2.5 * np.random.rand(1) - 0.5
    x_A = tmp
# learn from two pretrained models
path1 = "plots_1200/11_56_24_22_train/success_path.pt"
path2 = "plots_1200/23_54_22_16_train/success_path.pt"

# load a pretrained model
model_path = f"plots_1200/20_02_45_13_train/LearnExist_last_model.pt"

# combine two models directly
model1 = "plots_1200/23_54_22_16_train/last_model.pt"
model2 = "plots_1200/11_56_24_22_train/last_model.pt"