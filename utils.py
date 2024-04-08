import numpy as np
import torch
import torchvision
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from config import *
from scipy.optimize import approx_fprime

# Hyperbolic tangent activation function
hyperbolic_tan = torch.nn.Tanh()

noise_eps = np.sqrt(2*kbT*dt)
energy_shift = 0.05
# Wei's potential energy function
def energy_wei(pos):
    x, y = pos
    term_1 = energy_shift * y
    term_2 = 4 * (1 - x**2 - y**2) ** 2
    term_3 = 2 * (x**2 - 2) ** 2
    term_4 = ((x + y) ** 2 - 1) ** 2
    term_5 = ((x - y) ** 2 - 1) ** 2
    eng = term_1 + ((term_2 + term_3 + term_4 + term_5 - 2.0) / 6.0)
    return eng

# Compute the gradient of Wei's potential function on x (1D)
def grad_wei(pos):
    x, y = pos
    grad = torch.zeros(2)
    x_square = torch.square(x)
    y_square = torch.square(y)
    xy_square = torch.square(x+y)
    xminusy_square = torch.square(x-y)
    grad[0] = (-16*(1-x_square-y_square)*x+8*(x_square-2)*x+4*(xy_square-1)*(x+y)+4*(xminusy_square-1)*(x-y))/6.0
    grad[1] = energy_shift+1/6.0*(-16*(1-x_square-y_square)*y+4*(xy_square-1)*(x+y)-4*(xminusy_square-1)*(x-y))

    return grad

# Higher dimension (parallel) of computing gradient
def grad_wei_nd(pos):
    x, y = pos[:,0], pos[:,1]
    grad = torch.zeros((pos.shape[0],2))
    x_square = torch.square(x)
    y_square = torch.square(y)
    xy_square = torch.square(x+y)
    xminusy_square = torch.square(x-y)
    term1 = 1-x_square-y_square
    
    grad[:,0] = 1/6.0*(-16*torch.multiply(term1, x)+8*(x_square-2)*x+4*(xy_square-1)*(x+y)+4*(xminusy_square-1)*(x-y))
    grad[:,1] = energy_shift+1/6.0*(-16*torch.multiply(term1, y)+4*(xy_square-1)*(x+y)-4*(xminusy_square-1)*(x-y))

    return grad

# Numpy version of computing gradient
def grad_wei_np(pos):
    grad = np.zeros((pos.shape[0],2))
    x,y=pos[:,0],pos[:,1]
    x_square = np.square(x)
    y_square = np.square(y)
    xy_square = np.square(x+y)
    xminusy_square = np.square(x-y)
    grad[:,0] = 1/6.0*(-16*(1-x_square-y_square)*x+8*(x_square-2)*x+4*(xy_square-1)*(x+y)+4*(xminusy_square-1)*(x-y))
    grad[:,1] = energy_shift+1/6.0*(-16*(1-x_square-y_square)*y+4*(xy_square-1)*(x+y)-4*(xminusy_square-1)*(x-y))

    return grad

# credit: Haoya Li
def potential_ruggered_muller(x):  # rugged muller potential; outputs potential

    # parameters
    aa = [-1, -1, -6.5, 0.7]
    bb = [0, 0, 11, 0.6]
    cc = [-10, -10, -6.5, 0.7]
    AA = [-200, -100, -170, 15]

    XX = [1, 0, -0.5, -1]
    YY = [0, 0.5, 1.5, 1]

    d = 10  # dimension; might be changed
    k = 5  # frequency of sin terms
    amp1 = 9  # mode of sin terms
    sgm = 0.05  # variance of other dimensions

    # rugged part; first two dimensions involved
    V1 = AA[0] * np.exp(
        aa[0] * (x[0] - XX[0]) ** 2 + bb[0] * (x[0] - XX[0]) * (x[1] - YY[0]) + cc[0] * (x[1] - YY[0]) ** 2)
    for j in range(1, 4):
        V1 += AA[j] * np.exp(
            aa[j] * (x[0] - XX[j]) ** 2 + bb[j] * (x[0] - XX[j]) * (x[1] - YY[j]) + cc[j] * (x[1] - YY[j]) ** 2)

    V1 = V1 + amp1 * np.sin(2 * math.pi * k * x[0]) * np.sin(2 * math.pi * k * x[1])

    # other dimensions
    V2 = 0
    for j in range(2, d):
        V2 += x[j] ** 2
    V2 /= 2 * sgm ** 2
    V = V1 + V2
    return V

# credit: Haoya Li
def grad_ruggered_muller(x): # rugged muller potential; outputs its gradient
    # parameters
    aa = [-1, -1, -6.5, 0.7]
    bb = [0, 0, 11, 0.6]
    cc = [-10, -10, -6.5, 0.7]
    AA = [-200, -100, -170, 15]

    XX = [1, 0, -0.5, -1]
    YY = [0, 0.5, 1.5, 1]

    d = 10  # dimension; might be changed
    k = 5  # frequency of sin terms
    amp1 = 9  # mode of sin terms
    sgm = 0.05  # variance of other dimensions
    
    # force(gradient); first dimension
    fx = AA[0] * (2 * aa[0] * (x[0] - XX[0]) + bb[0] * (x[1] - YY[0])) \
         * np.exp(aa[0] * (x[0] - XX[0]) ** 2 + bb[0] * (x[0] - XX[0]) * (x[1] - YY[0]) + cc[0] * (x[1] - YY[0]) ** 2) + \
         AA[1] * (2 * aa[1] * (x[0] - XX[1]) + bb[1] * (x[1] - YY[1])) \
         * np.exp(aa[1] * (x[0] - XX[1]) ** 2 + bb[1] * (x[0] - XX[1]) * (x[1] - YY[1]) + cc[1] * (x[1] - YY[1]) ** 2) + \
         AA[2] * (2 * aa[2] * (x[0] - XX[2]) + bb[2] * (x[1] - YY[2])) \
         * np.exp(aa[2] * (x[0] - XX[2]) ** 2 + bb[2] * (x[0] - XX[2]) * (x[1] - YY[2]) + cc[2] * (x[1] - YY[2]) ** 2) + \
         AA[3] * (2 * aa[3] * (x[0] - XX[3]) + bb[3] * (x[1] - YY[3])) \
         * np.exp(aa[3] * (x[0] - XX[3]) ** 2 + bb[3] * (x[0] - XX[3]) * (x[1] - YY[3]) + cc[3] * (x[1] - YY[3]) ** 2)

    # force(gradient); second dimension
    fy = AA[0] * (2 * cc[0] * (x[1] - YY[0]) + bb[0] * (x[0] - XX[0])) \
         * np.exp(aa[0] * (x[0] - XX[0]) ** 2 + bb[0] * (x[0] - XX[0]) * (x[1] - YY[0]) + cc[0] * (x[1] - YY[0]) ** 2) + \
         AA[1] * (2 * cc[1] * (x[1] - YY[1]) + bb[1] * (x[0] - XX[1])) \
         * np.exp(aa[1] * (x[0] - XX[1]) ** 2 + bb[1] * (x[0] - XX[1]) * (x[1] - YY[1]) + cc[1] * (x[1] - YY[1]) ** 2) + \
         AA[2] * (2 * cc[2] * (x[1] - YY[2]) + bb[2] * (x[0] - XX[2])) \
         * np.exp(aa[2] * (x[0] - XX[2]) ** 2 + bb[2] * (x[0] - XX[2]) * (x[1] - YY[2]) + cc[2] * (x[1] - YY[2]) ** 2) + \
         AA[3] * (2 * cc[3] * (x[1] - YY[3]) + bb[3] * (x[0] - XX[3])) \
         * np.exp(aa[3] * (x[0] - XX[3]) ** 2 + bb[3] * (x[0] - XX[3]) * (x[1] - YY[3]) + cc[3] * (x[1] - YY[3]) ** 2)

    # sin part
    fx = fx + amp1 * (2 * math.pi * k) * np.cos(2 * math.pi * k * x[0]) * np.sin(2 * math.pi * k * x[1])
    fy = fy + amp1 * (2 * math.pi * k) * np.cos(2 * math.pi * k * x[1]) * np.sin(2 * math.pi * k * x[0])
    fz = x[2:] / sgm ** 2  # other dimensions
    F = np.hstack((fx, fy, fz))  # stack them together
    return F

# this function is negative inside a circle with center B with radius r, positive outside
def position_cost(x_trajectory, r, F_scale): 
    loss = torch.tensor(0.0, dtype=torch.float32)
    # goes over all the samples in one batch
    for i in range(x_trajectory.shape[1]):
        meet_B = False
        # goes over all the steps, see which one is close to B
        dist_B = torch.norm(x_trajectory[:,i,:2]-B, dim=1)
        for j in range(x_trajectory.shape[0]):
            if dist_B[j]<r:
                loss += F_scale*(hyperbolic_tan(dist_B[j]**2-r**2))
                meet_B = True
                break

        if not meet_B:
            pos = x_trajectory[-1,i,:2]
            dist_B = torch.norm(pos-B)
            loss += F_scale*(hyperbolic_tan(dist_B**2-r**2))

    # dist_B = torch.norm(x_trajectory[-1]-B, dim=1)

    # loss = torch.sum(F_scale*(hyperbolic_tan(2*(dist_B-r-0.02))))
    return loss

def langevin_montecarlo_np(pos, grad_fn):
    grad = grad_fn(pos)
    noise = np.random.normal(size=(pos.shape[0],2))
    return pos-grad*dt + noise*noise_eps, noise

# Brownian motion using a neural network with time in batch
def langevin_network_batch(x, G, dt_update):
    noise = torch.normal(mean = 0, std = 1, size=(x.shape[0],2))
    grad = G.dvdx_batch(x)
    return x-grad*dt_update + noise*noise_eps, noise

def langevin_network_batch_withNoise(x, G, dt_update, noise):
    grad = G.dvdx_batch(x)
    return x-grad*dt_update + noise*noise_eps, noise

def langevin_onestep_groundtruth(x, get_energy, bias_ratio, noise): 
    # accept = False
    # while(not accept):
    # noise = torch.normal(mean = 0, std = 1, size = x.shape)
    bias_potential_grad = torch.from_numpy(approx_fprime(x.detach().numpy(), get_energy, epsilon=1e-8))
    grad_energy = grad_wei(x)
    grad = bias_potential_grad*bias_ratio+grad_energy
    return x-grad*dt + noise*noise_eps, noise
  
def save_traces(init_x, x_history_numpy, path):
    init_x = init_x.cpu().numpy()
    fig, ax = plt.subplots()
    
    plt.plot([A.numpy()[0],B.numpy()[0]], [A.numpy()[1],B.numpy()[1]],'ro',label="local minima")
    plt.plot(init_x[0], init_x[1],'go', label = "starting point")

    plt.scatter(np.array(x_history_numpy)[:,0], np.array(x_history_numpy)[:,1], s=3)
    plt.plot(np.array(x_history_numpy)[:,0], np.array(x_history_numpy)[:,1], linewidth=1,alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Trajectory of points following the gradient and Brownian motion")
    plt.legend()
    plt.savefig(path)
    plt.close()

def plot_traces(init_x, x_history_numpy):
    init_x = init_x.cpu().numpy()
    plot_traces_np(init_x, x_history_numpy)
    
def plot_traces_np(init_x, x_history_numpy):
    fig, ax = plt.subplots()
    
    plt.plot([A.numpy()[0],B.numpy()[0]], [A.numpy()[1],B.numpy()[1]],'ro',label="local minima")
    plt.plot(init_x[0], init_x[1],'go', label = "starting point")

    plt.scatter(np.array(x_history_numpy)[:,0], np.array(x_history_numpy)[:,1], s=3)
    plt.plot(np.array(x_history_numpy)[:,0], np.array(x_history_numpy)[:,1], linewidth=1,alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Trajectory of points following the gradient and Brownian motion")
    plt.legend()
    plt.show()

@torch.no_grad()
def draw_loss_curve(KL_loss_history, pos_loss_history, start_iter, path, name):
    x_epoch = np.arange(start_iter,start_iter+len(KL_loss_history))

    plt.plot(x_epoch, KL_loss_history, label='KL loss')
    plt.plot(x_epoch, pos_loss_history, label='F_smooth function')
    plt.plot(x_epoch, np.array(KL_loss_history)+np.array(pos_loss_history), label='Objective')
    
    plt.xlabel("Step number")
    plt.title("Loss function (KL_loss + smooth indicator function)")
    plt.legend()
    plt.savefig(os.path.join(path, name))
    plt.close()

@torch.no_grad()
def draw_probability_curve(prob_history, path):
    x_epoch = np.arange(0, len(prob_history), 1)
    plt.plot(x_epoch, prob_history)
    plt.xlabel("Step number")
    plt.title("Probability of the rare event")
    plt.savefig(os.path.join(path, 'prob.jpg'))
    plt.close()

@torch.no_grad()
def plot_F_function(r):
    x = np.arange(-1,1.2,0.1)
    y = np.arange(-1.5,1.5,0.1)

    X,Y = np.meshgrid(x, y) # grid of point
    XY = np.stack([X,Y]).reshape(2,-1).transpose()
    p = np.zeros((XY.shape[0]))

    for i in range(XY.shape[0]):
        dist_B = np.linalg.norm(XY[i]-B.numpy())
        p[i] = 10*(np.tanh(5*(dist_B-r-0.02)))
    p = p.reshape(X.shape)

    
    fig, ax = plt.subplots()
    ax.set_title('Function F smooth.')

    plt.pcolormesh(X,Y,p, cmap='RdBu', vmin = -np.abs(p).max(), 
                                         vmax = np.abs(p).max(), shading="auto")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar()
    plt.savefig(f"position_fun_{r}.png")

@torch.no_grad()
def plot_levelset():
    # parameters for Lucy's potential
    xa=-1.0 
    ya=0.0
    xb=1.0
    yb=0.0
    gamma = 0.01
    par = np.array([xa,ya,xb,yb,gamma]) # centers of sets A and B
    
    def Lucy(xy,par):
        xa=par[0]
        ya=par[1]
        xb=par[2] 
        yb=par[3]
        gamma=par[4]
        x = xy[:,0]
        y = xy[:,1]
        v = energy_shift*y+(1/6.0)*(4*(1-x**2-y**2)**2+2*(x**2-2)**2+((x+y)**2-1)**2+((x-y)**2-1)**2-2)
        return v

    def fpot(pts):
        return Lucy(pts,par)

    # define face potential on a meshgrid
    nx,ny= (101,101)
    nxy = nx*ny
    xmin = -1.5
    xmax = 1.5
    ymin = -1.5
    ymax = 1.5
    x1 = np.linspace(xmin,xmax,nx)
    y1 = np.linspace(ymin,ymax,ny)
    x_grid, y_grid = np.meshgrid(x1,y1)
    x_vec = np.reshape(x_grid, (nxy,1))
    y_vec = np.reshape(y_grid, (nxy,1))

    xy = np.concatenate((x_vec,y_vec),axis=1)
    v = fpot(xy)
    vmin = np.amin(v)
    vmax = np.amax(v)
    v_grid = np.reshape(v,(nx,ny))    
    # graphics
    ls = plt.contour(x_grid,y_grid,v_grid,np.linspace(vmin,2.0,10))
    # plt.colorbar(label="Potential function", orientation="vertical")
    axes=plt.gca()
    axes.set_aspect(1)

# compute the probability ratio between G and original potential
def compute_probability(delta_prime_history, x_history):
    p = torch.tensor(0.0, dtype=torch.float32)
    for j in range(x_history.shape[0]-1):
        x = x_history[j].squeeze()
        x_next = x_history[j+1].squeeze()
        grad_energy = grad_wei(x_history[j]).squeeze()
        delta = (x_next-x+grad_energy*dt)
        delta_prime = delta_prime_history[j]*np.sqrt(2*kbT*dt)
        p += 0.5*(-torch.norm(delta)**2+torch.norm(delta_prime)**2)/(2*kbT*dt)
    return p, torch.exp(p)

@torch.no_grad()
def compute_probability_test(delta_prime_history, x_history):
    delta_prime_history.requires_grad = False
    x_history.requires_grad = False
    p = torch.tensor(0.0, dtype=torch.float32)
    for j in range(x_history.shape[0]-1):
        x = x_history[j].squeeze()
        x_next = x_history[j+1].squeeze()
        grad_energy = grad_wei(x_history[j]).squeeze()
        delta = (x_next-x+grad_energy*dt)
        delta_prime = delta_prime_history[j]*np.sqrt(2*kbT*dt)
        p += 0.5*(-torch.norm(delta)**2+torch.norm(delta_prime)**2)/(2*kbT*dt)
    return p, torch.exp(p)

# compute the probability ratio between two models
def compute_probability_model(x_history, G):
    x_history.requires_grad = True
    p = torch.tensor(0.0, dtype=torch.float32)
    grad_energy = G.dvdx_batch(x_history[1:]) 
    noise = (x_history[1:]-x_history[:-1]+grad_energy*dt)/np.sqrt(2*kbT*dt)
    delta = torch.norm(noise,dim=1)**2

    grad_original = grad_wei_nd(x_history[1:]) 
    noise_prime = (x_history[1:]-x_history[:-1]+grad_original*dt)/np.sqrt(2*kbT*dt)
    delta_prime = torch.norm(noise_prime, dim=1)**2
    log_ratio = 0.5*torch.sum(delta-delta_prime)
    return log_ratio, -0.5*torch.sum(delta)
