import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from modules import *
from tqdm import tqdm
from scipy import stats
from pathlib import Path
import copy
from config import *
from scipy.interpolate import *
import random
import seaborn as sns
import math
torch.autograd.set_detect_anomaly(False)

class RareEventNNSampler():
    def __init__(self, model_layers, save_path, size_batch, lr_rate = 1e-3):
        
        # Parameters
        self.KL_loss_weight = 1
        self.size_batch = size_batch
        self.model_layers = model_layers
        self.save_path = save_path

        # Initialize the model and optimizer
        self.G = Sequentialmodel_logp(model_layers).to(device)
        self.optimizer = optim.AdamW(self.G.parameters(), lr=lr_rate)

        # Other variables
        self.best_model = None   
        self.start_iter = 0    # the iteration that the parameters (F_scale, energy_scale, r) are constant
        self.best_epoch = 0

        self.setup_simulation()

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def setup_simulation(self):
        # Initialize variables for the simulation
        self.KL_loss_history = []
        self.pos_loss_history = []
        init_x = np.expand_dims(x_A,0)
        self.init_x = torch.from_numpy(np.repeat(init_x, self.size_batch, axis=0)).float().to(device)
        self.dt_update = dt
        self.noise_size = self.size_batch

        # Other variables
        self.min_loss = np.Inf
        self.num_steps = int(T_deadline/dt)
        self.total_probability = []
        self.noise_pos_trajectory = []
        self.validation_probability = []
        self.val_noise = torch.normal(mean = 0, std = 1, size=(self.num_steps, self.size_batch,2))


    def setup_each_step_simulation(self):
        x_history = torch.zeros((self.num_steps+1,self.size_batch,2))
        delta_prime_history = torch.zeros((self.num_steps,self.size_batch,2))
        return x_history, delta_prime_history

    def initialize_move_right(self, N=10000):
        pts = np.loadtxt('FEM_solution/Lucy_pts.csv', delimiter=',', dtype=float)
        pts_torch = torch.from_numpy(pts).float()
        loss = 1e6
        sim = 0
        while loss>1000 and sim<N:
            loss = 0
            sim+=1
            distance = self.G(pts_torch)+pts_torch[:,0]*2
            loss += torch.norm(distance)**2
            if (sim+1)%100==0 and self.save_path: 
                print(f"Loss at step {sim+1} is {loss}.")

            self.optimizer.zero_grad(set_to_none=True) 
            loss.backward()
            self.optimizer.step()

        self.save_G_plot(self.save_path)

    def train(self, N, KL_loss_division=10, continue_training=False):
        # Train the model to be an energy function that can drive the particles moving to B
        # two loss: position loss (smooth indicator function) + KL loss (make the two trajectory distributions close)
        self.setup_simulation()
        self.G.change_energy_scale(1.0) 
        init_path = Path(os.path.join(self.save_path, "init/"))
        init_path.mkdir() 
        self.save_G_plot(init_path)
        self.val_plot(1, 0)
        self.G.train()
        dt_index = 0
        for sim in tqdm(range(N)):  
            KL_loss = torch.tensor(0.0, dtype=torch.float32).to(device)
            x = self.init_x.detach().clone()
            x.requires_grad = True   
            x_history, delta_prime_history = self.setup_each_step_simulation()
            x_history[0] = x.clone()
            # this is to balance the loss between different batches
            batch_log_qs = []
            batch_KL_losses = []
            mask = np.zeros(x.shape[0])
            # batch_indices = torch.randint(0, self.noise_size, (self.size_batch,))

            for i in range(self.num_steps):
                # the x for the next step is (dt*gradient of NN+Eng + random noise + x)
                x, delta_prime = langevin_network_batch(x, self.G, self.dt_update)
                x_history[i+1] = x.clone()
                delta_prime_history[i]=delta_prime.clone()               

                # check each trajectory in the batch, if it reaches B, accumualate the KL loss  
                masked_indices = np.ma.masked_array(range(self.size_batch),mask).compressed()      
                for ii in masked_indices:
                    if torch.any(torch.isnan(x)):
                        print(x)
                    if torch.norm(x[ii,:]-B)<delta:
                        loss,log_q = self.G.KL_loss(delta_prime_history[:i,ii,:], x_history[:i+1,ii,:])  
                        batch_log_qs.append(log_q.unsqueeze(0).detach())   
                        batch_KL_losses.append(loss.unsqueeze(0))        
                        mask[ii]=1
                    if torch.abs(x[ii,0])>=1.5 or torch.abs(x[ii,1])>=1.5:
                        mask[ii]=1

                if np.all(mask==1):
                    break

            if batch_log_qs:
                batch_log_qs = torch.cat(batch_log_qs)/20    
                batch_log_qs -= torch.mean(batch_log_qs)

                if batch_log_qs.shape[0]==1:
                    batch_log_qs += 5

            if batch_KL_losses:
                batch_KL_losses = torch.cat(batch_KL_losses)
                KL_loss = torch.sum(torch.multiply(batch_KL_losses, torch.exp(batch_log_qs)))

            # compute the position loss for all trajectories
            # the loss does not have gradient for trajectories with endpoints inside B or far away from B 
            KL_loss = KL_loss/self.size_batch
            pos_loss = self.G.position_loss(x_history[:,mask==0,:])/self.size_batch

            accumulate_loss = pos_loss + KL_loss*self.KL_loss_weight
            self.optimizer.zero_grad(set_to_none=True) 
            accumulate_loss.backward()
            self.optimizer.step()  
            # increase the original energy scale (will be 1 eventually) 

            # print(f"Position loss={pos_loss}, KL loss={KL_loss}.")

            if continue_training:
                self.G.r =0.049720128493546116
                self.G.F_scale = 7289.048368510332 

                self.KL_loss_history.append(KL_loss.item())
                self.pos_loss_history.append(pos_loss.item())
                if self.start_iter==0:
                    self.start_iter = sim
                if accumulate_loss<self.min_loss:
                    self.min_loss = accumulate_loss.item()
                    self.best_model = copy.deepcopy(self.G.state_dict())
                    self.best_epoch = sim

            else:
                if self.G.r>delta:                         
                    self.G.multiply_r_withrate(0.95, 1.1)                
                
                else:
                    self.KL_loss_history.append(KL_loss.item())
                    self.pos_loss_history.append(pos_loss.item())
                    if self.start_iter==0:
                        self.start_iter = sim
                    if accumulate_loss<self.min_loss:
                        self.min_loss = accumulate_loss.item()
                        self.best_model = copy.deepcopy(self.G.state_dict())
                        self.best_epoch = sim

            if (sim+1)%5==0 and self.save_path:               
                self.save_G_plot(self.save_path, str(sim+1)+".png")
                self.val_plot(1, sim+1)
                self.G.train()
                if (sim+1)%100==0:
                    plot_dir = Path(os.path.join(self.save_path, str(sim+1)))     
                    plot_dir.mkdir(exist_ok=True)
                    torch.save(self.G.state_dict(), os.path.join(plot_dir,f"model_{sim+1}.pt"))


        if self.save_path:
            print(f"Final parameters: r={self.G.r}, F_scale={self.G.F_scale}, KL weight={self.KL_loss_weight}.")
            torch.save({
            'epoch': N,
            'model_state_dict': self.G.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'KL_loss': KL_loss.item(),
            'pos_loss': pos_loss.item(),
            }, os.path.join(self.save_path,"last_model.pt"))

            torch.save(self.KL_loss_history, os.path.join(self.save_path,"KL_loss.pt"))
            torch.save(self.pos_loss_history, os.path.join(self.save_path,"pos_loss.pt"))     
            draw_loss_curve(self.KL_loss_history, self.pos_loss_history, self.start_iter, self.save_path, "loss.png")

    def load_model(self,PATH):
        # Load a saved model from a file

        checkpoint = torch.load(PATH)
        with open(os.path.join(self.save_path, "testResults.txt"), "a") as file:
            file.write(f"loading checkpoint from {PATH}.\n")

        # self.G.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.G.load_state_dict(checkpoint)
        self.G.train()

    def test(self, N):
        # Train the model to be an energy function that can drive the particles moving to B
        # two loss: position loss (smooth indicator function) + KL loss (make the two trajectory distributions close)
        self.G.eval()
        self.setup_simulation()
        self.G.change_energy_scale(1.0)
        self.save_G_plot(self.save_path)
        
        up = 0
        down = 0

        success_path = []
        noise_pos_trajectory = []
        alphas = []
        self.total_probability = []
        for sim in tqdm(range(N)):  
            success_num =0      
            x = self.init_x.detach().clone()
            x.requires_grad = True   
            x_history, delta_prime_history = self.setup_each_step_simulation()
            x_history[0] = x.clone()
            mask = np.zeros(x.shape[0]).astype(int)
        
            for i in range(self.num_steps):
                # the x for the next step is (dt*gradient of NN+Eng + random noise + x)
                x = x.clone().detach()
                x.requires_grad = True
                x, delta_prime = langevin_network_batch(x, self.G, self.dt_update)
                x_history[i+1] = x.clone()
                delta_prime_history[i]=delta_prime.detach().clone()

                # check each trajectory in the batch, if it reaches B, accumualate the KL loss
                masked_indices = np.ma.masked_array(range(self.size_batch),mask).compressed()
                for ii in masked_indices:
                    # the delta is the size of the A, B balls
                    if torch.norm(x[ii,:]-B)<delta:
                        mask[ii]=1
                        success_num +=1

                        if torch.sum(x_history[:,ii,1])>0: 
                            up+=1
                        else:
                            down+=1
                        _, p = compute_probability_test(delta_prime_history[:i,ii].detach(), x_history[:i+1,ii].detach()) 

                        self.total_probability.append(p.item())

                        success_path.append(x_history[:(i+1),ii].detach().numpy())

                    if torch.norm(x-A)<delta:
                        mask[ii]=1

                if np.all(mask==1):
                    break
            
            self.total_probability += [0]*(self.size_batch-success_num)
            success_path+=[0]*(self.size_batch-success_num)

        with open(os.path.join(self.save_path, "testResults.txt"), "a") as file:
            file.write(f"Two channels: {up}, {down}\n")

        torch.save(success_path, os.path.join(self.save_path, "success_path.pt"))
        final_p = self.plot_paths_compute_stats(success_path, N*self.size_batch)

        return final_p

    def val_plot(self, N, sim_iteration):
        # Train the model to be an energy function that can drive the particles moving to B
        # two loss: position loss (smooth indicator function) + KL loss (make the two trajectory distributions close)
        self.G.eval()
        self.G.change_energy_scale(1.0)

        success_path = []
        self.total_probability = []
        noise_pos_trajectory = []
        alphas = []
        for sim in range(N):  
            success_num =0      
            x = self.init_x.detach().clone()
            x.requires_grad = True   
            x_history, delta_prime_history = self.setup_each_step_simulation()
            x_history[0] = x.clone()
            mask = np.zeros(x.shape[0]).astype(int)
            # batch_indices = torch.randint(0, self.noise_size, (self.size_batch,))
            for i in range(self.num_steps):
                # the x for the next step is (dt*gradient of NN+Eng + random noise + x)
                x = x.clone().detach()
                x.requires_grad = True
                x, delta_prime = langevin_network_batch_withNoise(x, self.G, self.dt_update, self.val_noise[i])
                x_history[i+1] = x.clone()
                delta_prime_history[i]=delta_prime.detach().clone()

                # check each trajectory in the batch, if it reaches B, accumualate the KL loss
                masked_indices = np.ma.masked_array(range(self.size_batch),mask).compressed()
                for ii in masked_indices:
                    # the delta is the size of the A, B balls
                    if torch.norm(x[ii,:]-B)<delta:
                        mask[ii]=1

                        success_path.append(x_history[:(i+1),ii,:].detach().numpy())
                        _, p = compute_probability_test(delta_prime_history[:i,ii].detach(), x_history[:i+1,ii].detach()) 
                        self.total_probability.append(p)
                if np.all(mask==1):
                    break

            for ii in range(self.size_batch):
                if mask[ii]==0:
                    success_path.append(x_history[:,ii].detach().numpy())
            self.total_probability += [0]*(self.size_batch-success_num)

        _ = self.plot_paths_compute_stats(success_path, N*self.size_batch, sim_iteration, compute_stats=False)
        self.G.train()

    # learn from successful paths
    def learn_existing_paths(self, N, paths_dirs):
        self.setup_simulation()
        self.val_plot(1, 0)
        self.G.train()

        up_path_collections = []
        down_path_collections = []
        for paths_dir in paths_dirs:
            # load in numpy files or pytorch files
            if paths_dir.endswith('.npy'):
                torch_results =  np.load(paths_dir, allow_pickle=True)
                for i in range(len(torch_results)):
                    if np.sum(torch_results[i,1])>0:
                        up_path_collections.append((torch.from_numpy(torch_results[i,0]).float(), torch.from_numpy(torch_results[i,1]).float()))
                    else:
                        down_path_collections.append((torch.from_numpy(torch_results[i,0]).float(), torch.from_numpy(torch_results[i,1]).float()))
            else:
                torch_results =  torch.load(paths_dir)
                for i in range(len(torch_results)):
                    if not np.isscalar(torch_results[i]):
                        path = torch.from_numpy(torch_results[i])
                        if torch.sum(path[:,1])>0:
                            up_path_collections.append(path.float())
                        else:
                            down_path_collections.append(path.float())
        print(len(up_path_collections), len(down_path_collections))

        init_path = Path(os.path.join(self.save_path, "LearnExist_init/"))
        init_path.mkdir() 

        self.save_G_plot(init_path)
        KL_loss_history = []
        probabilities_history = []
        for sim in tqdm(range(N)):  
            loss = torch.tensor(0.0, dtype=torch.float32).to(device)
            path_log_ps = []
            entropys = []
            for _ in range(self.size_batch):
                random_num = np.random.rand()
                if np.random.rand()>0.5:
                    x_history = random.choice(up_path_collections)
                else:
                    x_history = random.choice(down_path_collections)

                log_ratio, logp = compute_probability_model(x_history.detach(), self.G)
                entropys.append(log_ratio.unsqueeze(0))
                path_log_ps.append(logp.unsqueeze(0))

            path_log_ps = torch.cat(path_log_ps)/20  
            entropys = torch.cat(entropys)

            mean = torch.mean(path_log_ps).item()
            probabilities_history.append(mean)
            path_log_ps -= mean

            if path_log_ps.shape[0]==1:
                path_log_ps += 5

            loss = torch.sum(torch.multiply(entropys, torch.exp(path_log_ps)))

            self.KL_loss_history.append(loss.item())        
            self.optimizer.zero_grad(set_to_none=True) 
            loss.backward()
            self.optimizer.step()  

            if (sim+1)%5==0 and self.save_path: 
                self.val_plot(1, sim+1)
                self.save_G_plot(self.save_path, str(sim+1)+".png")
                self.G.train()

        torch.save(self.G.state_dict(), os.path.join(self.save_path,"LearnExist_last_model.pt"))
        torch.save(self.KL_loss_history, os.path.join(self.save_path,"LearnExist_KL_loss.pt"))     
        draw_loss_curve(self.KL_loss_history, [0]*len(self.KL_loss_history), self.start_iter, self.save_path, "LearnExist_loss.png")
        draw_probability_curve(probabilities_history, self.save_path)

    def combine_two_learned_models(self, model_1_dict, model_2_dict):
        self.setup_simulation()
        model1 = Sequentialmodel_logp(self.model_layers).to(device)
        checkpoint1 = torch.load(model_1_dict)
        model1.load_state_dict(checkpoint1["model_state_dict"])
        model2 = Sequentialmodel_logp(self.model_layers).to(device)
        checkpoint2 = torch.load(model_2_dict)
        model2.load_state_dict(checkpoint2["model_state_dict"])
        self.G = Combined_model(model1, model2)


    # learn from the ground truth directly, use L2 loss
    def train_learn_ground_truth_directly(self, N):
        print("the learning rate is ", self.get_lr())
        pts = np.loadtxt('FEM_solution/Lucy_pts.csv', delimiter=',', dtype=float)
        bias_pot = np.load(f"FEM_solution/bias_potential_{temp}.npy")

        for sim in tqdm(range(N)):
            loss = 0
            pts_torch = torch.from_numpy(pts).float()

            distance = self.G(pts_torch)-torch.from_numpy(bias_pot)
            loss += torch.norm(distance)**2
            if (sim+1)%100==0 and self.save_path: 
                print(f"Loss at step {sim+1} is {loss}.")

            self.optimizer.zero_grad(set_to_none=True) 
            loss.backward()
            self.optimizer.step()

        self.save_G_plot(self.save_path)
        torch.save(self.G.state_dict(), os.path.join(self.save_path,"LearnGT_last_model.pt"))
        torch.save(self.KL_loss_history, os.path.join(self.save_path,"LearnGT_KL_loss.pt"))     
        draw_loss_curve(self.KL_loss_history, [0]*len(self.KL_loss_history), self.start_iter, self.save_path, "LearnExist_loss.png")

    # test if our FEM solved ground truth bias potential is good
    def test_ground_truth(self, N, bias_ratio=1.0):
        self.size_batch = 1
        self.setup_simulation()      
        self.save_G_plot(self.save_path)

        pts = np.loadtxt('FEM_solution/Lucy_pts.csv', delimiter=',', dtype=float)
        bias_pot = np.load(f"FEM_solution/bias_potential_{temp}.npy")
        interp = LinearNDInterpolator(pts,bias_pot)
        def get_energy(point):
            X, Y = point
            Z = interp(X, Y)
            return Z

        up = 0
        down = 0
        paths = []
        log_probs = []

        noise_pos_trajectory = []
        
        for sim in tqdm(range(N)):  
            success =False      
            # x = torch.normal(0,1,size=(2,))*delta+A
            x = self.init_x.squeeze().detach().clone()
            x_history = torch.zeros((self.num_steps+1, 2))
            x_history[0] = x.clone()
            delta_prime_history = torch.zeros((self.num_steps, 2))

            mask = np.zeros(x.shape[0])
            first_noise = torch.zeros(2)
            for i in range(self.num_steps):
                # the x for the next step is (dt*gradient of NN+Eng + random noise + x)
                x_update, delta_prime = langevin_onestep_groundtruth(x, get_energy, bias_ratio)

                x = x_update.detach().clone().float()

                x_history[i+1] = x.clone()
                delta_prime_history[i]=delta_prime.detach().clone()
                if torch.norm(x-B)<delta:
                    success =True
                    if torch.sum(x_history[:,1])>0:
                        up+=1
                    else:
                        down +=1
                    
                    _, p = compute_probability_test(delta_prime_history[:i].detach(), x_history[:i+1].detach()) 

                    self.total_probability.append(p.item())
                    paths.append(x_history[:(i+1)].detach().numpy())
                    # noise_pos_trajectory.append((delta_prime_history[:i].detach(), x_history[:i+1].detach()))

                    break
                if torch.norm(x-A)<delta:
                    break
                if np.abs(x[0])>=1.5 or np.abs(x[1])>=1.5:
                    break

            if not success:
                self.total_probability.append(0)

        with open(os.path.join(self.save_path, "testResults.txt"), "a") as file:
            file.write(f"Two channels: {up}, {down}\n")
        # torch.save(noise_pos_trajectory, os.path.join(self.save_path, "noise_x_"))

        final_p = self.plot_paths_compute_stats(paths, N*self.size_batch)

        return final_p

    # plot G with respect to x, y
    def save_G_plot(self, PATH, name=None):
        x = np.arange(-1.5,1.5,0.1)
        y = np.arange(-1.5,1.5,0.1)

        pts = np.loadtxt('FEM_solution/Lucy_pts.csv', delimiter=',', dtype=np.float32)
        tri = np.loadtxt('FEM_solution/Lucy_tri.csv', delimiter=',', dtype=int)
        bias_pot = np.load(f"FEM_solution/bias_potential_{temp}.npy")
        log_q = np.zeros(pts.shape[0])
        total_q = np.zeros(pts.shape[0])
        q_diff = np.zeros(pts.shape[0])

        for i in range(pts.shape[0]):
            pos_time = torch.from_numpy(pts[i])
            potential = self.G(pos_time.unsqueeze(0)).detach().cpu().numpy()
            potential_dff = abs(potential-bias_pot[i])
            log_q[i] = potential
            total_q[i] = potential+energy_wei(pts[i])
            q_diff[i]=potential_dff

        if np.any(np.isnan(log_q)):
            print(log_q[np.isnan(log_q)],"NAN!!!")
            return 

        plt.tricontourf(pts[:,0], pts[:,1],tri,log_q,levels=20)
        plt.title("Bias Potential function")
        plt.colorbar(orientation="vertical")
        axes=plt.gca()
        axes.set_aspect(1)
        plt.xlabel("x")
        plt.ylabel("y")
        if name:
            plt.savefig(os.path.join(PATH, name))
        else:
            plt.savefig(os.path.join(PATH, f'bias_potential.png'))
        plt.close()

        # plt.tricontourf(pts[:,0], pts[:,1],tri,q_diff, levels = 20)
        # plt.title("Bias Potential function difference")
        # plt.colorbar(orientation="vertical")
        # axes=plt.gca()
        # axes.set_aspect(1)
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.savefig(os.path.join(PATH, f'bias_potential_diff_abs.png'))
        # plt.close()

        # plt.tricontourf(pts[:,0], pts[:,1],tri,total_q)
        # plt.title("Total Potential function")
        # plt.colorbar(orientation="vertical")
        # axes=plt.gca()
        # axes.set_aspect(1)
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.savefig(os.path.join(PATH, f'total_potential.png'))
        # plt.close()

    @torch.no_grad()
    def save_G_plot_pts(self, PATH):
        x = np.arange(-1.5,1.5,0.05)
        y = np.arange(-1.5,1.5,0.05)

        X,Y = np.meshgrid(x, y) # grid of point
        XY = np.stack([X,Y]).reshape(2,-1).transpose()
        # p = np.zeros((XY.shape[0]))
         
        p = self.G(torch.from_numpy(XY).float())

        p-=torch.min(p)
        p = p.reshape(X.shape).detach().numpy()

        plt.pcolormesh(X,Y,p, shading="auto")
        plt.colorbar()
        plt.savefig(os.path.join(PATH, f'bias_potential.png')) 
        plt.close()

    @torch.no_grad()
    def plot_paths_compute_stats(self, paths, N, sim_iteration=None, compute_stats = True):
        final_p = 0
        self.total_probability = np.array(self.total_probability)
        if compute_stats:
            np.save(os.path.join(self.save_path, "likelihood_ratios"),self.total_probability)

            final_p, std = self.total_probability.mean(), self.total_probability.std()


            lower_interval, upper_interval = final_p-1.96*std/np.sqrt(N), final_p+1.96*std/np.sqrt(N)
            ess = np.sum(self.total_probability)**2/np.sum(np.square(self.total_probability)) 
            ess_ratio = ess/N

            with open(os.path.join(self.save_path, "testResults.txt"), "a") as file:
                file.write(f"step size: {dt}, time deadline: {T_deadline}\n")
                file.write(f"Empirical mean: {final_p}\n")
                file.write(f"Std of the empirical mean: {std/np.sqrt(N)}\n")
                file.write(f"Coefficient of Variation: {std/final_p}\n")
                file.write(f"confidence interval:({lower_interval},{upper_interval})\n")
                file.write(f"Efficient sample size (ESS): {ess}\n")
                file.write(f"ESS ratio: {ess_ratio}\n")

        compare_p = self.total_probability.sum()/N
        for p, path in zip(self.total_probability, paths):    
            if p>0:
                c = min(1.0, p/compare_p)
                plt.plot(np.array(path)[:,0], np.array(path)[:,1], linewidth=0.5,alpha=c)
            # else:
            #     plt.plot(np.array(path)[:,0], np.array(path)[:,1], linewidth=0.5,alpha=1)
                   
        plot_levelset()

        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim([-1.5,1.5])
        plt.ylim([-1.5,1.5])
        plt.title("Trajectories")
        if sim_iteration!=None: plt.savefig(os.path.join(self.save_path,f"traj_{sim_iteration}.png"))
        else: plt.savefig(os.path.join(self.save_path,"traj.png"))
        plt.close()
        # self.save_G_plot(self.save_path)
        return final_p