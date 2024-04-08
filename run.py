import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from time import time
from torch import nn, optim
import argparse
import random
import os
from config import *
from utils import *
from modules import *
from train import *
from pathlib import Path
from datetime import datetime
from scipy import stats
import time

torch.set_default_tensor_type(torch.FloatTensor)
# number of simulations to train with

def main():
    print(f"batch size: {batch_size}")
    # Setting random seed for reproducibility
    st = time.time()
    random_seed = np.random.randint(0, 100)
    print("random seed", random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Getting current time for creating a plot directory
    time_now = datetime.now()
    current_time = time_now.strftime("%H_%M_%S")+"_"+str(random_seed)+"_train"
    plot_dir = Path(os.path.join(f"plots_{temp}/", current_time))
    plot_dir.mkdir(exist_ok=True)
    print(f"saving results to {plot_dir}")

    # Creating an instance of RareEventNNSampler with specified conditions
    main_sampler = RareEventNNSampler([4,8,4,2], plot_dir, batch_size, lr_rate)
    main_sampler.KL_loss_weight = KL_weight


    # Training the sampler
    # main_sampler.learn_existing_paths(N, [path1, path2])

    # # Uncomment the lines below if you want to load a pre-trained model
    print(f"load model from {model_path}")
    # main_sampler.load_model(model_path)
    # main_sampler.train(N, continue_training=True)
    # main_sampler.train_learn_ground_truth_directly(10)
    # main_sampler.initialize_move_right()
    # main_sampler.train(N)
    main_sampler.combine_two_learned_models(model1, model2)
    # Testing the trained sampler
    et = time.time()
    training_time = et - st
    p = main_sampler.test(test_N)
    # p = main_sampler.test_ground_truth(1000)

    # Saving the generated plot of G
    et2 = time.time()

    # get the execution time
    testing_time = et2 - et

    with open(os.path.join(main_sampler.save_path, "testResults.txt"), "a") as file:
        file.write(f"Training time: {training_time}s, testing time: {testing_time}s.\n")
        file.write(f"Trained for {N} steps, batch size {batch_size}, Test for {test_N} steps.\n")
        file.write(f"learning rate: {lr_rate}, KL_weight: {KL_weight}\n")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Process some integers.')
    
    # # Define the batch_size argument
    # parser.add_argument('--batch_size', type=int, default=256, help='an integer for the batch size')

    # # Parse arguments
    # args = parser.parse_args()

    # Call the main function with the parsed arguments
    main()
