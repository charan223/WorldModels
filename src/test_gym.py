#testing gym environment

#import controller

import gym
import numpy as np


import pickle
import argparse
from os.path import join, exists
from models.convVAE import ConvVAE
from models.lstm_mdn import LSTM_MDN
import torch
import torch.nn as nn
import multiprocessing
import matplotlib.pyplot as plt
from utils.train_utils import load_model
import sys

import controller

#not sure how to use this functionality: #not that it is needed
#from gym.envs import box2d
#box2d.AntEnv


LATENT_SIZE = 256
HIDDEN_SIZE = 32
ACTION_SIZE = 3
ONLY_VAE = True


class Controller(nn.Module):
    """ Controller """
    def __init__(self, latent_size, hidden_size, action_size, only_vae):
        super().__init__()
        if only_vae:
            self.fc = nn.Linear(latent_size, action_size)
        else:
            self.fc = nn.Linear(latent_size + hidden_size, action_size)
           
    def forward(self, *inputs):
        cat_in = torch.cat(inputs, dim=0)
        fc1 = self.fc(cat_in)
        th1 = torch.tanh(fc1)
        th1[1] = (th1[1] + 1)/2
        th1[2] = (th1[2] + 1)/2
        return th1
        
        
def unflatten_parameters(params, example, device):
    """ Unflatten parameters.
    :args params: parameters as a single 1D np array
    :args example: generator of parameters (as returned by module.parameters()),
        used to reshape params
    :args device: where to store unflattened parameters
    :returns: unflattened parameters
    """
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened
    

def load_parameters(params, controller):
    """ Load flattened parameters into controller.
    :args params: parameters as a single 1D np array
    :args controller: module in which params is loaded
    """
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)


with open('evo_vae_28_pop_size_800_length_4_avg_rollout.pkl', 'rb') as f:
    info_loaded = pickle.load(f)
    
solver_state = info_loaded[0]
best_param = solver_state.result[0]




controller_test = Controller(LATENT_SIZE, HIDDEN_SIZE, ACTION_SIZE, ONLY_VAE)
#controller_test.load_state_dict(controllers[0].state_dict())

load_parameters(best_param, controller_test)


device = torch.device("cpu")
vae_file = checkpoints/random/model_7.pth
vae = ConvVAE()
vae.load_state_dict(torch.load(vae_file, map_location=device))

if not ONLY_VAE:
    lstm_model_path = "src/saved_models/lstm/49500/1576236505.pth.tar"
    lstm_mdn = LSTM_MDN()
    load_model(lstm_model_path, lstm_mdn)


#env = gym.make('MountainCar-v0')
env = gym.make('CarRacing-v0')
env.reset()


counter = 0

#s = controller.Controller #Will not work because I do not have inputs.
#s.action_rand()
#s.action(z,h)

for _ in range(1000):
    
    env.render()
    
    batch = np.array([cv2.resize(obs, (64, 64))]).astype(np.float)/255.0
    batch = torch.from_numpy(batch).permute(0,3,1,2).float()
    z, _, _ = vae.encode(batch)#Take first argument
    z_vector = z.detach()
    if not ONLY_VAE:
        lstm_input = torch.cat((z, torch.tensor(a, dtype=torch.float)))
        _ = lstm_mdn(lstm_input.view(1, 1, 35))
        _, hidden = lstm_mdn.hidden
        a = controller_test(z_vector, torch.squeeze(hidden[0]))
    else:
        a = controller_test(z_vector)
    
    obs, reward, done, _ = envir.step(a.detach().numpy())
    
    
    #t = controller.Controller_Simple.action() #(not initialising this class..)
    #c = env.action_space.sample()
    #a = env.step(t) # take a random action use r if wanting to sample from controller
    #counter +=1 #I do not want to change their for loop for now
    #if counter % 5 == 1: print(a[1:], '\n', r,'\n', c,'\n', t) #not printing the observation
    
env.close()

print(len(a))
print(type(a))
print(type(a[0]))
print(a[0].shape)