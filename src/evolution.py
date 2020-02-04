import gym
import numpy as np
import controller
import cma
import torch
import cv2
#import json
import pickle
#import matplotlib #not necessary, but how do I get logger to plot

import sys
#Change this path to your computer
sys.path.insert(1, '/users/alberto/projects/WorldModels/models')
import convVAE
import lstm_mdn


#Full controller training scripts
#to run (from outside src for paths to work) #actually not sure it is thhe case anymore
#python3 src/evolution.py

##PENDING
#Size parameters W and bias (0 to 1)?

#To DO
#better saving format for data
#FIX PLOTS

#issue
#solver.looger.plot() is not plotting


#Parameters experiment:
POPULATION_SIZE = 24
NUMBER_ROLLS = 5
GENERATION_LIMIT = 32
SCORE_LIMIT = 100
MAX_STEPS = 200 #each run should actually has 1000 steps, but this can give us time

vae = convVAE.ConvVAE()
lstm = lstm_mdn.LSTM_MDN()


#line 37 train_lstm?
#mdn = need
from datasets.generate_lstm_training import LSTMDataset #line 63 train_lstm
sys.path.insert(1, '/users/alberto/projects/WorldModels/utils')
import train_utils
train_utils.load_model()



#Using this, as recommended by paper
#http://blog.otoro.net/2017/11/12/evolving-stable-strategies/
def rollout(k, env):
    # k is a controller instance
    # env is the car racing environment
    
    obs = env.reset()

    done = False
    total_reward = 0
    
    #while not done:
    while step_counter < MAX_STEPS:
        step_counter += 1
        
        batch = np.array([cv2.resize(obs, (64, 64))]).astype(np.float)/255.0
        batch = torch.from_numpy(batch).permute(0,3,1,2).float()
        z, mu, logvar = vae.encode(batch) #gets z vector
        z_vector = z.detach().numpy()
        
        #h vector processing
        #need to adapt to my code
        inp = torch.cat((x['encoded'], x['actions']), dim=1) #line 30 train_lstm
        inp = inp.view(-1, lstm.seq_size, lstm.z_dim + 3)
        i, sigma, mu = lstm(inp)
        h = lstm.hidden[0]
        #processing required?
        
        #also require h vector
        k.observe(z_vector,h_vector)
        a = k.action()
        obs, reward, done, other = env.step(a)
        total_reward += reward
        
    return total_reward


env = gym.make('CarRacing-v0')

solver = cma.CMAEvolutionStrategy(876* [0.5], 0.2, {'popsize': POPULATION_SIZE,}) #876 total parameters
best_par_score = [] #list with best parameters and scores each round (solver format)
best_par_score2 = [] #(my format)
generation = 0

while True:
    generation += 1
    print('Generation: ', generation)
    
    solutions = solver.ask()
    #fitness_list = np.zeros(len(solutions)) #I think it is accepted
    fitness_list = []
    

    for i in range(0, len(solutions)): #loop for each of the solutions provided
        print('solution: ', i)
    
        #Could pack this into one function - eg: controller set..
        #need to adjust to our parameters (simple)
        w = solutions[i[:874]].reshape(3,288) #Check the slice has the correct
        b = solutions[i[874:]].reshape(3,)
        k = controller.Controller()
        k.set_parameters(w,b)
    
        #simulate agent in environment
        total_roll = 0
        for j in range(0, NUMBER_ROLLS):  #This could be parallelised (each instance runs its own)
    
            #This will not run, outputs will not match
            print('rollout: ', j)
            total_roll += rollout(k, env) #returns cumulative score each run
        
        average_roll = total_roll/(NUMBER_ROLLS)
        fitness_list.append(average_roll) #They should be appended in right order

        #add function here to monitor every so often state of evo algoirthm
        print('score: ', average_roll)
        
    solver.tell(fitness_list) #I think input is a list #Might need to check
    solver.tell(solutions, fitness_list)
    
    #solver save
    best = solver.result

    #my own save
    max_value = max(fitness_list)
    max_index = fitness_list.index(max_value)
    best2 = max_value, solutions[max_index]
    
    best_par_score.append(best)
    best_par_score2.append(best2)
    print('Best obtained: ', max_value)

    if generation == GENERATION_LIMIT or max_value > SCORE_LIMIT:
        #exit while loop
        #put condition appropiate
        break

final_solutions = best, best2 #not sure if this assignment is allowed
with open('evo_full_results.pkl', 'wb') as f: #save in current folder
    pickle.dump(final_solutions, f)

#f = open('evo_results.json')
#final_solutions = json.load(f)

print('end')   
solver.result_pretty()
solver.logger.plot()