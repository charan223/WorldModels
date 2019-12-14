import gym
import numpy as np
import controller
import cma
import torch
import cv2
#import json
import pickle
import matplotlib

import sys
sys.path.insert(1, '/users/alberto/projects/WorldModels/models')
import convVAE

#This works with just VAE
#to run (from outside src for paths to work)
#python3 src/evolution_vae.py

##PENDING
#Size parameters W and bias (0 to 1)?

#Using this, as recommended by paper
#http://blog.otoro.net/2017/11/12/evolving-stable-strategies/

#issue: solver.looger.plot() is not plotting

#Parameters experiment:
POPULATION_SIZE = 24
NUMBER_ROLLS = 5
GENERATION_LIMIT = 32
SCORE_LIMIT = 100
MAX_STEPS = 200 #each run should actually has 1000 steps, but this can give us time


vae = convVAE.ConvVAE()

def rollout(k, env):
    # k is a controller instance
    # env is the car racing environment
    
    obs = env.reset()
    #Is there another way to run the experiment without initialising
    #env.render() #for visualization, does not work well on my laptop
    
    #I am considering putting our own counter to easily change the length of rollouts (reduce them)
    #I counted 1000 steps, I reduced it to 100
    step_counter = 0
    
    done = False
    total_reward = 0
    
    #while not done:
    while step_counter < MAX_STEPS:
        step_counter += 1
        
        #obs_tensor = torch.from_numpy(obs.copy())
        batch = np.array([cv2.resize(obs, (64, 64))]).astype(np.float)/255.0
        batch = torch.from_numpy(batch).permute(0,3,1,2).float()
        z, mu, logvar = vae.encode(batch)#Take first argument
        #z_vector = z.numpy()
        z_vector = z.detach().numpy()
        
        k.observe(z_vector) 
        a = k.action()
        obs, reward, done, other = env.step(a)
        total_reward += reward
        
    return total_reward


env = gym.make('CarRacing-v0')
#env.reset() #adding these two to test behaviour of background visualization
#env.render() #could it speed up?

solver = cma.CMAEvolutionStrategy(99* [0.5], 0.2, {'popsize': POPULATION_SIZE,}) #876 total parameters (#99 in VAE model)
best_par_score = [] #list with best parameters and scores each round (solver format)
best_par_score2 = [] #(my format)
generation = 0

while True:
    generation += 1
    print('Generation: ', generation)
    
    solutions = solver.ask()
    #fitness_list = np.zeros(len(solutions)) #I think it is accepted
    fitness_list = []
    
    for i in range(0, len(solutions)): 
    #loop for each of the solutions provided , which is
    #determined by popsize in solver
        print('solution (instance): ', i)
    
        #Could pack this into one function - eg: controller set..
        w = solutions[i][:96].reshape(3,32) #arrange phenotype
        b = solutions[i][96:].reshape(3,)
        k = controller.Controller_VAE() #instantiate and assign parameters
        k.set_parameters(w,b)
    
        #simulate agent in environment
        total_roll = 0
        for j in range(0, NUMBER_ROLLS): #Hardcoded, 17 rollouts #This could be parallelised
            print('rollout: ', j)
            #This will not run, outputs will not match
            total_roll += rollout(k, env) #returns cumulative score each run
        
        average_roll = total_roll/(NUMBER_ROLLS)
        #fitness_list[i] = average_roll
        fitness_list.append(average_roll) #They should be appended in right order

        #add function here to monitor every so often state of evo algoirthm
        print('score: ', average_roll)
        
    solver.tell(solutions, fitness_list)
    solver.logger.add()
    solver.disp()
    
    #Solver know
    #bestsol, bestfit = solver.result()
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

#with open('evo_results.json', 'w') as f: #Seems not to be working
#    json.dump(final_solutions, f)
##f = open('evo_results.json')
##final_solutions = json.load(f)

with open('evo_results.pkl', 'wb') as f: #save in current folder
    pickle.dump(final_solutions, f)
#with open('evo_results.pkl', ‘rb’) as f:
#    stored_data = pickle.load(f)

#env.render() #would this allow log to plot?
#env.close() 

print('end')   
solver.result_pretty()
solver.logger.plot()
#scma.plot()
#solver.plot() 