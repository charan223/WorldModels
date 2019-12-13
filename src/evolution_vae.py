import gym
import numpy as np
import controller
import cma

import sys
sys.path.insert(1, '/users/alberto/projects/WorldModels/models')
import convVAE
#This works with just VAE


##PENDING
#Size parameters W and bias (0 to 1)?

#Using this, as recommended by paper
#http://blog.otoro.net/2017/11/12/evolving-stable-strategies/

network = convVAE.ConvVAE()

def rollout(k, env):
    #takes in the controller instance k
    
    obs = env.reset()
    #VAE should encode obs
    done = False
    total_reward = 0
    while not done:
        
        #feed z not obs into k.observe
        #z = network.decode(obs)
        k.observe(obs) 
        a = k.action()
        obs, reward, done = env.step(a)
        total_reward += reward
        
    return total_reward


env = gym.make('CarRacing-v0')

solver = cma.CMAEvolutionStrategy(36* [0.5], 0.2) #876 total parameters

generation = 0
while True:
    
    solutions = solver.ask()

    #one generation #need to do for many
    generation += 1
    fitness_list = np.zeros(len(solutions))
    
    for i in range(0, len(solutions)): #loop for each of the solutions provided
    
        #Could pack this into one function - eg: controller set..
        #need to adjust to our parameters (simple)
        w = solutions[i][:33].reshape(3,11) #Check the slice has the correct
        b = solutions[i][33:].reshape(3,)
        k = controller.Controller_VAE()
        k.set_parameters(w,b)
    
        #simulate agent in environment
        total_roll = 0
        for i in range(0,17): #Hardcoded, 17 rollouts #This could be parallelised
    
            #This will not run, outputs will not match
            total_roll += rollout(k, env) #returns cumulative score each run
        
        average_roll = total_roll/16
        fitness_list[i] = average_roll

        #add function here to monitor every so often state of evo algoirthm
        
    solver.tell(fitness_list)
    solver.logger.add()
    solver.disp()
    #Solver know
    bestsol, bestfit = solver.result()

    if generation == 1500:
        #exit while loop
        #put condition appropiate
        break

final_solutions = bestsol, bestfit #not sure if this assignment is allowed
with open('evo_results.json', 'w') as f:
    json.dump(final_solutions, f)

#f = open('evo_results.json')
#final_solutions = json.load(f)
    
solver.results.pretty()
cma.plot() 