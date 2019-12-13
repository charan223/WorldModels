import gym
import numpy as np
import controller

#This works with just VAE


##PENDING
#Size parameters W and bias (0 to 1)?
#


#Using this, as recommended by paper
#http://blog.otoro.net/2017/11/12/evolving-stable-strategies/

def rollout(agent, env):
    obs = env.reset()
    #VAE should encode obs
    done = False
    total_reward = 0
    while not done:
        
        #actually feed into controller VAE output
        a = controller.action(obs)
        obs, reward, done = env.step(a)
        total_reward += reward
        
    return total_reward


env = gym.make('CarRacing-v0')

solver = cma.CMAEvolutionStrategy(36* [0.5], 0.2) #876 total parameters


while True:
    
    solutions = solver.ask()

    #one generation #need to do for many
    generation += 1
    fitness_list = np.zeros(len(solutions))

    for i in range(0, len(solutions)): #loop for each of the solutions provided
    
        #Could pack this into one function - eg: controller set..
        #need to adjust to our parameters (simple)
        w = solutions[i[:34]].reshape(3,11) #Check the slice has the correct
        b = solutions[i[34:]].reshape(3,)
        k = controller.Controller()
        k.set_parameters(w,b)
    
        #simulate agent in environment
        for i in range(0,17): #Hardcoded, 17 rollouts #This could be parallelised
    
            #This will not run, outputs will not match
            total_roll += rollout(controller, env) #returns cumulative score each run
        
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