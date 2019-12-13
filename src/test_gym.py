#testing gym environment

#import controller

import gym
import numpy as np

import controller

#not sure how to use this functionality: #not that it is needed
#from gym.envs import box2d
#box2d.AntEnv

#env = gym.make('MountainCar-v0')
env = gym.make('CarRacing-v0')


#commented output on CarRacing-v0
print(env.action_space)
#> Box(3,) 
print(env.observation_space)
#> Box(96, 96, 3)

print(env.action_space.high)
#> [1. 1. 1.]
print(env.action_space.low)
#> [-1.  0.  0.]


env.reset()

counter = 0

s = controller.Controller #Will not work because I do not have inputs.
#s.action_rand()
#s.action(z,h)

for _ in range(100):
    env.render()
    
    r = np.random.rand(3,)
    r[0] = (r[0]  * 2) - 1
    
    t = controller.Controller_Simple.action() #(not initialising this class..)
    
    c = env.action_space.sample()
    a = env.step(t) # take a random action use r if wanting to sample from controller
    
    counter +=1 #I do not want to change their for loop for now
    if counter % 5 == 1: print(a[1:], '\n', r,'\n', c,'\n', t) #not printing the observation
    
env.close()

print(len(a))
print(type(a))
print(type(a[0]))
print(a[0].shape)