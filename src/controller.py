import numpy as np


#test class, ignore
class Controller_Simple:
    def action():
        r = np.random.rand(3,)
        r[0] = (r[0]  * 2) - 1
        return r

        
def clip(value, y_max, y_min):
    if value > y_max: value = y_max
    if value < y_min: value = y_min
    return value

class Controller:
    #create method to reset/better preset values for parameters/inputs?
    #later on, functionality with only VAE (or different controller)
    
    def __init__(self):
        self.visual = np.zeros((32,)) #for now I am fising sizes #would ones be more useful?
        self.hidden = np.zeros((256,))
        
        self.weights = np.zeros((3,288))
        self.bias = np.zeros((3,))

    def randomize_parameters(self):
        #randomize weights and bias (between 0 and 1)
        self.weights = np.random.rand(3,288)
        self.bias = np.random.rand(3,)
        
    def randomize_input(self):
        #randomize visual, hidden
        pass
        
    def observe (self, z, h):
        #update hidden and visual
        #anything can go in here so not great
        self.visual = z
        self.hidden = h
    
    def set_parameters (self, w, b):
        self.weights = w
        self.bias = b
        
    def action (self):
        #apply function to keep within boundaries?
        conc = np.concatenate((self.visual, self.hidden))
        action = np.dot(self.weights, conc ) + self.bias
        
        action[0] = clip(action[0], 1, -1)
        action[1] = clip(action[1], 1, 0)
        action[2] = clip(action[2], 2, 0)
        
        return action
    
    def action_rand(self):
        r = np.random.rand(3,)
        r[0] = (r[0] * 2) - 1
        return r
    
    def return_parameters(self):
        return self.weights, self.bias


#Just VAE network
class Controller_VAE:
    #create method to reset/better preset values for parameters/inputs?
    #later on, functionality with only VAE (or different controller)
    
    def __init__(self):
        self.visual = np.zeros((32,)) #for now I am fising sizes #would ones be more useful?
        self.padding = np.zeros((1,))
        
        self.weights = np.zeros((3,11))
        self.bias = np.zeros((3,))

    def randomize_parameters(self):
        #randomize weights and bias (between 0 and 1)
        self.weights = np.random.rand(3,11)
        self.bias = np.random.rand(3,)
        
    def randomize_input(self):
        #randomize visual, hidden
        pass
        
    def observe (self, z):
        #update hidden and visual
        #anything can go in here so not great
        self.visual = z
    
    def set_parameters (self, w, b):
        self.weights = w
        self.bias = b
        
    def action (self):
        #apply function to keep within boundaries?
        conc = np.concatenate((self.visual, self.padding))
        action = np.dot(self.weights, conc ) + self.bias
        
        action[0] = clip(action[0], 1, -1)
        action[1] = clip(action[1], 1, 0)
        action[2] = clip(action[2], 2, 0)
        
        return action
    
    def action_rand(self):
        r = np.random.rand(3,)
        r[0] = (r[0] * 2) - 1
        return r
    
    def return_parameters(self):
        return self.weights, self.bias
        

k = Controller()
k.randomize_parameters()
k.randomize_input()

#k.observe(input_VAE, input_MDN)

b = k.action()
print(b.shape)
print(b)

s = k.return_parameters()
#print(s)
    
#Controller_Simple.action()