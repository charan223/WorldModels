import gym
import numpy as np
#import controller
import cma
import torch
import cv2
#import json
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

parser = argparse.ArgumentParser(description='Controller for WorldModels')
parser.add_argument('--seed', type=int, default=123, metavar='N',
                    help='seed value')

parser.add_argument('--pop_size', type=int, default=32, metavar='N',
                    help='population size')
parser.add_argument('--num_rolls', type=int, default=4, metavar='N',
                    help='number of rolls')
parser.add_argument('--gen_limit', type=int, default=50, metavar='N',
                    help='generation limit')
parser.add_argument('--score_limit', type=int, default=1500, metavar='N',
                    help='score limit')
parser.add_argument('--max_steps', type=int, default=1000, metavar='N',
                    help='max steps')
parser.add_argument('--processes', type=int, default=8, metavar='N',
                    help='number of parallel processes')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--vae_model_path', type=str, default="checkpoints",
                    help='folder of vae model')
parser.add_argument('--action_type', type=str, default="",
                    help='policy type random or continuous')
parser.add_argument('--vae_model_file', type=str, default="final.pth",
                    help='vae model filename')
parser.add_argument('--only_vae', action='store_true', default=True,
                    help='if not using rnn only_vae will be True')
parser.add_argument('--latent_size', type=int, default=32, metavar='N',
                    help='latent size')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size')
parser.add_argument('--action_size', type=int, default=3, metavar='N',
                    help='action size')
parser.add_argument('--pooling', action='store_true', default=True,
                    help='true or false')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

""" Define controller """
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

'''
#Parameters experiment:
POPULATION_SIZE = 24 #paper 64
NUMBER_ROLLS = 6 #paper 16
GENERATION_LIMIT = 30 #Limit to number generations used (paper says 1800 needed)
SCORE_LIMIT = 200 #score we want it to reach before ending (900 is what we should aim for)
MAX_STEPS = 600 #each run should actually has 1000 steps, but this can give us time
'''

vae_file = join(args.vae_model_path, args.action_type, args.vae_model_file)
vae = ConvVAE()
vae.load_state_dict(torch.load(vae_file, map_location=device))

controllers = [Controller(args.latent_size, args.hidden_size, args.action_size, args.only_vae) for i in range(args.pop_size)]
for controller in controllers:
    controller.load_state_dict(controllers[0].state_dict())

if not args.only_vae:
    lstm_model_path = "src/saved_models/lstm/49500/1576236505.pth.tar"
    lstm_mdns = [LSTM_MDN(seq_size=1) for i in range(args.pop_size)]
    for lstm_mdn in lstm_mdns:
        load_model(lstm_model_path, lstm_mdn)


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

def score_aggregator(fitness_list):
    average_value = sum(fitness_list)/len(fitness_list)
    best_value = min(fitness_list)
    worst_value = max(fitness_list)
    #score_agg.extend([])
    score_agg =[best_value, average_value, worst_value]
    return score_agg
    
def save_current_state(solver, logger, solutions, fitness_list, list_points, generation):
    file_to_store = (solver, logger, (solutions, fitness_list, list_points, generation))
    with open('evo_vae_{0}_pop_size_{1}_length_{2}_avg_rollout.pkl'.format(args.pop_size, args.max_steps, args.num_rolls), 'wb') as f:
        pickle.dump(file_to_store, f)


#env = gym.make('CarRacing-v0')
#x = env.reset() #adding these two to test behaviour of background visualization

#screen = env.render(mode='rgb_array') #could it speed up?
#plt.imshow(screen)
#plt.show()

parameters = controllers[0].parameters()

solver = cma.CMAEvolutionStrategy(torch.cat([p.detach().view(-1) for p in parameters], dim=0).cpu().numpy(),
        0.1, {'popsize': args.pop_size}) #876 total parameters (#99 in VAE model)
        
logger_res = cma.CMADataLogger().register(solver) 
score_point_gen = []

best_par_score = [] #list with best parameters and scores each round (solver format)
best_par_score2 = [] #(my format)
generation = 0



def rollout(params, controller):
    # k is a controller instance
    # env is the car racing environment
    
    envir = gym.make('CarRacing-v0')
    obs = envir.reset()
    #Is there another way to run the experiment without initialising
    #env.render() #for visualization, does not work well on my laptop
    
    #I am considering putting our own counter to easily change the length of rollouts (reduce them)
    #I counted 1000 steps, I reduced it to 100
    step_counter = 0
    
    done = False
    total_reward = 0
    
    #load params into controller
    if params is not None:
        load_parameters(params, controller)


    #while not done:
    while step_counter < args.max_steps:
        step_counter += 1
        batch = np.array([cv2.resize(obs, (64, 64))]).astype(np.float)/255.0
        batch = torch.from_numpy(batch).permute(0,3,1,2).float()
        z, _, _ = vae.encode(batch)#Take first argument
        z_vector = z.detach()
        a = controller(z_vector)
        obs, reward, done, _ = envir.step(a.detach().numpy())
        total_reward += reward
        if done == True: break #print early break
    return total_reward



def multi_run_wrapper(args):
   return rollout_pooling(*args)

def rollout_pooling(s_id, params, controller, lstm_mdn=None, nb_frames=1, frame_gap=5):
    total_roll = 0
    for i in range(args.num_rolls):
        # k is a controller instance
        # env is the car racing environment

        envir = gym.make('CarRacing-v0')
        obs = envir.reset()
        batch = [(cv2.resize(obs, (64, 64))).astype(np.float)/255.0 for _ in range((nb_frames - 1) * frame_gap + 1)]
        #Is there another way to run the experiment without initialising
        #env.render() #for visualization, does not work well on my laptop
        #screen = env.render(mode='rgb_array') #could it speed up?
        #plt.imshow(screen)
        #plt.show()
        #I am considering putting our own counter to easily change the length of rollouts (reduce them)
        #I counted 1000 steps, I reduced it to 100
        step_counter = 0
        
        done = False
        total_reward = 0
        
        #load params into controller
        if params is not None:
            load_parameters(params, controller)

        a = torch.zeros(3,)
        #while not done:
        while step_counter < args.max_steps:
            step_counter += 1
            batch.append((cv2.resize(obs, (64, 64))).astype(np.float)/255.0)
            batch.pop(0)
            input_frms = [batch[i] for i in range((nb_frames - 1) * frame_gap, -1, - frame_gap)]
            vae_input = np.array([np.concatenate(input_frms, axis=2)])
            vae_input = torch.from_numpy(vae_input).permute(0,3,1,2).float()
            _, mu, _ = vae.encode(vae_input)#Take first argument
            mu_vector = mu.detach()
            if not args.only_vae:
                lstm_input = torch.cat((mu_vector, a.clone().detach()))
                _ = lstm_mdn(lstm_input.view(1, 1, 35))
                _, hidden = lstm_mdn.hidden
                a = controller(mu_vector, torch.squeeze(hidden[0]))
            else:
                a = controller(mu_vector)
            obs, reward, done, _ = envir.step(a.detach().numpy())
            total_reward += reward
            if done == True: break #print early break
        envir.close()
        total_roll += total_reward
        envir.close()
    average_roll = -total_roll/(args.num_rolls)
    print("Average roll in id {} is {}".format(s_id, average_roll))
    return s_id, average_roll

if __name__ == '__main__':

    multiprocessing.set_start_method('spawn')

    sys.stdout = open('logs_vae_trained_lstm_28_4_800', 'w')

    while True:
        generation += 1
        print('Generation: ', generation)
        sys.stdout.flush()
        solutions = solver.ask()

        pool = multiprocessing.Pool(processes=args.processes)
        if args.pooling:
            fitness_list = np.zeros(len(solutions))
            pool_inputs = []
            for s_id, params in enumerate(solutions):
                if args.only_vae:
                    pool_inputs.append((s_id, params, controllers[s_id]))
                else:
                    pool_inputs.append((s_id, params, controllers[s_id], lstm_mdns[s_id]))
            pool_outputs = pool.map(multi_run_wrapper, pool_inputs)
            pool.close()
            pool.join()

            for s_id, average_roll in pool_outputs:
                fitness_list[s_id] = average_roll
        else:
            fitness_list = []
            for i in range(0, len(solutions)):
            #loop for each of the solutions provided , which is
            #determined by popsize in solver
                print('solution (instance): ', i)
                sys.stdout.flush()
                #simulate agent in environment
                total_roll = 0
                for j in range(0, args.num_rolls):  #This could be parallelised (each instance runs its own)
                    print('    G: ', generation, 'rollout: ', j) #indent inwards?
                    sys.stdout.flush()
                    total_roll += rollout(solutions[i], controller[i]) #returns cumulative score each run
            
                average_roll = -total_roll/(args.num_rolls)
                #fitness_list[i] = average_roll
                fitness_list.append(average_roll) #They should be appended in right order

                #add function here to monitor every so often state of evo algoirthm
                print('score: ', average_roll)
                sys.stdout.flush()

        solver.tell(solutions, fitness_list)
        solver.logger.add()
        solver.disp()
    
        max_avg_min = score_aggregator(fitness_list)
        score_point_gen.append(max_avg_min)
        save_current_state(solver, logger_res, solutions, fitness_list, score_point_gen, generation)
    
        #Solver know
        #bestsol, bestfit = solver.result()
        best = solver.result
    
        #my own save
        min_index, min_value = np.argmin(fitness_list), np.min(fitness_list)
        best2 = min_value, solutions[min_index]
    
        best_par_score.append(best)
        best_par_score2.append(best2)
        print('Best obtained: ', min_value)
        sys.stdout.flush()
        if generation == args.gen_limit or -min_value > args.score_limit:
            #exit while loop
            #put condition appropiate
            break

    final_solutions = best, best2 #not sure if this assignment is allowed

    #with open('evo_results.json', 'w') as f: #Seems not to be working
    #    json.dump(final_solutions, f)
    ##f = open('evo_results.json')
    ##final_solutions = json.load(f)

    with open('endres_28_4_800_trained.pkl', 'wb') as f: #save in current folder
        pickle.dump(final_solutions, f)


    print('end')

    sys.stdout.flush()
    solver.result_pretty()
    solver.logger.plot()
    #scma.plot()
    #solver.plot()