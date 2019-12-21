from __future__ import print_function
import argparse
import os
from os.path import exists, join
import logging
import numpy as np
import cv2
import torch
from torch.utils import data
from torch import nn, optim
from torch.nn import functional as F
from datasets.dataloader import load_data
from models.convVAE import ConvVAE
from utils.train_utils import save_model, load_model

parser = argparse.ArgumentParser(description='ConvVAE for WorldModels')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')                    
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--N_z', type=int, default=32, metavar='N',
                    help='size of latent variable')
parser.add_argument('--split', type=float, default=0.9, metavar='N',
                    help='train size divided by dataset size(0.9 = 90/100)')
parser.add_argument('--rollouts', type=int, default=1000, metavar='N',
                    help='number of rollouts to be considered for training + testing')
parser.add_argument('--batch_rollout_size', type=int, default=100, metavar='N',
                    help='number of rollouts to be considered for each batch')
parser.add_argument('--model_file', type=str, default='final.pth', metavar='N',
                    help='final model for evaluation')
parser.add_argument('--model_path', type=str, default='checkpoints', metavar='N',
                    help='checkpoints paths')
parser.add_argument('--log_file', type=str, default='vae_train.log', metavar='N',
                    help='final model for evaluation')
parser.add_argument('--log_path', type=str, default='logs', metavar='N',
                    help='checkpoints paths')
parser.add_argument('--action_type', type=str, default='random', metavar='N',
                    help='random or continuous')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

logging.basicConfig(filename=join(args.log_path, args.action_type, args.log_file), level=logging.INFO)
logger = logging.getLogger('vae_train')

torch.manual_seed(args.seed)
# Fix numeric divergence due to bug in Cudnn (from ctallec repo)
#torch.backends.cudnn.benchmark = True

#for random flipping, shuffling
np.random.seed(100)


device = torch.device("cuda" if args.cuda else "cpu")

model = ConvVAE(N_z=args.N_z,
              batch_size=args.batch_size).to(device)

#Added scheduler to control learning rate(from ctallec repo)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def save_checkpoint(state, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

# L2 Loss + KL Loss as suggested in paper and are summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):

    L2_Loss = F.mse_loss(x, recon_x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KL_Loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return L2_Loss + KL_Loss

def train(epoch, data_folder, train_loss_arr, nb_frames=1, frame_gap=5):

    assert join(args.model_path, args.action_type)

    filename = join(args.model_path, args.action_type, args.model_file)
    if exists(filename):
        #TO DO: LOAD MODEL HERE and train code for left over rollouts
        #model.load_state_dict(torch.load(join(model_path, model_file)))
        state = torch.load(filename)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])

    model.train()
    train_loss = 0

    # if split is 0.9  train_rollouts = 900
    train_rollouts =  args.split * args.rollouts

    # fixed episode length for each rollout
    ep_length = 1000

    # load 100 rollouts at a time and shuffle among their 1000 * 100 frames
    batch_rollout_size = args.batch_rollout_size

    # num_rollout_batches = 9 
    num_rollout_batches = int(train_rollouts/batch_rollout_size)

    # num_batches = 1000 * 100/1
    num_batches = int(ep_length * batch_rollout_size/args.batch_size)
    
    for batch_rollout in range(num_rollout_batches):
            # obs has 1000 * 100 frames from 100 rollouts (taking only 100 rollouts(~ 2.7 GB) due to memory constraints)
            obs = load_data(data_folder, ep_length, batch_rollout_size, batch_rollout)

            # perform shuffling only in train
            if nb_frames == 1:
                np.random.shuffle(obs)
            for batch_idx in range(num_batches):
                if nb_frames != 1:
                    rollout_id = np.random.randint(int(batch_rollout_size))
                    frm_id = np.random.randint(ep_length - (nb_frames - 1) * frame_gap)
                    start_idx = rollout_id * ep_length + frm_id
                    frms = [obs[start_idx + i * frame_gap] for i in range(nb_frames)]
                    batch = np.array([np.concatenate(frms, axis=2)])
                else:
                    batch = obs[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]

                #add random flipping
                flip = np.random.choice([0,1], 1, p=[0.5,0.5])[0]                
                batch = [batch, np.flip(batch, axis = 2)][flip]

                # as the input pixels lie between 0, 1
                batch_obs = batch.astype(np.float)/255.0

                batch_obs = torch.from_numpy(batch_obs).permute(0,3,1,2).float()
                batch_obs = (batch_obs).to(device) 
                # for gradients to not accumulate
                optimizer.zero_grad()

                # forward pass
                recon_batch_obs, mu, logvar = model(batch_obs)

                #loss function
                loss = loss_function(recon_batch_obs, batch_obs, mu, logvar)

                # backpropagation
                loss.backward()

                train_loss += loss.item()

                # update parameters based on calculated gradients
                optimizer.step()

                if batch_idx % args.log_interval == 0:
                    logger.info('Train Epoch: {}, batch_rollout: [{}/{}] batch [{}/{}]\tLoss: {:.6f}'.format(
                        epoch, batch_rollout, num_rollout_batches, batch_idx * len(batch), len(obs),
                        loss.item() / len(batch)))
            
            # TO DO SAVE MODEL HERE
            #torch.save(model.state_dict(), join(model_path, model_file))
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'precision': test_loss,
                'optimizer': optimizer.state_dict()
            }, False, filename, None)

    train_loss = train_loss / (train_rollouts * len(obs))

    logger.info('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss ))
    train_loss_arr.append(train_loss)
    return train_loss_arr

def test(epoch, data_folder, test_loss_arr, nb_frames=1, frame_gap=5):

    assert join(args.model_path, args.action_type)

    filename = join(args.model_path, args.action_type, args.model_file)
    if exists(filename):
        #TO DO: LOAD MODEL HERE and train code for left over rollouts
        #model.load_state_dict(torch.load(join(model_path, model_file)))
        state = torch.load(filename)
        model.load_state_dict(state['state_dict'])

    model.eval()
    test_loss = 0

    # if split is 0.9  train_rollouts = 900
    train_rollouts =  args.split * args.rollouts

    # fixed episode length for each rollout
    ep_length = 1000

    # test rollouts = 100
    test_rollouts = args.rollouts - train_rollouts

    # load 100 rollouts at a time and shuffle among their 1000 * 100 frames
    batch_rollout_size = args.batch_rollout_size

    # num_rollout_batches = 1 
    num_rollout_batches = int(test_rollouts/batch_rollout_size)
    total_rollout_batches = int(args.rollouts/batch_rollout_size)

    # num_batches = 1000 * 100
    num_batches = int(ep_length * batch_rollout_size/args.batch_size)

    with torch.no_grad():
        for batch_rollout in range(total_rollout_batches - num_rollout_batches, total_rollout_batches):
                # obs has 1000 * 100 frames from 100 rollouts (taking only 100 rollouts(~ 2.7 GB) due to memory constraints)
                obs = load_data(data_folder, ep_length, batch_rollout_size, batch_rollout)
                
                if nb_frames == 1:
                    for batch_idx in range(num_batches):
                        batch = obs[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]

                        # as the input pixels lie between 0, 1
                        batch_obs = batch.astype(np.float)/255.0

                        batch_obs = torch.from_numpy(batch_obs).permute(0,3,1,2).float()
                        batch_obs = batch_obs.to(device)
                        batch_recon_obs, mu, logvar = model(batch_obs)

                        test_loss += loss_function(batch_recon_obs, batch_obs, mu, logvar).item()
                    
                        #if batch_idx < 3:
                        #    cv2.imwrite("results/" + str(batch_rollout) + "_" + str(batch_idx) + "_recon.jpg", np.squeeze(batch_recon_obs.permute(0,2,3,1).numpy())*255)
                        #    cv2.imwrite("results/" + str(batch_rollout) + "_" + str(batch_idx) + ".jpg", np.squeeze(batch_obs.permute(0,2,3,1).numpy())*255)

                else:
                    for rollout_id in batch_rollout_size:
                        for frm_id in range(ep_length - (nb_frames - 1) * frame_gap):
                            start_idx = rollout_id * ep_length + frm_id
                            frms = [obs[start_idx + i * frame_gap] for i in range(nb_frames)]
                            batch = np.array([np.concatenate(frms, axis=2)])
                            batch_obs = batch.astype(np.float) / 255.0

                            batch_obs = torch.from_numpy(batch_obs).permute(0, 3, 1, 2).float()
                            batch_obs = batch_obs.to(device)
                            batch_recon_obs, mu, logvar = model(batch_obs)

                            test_loss += loss_function(batch_recon_obs, batch_obs, mu, logvar).item()

        if nb_frames == 1:
            test_loss = test_loss / (test_rollouts * ep_length)
        else:
            test_loss = test_loss / (test_rollouts * (ep_length - (nb_frames - 1) * frame_gap))
        logger.info('====> Test set loss: {:.4f}'.format(test_loss))
        test_loss_arr.append(test_loss)
        return test_loss_arr



#data_folder = join("data/carracing" , args.action_type)
data_folder = os.path.join('..', 'data', 'carracing')
assert exists(data_folder)
train_loss_arr, test_loss_arr = [], []
cur_best = None
for epoch in range(1, args.epochs + 1):
        train_loss_arr = train(epoch, data_folder, train_loss_arr, nb_frames=5, frame_gap=5)
        test_loss_arr = test(epoch, data_folder, test_loss_arr, nb_frames=5, frame_gap=5)
        test_loss = test_loss_arr[len(test_loss_arr)-1]
        
        # checkpointing
        best_filename = join(args.model_path, args.action_type, 'best.pth')
        filename = join(args.model_path, args.action_type, 'checkpoint_' + str(epoch) + '.pth')
        
        is_best = not cur_best or test_loss < cur_best
        if is_best:
            cur_best = test_loss

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'precision': test_loss,
            'optimizer': optimizer.state_dict()
        }, is_best, filename, best_filename)

logger.info("Overall training loss is")
logger.info(train_loss_arr)
logger.info("Overall testing loss is")
logger.info(test_loss_arr)
