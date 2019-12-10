from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from models.convVAE import ConvVAE


parser = argparse.ArgumentParser(description='ConvVAE for WorldModels')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')                    
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--N_z', type=int, default=32, metavar='N',
                    help='size of latent variable')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")



model = ConvVAE(N_z=args.N_z,
              batch_size=args.batch_size,
              kl_tolerance=args.kl_tolerance,
              is_training=True,
              reuse=False,
              gpu_mode=True).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

# L2 Loss + KL Loss as suggested in paper and are summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):

    L2_Loss = F.mse_loss(x, recon_x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KL_Loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return L2_Loss + KL_Loss


def train(epoch):
    model_save_path = "torch_vae"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    #dataset loading
    dataset = create_dataset(N=10000, M=1000)
    np.random.shuffle(dataset)

    # split into batches:
    total_length = len(dataset)
    num_batches = int(np.floor(total_length/args.batch_size))

    model.train()
    train_loss = 0

        
    for batch_idx in range(num_batches):
        batch = dataset[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]

        # as the input pixels lie between 0, 1
        obs = batch.astype(np.float)/255.0

        # for gradients to not accumulate
        optimizer.zero_grad()

        # forward pass
        recon_obs, mu, logvar = model(obs)

        #loss function
        loss = loss_function(recon_obs, obs, mu, logvar)

        # backpropagation
        loss.backward()

        train_loss += loss.item()

        # update parameters based on calculated gradients
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(batch), len(dataset),
                100. * batch_idx / len(dataset),
                loss.item() / len(batch)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(dataset)))

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)