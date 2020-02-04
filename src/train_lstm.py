import time
import numpy as np
from models.lstm_mdn import LSTM_MDN
import torch
from torch.utils.data import DataLoader
from torch.distributions import Normal
from utils.train_utils import save_model, load_model
from datasets.generate_lstm_training import LSTMDataset


CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda") if CUDA else torch.device("cpu")


def loss_mdn(pi, sigma, mu, y, seq_size=100, z_dim=32):
    result = Normal(loc=mu, scale=sigma)
    y = y.view(-1, seq_size, 1, z_dim)
    result = torch.sum(torch.exp(result.log_prob(y)) * pi, dim=2)
    result = -torch.log(result + 1e-10)
    return torch.mean(result)


def train_batch(lstm, x, weight_decay=0., lr=0.0005):
    optimizer = torch.optim.Adam(lstm.parameters(), lr=lr, weight_decay=weight_decay)
    hidden = torch.zeros(1, lstm.seq_size, lstm.hidden_units, device=DEVICE)
    cell = torch.zeros(1, lstm.seq_size, lstm.hidden_units, device=DEVICE)
    lstm.hidden = hidden, cell

    ## This part was copied from dylandjian github repo
    inp = torch.cat((x['encoded'], x['actions']), dim=1) #.view(-1, 1) / 3), dim=1)
    inp = inp.view(-1, lstm.seq_size, lstm.z_dim + 3)
    last_x = x['encoded'][-1].view(-1, x['encoded'].size()[1])
    target = torch.cat((x['encoded'][1:x['encoded'].size()[0]], last_x,))
    ##

    optimizer.zero_grad()
    pi, sigma, mu = lstm(inp)
    loss = loss_mdn(pi, sigma, mu, target)
    loss.backward()
    optimizer.step()

    return float(loss)


## This function was copied from dylandjian github repo
def collate_fn(example):
    encoded = []
    actions = []

    for ex in example:
        encoded.extend(ex[0])
        actions.extend(ex[1])

    frames = torch.tensor(encoded, dtype=torch.float, device=DEVICE) / 255
    actions = torch.tensor(actions, dtype=torch.float, device=DEVICE) / 3
    return frames, actions
##


def train_lstm(lstm, dataset_name, max_iter=1000, load_path=None):

    dataset = LSTMDataset(name=dataset_name)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    if load_path is not None:
        load_model(load_path, lstm)

    n_iter = 0

    while n_iter < max_iter:

        train_loss = []

        for idx, (encoded, actions) in enumerate(dataloader):

            x = {'encoded': encoded, 'actions': actions}
            loss = train_batch(lstm, x)
            train_loss.append(loss)

            if n_iter % 5 == 0:
                print("[TRAIN] current iteration: {}, loss: {}".format(n_iter, loss))

            if (n_iter + 1) % 500 == 0:
                dir_path = './saved_models/'
                save_model(dir_path, lstm, 'lstm', str(n_iter), str(int(time.time())), {})

            n_iter += 1

        print("[TRAIN] Average backward pass loss : {}".format(np.mean(train_loss)))


if __name__ == "__main__":
    lstm = LSTM_MDN()
    train_lstm(lstm, "lstm_data_v0", max_iter=50000)
