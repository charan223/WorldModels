from models.convVAE import ConvVAE
import torch
import os
from torch.utils.data import Dataset
import numpy as np
import cv2


class LSTMDataset(Dataset):

    def __init__(self, name=None):
        self.encoded = []
        self.actions = []

        if name is not None:
            save_dir = "data/carracing_lstm/"
            dataset = np.load(os.path.join(save_dir, name))
            self.encoded = dataset['encoded']
            self.actions = dataset['actions']

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        sample_idx = np.random.randint(0, 900)
        return self.encoded[idx][sample_idx:sample_idx + 100], self.actions[idx][sample_idx:sample_idx + 100]


def generate_lstm_data(name, vae_file, nb_rollouts=1000):

    data_folder = "data/carracing/"
    save_dir = "data/carracing_lstm/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, name)

    vae = ConvVAE()
    vae.load_state_dict(torch.load(vae_file))
    encoded = []
    act = []

    with torch.no_grad():
        for k in range(nb_rollouts):
            dat = np.load(data_folder + "rollout_" + str(k) + ".npz")
            act.append(dat['actions'])
            obs = []
            for ob in dat['observations']:
                batch = np.array([cv2.resize(ob, (64, 64))]).astype(np.float)/255.0
                batch = torch.from_numpy(batch).permute(0,3,1,2).float()
                z, mu, logvar = vae.encode(batch)
                obs.append((mu, logvar))
            encoded.append(obs)
    with open(save_path, 'wb') as f:
        np.savez(f, encoded=encoded, actions=act)
