from models.convVAE import ConvVAE
import torch
import os
from torch.utils.data import Dataset
import numpy as np
import cv2

device = torch.device("cpu")


def visualize_images(name, vae_file, nb_rollouts=1):

    data_folder = "data/carracing/random"
    save_dir = "results/random"

    save_path = os.path.join(save_dir, name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    vae = ConvVAE()
    state = torch.load(vae_file, map_location=device)
    vae.load_state_dict(state['state_dict'])


    with torch.no_grad():
        for k in range(nb_rollouts):
            print(k)
            dat = np.load(os.path.join(data_folder , "rollout_" + str(k) + ".npz"))
            for batch_idx, ob in enumerate(dat['observations']):
                batch = np.array([cv2.resize(ob, (64, 64))]).astype(np.float)/255.0
                batch = torch.from_numpy(batch).permute(0,3,1,2).float()
                recon_batch, _,_ = vae(batch)
                if batch_idx < 10:
                        cv2.imwrite(os.path.join(save_path , str(k) + "_" + str(batch_idx) + "_recon.jpg"), np.squeeze(recon_batch.permute(0,2,3,1).numpy())*255)
                        cv2.imwrite(os.path.join(save_path , str(k) + "_" + str(batch_idx) + ".jpg"), np.squeeze(batch.permute(0,2,3,1).numpy())*255)
                
if __name__ == "__main__":

    name = "visualize"
    vae_file = "checkpoints/random/best.pth"
    visualize_images(name, vae_file)