import cv2
import numpy as np


def load_data(data_folder, ep_length, batch_rollout_size, batch_rollout):
        obs = np.zeros((ep_length * batch_rollout_size, 64, 64, 3))
        i, j  = batch_rollout * batch_rollout_size, (batch_rollout + 1) * batch_rollout_size
        for k in range(i, j):
            ob = np.load(data_folder + "rollout_" + str(k) + ".npz")['observations']
            for m in range(ep_length):
                obs[k * ep_length + m,:,:,:] = cv2.resize(ob[m,:,:,:], (64, 64))
        return obs


