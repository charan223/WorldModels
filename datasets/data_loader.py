import random
import numpy as np
import os

DATA_DIR = "record"


def create_dataset(N=10000, M=1000): # N is 10000 episodes, M is number of timesteps
  filelist = os.listdir(DATA_DIR)
  filelist.sort()
  filelist = filelist[0:10000]
  
  data = np.zeros((M*N, 64, 64, 3), dtype=np.uint8)
  idx = 0
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join("record", filename))['obs']
    l = len(raw_data)
    if (idx+l) > (M*N):
      data = data[0:idx]
      print('premature break')
      break
    data[idx:idx+l] = raw_data
    idx += l
    if ((i+1) % 100 == 0):
      print("loading file", i+1)
  return data









