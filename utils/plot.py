import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

with open('../Evo_vae_trained_lstm_28_4_800.pkl', 'rb') as f:
    trained = pickle.load(f)

with open('../Evo_vae_untrained_lstm_28_4_800.pkl', 'rb') as f:
    untrained = pickle.load(f)

with open('../evo_stacked_vae.pkl', 'rb') as f:
    stacked = pickle.load(f)

data_trained = trained[2][2]
maxes_trained = np.array(data_trained).T[0]

data_untrained = untrained[2][2]
maxes_untrained = np.array(data_untrained).T[0]

data_stacked = stacked[2][2]
maxes_stacked = np.array(data_stacked).T[0]

print(maxes_stacked)