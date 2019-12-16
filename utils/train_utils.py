import os
import torch


def save_model(dir_path, model, model_type, n_iter, time, state):

    save_dir = os.path.join(dir_path, model_type, n_iter)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, time + '.pth.tar')
    state['model'] = model.state_dict()
    torch.save(state, filename)


def load_model(filepath, model):
    model.load_state_dict(torch.load(filepath)['model'])