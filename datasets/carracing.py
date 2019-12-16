import argparse
from os.path import join, exists
import gym
import numpy as np
import math

#taken from ctallec repo
def sample_continuous_policy(action_space, seq_len, dt):
    """ Sample a continuous policy.
    Atm, action_space is supposed to be a box environment. The policy is
    sampled as a brownian motion a_{t+1} = a_t + sqrt(dt) N(0, 1).
    :args action_space: gym action space
    :args seq_len: number of actions returned
    :args dt: temporal discretization
    :returns: sequence of seq_len actions
    """
    actions = [action_space.sample()]
    for _ in range(seq_len):
        daction_dt = np.random.randn(*actions[-1].shape)
        actions.append(
            np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
                    action_space.low, action_space.high))
    return actions

def generate_data(rollouts, data_dir, action_type): 
    """ Generates data """
    assert exists(data_dir), "The data directory does not exist..."
    assert exists(join(data_dir, action_type))
    env = gym.make("CarRacing-v0")
    seq_len = 1000

    for i in range(rollouts):
        env.reset()
        #env.env.viewer.window.dispatch_events()
        if action_type == "random":
            a_rollout = [env.action_space.sample() for _ in range(seq_len)]
        else:
            a_rollout = sample_continuous_policy(env.action_space, seq_len, 1. / 50)
        s_rollout = []
        r_rollout = []
        d_rollout = []
        t = 0
        while True:
            action = a_rollout[t]
            t += 1
            s, r, done, _ = env.step(action)
            #env.env.viewer.window.dispatch_events()
            s_rollout += [s]
            r_rollout += [r]
            d_rollout += [done]
            if done:
                print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                np.savez(join(data_dir, action_type, 'rollout_{}'.format(i)),
                         observations=np.array(s_rollout),
                         rewards=np.array(r_rollout),
                         actions=np.array(a_rollout),
                         terminals=np.array(d_rollout))
                break
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, default=1000, help="Number of rollouts")
    parser.add_argument('--dir', type=str, default="data/carracing", help="Where to place rollouts")
    parser.add_argument('--action_type', type=str, default="random", help="random or continuous")
    args = parser.parse_args()
    generate_data(args.rollouts, args.dir, args.action_type)