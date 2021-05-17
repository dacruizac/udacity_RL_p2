from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
import argparse
import configparser
from ddpg_interaction import info, reset, step, ddpg
from agent_ddpg import Agent
import torch
import time

parser = argparse.ArgumentParser()
# input argument for single or multiple arms agents
parser.add_argument('--multi-agent', action='store_true',
                    help='To run 20 arms instead of 1')
# input argument to display learning curves
parser.add_argument('--display', action='store_true',
                    help='Display evolution of the scores')
parser.add_argument('--file',
                    help='filename of the trained weights')
parser.add_argument('--train', action='store_true',
                    help='run a pre-trainded neural network agent')
parser.add_argument('--continuing', action='store_true',
                    help='continue training a neural network agent')
parser.add_argument('--random', action='store_true',
                    help='run a tabula rasa agent')
parser.add_argument('--no-graph', action='store_true',
                    help='run a tabula rasa agent')
args = parser.parse_args()

# setting filename
if args.file==None:
    filename='checkpoint'
else:
    filename=args.file

# current configuration
configParser = configparser.ConfigParser()
configParser.read('params_example.ini')
config = {
    'n_episodes':    int(configParser['DEFAULT']['n_episodes']),
    'max_t':         int(configParser['DEFAULT']['max_t']),
    'print_every':   int(configParser['DEFAULT']['print_every']),
    'SEED':          int(configParser['DEFAULT']['SEED']),
    'BUFFER_SIZE':   int(float(configParser['DEFAULT']['BUFFER_SIZE'])),
    'BATCH_SIZE':    int(configParser['DEFAULT']['BATCH_SIZE']),
    'UPDATE_EVERY':  int(configParser['DEFAULT']['UPDATE_EVERY']),
    'GAMMA':         float(configParser['DEFAULT']['GAMMA']),
    'SIGMA':         float(configParser['DEFAULT']['SIGMA']),
    'TAU':           float(configParser['DEFAULT']['TAU']),
    'LR_ACTOR':      float(configParser['DEFAULT']['LR_ACTOR']),
    'LR_CRITIC':     float(configParser['DEFAULT']['LR_CRITIC']),
    'WEIGHT_DECAY':  float(configParser['DEFAULT']['WEIGHT_DECAY']),
    'FC1_ACTOR':     int(configParser['DEFAULT']['FC1_ACTOR']),
    'FC2_ACTOR':     int(configParser['DEFAULT']['FC2_ACTOR']),
    'FC1_CRITIC':    int(configParser['DEFAULT']['FC1_CRITIC']),
    'FC2_CRITIC':    int(configParser['DEFAULT']['FC2_CRITIC']),
}
# print configuration
print(' Config Parameters')
for k,v in config.items():
    print('{:<15}: {:>15}'.format(k,v))

# create environment
if args.multi_agent:
    env = UnityEnvironment(file_name='Reacher_Linux_20/Reacher.x86_64', seed=config['SEED'])
else:
    env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64', seed=config['SEED'])
# get info of the environment
num_agents, state_size, action_size = info(env)

# create an agent
agent = Agent(num_agents=num_agents, state_size=state_size, action_size=action_size,
              random_seed=config['SEED'],
              gamma=config['GAMMA'],
              sigma=config['SIGMA'],
              tau=config['TAU'],
              lr_actor=config['LR_ACTOR'],
              lr_critic=config['LR_CRITIC'],
              weight_decay=config['WEIGHT_DECAY'],
              fc1_a=config['FC1_ACTOR'],
              fc2_a=config['FC2_ACTOR'],
              fc1_c=config['FC1_CRITIC'],
              fc2_c=config['FC2_CRITIC'],
              buffer_size=config['BUFFER_SIZE'],
              batch_size=config['BATCH_SIZE'],
              update_every=config['UPDATE_EVERY'])
# continue training
if args.continuing:
    # load the weights from file
    agent.actor_local.load_state_dict(torch.load(filename+'.actor.pth'))
    agent.actor_target.load_state_dict(torch.load(filename+'.actor.pth'))
    agent.critic_local.load_state_dict(torch.load(filename+'.critic.pth'))
    agent.critic_target.load_state_dict(torch.load(filename+'.critic.pth'))
    print('Loaded {}:'.format(filename))

# learn or prove
if args.train:

    tic = time.perf_counter()
    scores, scores_avg = ddpg(env, agent,
                              n_episodes=config['n_episodes'],
                              max_t=config['max_t'],
                              print_every=config['print_every'],
                              filename=filename)
    toc = time.perf_counter()
    print(f"Time:  {toc - tic:0.4f} seconds")

    print(f"Time:  {(toc - tic)/60:0.4f} minutes")
    # save training curves
    np.savez(filename+'.npz', scores=scores, scores_avg=scores_avg, config=config)
    # plot learning curves output
    # display mode
    if args.display:
        # to continue ...
        print('Press [Q] on the plot window to continue ...')
        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.plot(np.arange(len(scores_avg)), scores_avg)
        plt.ylabel('Scores')
        plt.xlabel('Episode #')
        plt.show()
elif args.random:
    # choose random policy
    print('Run a Tabula Rasa or random agent')
else:
    # load the weights from file
    agent.actor_local.load_state_dict(torch.load(filename+'.actor.pth'))
    agent.actor_target.load_state_dict(torch.load(filename+'.actor.pth'))
    agent.critic_local.load_state_dict(torch.load(filename+'.critic.pth'))
    agent.critic_target.load_state_dict(torch.load(filename+'.critic.pth'))
    print('Loaded {}:'.format(filename))

# run a random controller through arms
if args.no_graph:
    print('no proves on controller')
else:
    for i in range(10):
        scores = np.zeros(num_agents)
        states = reset(env, train_mode=False)
        # check performance of the agent
        for j in range(1000):
            actions = agent.act(states, add_noise=False)
            next_states, rewards, dones = step(env, actions)
            states =  next_states
            scores += rewards
            if np.any(dones):
                break
        print('Total score (averaged over agents) this episode: {}'.
              format(np.mean(scores)))
env.close()