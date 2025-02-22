

## Import required libraries

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import warnings
from torch.distributions.categorical import Categorical
warnings.simplefilter("ignore")


## Initilaize Data store, Actor Network and Critic network


############################# Data Store ####################################################
class PPOMemory():
    """
    Memory for PPO
    """

    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.vals = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        ## suppose n_states=20 and batch_size = 4
        n_states = len(self.states)
        ##n_states should be always greater than batch_size
        ## batch_start is the starting index of every batch
        ## eg:   array([ 0,  4,  8, 12, 16]))
        batch_start = np.arange(0, n_states, self.batch_size)
        ## random shuffling if indexes
        # eg: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
        indices = np.arange(n_states, dtype=np.int64)
        ## eg: array([12, 17,  6,  7, 10, 11, 15, 13, 18,  9,  8,  4,  3,  0,  2,  5, 14,19,  1, 16])
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        ## eg: [array([12, 17,  6,  7]),array([10, 11, 15, 13]),array([18,  9,  8,  4]),array([3, 0, 2, 5]),array([14, 19,  1, 16])]
        return np.array(self.states), np.array(self.actions), \
            np.array(self.action_probs), np.array(self.vals), np.array(self.rewards), \
            np.array(self.dones), batches

    def store_memory(self, state, action, action_prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.action_probs.append(action_prob)
        self.rewards.append(reward)
        self.vals.append(val)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.vals = []
        self.dones = []


############################ Actor Network ######################################

## initialize actor network and critic network


class ActorNwk(nn.Module):
    def __init__(self, input_dim, out_dim,
                 adam_lr,
                 chekpoint_file,
                 hidden1_dim=256,
                 hidden2_dim=256
                 ):
        super(ActorNwk, self).__init__()

        self.actor_nwk = nn.Sequential(
            nn.Linear(*input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, out_dim),
            nn.Softmax(dim=-1)
        )

        self.checkpoint_file = chekpoint_file
        self.optimizer = torch.optim.Adam(params=self.actor_nwk.parameters(), lr=adam_lr)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        out = self.actor_nwk(state)
        dist = Categorical(out)
        return dist

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


############################### Crirtic Network ######################################

class CriticNwk(nn.Module):
    def __init__(self, input_dim,
                 adam_lr,
                 chekpoint_file,
                 hidden1_dim=256,
                 hidden2_dim=256
                 ):
        super(CriticNwk, self).__init__()

        self.critic_nwk = nn.Sequential(
            nn.Linear(*input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, 1),

        )

        self.checkpoint_file = chekpoint_file
        self.optimizer = torch.optim.Adam(params=self.critic_nwk.parameters(), lr=adam_lr)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        out = self.critic_nwk(state)
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


## Initilaize an Agent will will be able to train the model

############################# Agent ########################################3

## agent

class Agent():
    def __init__(self, gamma, policy_clip, lamda, adam_lr,
                 n_epochs, batch_size, state_dim, action_dim):

        self.gamma = gamma
        self.policy_clip = policy_clip
        self.lamda = lamda
        self.n_epochs = n_epochs

        self.actor = ActorNwk(input_dim=state_dim, out_dim=action_dim, adam_lr=adam_lr, chekpoint_file='tmp/actor')
        self.critic = CriticNwk(input_dim=state_dim, adam_lr=adam_lr, chekpoint_file='tmp/ctitic')
        self.memory = PPOMemory(batch_size)

    def store_data(self, state, action, action_prob, val, reward, done):
        self.memory.store_memory(state, action, action_prob, val, reward, done)

    def save_models(self):
        print('... Saving Models ......')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... Loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.actor.device)

        dist = self.actor(state)
        ## sample the output action from a categorical distribution of predicted actions
        action = dist.sample()
        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()

        ## value from critic model
        value = self.critic(state)
        value = torch.squeeze(value).item()

        return action, probs, value

    def calculate_advanatage(self, reward_arr, value_arr, dones_arr):
        time_steps = len(reward_arr)
        advantage = np.zeros(len(reward_arr), dtype=np.float32)
        # print(time_steps - 1)
        # exit()

        for t in range(0, time_steps ):
            print(t,dones_arr[t])
            discount = 1
            running_advantage = 0
            for k in range(t, time_steps):
                if int(dones_arr[k]) == 1:
                    running_advantage += reward_arr[k] - value_arr[k]
                else:
                    running_advantage += reward_arr[k] + (self.gamma * value_arr[k + 1]) - value_arr[k]

                running_advantage = discount * running_advantage
                # running_advantage += discount*(reward_arr[k] + self.gamma*value_arr[k+1]*(1-int(dones_arr[k])) - value_arr[k])
                discount *= self.gamma * self.lamda

            advantage[t] = running_advantage
        advantage = torch.tensor(advantage).to(self.actor.device)
        # print(advantage)
        exit()
        return advantage

    def learn(self):
        for _ in range(self.n_epochs):

            ## initially all will be empty arrays
            state_arr, action_arr, old_prob_arr, value_arr, \
                reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            advantage_arr = self.calculate_advanatage(reward_arr, value_arr, dones_arr)
            values = torch.tensor(value_arr).to(self.actor.device)

            for batch in batches:
                # print(batch,advantage_arr[batch])
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                # prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage_arr[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                     1 + self.policy_clip) * advantage_arr[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage_arr[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
        # exit()


### Train the model


import gym
import numpy as np
import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


if not os.path.exists('tmp'):
    os.makedirs('tmp')

env = gym.make('CartPole-v0')
N = 20
batch_size = 5
n_epochs = 4
alpha = 0.0003
agent = Agent(state_dim=env.observation_space.shape,
              action_dim=env.action_space.n,
              batch_size=batch_size,
              n_epochs=n_epochs,
              policy_clip=0.2,
              gamma=0.99, lamda=0.95,
              adam_lr=alpha)
n_games = 300
figure_file = 'cartpole.png'
best_score = env.reward_range[0]
score_history = []
learn_iters = 0
avg_score = 0
n_steps = 0
running_reward=10
for i in range(n_games):
    current_state, info = env.reset()
    terminated, truncated = False, False
    done = False
    score = 0


    while not done:
        action, prob, val = agent.choose_action(current_state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = 1 if (terminated or truncated) else 0
        n_steps += 1
        score += reward
        agent.store_data(current_state, action, prob, val, reward, done)
        if n_steps % N == 0:
            agent.learn()
            learn_iters += 1
        current_state = next_state
    score_history.append(score)
    running_reward = 0.05 * score + (1 - 0.05) * running_reward
    avg_score = np.mean(score_history[-100:])
    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()
    print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
          'time_steps', n_steps, 'learning_steps', learn_iters,' running reward',running_reward)

x = [i + 1 for i in range(len(score_history))]
plot_learning_curve(x, score_history, figure_file)

import matplotlib.pyplot as plt

from IPython.display import clear_output
from time import sleep

env = gym.make("CartPole-v1", render_mode="rgb_array")

state_dim = 4
action_dim = env.action_space.n


class ActorPredNwk(nn.Module):
    def __init__(self, input_dim, out_dim,
                 hidden1_dim=256,
                 hidden2_dim=256,
                 checkpoint_file='tmp/actor'
                 ):
        super(ActorPredNwk, self).__init__()

        self.checkpoint_file = checkpoint_file
        self.actor_nwk = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, out_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        x = torch.Tensor(state)
        out = self.actor_nwk(x)
        return out

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


policy_nwk = ActorPredNwk(input_dim=state_dim, out_dim=action_dim)
policy_nwk.load_checkpoint()

current_state, info = env.reset()

terminated, truncated = False, False
i = 1
while not (terminated or truncated):

    action = policy_nwk(current_state)
    action = torch.argmax(action).item()
    new_state, reward, terminated, truncated, info = env.step(action)
    print(current_state, new_state, action)
    print('New state: ', new_state, 'Reward: ', reward, 'Terminated: ', terminated, 'Truncated: ', truncated)
    current_state = new_state
    plt.title(f'Frame {i}')
    i += 1
    plt.imshow(env.render())
    plt.show()
    if i >= 20:
        break


