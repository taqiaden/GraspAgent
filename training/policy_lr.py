import numpy as np
import torch
from collections import deque


def policy_loss(new_policy_probs,old_policy_probs,advantages,epsilon=0.2):
    ratio = new_policy_probs / old_policy_probs
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    objective = torch.min(ratio * advantages, clipped_ratio * advantages)
    return objective

class PPOMemory():
    def __init__(self):
        self.actions_obj_list = deque([])

    def push(self, action_obj):
        self.actions_obj_list.append(action_obj)

    def generate_batches(self):
        return
        # ## suppose n_states=20 and batch_size = 4
        # n_states = len(self.states)
        # ##n_states should be always greater than batch_size
        # ## batch_start is the starting index of every batch
        # ## eg:   array([ 0,  4,  8, 12, 16]))
        # batch_start = np.arange(0, n_states, self.batch_size)
        # ## random shuffling if indexes
        # # eg: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
        # indices = np.arange(n_states, dtype=np.int64)
        # ## eg: array([12, 17,  6,  7, 10, 11, 15, 13, 18,  9,  8,  4,  3,  0,  2,  5, 14,19,  1, 16])
        # np.random.shuffle(indices)
        # batches = [indices[i:i + self.batch_size] for i in batch_start]
        # ## eg: [array([12, 17,  6,  7]),array([10, 11, 15, 13]),array([18,  9,  8,  4]),array([3, 0, 2, 5]),array([14, 19,  1, 16])]
        # return np.array(self.states), np.array(self.actions), \
        #     np.array(self.action_probs), np.array(self.vals), np.array(self.rewards), \
        #     np.array(self.dones), batches

    def pop(self):
        self.actions_obj_list.popleft()

    def get_all_buffer_files_ids(self):
        file_ids=[]
        for i in range(self.__len__()):
            file_ids.append(self.actions_obj_list[i].file_id)
        return file_ids

    def __len__(self):
        return len(self.actions_obj_list)

class PPOLearning():
    def __init__(self, model,n_epochs=4,policy_clip=0.2, gamma=0.99, lamda=0.95,batch_size=5):

        self.gamma = gamma
        self.policy_clip = policy_clip
        self.lamda = lamda
        self.n_epochs = n_epochs
        self.batch_size=batch_size
        self.model=model

        self.memory = PPOMemory()

    def store_data(self, state, action, action_prob, val, reward, done):
        self.memory.store_memory(state, action, action_prob, val, reward, done)

    # def save_models(self):
    #     print('... Saving Models ......')
    #     self.actor.save_checkpoint()
    #     self.critic.save_checkpoint()
    #
    # def load_models(self):
    #     print('... Loading models ...')
    #     self.actor.load_checkpoint()
    #     self.critic.load_checkpoint()

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

        for t in range(0, time_steps - 1):
            discount = 1
            running_advantage = 0
            for k in range(t, time_steps - 1):
                if int(dones_arr[k]) == 1:
                    running_advantage += reward_arr[k] - value_arr[k]
                else:

                    running_advantage += reward_arr[k] + (self.gamma * value_arr[k + 1]) - value_arr[k]

                running_advantage = discount * running_advantage
                # running_advantage += discount*(reward_arr[k] + self.gamma*value_arr[k+1]*(1-int(dones_arr[k])) - value_arr[k])
                discount *= self.gamma * self.lamda

            advantage[t] = running_advantage
        advantage = torch.tensor(advantage).to(self.actor.device)
        return advantage

    def learn(self):
        for _ in range(self.n_epochs):

            ## initially all will be empty arrays
            state_arr, action_arr, old_prob_arr, value_arr, \
                reward_arr, dones_arr, batches = self.memory.generate_batches(batch_size=self.batch_size)

            advantage_arr = self.calculate_advanatage(reward_arr, value_arr, dones_arr)
            values = torch.tensor(value_arr).to(self.model.device)

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.model.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.model.device)
                actions = torch.tensor(action_arr[batch]).to(self.model.device)

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

    ### Train the model
