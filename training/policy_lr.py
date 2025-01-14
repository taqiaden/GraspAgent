import numpy as np
import torch
from collections import deque
from Grasp_Agent_ import Action

minimum_buffer_size=20
max_buffer_size=50
gamma=0.99
lamda=0.95

def policy_loss(new_policy_probs,old_policy_probs,advantages,epsilon=0.2):
    ratio = new_policy_probs / old_policy_probs
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    objective = torch.min(ratio * advantages, clipped_ratio * advantages)
    return objective

class PPOMemory():
    def __init__(self):
        self.actions_obj_list = deque([])
        self.rewards=deque([])
        self.values=deque([])
        self.advantages=deque([])

        self.probs=deque([])
        self.last_ending_index=None # the end index of the last completed episode

        '''track episode data'''
        self.is_end_of_episode=deque([])
        # list elements takes the values [0,1,None]
            # 0 : the instance belong to the episode but is not the end of it
            # 1 : the instance is the end of the episode
            # None : the instance is not part of the episode, also means the instance is sampled following either a deterministic policy or a random policy

    def push(self, action_obj:Action):
        self.actions_obj_list.append(action_obj)
        self.values.append(action_obj.value)
        self.probs.append(action_obj.prob)
        self.update_rewards(action_obj)
        if action_obj.on_target:
            if len(self)>0 and self.is_end_of_episode[-1]==0:
                self.is_end_of_episode[-1]=1
            self.is_end_of_episode.append(None)
            self.last_ending_index=len(self)-1
            self.update_advantages()
        else:
            self.is_end_of_episode.append(0)


    def update_rewards(self,action_obj:Action):
        assert len(self.rewards)==len(self.actions_obj_list)-1
        '''Reward is computed in two steps'''
        if action_obj.on_target:
            # if the action grasp the target then it is not part of the episode as it follows a deterministic policy
            # the episode ends before this action step and more reward will be added to the former actions because it facilitates the current action to grasp the target
            self.rewards.append(None)
        else:
            '''1) penalize current action to reduce the effort needed to reach the target'''
            k = 2 if action_obj.is_synchronous else 1 # less penalty if the robot runs both arms at the same time
            self.rewards.append(-0.2/k)

        '''2) reward previous action/s if it cause to grasp the target in the current action'''
        if action_obj.on_target and len(self.actions_obj_list)>1:
            if self.actions_obj_list[-1].is_synchronous:
                # add rewards to the last two actions if both arms are used in the last run
                self.rewards[-2:]+=1
            else:
                self.rewards[-1:]+=1

    def generate_batches(self,batch_size):
        '''arrange batches to the end of the last completed episode'''
        n_states = self.last_ending_index+1
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + batch_size] for i in batch_start]
        return  batches

    def update_advantages(self):
        n_states = self.last_ending_index + 1
        advantage = np.zeros(n_states, dtype=np.float32)
        start_=len(self.advantages)

        for t in range(start_, n_states - 1):
            discount = 1
            running_advantage = 0
            for k in range(t, n_states - 1):
                if self.is_end_of_episode[k] == 1:
                    running_advantage += self.rewards[k] - self.values[k]
                else:
                    running_advantage += self.rewards[k] + (gamma * self.values[k + 1]) - self.values[k]

                running_advantage = discount * running_advantage
                discount *= gamma * lamda

                if self.is_end_of_episode[k] == 1: break

            self.advantages.append(running_advantage)
            # advantage[t] = running_advantage
        return advantage

    def pop(self):
        if len(self)>max_buffer_size:
            for i in range(3):
                self.actions_obj_list.popleft()
                self.rewards.popleft()
                self.values.popleft()
                self.probs.popleft()
                self.advantages.popleft()
                self.is_end_of_episode.popleft()
                self.last_ending_index-=1

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
