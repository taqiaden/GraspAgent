import numpy as np
import torch
from collections import deque
from Grasp_Agent_ import Action
from models.policy_net import PolicyNet

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
        self.is_synchronous=deque([])
        self.action_indexes=deque([])
        self.point_indexes=deque([])

        self.probs=deque([])
        self.last_ending_index=None # the end index of the last completed episode

        self.is_end_of_episode=deque([])
        self.episodes_counter=0 # track the number of completed episodes in the buffer

    def push(self, action_obj:Action):
        if action_obj.policy_index==0:
            self.actions_obj_list.append(action_obj)
            self.values.append(action_obj.value)
            self.probs.append(action_obj.prob)
            self.is_synchronous.append(action_obj.is_synchronous)
            self.action_indexes.append(action_obj.action_index)
            self.point_indexes.append(action_obj.point_index)
            '''task in process'''
            self.effort_penalty(action_obj)
            self.is_end_of_episode.append(0)
        elif action_obj.policy_index==1:
            '''task end successfully'''
            if len(self)>0 and self.is_end_of_episode[-1]==0:
                self.episodes_counter+=1
                self.last_ending_index = len(self) - 1
                self.positive_reward()
                self.is_end_of_episode[-1]=1
                self.update_advantages()
        elif action_obj.policy_index==2:
            '''task end with failure'''
            if len(self)>0 and self.is_end_of_episode[-1]==0:
                self.episodes_counter+=1
                self.last_ending_index = len(self) - 1
                self.negative_reward()
                self.is_end_of_episode[-1]=1
                self.update_advantages()

    def effort_penalty(self,action_obj:Action):
        assert len(self.rewards)==len(self.actions_obj_list)-1
        '''penalize each action to reduce the effort needed to reach the target'''
        k = 2 if action_obj.is_synchronous else 1 # less penalty if the robot runs both arms at the same time
        self.rewards.append(-0.4/k)

    def positive_reward(self):
        '''add reward to the last action/s that leads to grasp the target in the current action'''
        if len(self.probs)>1:
            if self.is_synchronous[-1]:
                # add rewards to the last two actions if both arms are used in the last run
                self.rewards[-2:]+=1
            else:
                self.rewards[-1:]+=1

    def negative_reward(self):
        '''dedict reward to the last action/s that leads to the disappearance of the target in the new state'''
        if len(self.probs)>1:
            if self.is_synchronous[-1]:
                # add rewards to the last two actions if both arms are used in the last run
                self.rewards[-2:]-=1
            else:
                self.rewards[-1:]-=1

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
                if self.is_end_of_episode[0]==1:
                    self.episodes_counter -= 1
                self.actions_obj_list.popleft()
                self.rewards.popleft()
                self.values.popleft()
                self.probs.popleft()
                self.action_indexes.popleft()
                self.point_indexes.popleft()
                self.advantages.popleft()
                self.is_end_of_episode.popleft()
                self.is_synchronous.popleft()
                self.last_ending_index-=1


    def get_all_buffer_files_ids(self):
        file_ids=[]
        for i in range(self.__len__()):
            file_ids.append(self.actions_obj_list[i].file_id)
        return file_ids

    def __len__(self):
        return len(self.actions_obj_list)

class PPOLearning():
    def __init__(self, model=None,buffer:PPOMemory=None,n_epochs=4,policy_clip=0.2, gamma=0.99, lamda=0.95,batch_size=5):

        self.gamma = gamma
        self.policy_clip = policy_clip
        self.lamda = lamda
        self.n_epochs = n_epochs
        self.batch_size=batch_size
        self.model=model

        self.memory = buffer

    def step(self,model:PolicyNet(),buffer:PPOMemory):
        self.model=model
        self.memory=buffer
        self.model.train(True)
        self.learn()
        self.model.eval()
        return self.model

    def learn(self):
        for _ in range(self.n_epochs):

            batches = self.memory.generate_batches(batch_size=self.batch_size)

            # advantage_arr = self.calculate_advanatage(reward_arr, value_arr, dones_arr)
            # values = torch.tensor(value_arr).to(self.model.device)

            '''load files'''

            for batch in batches:
                b=len(batch)
                '''load states'''
                states=None
                # states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.model.device)
                old_probs = torch.tensor(self.memory.probs[batch]).to(self.model.device)
                # actions = torch.tensor(action_arr[batch]).to(self.model.device)
                griper_grasp_score,suction_grasp_score,shift_affordance_classifier,q_values,action_probs=self.model(rgb,gripper_pose,suction_direction,quality_masks)

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