import numpy as np
from collections import deque
from action import Action
from lib.dataset_utils import online_data2

max_policy_buffer_size=50
max_quality_buffer_size=50
gamma=0.99
lamda=0.95
learning_rate=1e-4

online_data2=online_data2()

class PPOMemory():
    def __init__(self):
        '''episodic buffer containers'''''
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
        self.episodic_file_ids=deque([])

        self.non_episodic_file_ids=deque([])

    def append_to_policy_buffer(self, action_obj:Action):
        self.values.append(action_obj.value)
        self.probs.append(action_obj.prob)
        self.is_synchronous.append(action_obj.is_synchronous)
        self.action_indexes.append(action_obj.action_index)
        self.point_indexes.append(action_obj.point_index)
        self.episodic_file_ids.append(action_obj.file_id)

    def push(self, action_obj:Action):
        print('policy index = ',action_obj.policy_index)
        if action_obj.is_grasp:
            self.non_episodic_file_ids.append(action_obj.file_id)
        if action_obj.policy_index==0:
            '''1) action is sampled from the stochastic policy'''
            self.append_to_policy_buffer(action_obj)
            '''task in process'''
            self.effort_penalty(action_obj)
            self.is_end_of_episode.append(0)
        elif action_obj.policy_index==1:
            '''2) action is sampled from the deterministic policy'''
            '''task end successfully'''
            if len(self)>0 and self.is_end_of_episode[-1]==0:
                self.episodes_counter+=1
                self.last_ending_index = len(self) - 1
                '''reward +1'''
                self.positive_reward()
                self.is_end_of_episode[-1]=1
                self.update_advantages()
        elif action_obj.policy_index==2:
            '''3) action is sampled from the random policy'''
            '''task end with failure'''
            if len(self)>0 and self.is_end_of_episode[-1]==0:
                self.episodes_counter+=1
                self.last_ending_index = len(self) - 1
                '''reward -1'''
                self.negative_reward()
                self.is_end_of_episode[-1]=1
                self.update_advantages()


    def effort_penalty(self,action_obj:Action):
        assert len(self.rewards)==len(self.episodic_file_ids)-1
        '''penalize each action to reduce the effort needed to reach the target'''
        k = 2 if action_obj.is_synchronous else 1 # less penalty if the robot runs both arms at the same time
        self.rewards.append(-0.4/k)

    def positive_reward(self):
        '''add reward to the last action/s that leads to grasp the target in the current action'''
        if len(self.probs)>1:
            self.rewards[-1] += 1
            if self.is_synchronous[-1]:
                # add rewards to the last two actions if both arms are used in the last run
                self.rewards[-2]+=1

    def negative_reward(self):
        '''dedict reward to the last action/s that leads to the disappearance of the target in the new state'''
        if len(self.probs)>1:
            self.rewards[-1] -= 1
            if self.is_synchronous[-1]:
                # add rewards to the last two actions if both arms are used in the last run
                self.rewards[-2]-=1

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
        return advantage

    def pop_policy_buffer(self):
        if len(self.episodic_file_ids)>max_policy_buffer_size:
            for i in range(3):
                '''update episode counter'''
                if self.is_end_of_episode[0]==1:
                    self.episodes_counter -= 1
                '''pop oldest sample'''
                self.rewards.popleft()
                self.values.popleft()
                self.probs.popleft()
                self.action_indexes.popleft()
                self.point_indexes.popleft()
                self.advantages.popleft()
                self.is_end_of_episode.popleft()
                self.is_synchronous.popleft()
                self.episodic_file_ids.popleft()
                '''move the index of the last episode ending'''
                self.last_ending_index-=1

    def pop(self):
        self.pop_policy_buffer()

    def __len__(self):
        return len(self.episodic_file_ids)

if __name__ == "__main__":
    buffer=PPOMemory()
    x=len(buffer.file_ids)
    print(x)
