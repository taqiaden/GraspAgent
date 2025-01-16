import random

import numpy as np
import torch
from collections import deque
from Grasp_Agent_ import Action
from Online_data_audit.data_tracker2 import DataTracker2
from check_points.check_point_conventions import ModelWrapper
from models.policy_net import PolicyNet, policy_module_key
from records.training_satatistics import MovingRate

max_policy_buffer_size=50
max_quality_buffer_size=50
gamma=0.99
lamda=0.95
learning_rate=1e-4

def policy_loss(new_policy_probs,old_policy_probs,advantages,epsilon=0.2):
    ratio = new_policy_probs / old_policy_probs
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    objective = torch.min(ratio * advantages, clipped_ratio * advantages)
    return objective

class PPOMemory():
    def __init__(self):
        '''episodic buffer containers'''''
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
        self.episodic_buffer_file_ids=deque([])

        '''quality buffer containers'''''
        # This buffer will be used to train the quality head, specifically, we maintain track of grasp and shift success.
        self.is_grasp=deque([])
        self.use_gripper=deque([])
        self.grasp_result=deque([])
        self.shift_result=deque([])
        self.non_episodic_buffer_file_ids=deque([])

        '''track sampling rates'''
        self.g_p_sampling_rate=MovingRate('gripper_positive_sampling_rate')
        self.g_n_sampling_rate=MovingRate('gripper_negative_sampling_rate')
        self.s_p_sampling_rate=MovingRate('suction_positive_sampling_rate')
        self.s_n_sampling_rate=MovingRate('suction_negative_sampling_rate')

    def append_to_policy_buffer(self, action_obj:Action):
        self.actions_obj_list.append(action_obj)
        self.values.append(action_obj.value)
        self.probs.append(action_obj.prob)
        self.is_synchronous.append(action_obj.is_synchronous)
        self.action_indexes.append(action_obj.action_index)
        self.point_indexes.append(action_obj.point_index)
        self.episodic_buffer_file_ids.append(action_obj.file_id)

    def append_to_quality_buffer(self,action_obj:Action):
        self.is_grasp.append(action_obj.is_grasp)
        self.use_gripper.append(action_obj.use_gripper_arm)
        self.grasp_result.append(action_obj.grasp_result)
        self.shift_result.append(action_obj.shift_result)
        self.non_episodic_buffer_file_ids.append(action_obj.file_id)

    def push(self, action_obj:Action):
        if action_obj.policy_index==0:
            '''1) action is sampled from the stochastic policy'''
            self.append_to_policy_buffer(action_obj)
            '''task in process'''
            self.effort_penalty(action_obj)
            self.is_end_of_episode.append(0)
        elif action_obj.policy_index==1:
            '''2) action is sampled from the determinstic policy'''
            self.append_to_quality_buffer(action_obj)

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
            self.append_to_quality_buffer(action_obj)

            '''task end with failure'''
            if len(self)>0 and self.is_end_of_episode[-1]==0:
                self.episodes_counter+=1
                self.last_ending_index = len(self) - 1
                '''reward -1'''
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
        return advantage

    def pop_policy_buffer(self):
        if len(self.episodic_buffer_file_ids)>max_policy_buffer_size:
            for i in range(3):
                '''update episode counter'''
                if self.is_end_of_episode[0]==1:
                    self.episodes_counter -= 1
                '''pop oldest sample'''
                self.actions_obj_list.popleft()
                self.rewards.popleft()
                self.values.popleft()
                self.probs.popleft()
                self.action_indexes.popleft()
                self.point_indexes.popleft()
                self.advantages.popleft()
                self.is_end_of_episode.popleft()
                self.is_synchronous.popleft()
                self.episodic_buffer_file_ids.popleft()
                '''move the index of the last episode ending'''
                self.last_ending_index-=1

    def pop_quality_buffer(self):
        if len(self.non_episodic_buffer_file_ids) > max_quality_buffer_size:
            for i in range(3):
                self.is_grasp.popleft()
                self.use_gripper.popleft()
                self.grasp_result.popleft()
                self.shift_result.popleft()
                self.non_episodic_buffer_file_ids.popleft()

    def pop(self):
        self.pop_policy_buffer()
        self.pop_quality_buffer()

    def __len__(self):
        return len(self.actions_obj_list)

def sampling_p(sampling_rate,target_rate=0.25,exponent=10,k=0.75):
    if sampling_rate>target_rate:
        return ((target_rate - sampling_rate - 1)**exponent) *k
    else:
        return 1. - (sampling_rate/target_rate)*(1-k)


class PPOLearning():
    def __init__(self, model=None,buffer:PPOMemory=None,n_epochs=4,policy_clip=0.2, gamma=0.99, lamda=0.95,batch_size=5):

        self.gamma = gamma
        self.policy_clip = policy_clip
        self.lamda = lamda
        self.n_epochs = n_epochs
        self.batch_size=batch_size
        self.model_wrapper=self.init_model_wrapper()

        self.memory = buffer

        self.counter=0

    def init_model_wrapper(self):
        '''load  models'''
        policy_net = ModelWrapper(model=PolicyNet(),module_key=policy_module_key)

        '''optimizers'''
        policy_net.ini_adam_optimizer(learning_rate=learning_rate)
        return policy_net

    def step(self,model:PolicyNet(),buffer:PPOMemory,data_tracker:DataTracker2):
        self.model_wrapper.model=model
        self.memory=buffer
        self.model_wrapper.model.train(True)
        if self.counter%2==0:
            self.policy_learning()
        else:
            self.grasp_quality_learning(model,buffer,data_tracker)
        self.model_wrapper.model.eval()
        self.counter+=1
        return self.model_wrapper.model

    def sample_files_(self,buffer:PPOMemory,data_tracker:DataTracker2,batch_size,n_batches,online_ratio):
        total_size=int(batch_size*n_batches)
        online_size=int(total_size*online_ratio)
        replay_size=total_size-online_size

        '''sample from online pool'''
        indexes=random.sample(np.arange(len(buffer.non_episodic_buffer_file_ids),dtype=np.int64),online_size)
        online_ids=buffer.non_episodic_buffer_file_ids[indexes]

        '''sample from old experience'''
        g_p_pr=sampling_p(buffer.g_p_sampling_rate.moving_rate)
        g_n_pr=sampling_p(buffer.g_n_sampling_rate.moving_rate)
        s_p_pr=sampling_p(buffer.s_p_sampling_rate.moving_rate)
        s_n_pr=sampling_p(buffer.s_n_sampling_rate.moving_rate)
        replay_ids=data_tracker.selective_sampling(size=replay_size,sampling_probabilities=(g_p_pr,g_n_pr,s_p_pr,s_n_pr))

        return online_ids+replay_ids

    def grasp_quality_learning(self,model_wrapper:ModelWrapper(),buffer:PPOMemory,data_tracker:DataTracker2,batch_size=2,n_batches=4,online_ratio=0.3):

        file_ids=self.sample_files_( buffer, data_tracker, batch_size, n_batches, online_ratio)
        return
        pi = progress_indicator('Begin new training round: ', max_limit=len(self.data_loader))
        for i, batch in enumerate(self.data_loader, 0):
            depth,pose_7,pixel_index,score,normal,is_gripper,random_rgb= batch
            depth = depth.cuda().float()  # [b,1,480.712]
            pose_7 = pose_7.cuda().float()
            score = score.cuda().float()
            normal = normal.cuda().float()
            is_gripper = is_gripper.cuda().float()
            random_rgb = random_rgb.cuda().float().permute(0,3,1,2)

            b = depth.shape[0]

            '''zero grad'''
            self.policy_net.model.zero_grad()

            '''generated grasps'''
            with torch.no_grad():
                gripper_pose, suction_direction, griper_collision_classifier, suction_quality_classifier, shift_affordance_classifier \
                    ,background_class,depth_features= self.action_net.generator(depth.clone())
                target_object_mask = background_class.detach() <= 0.5
                '''process label'''
                for j in range(b):
                    pix_A = pixel_index[j, 0]
                    pix_B = pixel_index[j, 1]
                    gripper_pose[j, :, pix_A, pix_B] = pose_7[j]
                    suction_direction[j, :, pix_A, pix_B] = normal[j]

            griper_grasp_score,suction_grasp_score,shift_affordance_classifier,q_value=self.policy_net.model(random_rgb,depth_features,gripper_pose,suction_direction,target_object_mask)

            '''learning objectives'''
            gripper_grasp_loss = torch.tensor([0.],device=griper_grasp_score.device)
            suction_grasp_loss = torch.tensor([0.],device=griper_grasp_score.device)

            for j in range(b):
                pix_A = pixel_index[j, 0]
                pix_B = pixel_index[j, 1]
                label_ = score[j:j + 1]
                if is_gripper[j] == 1:
                    prediction_ = griper_grasp_score[j, :, pix_A, pix_B]
                    l=bce_loss(prediction_, label_)**2
                    # print(f'g {prediction_.item()}, {label_}')
                    gripper_grasp_loss += l
                    self.gripper_grasp_statistics.loss=l.item()
                    self.gripper_grasp_statistics.update_confession_matrix(label_,prediction_.detach())
                else:
                    prediction_ = suction_grasp_score[j, :, pix_A, pix_B]
                    l=bce_loss(prediction_, label_)**2
                    suction_grasp_loss += l
                    # print(f's {prediction_.item()}, {label_}')

                    self.suction_grasp_statistics.loss = l.item()
                    self.suction_grasp_statistics.update_confession_matrix(label_,prediction_.detach())


            loss = (gripper_grasp_loss+suction_grasp_loss)/b

            reversed_decay_= lambda scores:torch.clamp(torch.ones_like(scores)-scores,0).mean()
            decay_loss=reversed_decay_(griper_grasp_score)+reversed_decay_(suction_grasp_score)+reversed_decay_(shift_affordance_classifier)
            decay_loss*=0.3

            '''q value initialization'''
            q_value_loss=torch.tensor([0.],device=q_value.device)

            for j in range(b):
                target_object_mask_j=target_object_mask[j,0]
                permuted_q_value=q_value[j,0:2].permute(1,2,0)
                q_value_loss += (torch.clamp(
                    permuted_q_value[~target_object_mask_j] - permuted_q_value[target_object_mask_j].mean(), 0.) ** 2).mean()

                grasp_q_values=q_value[j,0:2]
                shift_q_values=q_value[j,2:]

                '''initialize lower q-value for shift compared to grasp'''
                q_value_loss+=(torch.clamp(shift_q_values-grasp_q_values.mean(),0.)**2).mean()

                x1=np.random.randint(0,711)
                x2=np.random.randint(0,711)

                max_index=max(x1,x2)
                min_index=min(x1,x2)

                '''grasp q-value initialization'''
                q_value_loss+=(torch.clamp(grasp_q_values[0,:,max_index].mean()-grasp_q_values[0,:,min_index].mean(),0.)**2).mean()
                q_value_loss+=(torch.clamp(grasp_q_values[1,:,min_index].mean()-grasp_q_values[1,:,max_index].mean(),0.)**2).mean()

                '''shift q-value initialization'''
                q_value_loss+=(torch.clamp(shift_q_values[0,:,max_index].mean()-shift_q_values[0,:,min_index].mean(),0.)**2).mean()
                q_value_loss+=(torch.clamp(shift_q_values[1,:,min_index].mean()-shift_q_values[1,:,max_index].mean(),0.)**2).mean()

            loss=loss+decay_loss+q_value_loss
            loss.backward()
            self.policy_net.optimizer.step()

            self.swiped_samples += b
            if i%100==0 and i!=0:
                self.export_check_points()
                self.view()

            pi.step(i)
        pi.end()

        self.export_check_points()
        self.view()
        self.clear( )

    def policy_learning(self):
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