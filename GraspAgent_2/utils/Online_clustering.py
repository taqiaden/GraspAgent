import os

import torch
import torch.nn.functional as F
from colorama import Fore

from GraspAgent_2.utils.quat_operations import quaternion_angular_distance, quaternion_pairwise_angular_distance, \
    combine_quaternions
def z_score(data):
    relative_weights=data/data.sum()
    std_=relative_weights.std()+1e-5
    mean_=relative_weights.mean()
    return (relative_weights-mean_)/std_

def pairwise_euclidean_distance(X):
    """
    Compute pairwise Euclidean distance for a tensor X of shape [n, m].

    Args:
        X: Tensor of shape [n, m]

    Returns:
        dist_matrix: Tensor of shape [n, n] with Euclidean distances
    """
    # Compute squared norms of each row: [n, 1]
    norms = (X ** 2).sum(dim=1, keepdim=True)  # [n,1]

    # Compute squared Euclidean distance: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
    dist_squared = norms + norms.T - 2 * X @ X.T

    # Clamp to zero to prevent negative values due to numerical errors
    dist_squared = torch.clamp(dist_squared, min=0.0)

    # Take square root
    dist_matrix = torch.sqrt(dist_squared)

    return dist_matrix


def pairwise_cosine_distance(X, eps=1e-8):
    """
    Compute pairwise cosine distance for a tensor X of shape [n, m].

    Args:
        X: Tensor of shape [n, m]
        eps: Small value for numerical stability

    Returns:
        dist_matrix: Tensor of shape [n, n] with cosine distances
    """
    # Normalize rows to unit vectors
    X_norm = F.normalize(X, p=2, dim=1, eps=eps)  # [n, m]

    # Compute pairwise cosine similarity
    sim_matrix = X_norm @ X_norm.T  # [n, n]

    # Cosine distance
    dist_matrix = 1.0 - sim_matrix

    return dist_matrix

class OnlingClustering():
    def __init__(self,key_name,number_of_centers=10,vector_size=3,decay_rate=0.01,use_euclidean_dist=False,inti_centers=None,is_quat=None,dist_threshold=None):
        self.N=number_of_centers
        self.key_name=key_name
        self.vector_size=vector_size
        self.decay_rate=decay_rate

        self.dist_threshold=dist_threshold

        self.save_path=self.key_name+'_online_centers'
        self.centers=self.load(inti_centers)

        self.update_rates=self.load_update_rates()
        self.is_quat=is_quat if is_quat is not None else False
        self.use_euclidean_dist=use_euclidean_dist

        if self.use_euclidean_dist: assert self.is_quat==False

        self.repulse_lr=1e-3

        self.metrics=self.load_metrics()
    def load_metrics(self):
        if os.path.exists(self.save_path+'_metrics'):
            return torch.load(self.save_path+'_metrics')
        else:
            return torch.zeros((10,),device='cuda')

    def load(self,inti_centers=None):
        if os.path.exists(self.save_path):
            return torch.load(self.save_path).cuda()
        else:
            return inti_centers

    def load_update_rates(self):
        if os.path.exists(self.save_path+'_update_rates'):
            return torch.load(self.save_path+'_update_rates')
        else:
            if self.centers is None: return None
            return torch.ones((self.centers.shape[0],),device='cuda')

    def save(self):
        torch.save(self.centers,self.save_path)
        torch.save(self.update_rates,self.save_path+'_update_rates')
        torch.save(self.metrics,self.save_path+'_metrics')


    def interpolate(self,old_center,new_center):
        if self.is_quat:
            updated_center=combine_quaternions(q1=old_center, q2=new_center, r1=(1 - self.decay_rate), r2=self.decay_rate)
        else:
            updated_center = (1 - self.decay_rate) * old_center + self.decay_rate * new_center
        return updated_center


    def centers_separation(self):
        if self.is_quat:
            dist=quaternion_pairwise_angular_distance(self.centers)
        elif self.use_euclidean_dist:
            dist=pairwise_euclidean_distance(self.centers)
        else:
            dist = pairwise_cosine_distance(self.centers)

        dist.fill_diagonal_(0.0)
        mean_dist_per_center = dist.sum(dim=1) / (self.centers.shape[0] - 1)
        # print(mean_per_center)

        dist.fill_diagonal_(float('inf'))
        min_dist_per_center, _ = dist.min(dim=1)
        # print(min_dist_per_center)
        return mean_dist_per_center,min_dist_per_center

    def conditional_drop(self):

        mean_dist_per_center, min_dist_per_center=self.centers_separation()

        update_rate_percentage=((self.update_rates[:self.centers.shape[0]] / self.update_rates[:self.centers.shape[0]].sum()) * 100)

        # mean_dist_z_score=z_score(mean_dist_per_center)
        # min_dist_z_score=z_score(min_dist_per_center)
        # update_rates_z_score=z_score(self.update_rates[:self.centers.shape[0]])

        # print(f'update_rates_z_score={update_rates_z_score.cpu().numpy()}')
        # print(f'min_dist_z_score={min_dist_z_score.cpu().numpy()}')
        # print(f'mean_dist_z_score={mean_dist_z_score.cpu().numpy()}')

        if self.centers.shape[0]>self.N and min_dist_per_center.min()<self.metrics[0]:
            index = torch.argmin(min_dist_per_center)
        elif (update_rate_percentage<0.1).any() :
            index=torch.argmin(update_rate_percentage)
            # print(f'relative_update_rate: ',update_rates_z_score)
        # elif (min_dist_z_score<-1).any() :
        #     index=torch.argmin(min_dist_z_score)
        #     # print(f'relative_mean_dist: ',min_dist_z_score)
        # elif (mean_dist_z_score<-1).any() :
        #     index=torch.argmin(mean_dist_z_score)
        #     # print(f'relative_min_dist: ',mean_dist_z_score)
        else:
            self.metrics[3]=min_dist_per_center.min()
            self.metrics[4]=min_dist_per_center.mean()

            return


        self.centers=torch.cat([self.centers[:index], self.centers[index+1:]], dim=0)
        self.update_rates=torch.cat([self.update_rates[:index], self.update_rates[index+1:]], dim=0)


        print(Fore.RED,f'Drop a center for {self.key_name} at index {index}',Fore.RESET)

        self.save()
    def step_decay_rate(self,index):
        step=torch.zeros_like(self.update_rates)
        step[index]=1

        self.update_rates = (1 - self.decay_rate) * self.update_rates + self.decay_rate*step

    def update(self,new_vector):
        if self.centers is None:
            self.centers=new_vector.clone()[None,:]
            self.update_rates=torch.ones_like(self.centers[:,0:1])
        elif self.centers.shape[0]<self.N:
            self.centers=torch.cat([self.centers,new_vector[None,:]],dim=0)
            self.update_rates=torch.cat([self.update_rates,self.update_rates.mean(dim=0,keepdim=True)],dim=0)
        else:
            self.conditional_drop()
            '''get min dist'''
            index,min_dist=self.get_closest_center(new_vector)

            if self.centers.shape[0]<self.N*1.5 and min_dist>min(self.metrics[0].item(),self.metrics[4].item()):
                self.centers = torch.cat([self.centers, new_vector[None, :]], dim=0)
                self.update_rates=torch.cat([self.update_rates,self.update_rates.mean(dim=0,keepdim=True)],dim=0)
                print(Fore.GREEN,f'add new center, found min dist {min_dist}', Fore.RESET)
            else:
                '''update'''
                old_center = self.centers[index]
                new_center = self.interpolate( old_center, new_vector)
                self.centers[index] = new_center
                self.step_decay_rate(index)

        self.save()

    def get_closest_center(self,new_values):
        if self.is_quat:
            dist = quaternion_angular_distance(self.centers, new_values)
            index=dist.argmin()
        elif self.use_euclidean_dist:
            dist = torch.norm(self.centers - new_values.unsqueeze(0), dim=1)  # [n]
            index = torch.argmin(dist)
        else:
            dist = -torch.matmul(self.centers, new_values)  # [n]
            index = torch.argmin(dist)

        self.metrics[0]=(1-self.decay_rate)*self.metrics[0]+self.decay_rate*dist.mean()
        self.metrics[1]=(1-self.decay_rate)*self.metrics[1]+self.decay_rate*dist.std()
        self.metrics[2]=(1-self.decay_rate)*self.metrics[2]+self.decay_rate*dist.min()

        return index,dist.min()


    def vector_repulsive_update(self):
        # centers: [n, d], assumed normalized
        S = self.centers @ self.centers.T  # [n,n], cosine similarity
        S.fill_diagonal_(0.0)  # zero out self-similarity

        # Compute directional repulsion
        # Each center pushed by sum of nearby centers weighted by their similarity
        repulse_vec = S @ self.centers / self.centers.shape[0]  # [n,d]

        # Update centers
        self.centers = self.centers - self.repulse_lr * repulse_vec

        # Re-normalize to unit length
        if self.is_quat :
            self.centers = self.centers / self.centers.norm(dim=1, keepdim=True)


    def view(self):
        if self.centers is not None:
            print(f'new_centers of {self.key_name}')
            print(self.centers)
            print('Update_rates %:')
            print(((self.update_rates/self.update_rates.sum())*100).tolist())
            print(f'metrics: {self.metrics}')
            if self.centers is not None and self.centers.shape[0]>=self.N:
                try:
                    self.conditional_drop()
                except Exception as e:
                    print(str(e))

                # self.vector_repulsive_update()


if __name__ == "__main__":
    c=torch.tensor([[ 0.5490, -0.0307,  0.8011,  0.2365],
        [ 0.4494,  0.4722,  0.6385, -0.4091],
        [-0.1455, -0.9442,  0.2194,  0.1978],
        [ 0.2060, -0.6383,  0.6059,  0.4278],
        [ 0.1978,  0.1919,  0.8704,  0.4079],
        [-0.4529, -0.2628, -0.8249,  0.2129],
        [ 0.0187,  0.1722, -0.0686,  0.9825],
        [ 0.3251, -0.3824,  0.7818,  0.3699],
        [ 0.2912,  0.8662, -0.0987,  0.3939],
        [-0.0302, -0.1547,  0.9845,  0.0776],
        [ 0.2581,  0.2944,  0.5110, -0.7652],
        [-0.0589,  0.6149,  0.6496, -0.4432],
        [-0.4704,  0.7571, -0.3366,  0.3036],
        [-0.5652,  0.7308, -0.3357,  0.1840],
        [-0.2978, -0.5544,  0.7098,  0.3166],
        [ 0.5564,  0.1399,  0.8110, -0.1148]], device='cuda:0')


    x=OnlingClustering(key_name= '_test', number_of_centers=16, vector_size=4, decay_rate=0.01,inti_centers=c,use_euclidean_dist=False,is_quat=True)
    x.update_rates=torch.tensor([12.490534782409668, 6.558574676513672, 7.700675964355469, 4.430337905883789, 5.927631378173828, 7.891852378845215, 4.233270645141602, 12.770530700683594, 0.007824794389307499, 4.7287678718566895, 8.21914005279541, 0.13321208953857422, 1.195000410079956, 0.03290238231420517, 6.711850166320801, 16.967897415161133]
,device='cuda:0')
    # x.update(torch.tensor([0.0962, 0.2420, 0.8080, -0.5285]).cuda())
    x.view()





