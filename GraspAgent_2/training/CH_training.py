import argparse
import configparser
import os
import torch.nn.functional as F
from GraspAgent_2.model.CH_model import CH_model_key, CH_D, CH_G
from GraspAgent_2.sim_hand_s.Casia_hand_env import CasiaHandEnv
from GraspAgent_2.training.abstract_training_module import AbstractGraspAgentTraining
from GraspAgent_2.training.sample_random_grasp import ch_pose_interpolation
from GraspAgent_2.utils.model_init import init_weights_he_normal
from GraspAgent_2.utils.quat_operations import  grasp_frame_to_quat, quat_between
from check_points.check_point_conventions import GANWrapper
from lib.cuda_utils import cuda_memory_report
import torch

def process_pose(target_point, target_pose, view=False):
    target_pose_ = target_pose.clone()
    target_point_ = target_point.cpu().numpy() if torch.is_tensor(target_point) else target_point
    delta=target_pose_[5:5 + 3].cpu().numpy()/30

    target_point_=target_point_+delta

    # quat = target_pose_[:4].cpu().tolist()
    alpha=target_pose_[:3]
    # alpha[-1]=torch.clip(alpha[-1],max=0.)

    # beta=half_way_unit_vector(target_pose_[3:5])
    beta=target_pose_[3:5]

    alpha = F.normalize(alpha, p=2, dim=0, eps=1e-8)
    beta = F.normalize(beta, p=2, dim=0, eps=1e-8)

    approach_ref=torch.tensor([0.866, -0.5, 0],device='cuda')

    default_quat = quat_between(approach_ref, torch.tensor([0., 0., -1.],device='cuda'))
    quat=grasp_frame_to_quat(alpha, beta, default_quat).cpu().tolist()

    fingers = target_pose[5+3:].cpu().numpy().tolist()

    fingers=[min(1,max(0,x+0.5)) for x in fingers]

    # transition=torch.clip(target_pose_[5+3:5+3+1],0,1)
    # transition=target_pose_[5+3:5+3+1]
    # transition = (transition.cpu().numpy()) / 10
    # projected_transition = quat_rotate_vector(quat, approach_ref.tolist())*transition[0]
    # projected_transition = quat_rotate_vector(quat, approach_ref.tolist())*0.03

    # approach = quat_rotate_vector(np.array(quat), np.array([0, 0, 1]))
    # projected_transition = approach * transition

    # shifted_point = (target_point_ + projected_transition).tolist()
    # shifted_point = (target_point_ ).tolist()

    assert all(x == x for x in quat), f"quat contains NaN, {quat}"
    assert all(x == x for x in fingers), f"fingers contains NaN, {fingers}"
    # assert all(x == x for x in shifted_point), f"shifted_point contains NaN, {shifted_point}"

    if view:
        print()
        print('quat: ',quat)
        print('fingers: ',fingers)
        # print('transition: ',transition)
        print('target_point_: ',target_point_)
        # print('projected_transition: ',projected_transition)
        # print('shifted_point: ',shifted_point)

    return quat,fingers,target_point_.tolist()

class TrainGraspGAN(AbstractGraspAgentTraining):
    def __init__(self, args,n_samples=None, epochs=1, learning_rate=5e-5):

        super().__init__(args=args, n_samples=n_samples, epochs=epochs ,model_key=CH_model_key,
                         test_mode=False,pose_interpolation=ch_pose_interpolation,
                         process_pose=process_pose,n_param=11)

        '''model wrapper'''
        self.gan = self.prepare_model_wrapper()


        root_dir = os.getcwd()  # current working directory

        self.sim_env = CasiaHandEnv(root=root_dir + "/GraspAgent_2/sim_hand_s/speed_hand/",max_obj_per_scene=10)

    def prepare_model_wrapper(self):
        '''load  models'''
        gan = GANWrapper(CH_model_key, CH_G, CH_D)
        gan.ini_models(train=True)

        # gan.generator.back_bone.apply(init_weights_he_normal)
        # gan.generator.CH_PoseSampler.apply(init_weights_he_normal)


        # gan.generator.back_bone2_.apply(init_weights_he_normal)
        # gan.generator.grasp_quality_.apply(init_weights_he_normal)
        # gan.generator.grasp_collision_.apply(init_weights_he_normal)
        # gan.generator.grasp_collision2.apply(init_weights_he_normal)
        # gan.generator.grasp_collision3.apply(init_weights_he_normal)

        sampler_params = []
        sampler_params += list(gan.generator.CH_PoseSampler.parameters())
        sampler_params += list(gan.generator.back_bone.parameters())

        policy_params = []
        policy_params += list(gan.generator.grasp_quality_.parameters())

        policy_params += list(gan.generator.grasp_collision_.parameters())
        policy_params += list(gan.generator.grasp_collision2.parameters())
        policy_params += list(gan.generator.grasp_collision3.parameters())
        policy_params += list(gan.generator.back_bone2_.parameters())
        # policy_params += list(gan.generator.back_bone3_.parameters())

        gan.critic_adam_optimizer(learning_rate=self.args.lr, beta1=0.9, beta2=0.999)
        # gan.critic_sgd_optimizer(learning_rate=self.args.lr*10,momentum=0.,weight_decay_=0.)
        # gan.generator_adam_optimizer(param_group=policy_params,learning_rate=self.args.lr, beta1=0.9, beta2=0.999)
        gan.generator_sgd_optimizer(param_group=policy_params,learning_rate=self.args.lr*10,momentum=0.,weight_decay_=0.)
        gan.sampler_optimizer = torch.optim.SGD(sampler_params, lr=self.args.lr*10,
                                               momentum=0)
        # gan.sampler_adam_optimizer(param_group=sampler_params,learning_rate=self.args.lr,beta1=0.9, beta2=0.999,weight_decay_=0.)

        # gan.sampler_optimizer =torch.optim.Adam(sampler_params, lr=self.args.lr   )

        return gan



def train_N_grasp_GAN(args,n=1):
    lr = 1e-5

    Train_grasp_GAN = TrainGraspGAN(args,n_samples=None)
    torch.cuda.empty_cache()

    for i in range(n):
        cuda_memory_report()
        Train_grasp_GAN.initialize(n_samples=None)
        # fix_weight_scales(Train_grasp_GAN.gan.generator.grasp_collision_)
        # exit()
        # scale_all_weights(Train_grasp_GAN.gan.generator.back_bone_,5)
        # Train_grasp_GAN.export_check_points()
        # exit()
        Train_grasp_GAN.begin(iterations=10)

def read_config(path):
    config = configparser.ConfigParser()
    config.read(path)
    return config

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="config.ini",
        help="Path to the config file"
    )

    parser.add_argument(
        "--load_last_optimizer",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Load last optimizer state (default: True). Use true/false."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )


    parser.add_argument(
        "--catch_exceptions",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Wrap the execution with try and except (default: True). Use true/false."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Normalize filename (avoid config.ini.ini)
    config_path = args.config
    if not config_path.lower().endswith(".ini"):
        config_path += ".ini"

    # Read config
    config = read_config(config_path)

    print("Config path:", os.path.abspath(config_path))
    print("load_last_optimizer:", args.load_last_optimizer)

    train_N_grasp_GAN(args,n=10000)
