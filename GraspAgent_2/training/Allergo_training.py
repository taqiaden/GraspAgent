import argparse
import configparser
import os
import torch.nn.functional as F
from GraspAgent_2.model.allergo_model import Allergo_model_key, Allergo_G, Allergo_D
from GraspAgent_2.sim_dexee.allegro_hand_env import AllegroHandEnv
from GraspAgent_2.training.abstract_training_module import AbstractGraspAgentTraining
from GraspAgent_2.training.sample_random_grasp import allergo_pose_interpolation
from GraspAgent_2.utils.quat_operations import  grasp_frame_to_quat, quat_between
from check_points.check_point_conventions import GANWrapper
from lib.cuda_utils import cuda_memory_report
import torch

def process_fingers(target_pose_):
    fingers = target_pose_[5+3: ]

    ''''''
    fingers[1:2]*=1.806
    fingers[1:2]-=0.196

    fingers[2:3]*=1.884
    fingers[2:3]-=0.174

    fingers[3:4]*=1.847
    fingers[3:4]-=0.227

    ''''''
    fingers[5:6]*=1.806
    fingers[5:6]-=0.196

    fingers[6:7]*=1.884
    fingers[6:7]-=0.174

    fingers[7:8]*=1.847
    fingers[7:8]-=0.227

    ''''''

    fingers[9:10]*=1.806
    fingers[9:10]-=0.196

    fingers[10:11]*=1.884
    fingers[10:11]-=0.174

    fingers[11:12]*=1.847
    fingers[11:12]-=0.227

    '''thumb'''
    fingers[11:12]*=1.137
    fingers[11:12]+=0.263

    fingers[12:13]*=1.265
    fingers[12:13]-=0.105

    fingers[13:14]*=1.882
    fingers[13:14]-=0.162


    return fingers

def process_pose(target_point, target_pose, view=False):
    target_pose_ = target_pose.clone()
    target_point_ = target_point.cpu().numpy() if torch.is_tensor(target_point) else target_point
    delta=target_pose_[5:5 + 3].cpu().numpy()/15

    target_point_=target_point_+delta

    # quat = target_pose_[:4].cpu().tolist()
    alpha = target_pose_[:3]
    # alpha[-1]=torch.clip(alpha[-1],max=0.)

    # beta=half_way_unit_vector(target_pose_[3:5])
    beta = target_pose_[3:5]

    alpha = F.normalize(alpha, p=2, dim=0, eps=1e-8)
    beta = F.normalize(beta, p=2, dim=0, eps=1e-8)

    approach_ref = torch.tensor([0.0, 0., 1.0], device='cuda')

    default_quat = quat_between(approach_ref, torch.tensor([0., 0., -1.], device='cuda'))
    quat = grasp_frame_to_quat(alpha, beta, default_quat).cpu().tolist()

    fingers = process_fingers(target_pose_).cpu().tolist()

    assert all(x == x for x in quat), f"quat contains NaN, {quat, alpha, beta}"
    assert all(x == x for x in fingers), f"fingers contains NaN, {fingers}"

    if view:
        print()
        print('alpha: ', alpha)
        print('beta: ', beta)
        print('delta: ', delta)

        print('target_pose: ', target_pose)

        print('fingers: ', fingers)
        print('target_point_: ', target_point_)

    return quat, fingers, target_point_.tolist()

class TrainGraspGAN(AbstractGraspAgentTraining):
    def __init__(self, args,n_samples=None, epochs=1, learning_rate=5e-5):

        super().__init__(args=args, n_samples=n_samples, epochs=epochs ,model_key=Allergo_model_key,
                         test_mode=False,pose_interpolation=allergo_pose_interpolation,
                         process_pose=process_pose,n_param=24)

        '''model wrapper'''
        self.gan = self.prepare_model_wrapper()

        root_dir = os.getcwd()  # current working directory

        self.sim_env = AllegroHandEnv(root=root_dir + "/GraspAgent_2/sim_dexee/hands_and_objects/",max_obj_per_scene=10)

    def prepare_model_wrapper(self):
        '''load  models'''
        gan = GANWrapper(Allergo_model_key, Allergo_G, Allergo_D)
        gan.ini_models(train=True)

        # gan.generator.back_bone.apply(init_weights_he_normal)
        # gan.generator.AllergoPoseSampler_.apply(init_weights_he_normal)

        # gan.generator.back_bone2_.apply(init_weights_he_normal)
        # gan.generator.grasp_quality_.apply(init_weights_he_normal)
        # gan.generator.grasp_collision_.apply(init_weights_he_normal)
        # gan.generator.grasp_collision2.apply(init_weights_he_normal)
        # gan.generator.grasp_collision3.apply(init_weights_he_normal)

        sampler_params = []
        sampler_params += list(gan.generator.AllergoPoseSampler_.parameters())
        sampler_params += list(gan.generator.back_bone.parameters())

        policy_params = []
        policy_params += list(gan.generator.grasp_quality_.parameters())

        policy_params += list(gan.generator.grasp_collision_.parameters())
        policy_params += list(gan.generator.grasp_collision2.parameters())
        policy_params += list(gan.generator.grasp_collision3.parameters())
        policy_params += list(gan.generator.back_bone2_.parameters())
        # policy_params += list(gan.generator.back_bone3_.parameters())

        # gan.critic_adam_optimizer(learning_rate=self.args.lr, beta1=0.9, beta2=0.999)
        gan.critic_sgd_optimizer(learning_rate=self.args.lr*10,momentum=0.,weight_decay_=0.)
        gan.generator_adam_optimizer(param_group=policy_params,learning_rate=self.args.lr, beta1=0., beta2=0.999)
        # gan.generator_sgd_optimizer(param_group=policy_params,learning_rate=self.args.lr*10,momentum=0.,weight_decay_=0.)
        gan.sampler_optimizer = torch.optim.SGD(sampler_params, lr=self.args.lr*10,
                                               momentum=0)

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
