import os

from GraspAgent_2.utils.Multi_finger_hand_env import MojocoMultiFingersEnv

class CasiaHandEnv(MojocoMultiFingersEnv):
    def __init__(self,root):
        super().__init__(root=root)

        self.default_finger_joints = [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0.2, 0, -0.07, 1, 0, 0, 0, 0, 0, -0.07, 1, 0, 0, 0]


if __name__ == "__main__":
    root_dir = os.getcwd()  # current working directory


    env=CasiaHandEnv(root=root_dir + "/speed_hand/")

    env.manual_view()

