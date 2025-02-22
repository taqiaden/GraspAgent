
# GraspAgent

This repository is under construction.

Stay tuned for the final issue of GraspAgent 2.0.


## Conventions
    - Normal defines a vector emerge out of the surface
    - Approach direction defines a vector pointing to the surface
    - Approach = -1 * normal
    - The first three parameters of the generated pose for tyhe gripper are approach[0] and approach [1] and -1* approach[2]
    - The suction sampler outputs the normal direction
    - T_0 refers to a gripper head transformation matrix with zero penetration while T_d embeds the distance term
    - for any sequence we will always give the gripper the first index followed by the suction, e.g. if gripper grasp score locate at the (i) channel then the suction is located at (i+1) channel
    - if the action is shift, it is always saved in the first action object and no action is assigned to the second action object
    - Executing both arms at the same time is only allowed when both actions are grasp
    - A single run may include one action or two actions (moving both arms)
    - After execution, the robot rises three flags:
        - succeed: the plan has been executed completely
        - failed: Unable to execute part or full of the path
        - reset: path plan is found but execution terminated due to an error


## Acknowledgements
The collision detection unit and the ROS communication module are mainly handled by by Li Daheng, a senior member at RLIS Lab, CAISA, Bejing, China.

## Citations
If you find this project helpful for your research, please consider citing the following BibTeX entry.
```BibTex
@article{alshameri2024graspagent,
  title={GraspAgent 1.0: Adversarial Continual Dexterous Grasp Learning},
  author={Alshameri, Taqiaden and Wang, Peng and Li, Daheng and Wei, Wei and Duan, Haonan and Huang, Yayu and Alfarzaeai, Murad Saleh},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}

```

## Prerequisites
The following pakages version are used during the development of this repository:
```
python=3.10.16
torch=2.5.1
open3d=0.18.0
cuda=12.6
```

- The segmentation queries are forwarded to a separate repository where a mask of the target object/s is retrieved.
- We used the open source repository Grounded SAM 2.0 for the dedication and segmentation tasks:
    - [Grounded SAM 2.0](https://github.com/IDEA-Research/Grounded-SAM-2)

