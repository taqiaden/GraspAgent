
# GraspAgent

This repositry is under construction.

Stay tuned for the final code package of GraspAgnet 2.0.


# Conventions
    - Normal defines a vector emerge out of the surface
    - Approach direction defines a vector pointing to the surface
    - Approach = -1 * normal
    - The first three parameters of the gripper pose are approach[0] and approach [1] and -1* approach[2]
    - The suction sampler outputs the normal direction
    - T_0 refers to a gripper head transformation matrix with zero penetration while T_d embeds the distance term
    - for any sequence we will always give the gripper the first index followed by the suction, e.g. if gripper grasp score locate at the (i) channel then the suction is located at (i+1) channel
    - if the action is shift, it is always saved in the first action object and no action is assigned to the second action object
    - Executing both arms at the same time is only allowed when both actions are grasp
    - A single run may include one action or two actions (moving both arms)
    - After execution, the robot rises three flags:
        # succeed: the plan has been executed completely
        # failed: Unable to execute part or full of the path
        # reset: path plan is found but execution terminated due to an error


## Acknowledgements
The collision detection unit and the ROS communication module are mainly handled by by Li Daheng, a senior member at RLIS Lab, CAISA, Bejing, China.

## Contact


## Prerqusties

```
python=3.10.16
torch=2.5.1
open3d=0.18.0
cuda=12.6
```

- The object detetction and segmentation queries are forwarded to a seperate repository where a mask of the traget object/s is retrived.
- We used the open source repository Grounded SAM 2.0 for this task:
    - [Grounded SAM 2.0](https://github.com/IDEA-Research/Grounded-SAM-2)

