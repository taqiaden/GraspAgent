
# GraspAgent

This repositry is under construction.

Stay tuned for the final code package of GraspAgnet 2.0.


# Conventions
    # normal is a vector emerge out of the surface
    # approach direction is a vector pointing to the surface
    # approach = -1 * normal
    # The first three parameters of the gripper pose are approach[0] and approach [1] and -1* approach[2]
    # the suction sampler outputs the normal direction
    # T_0 refers to a gripper head transformation matrix with zero penetration while T_d embeds the distance term
    # for any sequence we will always give the gripper the first index followed by the suction, e.g. if gripper grasp score locate at the (i) channel then the suction is located at (i+1) channel
    # if the action is shift, it is always saved in the first action object and no action is assigned to the second action object
    # executing both arms at the same time is only allowed when both actions are grasp
    # a single run may include one action or two actions (moving both arms)
    # After execution, the robot rises three flags:
    # succeed: the plan has been executed completely
    # failed: Unable to execute part or full of the path
    # reset: path plan is found but execution terminated due to an error


## Acknowledgements
The collision detection unit and the ROS communication module are mainly handled by by Li Daheng, a senior member at RLIS Lab, CAISA, Bejing, China.

## Contact


## Prerqusties

```http
```

