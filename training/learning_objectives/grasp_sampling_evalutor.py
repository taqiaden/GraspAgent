

def gripper_sampler_loss(pixel_index,j,collision_state_,out_of_scope,firmness_state,generated_critic_score):
    pix_A = pixel_index[j, 0]
    pix_B = pixel_index[j, 1]


    # label_score = label_critic_score[j, 0, pix_A, pix_B]
    prediction_score = generated_critic_score[j, 0, pix_A, pix_B]

    # loss=torch.clamp(label_score - prediction_score - 1 * (1 - bad_state_grasp), min=0.)
    if collision_state_ or out_of_scope:
        weight=1
    # elif not firmness_state:
    #     weight=0.01
    else:
        weight=0.
    # weight=1 if bad_state_grasp else 0.1
    loss    =   - prediction_score*weight
    return loss