import torch


def gripper_sampler_loss(pixel_index,j,collision_state_list,out_of_scope_list,label_critic_score,generated_critic_score):
    pix_A = pixel_index[j, 0]
    pix_B = pixel_index[j, 1]
    collision_state_ = collision_state_list[j]
    out_of_scope = out_of_scope_list[j]
    bad_state_grasp = collision_state_ or out_of_scope

    label_score = label_critic_score[j, 0, pix_A, pix_B]
    prediction_score = generated_critic_score[j, 0, pix_A, pix_B]

    return torch.clamp(label_score - prediction_score - 1 * (1 - bad_state_grasp), min=0.)**2