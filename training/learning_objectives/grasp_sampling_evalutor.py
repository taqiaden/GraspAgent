import torch


def gripper_sampler_loss(pixel_index,j,collision_state_,out_of_scope,firmness_state,generated_critic_score,background_class,critic_score_label,threshold=0.01):
    pix_A = pixel_index[j, 0]
    pix_B = pixel_index[j, 1]

    # label_score = label_critic_score[j, 0, pix_A, pix_B]
    prediction_score = generated_critic_score[j, 0, pix_A, pix_B]

    # loss=torch.clamp(label_score - prediction_score - 1 * (1 - bad_state_grasp), min=0.)
    if collision_state_ or out_of_scope :
        # return -prediction_score
        return  torch.clamp(critic_score_label-prediction_score,0)

    else:
        return prediction_score*0.
        # return torch.tensor([0],device=generated_critic_score.device)
        # mask_ = background_class[j, 0]<0.5
        # return (-1* generated_critic_score[j,0][mask_]).mean()