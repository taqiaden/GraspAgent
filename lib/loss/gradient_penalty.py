import torch


def gradient_penalty(pc,critic,generated_pose_7,index,real,fake):
    batch_size,p=real.shape
    epsilon=torch.rand((batch_size,1)).repeat(1,p).to('cuda')
    interpolated_poses=real*epsilon+fake*(1-epsilon)
    interpolated_poses=interpolated_poses.detach().clone()
    interpolated_poses.requires_grad=True
    dense_pose_7 = generated_pose_7.detach().clone()
    # dense_pose_7.requires_grad=True

    for ii, j in enumerate(index):
        dense_pose_7[ii, :, j] = interpolated_poses[ii]
    critic_scores, realty_output = critic(pc, dense_pose_7)
    mixed_score = torch.stack([critic_scores[i, :, j] for i, j in enumerate(index)])

    gradient=torch.autograd.grad(inputs=interpolated_poses,
                                 outputs=mixed_score,
                                 grad_outputs=torch.ones_like(mixed_score),
                                 create_graph=True,
                                 retain_graph=True)[0]
    gradient=gradient.view(gradient.shape[0],-1)
    gradient_norm=gradient.norm(2,dim=1)

    gradient_penalty=torch.mean((gradient_norm-1)**2)


    return gradient_penalty
