import numpy as np
import torch
from colorama import Fore
from torch import nn
from torch.nn.functional import smooth_l1_loss
from Online_data_audit.data_tracker import sample_positive_buffer, gripper_grasp_tracker
from check_points.check_point_conventions import ModelWrapper
from dataloaders.background_seg_dl import BackgroundSegDataset
from interpolate_bin import estimate_object_mask
from lib.IO_utils import custom_print
from lib.Multible_planes_detection.plane_detecttion import bin_planes_detection, view_statistics
from lib.dataset_utils import online_data
from lib.depth_map import depth_to_point_clouds
from lib.loss.D_loss import binary_smooth_l1, binary_l1
from lib.loss.focal_loss import FocalLoss
from lib.report_utils import progress_indicator
from models.background_seg import BackgroundSegNet
from records.training_satatistics import TrainingTracker
from registration import transform_to_camera_frame, camera
from visualiztion import view_npy_open3d

instances_per_sample=16

module_key='background_seg'
training_buffer = online_data()

training_buffer.main_modality=training_buffer.depth
print=custom_print
BATCH_SIZE=1
learning_rate=5*1e-5
EPOCHS = 3
weight_decay = 0.000001
workers=2

sigmoid = nn.Sigmoid()
bce_loss=nn.BCELoss()

def manual_labeling(j,target_indexes,label,prediction,pc):
    target_index = target_indexes[j]
    threshold = 0.5
    if not training_buffer.label.exist(target_index):
        predicted_mask = label.detach().cpu().numpy() > threshold
        colors = np.zeros_like(pc)
        colors[predicted_mask, 0] += 1.
        view_npy_open3d(pc, color=colors)
        text = input('press (a) if label is accepted')
        if text == 'a':
            label = prediction.detach().cpu().numpy()
            label[label > threshold] = 1.
            label[label <= threshold] = 0.
            training_buffer.label.save(label, target_index)
            print(Fore.CYAN, f'Manual labeling of file with id {target_index}', Fore.RESET)
        else:
            predicted_mask = prediction.detach().cpu().numpy() > threshold
            colors = np.zeros_like(pc)
            colors[predicted_mask, 0] += 1.
            view_npy_open3d(pc, color=colors)
            text = input('press (a) if label is accepted')
            if text == 'a':
                label = prediction.detach().cpu().numpy()
                label[label > threshold] = 1.
                label[label <= threshold] = 0.
                training_buffer.label.save(label, target_index)
                print(Fore.CYAN, f'Manual labeling of file with id {target_index}', Fore.RESET)
def train_(file_ids):
    '''dataloader'''
    dataset = BackgroundSegDataset(data_pool=training_buffer,file_ids=file_ids)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=workers, shuffle=True)

    '''model'''
    background_seg=ModelWrapper(model=BackgroundSegNet,module_key=module_key)
    background_seg.ini_model()

    '''optimizer'''
    background_seg.ini_adam_optimizer(learning_rate=learning_rate)

    for epoch in range(EPOCHS):
        pi = progress_indicator('EPOCH {}: '.format(epoch + 1), max_limit=len(data_loader))
        statistics=TrainingTracker(name='',iterations_per_epoch=len(data_loader),samples_size=len(dataset))

        for i, batch in enumerate(data_loader, 0):
            depth,file_index= batch
            depth=depth.cuda().float()
            b = depth.shape[0]
            '''get predictions'''
            background_seg.model.zero_grad()
            predicted_seg_scores=background_seg.model(depth.clone()).permute(0,2, 3, 1)

            '''compute loss'''
            loss=0.
            c=0
            for j in range(b):
                pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)

                pc = transform_to_camera_frame(pc, reverse=True)
                prediction = predicted_seg_scores[j, :, :, 0][mask]

                # manual_mask=estimate_object_mask(pc)
                bin_mask=bin_planes_detection(pc,sides_threshold = 0.0035,floor_threshold=0.002,view=True,file_index=file_index[j])

                '''view'''
                # predicted_mask = prediction.detach().cpu().numpy() > 0.5
                # colors = np.zeros_like(pc)
                # colors[predicted_mask, 0] += 1.
                # view_npy_open3d(pc, color=colors)

                if bin_mask is None:
                    print('Unable to detect the bin')
                    continue
                else:
                    c+=1

                label=torch.from_numpy(bin_mask).to(predicted_seg_scores.device).float()
                positive_cls_mask=label>0.5
                positive_loss=bce_loss(prediction[positive_cls_mask], label[positive_cls_mask])
                negative_loss=bce_loss(prediction[~positive_cls_mask], label[~positive_cls_mask])
                loss += positive_loss+negative_loss

                '''view and labeling'''
                # manual_labeling(j, file_index, label, prediction, pc)
            if c==0 : continue
            loss=loss/c

            '''optimize'''
            loss.backward()
            background_seg.optimizer.step()

            statistics.running_loss += loss.item()
            pi.step(i)

            if i%100==0 and i!=0:        background_seg.export_model()


        pi.end()
        statistics.print()

        '''export check points'''
        background_seg.export_model()
        background_seg.export_optimizer()

if __name__ == "__main__":
    while True:
        # training_buffer.clear()
        # if len(training_buffer) == 0:
        #     load_training_buffer(size=10000)
        file_ids = sample_positive_buffer(size=10, dict_name=gripper_grasp_tracker)
        train_(file_ids)
        # training_buffer.clear()


