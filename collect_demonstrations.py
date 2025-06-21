# -*- coding: utf-8 -*-
import numpy as np
from Configurations.dynamic_config import get_int, save_key
from lib.dataset_utils import configure_smbclient
from lib.image_utils import view_image
from process_perception import trigger_new_perception,  get_scene_depth, get_scene_RGB
from lib.dataset_utils import demonstrations_data,online_data2

configure_smbclient()

demonstrations_counter_key='demonstrations_counter'
online_data2=online_data2()
demonstrations_data=demonstrations_data()


def save_new_demonstration(rgb,depth,label):
    '''set unique identifier'''
    index = get_int(demonstrations_counter_key) + 1

    '''save labeled sample'''
    demonstrations_data.rgb.save_as_image(rgb, idx=index)
    demonstrations_data.depth.save(depth, idx=index)
    demonstrations_data.label.save(label, idx=index)

    print('Save new demonstration, index: ',index)

    '''update index'''
    save_key(demonstrations_counter_key, index)

label=np.array([np.nan]*10,dtype=object)
trigger_new_perception()
'''get modalities'''
depth=get_scene_depth()
rgb=get_scene_RGB()
# view_image(rgb)
# view_image(depth)
label[2]=1.

assert (label!=None).sum()>0,f'{label}'
save_new_demonstration(rgb,depth,label)
# view_colored_point_cloud(rgb,depth)

# [0]: 0: first contrastive scene, 1: second contrastive scene, None: not a contrastive
# [1]: 1: No grasp points, None: not defined
# [2]: 1: No suction points, None: not defined
# [3]: 1: Priority to grasp, None: not defined
# [4]: 1: Priority to suction, None: not defined
# [5]: 1: high grasp score,
# [6]: 1: No grasp, No suction