from lib.dataset_utils import training_data, online_data

online_data=online_data()

pc_names=online_data.point_clouds.get_names()
for pc_name_ in pc_names:
    old_name = online_data.point_clouds.dir + pc_name_
    index=online_data.get_index(pc_name_)
    new_name = online_data.point_clouds.dir + index+online_data.point_clouds.sufix
    # print(old_name)
    # print(new_name)
    online_data.point_clouds.os.rename(old_name,new_name)