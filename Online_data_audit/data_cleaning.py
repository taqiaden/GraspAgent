import numpy as np
import torch
from label_unpack import LabelObj
from lib.dataset_utils import online_data
from lib.models_utils import initialize_model_state
from models.scope_net import scope_net_vanilla, suction_scope_model_state_path

online_data=online_data()

indexes=online_data.get_indexes()
model = initialize_model_state(scope_net_vanilla(in_size=6), suction_scope_model_state_path)
model.eval()

counter=0
total_counter=0
for index in indexes:
    label=online_data.label.load(index)
    label_obj = LabelObj(label=label)
    if label_obj.is_gripper or label_obj.failure: continue
    approach=label_obj.normal
    transition=label_obj.target_point

    input = np.concatenate([transition, approach])
    input=torch.from_numpy(input).to('cuda')[None,...].float()

    # print(input.shape)
    score=model(input)
    total_counter+=1
    if score<0.5:
        counter+=1
        print(score.item())

print(counter)
print(total_counter)
