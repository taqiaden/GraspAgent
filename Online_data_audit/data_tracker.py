from Online_data_audit.dictionary_utils import load_dict, save_dict
from lib.statistics import moving_momentum

dictionary_directory=r'Online_data_audit/'

class DataTracker():
    def __init__(self,name='',list_size=3):
        self.name=name
        self.path=dictionary_directory+name+',pkl'
        self.dict=load_dict(self.path)

        self.list_size=list_size
        self.empty_list=[0]*list_size

        self.truncate_factor=1000

    def get_value(self,key):
        if key in self.dict:
            return self.dict[key]
        else:
            return self.empty_list

    def update_record(self,file_ids,losses):
        for j in range(len(file_ids)):
            '''old record'''
            old_record=self.get_value(file_ids[j])

            '''compute'''
            first_moment = moving_momentum(old_record[1], losses[j].item(), decay_rate=0.99, exponent=1)
            second_moment = moving_momentum(old_record[2], losses[j].item(), decay_rate=0.99, exponent=2)

            '''truncated update'''
            new_record = old_record
            new_record[0]=float(int(losses[j] * self.truncate_factor))/self.truncate_factor
            new_record[1] = float(int(first_moment * self.truncate_factor))/self.truncate_factor
            new_record[2] = float(int(second_moment * self.truncate_factor))/self.truncate_factor

            # print(new_record)

            '''update'''
            self.dict[file_ids[j]] = new_record

    def save(self):
        save_dict(self.dict, self.path)
