import torch
from colorama import Fore
from Configurations.dynamic_config import save_key, get_float

def confession_mask(label,prediction_,pivot_value=0.5):
    TP_mask = (label > pivot_value) & (prediction_ > pivot_value)
    FP_mask = (label < pivot_value) & (prediction_ > pivot_value)
    FN_mask = (label > pivot_value) & (prediction_ <= pivot_value)
    TN_mask = (label < pivot_value) & (prediction_ <= pivot_value)

    return TP_mask, FP_mask, FN_mask, TN_mask

class ConfessionMatrix():
    def __init__(self,TP=0,FP=0,FN=0,TN=0):
        '''confession matrix'''
        self.TP = TP
        self.FP = FP
        self.FN = FN
        self.TN = TN

    def update_confession_matrix(self,label,prediction_,pivot_value=0.5):
        '''masks'''
        TP_mask,FP_mask,FN_mask,TN_mask=confession_mask(label,prediction_,pivot_value=pivot_value)
        self.TP += (TP_mask).sum()
        self.FP += (FP_mask).sum()
        self.FN += (FN_mask).sum()
        self.TN += (TN_mask).sum()

        return TP_mask,FP_mask,FN_mask,TN_mask

    def correct_classification(self):
        return self.TP+self.TN
    def total_classification(self):
        return self.TP+self.TN+self.FP+self.FN
    def accuracy(self):
        return self.correct_classification()/self.total_classification()
    def recall(self):
        return self.TP/(self.TP+self.FN)
    def tpr(self):
        return self.recall()
    def fpr(self):
        return self.FP/(self.FP+self.TN)
    def precision(self):
        return self.TP/(self.TP+self.FP)

    def view(self):
        total=self.total_classification()
        print(f'TP={int((self.TP/total)*1000)/10}%, FP={int((self.FP/total)*1000)/10}%, FN={int((self.FN/total)*1000)/10}%, TN={int((self.TN/total)*1000)/10}%')

class MovingRate():
    def __init__(self,name='000'):
        self.name=name
        self.sub_name=name+'_moving_rate'
        self.balance_indicator=0.0

        self.decay_rate = 0.001

        self.counter = 0

        self.moving_rate=0.0

        '''load latest'''
        self.upload()


        self.set_decay_rate()
        self.truncate_factor=10/self.decay_rate

    @property
    def val(self):
        return self.moving_rate

    def update(self,value,pivot_value=0.5):
        if value>pivot_value:
            self.moving_rate=(1-self.decay_rate)*self.moving_rate+self.decay_rate
        else:
            self.moving_rate = (1 - self.decay_rate) * self.moving_rate
        self.counter+=1


    def set_decay_rate(self):
        x=0.1*(1-0.003)**self.counter
        self.decay_rate=max(x,0.005)
        self.truncate_factor = 10 / self.decay_rate

    def save(self):
        save_key(self.sub_name, self.moving_rate, section=self.name)
        save_key('counter_'+self.sub_name, self.counter, section=self.name)

    def upload(self):
        self.moving_rate=get_float(self.sub_name,section=self.name)
        self.counter = get_float('counter_'+self.sub_name, section=self.name)

    def view(self):
        self.set_decay_rate()
        print(Fore.LIGHTBLUE_EX,end='')
        truncated_moving_rate=float(int(self.moving_rate* self.truncate_factor)) / self.truncate_factor
        print(f'{self.sub_name} = {truncated_moving_rate}',end='')
        print(Fore.RESET)

class TrainingTracker:
    def __init__(self,name='',iterations_per_epoch=None,track_label_balance=False,track_prediction_balance=False):
        self.name=name
        self.iterations_per_epoch=iterations_per_epoch

        self.running_loss_ = None

        '''confession matrix'''
        self.confession_matrix=ConfessionMatrix()

        '''balance indicator'''
        self.label_balance_indicator=self.load_label_balance_indicator() if track_label_balance else None
        self.prediction_balance_indicator=self.load_prediction_balance_indicator() if track_prediction_balance else None

        self.loss_moving_average_=self.load_loss_moving_average()
        self.convergence=self.load_convergence()
        self.decay_rate=0.001

        self.counter=self.load_counter()
        self.last_loss=None

        self.set_decay_rate()

    def set_decay_rate(self):
        x=0.1*(1-0.003)**self.counter
        self.decay_rate=max(x,0.005)


    @property
    def loss(self):
        return None

    @loss.setter
    def loss(self,value):
        self.running_loss_= value if self.running_loss_ is None else self.running_loss_+ value
        self.loss_moving_average_ = self.decay_rate * value + self.loss_moving_average_ * (1 - self.decay_rate)
        if self.last_loss is not None:
            change=value-self.last_loss
            self.convergence=self.decay_rate * change + self.convergence * (1 - self.decay_rate)
        self.last_loss=value
        self.counter+=1


    def update_confession_matrix(self,label,prediction_,pivot_value=0.5):
        with torch.no_grad():
            TP_mask,FP_mask,FN_mask,TN_mask=self.confession_matrix.update_confession_matrix(label,prediction_,pivot_value)
            if self.label_balance_indicator is not None: self.update_label_balance_indicator(label)
            if self.prediction_balance_indicator is not None: self.update_prediction_balance_indicator(label)
            return TP_mask,FP_mask,FN_mask,TN_mask

    def load_label_balance_indicator(self):
        return get_float('label_balance_indicator',section=self.name)

    def load_prediction_balance_indicator(self):
        return get_float('prediction_balance_indicator',section=self.name)

    def load_loss_moving_average(self):
        return get_float('loss_moving_average',section=self.name)

    def load_convergence(self):
        return get_float('convergence',section=self.name)

    def load_counter(self):
        return get_float('counter',section=self.name)

    def update_label_balance_indicator(self,label,pivot_value=0.5):
        if label>pivot_value:
            self.label_balance_indicator=(1-self.decay_rate)*self.label_balance_indicator+self.decay_rate
        else:
            self.label_balance_indicator = (1 - self.decay_rate) * self.label_balance_indicator - self.decay_rate

    def update_prediction_balance_indicator(self,prediction,pivot_value=0.5):
        if prediction>pivot_value:
            self.prediction_balance_indicator=(1-self.decay_rate)*self.prediction_balance_indicator+self.decay_rate
        else:
            self.prediction_balance_indicator = (1 - self.decay_rate) * self.prediction_balance_indicator - self.decay_rate

    def print(self,swiped_samples=None):
        self.set_decay_rate()
        print(Fore.LIGHTBLUE_EX,f'statistics report for {self.name}')
        size=swiped_samples if swiped_samples is not None else self.iterations_per_epoch
        if self.running_loss_ is not None:
            if size is not None:
                print(f'Average loss = {self.running_loss_/size}')
            else:
                print(f'Running loss = {self.running_loss_}')

        print(f'Loss (moving average) = {self.loss_moving_average_}')

        if self.confession_matrix.total_classification()>0:
            self.confession_matrix.view()

        if self.label_balance_indicator is not None:
            print(f'label balance indicator = {self.label_balance_indicator}')

        if self.prediction_balance_indicator is not None:
            print(f'prediction balance indicator = {self.prediction_balance_indicator}')

        if self.convergence is not None:
            print(f'Convergence = {self.convergence}')


        print(Fore.RESET)

    def save(self):
        save_key('label_balance_indicator', self.label_balance_indicator, section=self.name)
        save_key('prediction_balance_indicator', self.prediction_balance_indicator, section=self.name)

        save_key('loss_moving_average', self.loss_moving_average_, section=self.name)
        save_key('convergence', self.convergence, section=self.name)

        save_key('counter', self.counter, section=self.name)


    def clear(self):
        self.running_loss_=0
        self.confession_matrix=ConfessionMatrix()
