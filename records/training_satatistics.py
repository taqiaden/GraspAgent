import torch
from colorama import Fore
from Configurations.dynamic_config import save_key, get_float
from lib.report_utils import save_new_data_point


def truncate(x,k=10000):
    return int(x * k) / k

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

        self.epsilon = 0.00001

    def update_confession_matrix(self,label,prediction_,pivot_value=0.5):
        '''masks'''
        TP_mask,FP_mask,FN_mask,TN_mask=confession_mask(label,prediction_,pivot_value=pivot_value)
        self.TP += (TP_mask).sum()
        self.FP += (FP_mask).sum()
        self.FN += (FN_mask).sum()
        self.TN += (TN_mask).sum()


        return TP_mask,FP_mask,FN_mask,TN_mask
    @property
    def correct_classification(self):
        return self.TP+self.TN

    @property
    def total_classification(self):
        return self.TP+self.TN+self.FP+self.FN

    @property
    def accuracy(self):
        return self.correct_classification/(self.total_classification+self.epsilon)

    @property
    def recall(self):
        return self.TP/(self.TP+self.FN)

    @property
    def tpr(self):
        return self.recall

    @property
    def fpr(self):
        return self.FP/(self.FP+self.TN)

    @property
    def precision(self):
        return self.TP/(self.TP+self.FP)

    def view(self):
        total=self.total_classification
        print(f'TP={int((self.TP/total)*1000)/10}%, FP={int((self.FP/total)*1000)/10}%, FN={int((self.FN/total)*1000)/10}%, TN={int((self.TN/total)*1000)/10}%')

class MovingRate():
    def __init__(self,name='000'):
        self.name=name

        self.decay_rate = 0.001
        self.counter = 0
        self.moving_rate=0.0
        self.momentum=0.0
        self.convergence=0.0

        '''load latest'''
        self.upload()
        self.set_decay_rate()

        self.last_value=None


    @property
    def val(self):
        return self.moving_rate

    def update(self,value):
        with torch.no_grad():
            self.moving_rate=(1-self.decay_rate)*self.moving_rate+self.decay_rate*value
            self.momentum=(1-self.decay_rate)*self.momentum+self.decay_rate*(value**2)
            if self.last_value is not None:
                change = value - self.last_value
                self.convergence = self.decay_rate * change + self.convergence * (1 - self.decay_rate)
            self.last_value=value
            self.counter+=1


    def set_decay_rate(self):
        x=0.1*(1-0.0045)**self.counter
        self.decay_rate=max(x,0.001)

    def save(self):
        save_key('moving_rate_', self.moving_rate, section=self.name)
        save_key('counter_', self.counter, section=self.name)
        save_key('momentum_', self.momentum, section=self.name)
        save_key('convergence_', self.convergence, section=self.name)

        '''append to history records'''
        save_new_data_point(self.moving_rate, self.name + '_moving_rate.txt')
        save_new_data_point(self.counter, self.name + '_counter.txt')
        save_new_data_point(self.momentum, self.name + '_momentum.txt')
        save_new_data_point(self.convergence, self.name + '_convergence.txt')


    def upload(self):
        self.moving_rate=get_float('moving_rate_',section=self.name)
        self.counter = get_float('counter_', section=self.name)
        self.momentum = get_float('momentum_', section=self.name)
        self.convergence = get_float('convergence_', section=self.name)

    def view(self):
        self.moving_rate=truncate(self.moving_rate)
        self.momentum=truncate(self.momentum)
        self.convergence=truncate(self.convergence)

        self.set_decay_rate()
        print(Fore.LIGHTBLUE_EX,end='')
        print(f'{self.name} moving rate = {self.moving_rate}, momentum = {self.momentum}, decay rate = {self.decay_rate}, convergence={self.convergence}',end='')
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
        self.momentum=self.load_momentum()
        self.decay_rate=0.001

        self.counter=self.load_counter()
        self.last_loss=None

        self.set_decay_rate()
        self.tmp_counter=0

    def set_decay_rate(self):
        x=0.1*(1-0.0045)**self.counter
        self.decay_rate=max(x,0.001)
    @property
    def accuracy(self):
        return self.confession_matrix.accuracy

    @property
    def loss(self):
        return None

    @loss.setter
    def loss(self,value):
        self.running_loss_= value if self.running_loss_ is None else self.running_loss_+ value
        self.loss_moving_average_ = self.decay_rate * value + self.loss_moving_average_ * (1 - self.decay_rate)
        self.momentum = self.decay_rate * (value**2) + self.momentum * (1 - self.decay_rate)

        if self.last_loss is not None:
            change=value-self.last_loss
            self.convergence=self.decay_rate * change + self.convergence * (1 - self.decay_rate)
        self.last_loss=value
        self.counter+=1
        self.tmp_counter+=1


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

    def load_momentum(self):
        return get_float('momentum',section=self.name)

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

    def print(self):
        self.set_decay_rate()
        print(Fore.LIGHTBLUE_EX,f'statistics for {self.name}')
        # if self.running_loss_ is not None:
        #     self.running_loss_ = truncate(self.running_loss_, k=100000)
        #     print(f'Average loss = {self.running_loss_/self.tmp_counter}, Running loss = {self.running_loss_}')

        self.loss_moving_average_ = truncate(self.loss_moving_average_,k=100000)
        self.convergence = truncate(self.convergence,k=100000)
        self.momentum = truncate(self.momentum,k=100000)
        print(f'Moving average loss= {self.loss_moving_average_}, Convergence = {self.convergence}, momentum = {self.momentum}')

        if self.confession_matrix.total_classification>0:
            self.confession_matrix.view()

        if self.label_balance_indicator is not None:
            self.label_balance_indicator = truncate(self.label_balance_indicator)
            print(f'label balance indicator = {self.label_balance_indicator}')

        if self.prediction_balance_indicator is not None:
            self.prediction_balance_indicator = truncate(self.prediction_balance_indicator)
            print(f'prediction balance indicator = {self.prediction_balance_indicator}')

        print(Fore.RESET,'-------------------------------------------------------------------------')

    def save(self):
        save_key('label_balance_indicator', self.label_balance_indicator, section=self.name)
        save_key('prediction_balance_indicator', self.prediction_balance_indicator, section=self.name)
        save_key('loss_moving_average', self.loss_moving_average_, section=self.name)
        save_key('convergence', self.convergence, section=self.name)
        save_key('momentum', self.momentum, section=self.name)
        save_key('counter', self.counter, section=self.name)

        '''append to history records'''
        save_new_data_point(self.label_balance_indicator, self.name+'_label_balance_indicator.txt')
        save_new_data_point(self.prediction_balance_indicator, self.name+'_prediction_balance_indicator.txt')
        save_new_data_point(self.loss_moving_average_, self.name+'_loss_moving_average_.txt')
        save_new_data_point(self.convergence, self.name+'_convergence.txt')
        save_new_data_point(self.momentum, self.name+'_momentum.txt')
        save_new_data_point(self.counter, self.name+'_counter.txt')

        save_new_data_point(self.confession_matrix.TP, self.name+'_TP.txt')
        save_new_data_point(self.confession_matrix.FP, self.name+'_FP.txt')
        save_new_data_point(self.confession_matrix.TN, self.name+'_TN.txt')
        save_new_data_point(self.confession_matrix.FN, self.name+'_FN.txt')

    def clear(self):
        self.running_loss_=0
        self.confession_matrix=ConfessionMatrix()

if __name__ == '__main__':
    save_new_data_point(torch.Tensor([3, 4, 5, 6]), 'my_file.txt')
