from colorama import Fore
from Configurations.dynamic_config import save_key, add_to_value, get_value, get_float
from lib.loss.D_loss import binary_l1

class ConfessionMatrix():
    def __init__(self,TP=0,FP=0,FN=0,TN=0):
        '''confession matrix'''
        self.TP = TP
        self.FP = FP
        self.FN = FN
        self.TN = TN

    def update_confession_matrix(self,label,prediction_,pivot_value=0.5):
        self.TP += ((label > pivot_value) & (prediction_ > pivot_value)).sum()
        self.FP += ((label < pivot_value) & (prediction_ > pivot_value)).sum()
        self.FN += ((label > pivot_value) & (prediction_ <= pivot_value)).sum()
        self.TN += ((label < pivot_value) & (prediction_ <= pivot_value)).sum()

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
        print(f'TP={int((self.TP/total)*100)}%, FP={int((self.FP/total)*100)}%')
        print(f'FN={int((self.FN/total)*100)}%, TN={int((self.TN/total)*100)}%')

class MovingMetrics():
    def __init__(self,name='000',decay_rate=0.001):
        self.name=name

        self.balance_indicator=0.0

        '''moving rates'''
        self.decay_rate=decay_rate
        self.tpr=1.0
        self.tnr=1.0

        '''load latest'''
        self.upload()


    def update(self,label,prediction_,pivot_value=0.5):
        confession_matrix = ConfessionMatrix()
        confession_matrix.update_confession_matrix(label, prediction_, pivot_value)
        if confession_matrix.TP==1:
            self.tpr=(1-self.decay_rate)*self.tpr+self.decay_rate
        elif confession_matrix.FN==1:
            self.tpr = (1 - self.decay_rate) * self.tpr
        elif confession_matrix.TN==1:
            self.tnr=(1-self.decay_rate)*self.tnr+self.decay_rate
        else:
            self.tnr = (1 - self.decay_rate) * self.tnr

    def save(self):
        save_key("tpr", self.tpr, section=self.name)
        save_key("tnr", self.tnr, section=self.name)

    def upload(self):
        self.tpr=get_float('tpr',section=self.name)
        self.tnr=get_float('tnr',section=self.name)

    def view(self):
        print(Fore.LIGHTBLUE_EX)
        print(f'Moving true positive rate = {self.tpr}')
        print(f'Moving true negative rate = {self.tnr}')
        print(Fore.RESET)

class MovingRate():
    def __init__(self,name='000',decay_rate=0.001):
        self.name=name
        self.sub_name=name+'_moving_rate'
        self.balance_indicator=0.0

        self.decay_rate=decay_rate

        self.moving_rate=0.0

        self.truncate_factor=10/self.decay_rate


        '''load latest'''
        self.upload()

    def update(self,value,pivot_value=0.5):
        if value>pivot_value:
            self.moving_rate=(1-self.decay_rate)*self.moving_rate+self.decay_rate
        else:
            self.moving_rate = (1 - self.decay_rate) * self.moving_rate

    def save(self):
        save_key(self.sub_name, self.moving_rate, section=self.name)

    def upload(self):
        self.moving_rate=get_float(self.sub_name,section=self.name)

    def view(self):
        print(Fore.LIGHTBLUE_EX,end='')
        truncated_moving_rate=float(int(self.moving_rate* self.truncate_factor)) / self.truncate_factor
        print(f'{self.sub_name} = {truncated_moving_rate}',end='')
        print(Fore.RESET)

class TrainingTracker():
    def __init__(self,name='',iterations_per_epoch=None,samples_size=None,track_label_balance=False,track_prediction_balance=False):
        self.name=name
        self.iterations_per_epoch=iterations_per_epoch
        self.samples_size=samples_size

        self.running_loss = 0.

        '''confession matrix'''
        self.confession_matrix=ConfessionMatrix()
        self.labels_with_zero_loss = 0

        '''track_discrimination_loss'''
        self.positive_loss=0.0
        self.negative_loss=0.0

        '''balance indicator'''
        self.label_balance_indicator=self.load_label_balance_indicator() if track_label_balance else None
        self.prediction_balance_indicator=self.load_prediction_balance_indicator() if track_prediction_balance else None


    def update_cumulative_discrimination_loss(self,prediction,label,exponent=2.0):
        if label > 0.5: self.positive_loss+=binary_l1(prediction, label).item()**exponent
        else:self.negative_loss+=binary_l1(prediction, label).item()**exponent

    def update_confession_matrix(self,label,prediction_,pivot_value=0.5):
        self.confession_matrix.update_confession_matrix(label,prediction_,pivot_value)
        if self.label_balance_indicator is not None: self.update_label_balance_indicator(label)
        if self.prediction_balance_indicator is not None: self.update_prediction_balance_indicator(prediction_)


    def load_label_balance_indicator(self):
        return get_float('label_balance_indicator',section=self.name)

    def load_prediction_balance_indicator(self):
        return get_float('prediction_balance_indicator',section=self.name)

    def update_label_balance_indicator(self,label,pivot_value=0.5,decay_rate=0.001,use_momentum=False):
        adapted_decay_rate=max(decay_rate,self.label_balance_indicator*decay_rate) if use_momentum else decay_rate
        if label>pivot_value:
            self.label_balance_indicator=(1-adapted_decay_rate)*self.label_balance_indicator+adapted_decay_rate
        else:
            self.label_balance_indicator = (1 - adapted_decay_rate) * self.label_balance_indicator - adapted_decay_rate

    def update_prediction_balance_indicator(self,prediction,pivot_value=0.5,decay_rate=0.001,use_momentum=False):
        adapted_decay_rate=max(decay_rate,self.prediction_balance_indicator*decay_rate) if use_momentum else decay_rate
        if prediction>pivot_value:
            self.prediction_balance_indicator=(1-adapted_decay_rate)*self.prediction_balance_indicator+adapted_decay_rate
        else:
            self.prediction_balance_indicator = (1 - adapted_decay_rate) * self.prediction_balance_indicator - adapted_decay_rate

    def print(self):
        print(Fore.LIGHTBLUE_EX,f'statistics report for {self.name}')
        if self.iterations_per_epoch is not None:
            print(f'Average loss = {self.running_loss/self.iterations_per_epoch}')
        else:
            print(f'Running loss = {self.running_loss}')

        if self.confession_matrix.total_classification()>0:
            self.confession_matrix.view()

        if self.labels_with_zero_loss>0:
            print(f'Number of labels with zero loss = {self.labels_with_zero_loss}')

        if self.samples_size is not None:
            print(f'Total number of samples= {self.samples_size}')

        if self.positive_loss+self.negative_loss>0.0:
            print(f'Running positive loss = {self.positive_loss}')
            print(f'Running negative loss = {self.negative_loss}')

        if self.label_balance_indicator is not None:
            print(f'label balance indicator = {self.label_balance_indicator}')

        if self.prediction_balance_indicator is not None:
            print(f'prediction balance indicator = {self.prediction_balance_indicator}')

        print(Fore.RESET)

    def save(self):
        save_key('label_balance_indicator', self.label_balance_indicator, section=self.name)
        save_key('prediction_balance_indicator', self.prediction_balance_indicator, section=self.name)
