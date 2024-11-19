from colorama import Fore

from Configurations.dynamic_config import save_key, add_to_value, get_value, get_float

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
        print(f'TP={self.TP}, FP={self.FP}')
        print(f'FN={self.FN}, TN={self.TN}')

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

class TrainingTracker():
    def __init__(self,name='',iterations_per_epoch=None,samples_size=None):
        self.name=name
        self.iterations_per_epoch=iterations_per_epoch
        self.samples_size=samples_size

        self.running_loss = 0.

        '''confession matrix'''
        self.confession_matrix=ConfessionMatrix()

        self.labels_with_zero_loss = 0

    def update_confession_matrix(self,label,prediction_,pivot_value=0.5):
        self.confession_matrix.update_confession_matrix(label,prediction_,pivot_value)


    def print(self):
        print(Fore.LIGHTBLUE_EX)
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

        print(Fore.RESET)
