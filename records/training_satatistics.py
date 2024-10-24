from colorama import Fore

class TrainingTracker():
    def __init__(self,name='',iterations_per_epoch=None,samples_size=None):
        self.name=name
        self.iterations_per_epoch=iterations_per_epoch
        self.samples_size=samples_size

        self.running_loss = 0.

        '''confession matrix'''
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0

        self.labels_with_zero_loss = 0

    def update_confession_matrix(self,label,prediction_):
        self.TP += ((label > 0.5) & (prediction_ > 0.5)).sum()
        self.FP += ((label < 0.5) & (prediction_ > 0.5)).sum()
        self.FN += ((label > 0.5) & (prediction_ <= 0.5)).sum()
        self.TN += ((label < 0.5) & (prediction_ <= 0.5)).sum()

    def print(self):
        print(Fore.LIGHTBLUE_EX)
        if self.iterations_per_epoch is not None:
            print(f'Average loss = {self.running_loss/self.iterations_per_epoch}')
        else:
            print(f'Running loss = {self.running_loss}')

        print(f'TP={self.TP}, FP={self.FP}')
        print(f'FN={self.FN}, TN={self.TN}')
        print(f'Number of labels with zero loss = {self.labels_with_zero_loss}')

        if self.samples_size is not None:
            print(f'Total number of samples= {self.samples_size}')

        print(Fore.RESET)
