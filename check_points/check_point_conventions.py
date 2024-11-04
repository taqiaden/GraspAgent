import torch
from lib.models_utils import initialize_model, export_model_state
from lib.optimizer import export_optm, load_opt

class ModelWrapper():
    def __init__(self,model=None,optimizer=None,model_name='',optimizer_name=''):
        self.model=model
        self.optimizer=optimizer

        self.model_name=model_name
        self.optimizer_name=optimizer_name

        self.weight_decay = 0.000001
        self.learning_rate=1*1e-5

    '''model operations'''
    def ini_model(self,train=True,file_index=None):
        file_name = self.model_name if file_index is None else str(file_index) + self.model_name
        self.model = initialize_model(self.model, file_name)
        self.model.train(train)
        return self.model

    def export_model(self,file_index=None):
        file_name=self.model_name if file_index is None else str(file_index)+self.model_name
        export_model_state(self.model, file_name)

    '''optimizer operations'''
    def ini_adam_optimizer(self,learning_rate=None,file_index=None):
        file_name = self.optimizer_name if file_index is None else str(file_index) + self.optimizer_name
        if learning_rate is not None: self.learning_rate=learning_rate

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8,
                                     weight_decay=self.weight_decay)
        self.optimizer = load_opt(self.optimizer, file_name)
        return self.optimizer

    def ini_sgd_optimizer(self,learning_rate=None):
        if learning_rate is not None: self.learning_rate=learning_rate
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return self.optimizer

    def export_optimizer(self,file_index=None):
        file_name = self.optimizer_name if file_index is None else str(file_index) + self.optimizer_name
        export_optm(self.optimizer, file_name)


