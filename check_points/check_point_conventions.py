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

    def ini_model(self,train=True):
        self.model = initialize_model(self.model, self.model_name)
        self.model.train(train)
        return self.model

    def ini_adam_optimizer(self,learning_rate=None):
        if learning_rate is not None: self.learning_rate=learning_rate

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8,
                                     weight_decay=self.weight_decay)
        self.optimizer = load_opt(self.optimizer, self.optimizer_name)
        return self.optimizer


    def export_optimizer(self):
        export_optm(self.optimizer, self.optimizer_name)

    def export_model(self):
        export_model_state(self.model, self.optimizer_name)
