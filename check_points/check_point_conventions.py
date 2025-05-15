import torch

from Configurations.config import weight_decay
from lib.models_utils import initialize_model, export_model_state, get_model_time_stamp
from lib.optimizer import export_optm, load_opt


class ModelWrapper():
    def __init__(self,model=None,optimizer=None,module_key=''):
        self.model=model
        self.optimizer=optimizer

        self.model_name=module_key+'_model'
        self.optimizer_name=module_key+'_optimizer'

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

        print(f'{self.model_name} check point exported')

    def model_time_stamp(self):
        return  get_model_time_stamp(self.model_name)

    '''optimizer operations'''
    def ini_adam_optimizer(self,params_group=None,learning_rate=None,file_index=None,beta1=0.9):
        file_name = self.optimizer_name if file_index is None else str(file_index) + self.optimizer_name
        if learning_rate is not None: self.learning_rate=learning_rate
        if params_group is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(beta1, 0.999), eps=1e-8,
                                         weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(params_group, betas=(beta1, 0.999),
                                              eps=1e-8,
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


class GANWrapper():
    def __init__(self,module_key,generator,critic=None):
        self.module_key=module_key
        self.critic=critic
        self.generator=generator
        self.critic_optimizer=None
        self.generator_optimizer=None
        self.learning_rate=1*1e-5

    '''model operations'''
    def ini_generator(self,train=True,wait=True):
        self.generator = initialize_model(self.generator(), self.module_key+'_generator',wait=wait)
        self.generator.train(train)

    def ini_critic(self,train=True,wait=True):
        self.critic = initialize_model(self.critic(), self.module_key+'_critic',wait=wait)
        self.critic.train(train)

    def ini_models(self,train=True):
        self.ini_generator(train)
        self.ini_critic(train)

    def export_models(self,file_index=None):
        export_model_state(self.generator, self.module_key+'_generator')
        export_model_state(self.critic,  self.module_key+'_critic')

    '''optimizer operations'''
    def critic_adam_optimizer(self,learning_rate=None,beta1=0.9):
        if learning_rate is not None: self.learning_rate=learning_rate

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate, betas=(beta1, 0.999), eps=1e-8,
                                     weight_decay=weight_decay)
        self.critic_optimizer = load_opt(self.critic_optimizer, self.module_key+'_critic_optimizer')

    def generator_adam_optimizer(self,param_group=None,learning_rate=None,beta1=0.9):
        if learning_rate is not None: self.learning_rate=learning_rate
        if param_group is None:
            self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(beta1, 0.999), eps=1e-8,
                                         weight_decay=weight_decay)
        else:
            self.generator_optimizer = torch.optim.Adam(param_group,
                                                        betas=(beta1, 0.999), eps=1e-8,
                                                        weight_decay=weight_decay)
        self.generator_optimizer = load_opt(self.generator_optimizer, self.module_key+'_generator_optimizer')

    def critic_sgd_optimizer(self, learning_rate=None):
        if learning_rate is not None: self.learning_rate = learning_rate

        self.critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=self.learning_rate,
                                                 weight_decay=weight_decay)
        self.critic_optimizer = load_opt(self.critic_optimizer, self.module_key + '_critic_optimizer')

    def critic_rmsprop_optimizer(self, learning_rate=None):
        if learning_rate is not None: self.learning_rate = learning_rate

        self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(), lr=self.learning_rate,
                                                 weight_decay=weight_decay)
        self.critic_optimizer = load_opt(self.critic_optimizer, self.module_key + '_critic_optimizer')

    def generator_sgd_optimizer(self, learning_rate=None):
        if learning_rate is not None: self.learning_rate = learning_rate

        self.generator_optimizer = torch.optim.SGD(self.generator.parameters(), lr=self.learning_rate,
                                                    weight_decay=weight_decay)
        self.generator_optimizer = load_opt(self.generator_optimizer, self.module_key + '_generator_optimizer')

    def export_optimizers(self):
        export_optm(self.generator_optimizer, self.module_key + '_generator_optimizer')
        export_optm(self.critic_optimizer, self.module_key + '_critic_optimizer')



