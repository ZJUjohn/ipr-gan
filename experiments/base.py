from abc import ABC, abstractmethod # 导入抽象基类和抽象方法
from experiments.util import Logger # 导入日志记录器
from tqdm import tqdm # 导入进度条
import importlib 
import os
import torch

class Experiment(ABC):
    '''
    这是一个实验的基类，所有实验都应该继承这个类。
    这个类定义了一些基本的方法和属性，用于配置设备、数据集、模型等。
    具体的实验类需要实现这些方法。
    这个类还提供了一些基本的功能，比如加载状态字典、训练模型、评估模型等。
    这个类使用了抽象基类（ABC）和抽象方法（abstractmethod），
    这意味着这个类不能被实例化，子类必须实现所有的抽象方法。
    '''

    def __init__(self, config):
        super(Experiment, self).__init__() # 调用父类的构造函数
        assert config is not None, '"config" is undefined' # assert语句用于检查条件是否为真，如果不为真则抛出异常
        self.config = config # 将配置文件赋值给实例变量
        self.logger = Logger(config) # 创建日志记录器
        config_path = os.path.join(config.log.path, 'config.yaml') # 设置配置文件路径
        with open(config_path, 'w') as f: # 打开配置文件
            f.write(config.to_yaml()) # 将配置文件写入磁盘

        self.init_step = 1 # 初始化步骤

        self.configure_device() # 配置设备

    @abstractmethod # 这是一个抽象方法，子类必须实现
    def configure_dataset(self): pass # pass表示不执行任何操作

    def configure_device(self):
        print('*** DEVICE ***')
        
        # 是否使用GPU
        use_gpu = self.config.resource.gpu # 从配置文件中获取是否使用GPU的设置
        has_gpu = torch.cuda.is_available() # 检查是否有可用的GPU
        if has_gpu and use_gpu:
            gpu_count = torch.cuda.device_count()
            gpu_count = min(gpu_count, self.config.resource.get('ngpu', 1))
            self.device = [torch.device(f'cuda:{i}') for i in range(gpu_count)]
        else:
            self.device = [torch.device('cpu'), ]

        if 'pretrain_iter' in self.config.hparam.to_dict():
            self.config.hparam.pretrain_iter //= len(self.device)
        self.config.hparam.iteration //= len(self.device)
        self.config.hparam.bsz *= len(self.device)

        for i, device in enumerate(self.device):
            print(f'{i}: {str(device).upper()}')
        print()

    @abstractmethod
    def configure_protection(self): pass

    @abstractmethod
    def configure_model(self): pass

    @abstractmethod
    def checkpoint(self): pass

    @abstractmethod
    def evaluate(self): pass
    
    # load_state_dict方法用于加载模型的状态字典
    def load_state_dict(self, state_dict, strict=False):
        assert hasattr(self, 'model'), '"model" not defined'
        self.model.load_state_dict(state_dict, strict=strict)
        if state_dict['step'] == 'END':
            total_iter = self.config.hparam.get('pretrain_iter', 0)
            total_iter += self.config.hparam.iteration
            self.init_step = total_iter
        else:
            self.init_step = state_dict['step'] + 1

    @abstractmethod
    def train(self, **kwargs): pass

    def start(self):
        pretrain_iteration = self.config.hparam.get('pretrain_iter', 0)
        iteration = self.config.hparam.iteration

        print('*** TRAINING ***')
        rng = range(self.init_step, pretrain_iteration + iteration + 1)
        for step in tqdm(rng):
            self._step = step
            self.train()
            self.checkpoint()

        self._step = 'end'
        self.checkpoint()
        print()