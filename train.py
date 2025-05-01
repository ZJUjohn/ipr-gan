from configs import Config # 导入配置文件，用于设置训练参数

import argparse # 用于解析命令行参数
import experiments # 导入实验模块，包含训练和评估的具体实现
import numpy as np # 用于生成随机数

import os # 用于处理文件和目录
import random # 用于生成随机数
import torch # 用于深度学习框架PyTorch

# 设置当前工作目录为脚本所在目录
parser = argparse.ArgumentParser(description='IPR-GAN training script')

# 添加命令行参数
ConfigFile = lambda path: Config.parse(path) # 定义一个函数，用于解析配置文件路径

parser.add_argument('-c', '--config', required=True, type=ConfigFile,
                                     metavar='PATH', help='Path to YAML config file') # 配置文件路径
args = parser.parse_args() # 解析命令行参数

def main(config):
    
    if not config.resource.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '' # 设置CUDA可见设备为空，表示不使用GPU

    Experiment = getattr(experiments, config.experiment) # 根据配置文件中的实验名称获取对应的实验类
    experiment = Experiment(config) # 实例化实验类

    ckpt_path = os.path.join(config.log.path, 'checkpoint.pt') # 设置检查点路径
    if os.path.exists(ckpt_path): # 如果检查点文件存在
        print('*** LOAD CHECKPOINT ***')
        state_dict = torch.load(ckpt_path) # 加载检查点文件
        experiment.load_state_dict(state_dict) # 将检查点的状态字典加载到实验中
        print(f'From Step: {experiment.init_step}\n') # 打印加载的步骤

    experiment.start() # 开始训练

    # save evaluation metrics into JSON file
    eval_metrics_fpath = os.path.join(config.log.path, 'metrics.json')
    experiment.evaluate(eval_metrics_fpath) # 评估模型并保存评估指标到JSON文件
    print(f'Result saved to: {eval_metrics_fpath}')

if __name__ == '__main__':
    config = args.config # 获取配置文件

    torch.manual_seed(config.seed) # 设置随机种子
    torch.backends.cudnn.deterministic = True # 设置CUDNN为确定性模式
    torch.backends.cudnn.benchmark = True # 设置CUDNN为基准模式
    np.random.seed(config.seed) # 设置NumPy随机种子
    random.seed(config.seed) # 设置Python随机种子

    main(config) # 调用主函数
