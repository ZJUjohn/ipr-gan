from scipy import linalg
from scipy.stats import entropy
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import functional as TF
# from tensorboardX import SummaryWriter
import numpy as np
import os

class Logger:
    def __init__(self, config):
        super(Logger, self).__init__()
        self.directory = config.log.path
        os.makedirs(self.directory, exist_ok=True)
        
        self.writer = SummaryWriter(self.directory)

        print('*** LOG ***')
        print(f'Path: {self.directory}\n')

    def write_scalar(self, metrics, index):
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, index)
        # self.writer.flush()

    def save_images(self, images, index):
        self.writer.add_images('sample', images, index)
        # self.writer.flush()

class ImageWriter:
    def __init__(self, directory):
        self.directory = directory
        self.counter = {}
    
    def __call__(self, tensor, prefix=None, suffix=None):
        prefix = prefix + '_' if prefix else ''
        suffix = '_' + suffix if suffix else ''
        key = prefix + suffix
        if not key in self.counter: self.counter[key] = 0
        hex_id = hex(self.counter[key])[2:].zfill(6)
        fpath = os.path.join(self.directory, f'{prefix}{hex_id}{suffix}.png')
        TF.to_pil_image(tensor).save(fpath)
        self.counter[key] += 1

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """ code taken from https://github.com/mseitzer/pytorch-fid
    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """
    # 这个函数用于计算两个高斯分布之间的Frechet距离
    # mu1, sigma1: 生成样本的均值和协方差矩阵
    # mu2, sigma2: 真实样本的均值和协方差矩阵
    # eps: 防止数值不稳定的微小值
    # 高斯分布是指具有均值和协方差矩阵的多元正态分布
    # Frechet距离是用于比较两个高斯分布的距离度量，计算公式为
    # d^2 = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    # 其中||mu1 - mu2||^2是均值之间的欧几里得距离的平方

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
             + np.trace(sigma2) - 2 * tr_covmean)

def calculate_inception_score(prob, splits=10):
    scores = []
    length = prob.shape[0]
    step = length // splits
    for k in range(0, length, step):
        part = prob[k:k+step, :]
        py = np.mean(part, axis=0)
        _score = [entropy(part[i, :], py) for i in range(part.shape[0])]
        _score = np.exp(np.mean(_score))
        scores.append(_score)
    return np.mean(scores), np.std(scores)
