import os
import torch
import inspect
import yaml
import shutil
from datetime import datetime
from loguru import logger


def load_yaml(config_path):
    filepath = os.path.join(os.getcwd(), config_path)
    with open(filepath, 'r', encoding='UTF-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    return configs


def copy_file_to_dir(src_file, dst_dir):
    """
    Copy source files to target directory
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    shutil.copy(src_file, dst_dir)


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_experiment(args, runner_name=None, exp_id=None):
    """
    Get filepath of calling script
    The code is from SimGCD
    https://github.com/CVMI-Lab/SimGCD/blob/main/util/general_utils.py
    """
    if runner_name is None:
        runner_name = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split(".")[-2:]

    root_dir = os.path.join(args.exp_root, *runner_name)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Either generate a unique experiment ID, or use one which is passed
    if exp_id is None:

        if args.exp_name is None:
            raise ValueError("Need to specify the experiment name")
        
        # Unique identifier for experiment
        now = '{}_{}_({}.{:02d}.{:02d}_{:02d}-{:02d})'.format(
            args.exp_name,
            args.category,
            datetime.now().year,
            datetime.now().month,
            datetime.now().day,
            datetime.now().hour,
            datetime.now().minute
        )

        log_dir = os.path.join(root_dir, 'log', now)

    else:

        log_dir = os.path.join(root_dir, 'log', f'{exp_id}')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
        
    logger.add(os.path.join(log_dir, 'log.txt'))
    args.logger = logger
    args.log_dir = log_dir

    # Instantiate directory to save models to
    model_root_dir = os.path.join(args.log_dir, 'checkpoints')
    if not os.path.exists(model_root_dir):
        os.mkdir(model_root_dir)

    args.model_dir = model_root_dir
    args.model_path = os.path.join(args.model_dir, 'model.pt')

    print(f'Experiment saved to: {args.log_dir}')

    hparam_dict = {}

    for k, v in vars(args).items():
        if isinstance(v, (int, float, str, bool, torch.Tensor)):
            hparam_dict[k] = v

    return args
