import yaml
from easydict import EasyDict
import os
from .logger import print_log

SHAPELLM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _resolve_config_path(path, base_dir):
    if os.path.isabs(path):
        return path

    if base_dir:
        candidate = os.path.join(base_dir, path)
        if os.path.exists(candidate):
            return candidate

    if SHAPELLM_ROOT:
        candidate = os.path.join(SHAPELLM_ROOT, path)
        if os.path.exists(candidate):
            return candidate

    return os.path.abspath(path)


def log_args_to_file(args, pre='args', logger=None):
    for key, val in args.__dict__.items():
        print_log(f'{pre}.{key} : {val}', logger=logger)


def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            print_log(f'{pre}.{key} = edict()', logger=logger)
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        print_log(f'{pre}.{key} : {val}', logger=logger)


def merge_new_config(config, new_config, base_dir=None):
    for key, val in new_config.items():
        if key == '_base_':
            base_path = _resolve_config_path(val, base_dir)
            with open(base_path, 'r') as f:
                try:
                    base_config = yaml.load(f, Loader=yaml.FullLoader)
                except:
                    base_config = yaml.load(f)
            merge_new_config(
                config,
                base_config,
                base_dir=os.path.dirname(os.path.abspath(base_path))
            )
            continue

        if isinstance(val, dict):
            if key not in config:
                config[key] = EasyDict()
            merge_new_config(config[key], val, base_dir=base_dir)
        else:
            config[key] = val
    return config


def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    cfg_dir = os.path.dirname(cfg_file)
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    merge_new_config(config=config, new_config=new_config, base_dir=cfg_dir)
    return config


def get_config(args, logger=None):
    if args.resume:
        cfg_path = os.path.join(args.experiment_path, 'config.yaml')
        if not os.path.exists(cfg_path):
            print_log("Failed to resume", logger=logger)
            raise FileNotFoundError()
        print_log(f'Resume yaml from {cfg_path}', logger=logger)
        args.config = cfg_path
    config = cfg_from_yaml_file(args.config)
    if not args.resume and args.local_rank == 0:
        save_experiment_config(args, config, logger)
    return config


def save_experiment_config(args, config, logger=None):
    config_path = os.path.join(args.experiment_path, 'config.yaml')
    os.system('cp %s %s' % (args.config, config_path))
    print_log(f'Copy the Config file from {args.config} to {config_path}', logger=logger)
