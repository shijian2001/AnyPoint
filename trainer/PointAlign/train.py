"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb
import yaml



def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", default='./train_configs/MiniGPT_3D/stage_2.yaml',
                        help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()

    return args


def setup_seeds(config):
    from minigpt4.common.dist_utils import get_rank

    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    from minigpt4.common.registry import registry

    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def _resolve_cfg_path(raw_cfg_path: str) -> Path:
    cfg_path = Path(raw_cfg_path)
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
    return cfg_path


def _resolve_maybe_relative(path_value: str, cfg_dir: Path) -> str:
    p = Path(path_value)
    if p.is_absolute():
        return str(p)
    return str((cfg_dir / p).resolve())


def _set_path_env_from_cfg(cfg_path: Path) -> None:
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg_data = yaml.safe_load(f) or {}

    model_cfg = cfg_data.get("model", {}) if isinstance(cfg_data, dict) else {}
    datasets_cfg = cfg_data.get("datasets", {}) if isinstance(cfg_data, dict) else {}
    cfg_dir = cfg_path.parent

    os.environ["POINTALIGN_CFG_PATH"] = str(cfg_path)
    os.environ["POINTALIGN_CFG_DIR"] = str(cfg_dir)

    llama_model = model_cfg.get("llama_model")
    if isinstance(llama_model, str) and llama_model.strip():
        resolved = _resolve_maybe_relative(llama_model.strip(), cfg_dir)
        os.environ["POINTALIGN_LLAMA_MODEL"] = resolved
        os.environ["POINTALIGN_TOKENIZER_PATH"] = resolved

    pc_encoder_ckpt = model_cfg.get("pc_encoder_ckpt", "./params_weight/pc_encoder/point_model.pth")
    if isinstance(pc_encoder_ckpt, str) and pc_encoder_ckpt.strip():
        os.environ["POINTALIGN_PC_ENCODER_CKPT"] = _resolve_maybe_relative(pc_encoder_ckpt.strip(), cfg_dir)

    bert_base_path = model_cfg.get("bert_model_path", "./params_weight/bert-base-uncased")
    if isinstance(bert_base_path, str) and bert_base_path.strip():
        os.environ["POINTALIGN_BERT_BASE_PATH"] = _resolve_maybe_relative(bert_base_path.strip(), cfg_dir)

    if isinstance(datasets_cfg, dict):
        for dataset_cfg in datasets_cfg.values():
            build_info = dataset_cfg.get("build_info") if isinstance(dataset_cfg, dict) else None
            data_path = build_info.get("data_path") if isinstance(build_info, dict) else None
            if isinstance(data_path, str) and data_path.strip():
                os.environ["POINTALIGN_OBJAVERSE_DATA_PATH"] = _resolve_maybe_relative(data_path.strip(), cfg_dir)
                break


def main():
    args = parse_args()
    cfg_path = _resolve_cfg_path(args.cfg_path)
    _set_path_env_from_cfg(cfg_path)
    args.cfg_path = str(cfg_path)

    # Delay heavy imports until after config-derived env vars are set.
    import minigpt4.tasks as tasks
    from minigpt4.common.config import Config
    from minigpt4.common.dist_utils import init_distributed_mode
    from minigpt4.common.logger import setup_logger
    import minigpt4.common.optims  # noqa: F401
    import minigpt4.datasets.builders  # noqa: F401
    import minigpt4.models  # noqa: F401
    import minigpt4.processors  # noqa: F401
    import minigpt4.runners  # noqa: F401

    from minigpt4.common.utils import now

    job_id = now()
    cfg = Config(args)

    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)

    setup_logger()
    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    print("")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter {name} will be updated.")

    print()
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Number of trainable parameters: {num_trainable_params}")

    num_trainable_params = sum(p.numel() for p in model.llama_model.parameters() if p.requires_grad)
    print(f"    llama_model: Number of trainable parameters: {num_trainable_params}")

    num_trainable_params = sum(p.numel() for p in model.llama_proj.parameters() if p.requires_grad)
    print(f"    llama_proj: Number of trainable parameters: {num_trainable_params}")

    num_trainable_params = sum(p.numel() for p in model.llama_proj2.parameters() if p.requires_grad)
    print(f"    llama_proj2: Number of trainable parameters: {num_trainable_params}")

    num_trainable_params = sum(p.numel() for p in model.point_2_Qformer_proj.parameters() if p.requires_grad)
    print(f"    point_2_Qformer_proj: Number of trainable parameters: {num_trainable_params}")

    num_trainable_params = sum(p.numel() for p in model.Qformer.parameters() if p.requires_grad)
    print(f"    Qformer: Number of trainable parameters: {num_trainable_params}")

    if hasattr(model, 'alignment_projector'):
        num_trainable_params = sum(p.numel() for p in model.alignment_projector.parameters() if p.requires_grad)
        print(f"    alignment_projector: Number of trainable parameters: {num_trainable_params}")

    print("")

    if cfg.run_cfg.wandb_log:
        wandb.login()
        wandb.init(project="pointalign", name=cfg.run_cfg.job_name)
        wandb.watch(model)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()
