"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import sys

# expose this embedded package under the canonical name `minigpt4`
# so internal absolute imports like `import minigpt4...` keep working
if __name__ != "minigpt4":
	sys.modules["minigpt4"] = sys.modules[__name__]

from omegaconf import OmegaConf

from .common.registry import registry

# use relative imports so the package works when embedded under PointQA_Eval
from .datasets.builders import *
from .models import *
from .processors import *
from .tasks import *


root_dir = os.path.dirname(os.path.abspath(__file__))
default_cfg = OmegaConf.load(os.path.join(root_dir, "configs/default.yaml"))

registry.register_path("library_root", root_dir)
repo_root = os.path.join(root_dir, "..")
registry.register_path("repo_root", repo_root)
cache_root = os.path.join(repo_root, default_cfg.env.cache_root)
registry.register_path("cache_root", cache_root)

registry.register("MAX_INT", sys.maxsize)
registry.register("SPLIT_NAMES", ["train", "val", "test"])
