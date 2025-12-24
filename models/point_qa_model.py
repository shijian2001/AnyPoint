import os
import sys
import re
import torch
import argparse
import numpy as np
from typing import Dict, List, Any, Callable, Union, Sequence, Mapping
from collections import OrderedDict

from .base_qa_model import QAModel, QAModelInstance, load_point_cloud


point_qa_models = {

}

def list_point_qa_models() -> List[str]:
    return list(point_qa_models.keys())
    

class PointQAModel(QAModel):
    def __init__(
        self,
        model_name: str,
        checkpoint_path: str,
        prompt_name: str = "default",
        prompt_func: Callable = None,
        choice_format: str = 'letter',
        cache_path: str = None,
        device: str = None,
        **kwargs,
    ):
        if prompt_func is None:
            prompt_func = self._default_prompt_func

        super().__init__(
            model_name=model_name,
            prompt_name=prompt_name,
            prompt_func=prompt_func,
            choice_format=choice_format,
            enable_choice_search=True,
            cache_path=cache_path,
        )

        if model_name not in point_qa_models:
            raise ValueError(f"Unknown point QA model: {model_name}")

        model_class_name = point_qa_models[model_name]
        print(f"Loading {model_name}...")
        if isinstance(model_class_name, (tuple, list)):
            model_class_name = model_class_name[0]
        if isinstance(model_class_name, str):
            ModelClass = globals().get(model_class_name)
            if ModelClass is None:
                raise ValueError(f"Model class '{model_class_name}' not found in globals().")
        else:
            ModelClass = model_class_name
        runtime_kwargs = dict(kwargs)
        runtime_kwargs.setdefault('checkpoint_path', checkpoint_path)
        runtime_kwargs.setdefault('device', device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.model = ModelClass(**runtime_kwargs)

    @staticmethod
    def _default_prompt_func(question: str, options: List[str] = None) -> str:
        if not options:
            return question
        options_text = "\n".join(options)
        return f"{question}\n\n{options_text}\n\nPlease answer with the letter of the correct option."

    def _data_to_str(self, data: Dict[str, Any]) -> str:
        if 'point_cloud_path' in data:
            return data['point_cloud_path']
        if 'point_cloud' in data:
            pc = load_point_cloud(data['point_cloud'])
            return str(hash(pc.tobytes()))
        return "unknown"