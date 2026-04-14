import os
import torch
import numpy as np
from typing import Dict, List, Any, Callable, Union

from .base_qa_model import QAModel, QAModelInstance, load_point_cloud

point_qa_models = {
    "shapellm": ("ShapeLLM"),
    "pointllm": ("PointLLM"),
    "onellm": ("OneLLM"),
    "minigpt3d": ("MiniGPT3D"),
    "pointalign": ("PointAlign"),
    "greenplm": ("GreenPLM"),
}


def _get_point_cloud_input(data: Dict[str, Any]) -> Any:
    """Return in-memory point cloud first, otherwise fall back to its path."""
    point_cloud = data.get('point_cloud')
    if point_cloud is not None:
        return point_cloud
    return data.get('point_cloud_path')


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
        if options:
            options_text = "\n".join(options)
            return (
                "Answer the question based on the provided point cloud.\n"
                f"Question: {question}\n"
                f"{options_text}\n"
                # "Output only the answer option, such as: <answer>A</answer>.\n"
                # "Can you see the point cloud?"
            )
        return f"Answer the question based on the provided point cloud.\nQuestion: {question}\nOutput only the answer."

    def _data_to_str(self, data: Dict[str, Any]) -> str:
        if 'point_cloud_path' in data:
            return data['point_cloud_path']
        if 'point_cloud' in data:
            pc = load_point_cloud(data['point_cloud'])
            return str(hash(pc.tobytes()))
        return "unknown"


class ShapeLLM(QAModelInstance):
    def __init__(self, **kwargs):
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = kwargs.get('checkpoint_path')
        if self.model_path is None:
            raise ValueError("ShapeLLM requires checkpoint_path")
        
        self.model_base = kwargs.get('model_base')
        self.conv_mode = kwargs.get('conv_mode', 'llava_v1')
        self.temperature = kwargs.get('temperature', 0.2)
        self.top_k = kwargs.get('top_k', 1)
        self.top_p = kwargs.get('top_p', None)
        self.num_beams = kwargs.get('num_beams', 1)
        self.max_new_tokens = kwargs.get('max_new_tokens', 2048)

        try:
            from models.dependence.shapellm.llava.utils import disable_torch_init
            from models.dependence.shapellm.llava.model.builder import load_pretrained_model
            from models.dependence.shapellm.llava.mm_utils import tokenizer_point_token, get_model_name_from_path, load_pts, process_pts
            from models.dependence.shapellm.llava.constants import POINT_TOKEN_INDEX, DEFAULT_POINT_TOKEN, DEFAULT_PT_START_TOKEN, DEFAULT_PT_END_TOKEN
            from models.dependence.shapellm.llava.conversation import conv_templates, SeparatorStyle

            self.tokenizer_point_token = tokenizer_point_token
            self.load_pts = load_pts
            self.process_pts = process_pts
            self.POINT_TOKEN_INDEX = POINT_TOKEN_INDEX
            self.DEFAULT_POINT_TOKEN = DEFAULT_POINT_TOKEN
            self.DEFAULT_PT_START_TOKEN = DEFAULT_PT_START_TOKEN
            self.DEFAULT_PT_END_TOKEN = DEFAULT_PT_END_TOKEN
            self.conv_templates = conv_templates
            self.SeparatorStyle = SeparatorStyle
        except ImportError as exc:
            raise ImportError("ShapeLLM dependencies missing. Please ensure 'llava' and related packages are installed and importable as a Python package (e.g. in models.dependence.llava)") from exc

        disable_torch_init()
        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.context_len = load_pretrained_model(
            self.model_path, self.model_base, model_name
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def _prepare_point_cloud(self, point_cloud: Union[np.ndarray, torch.Tensor, str], point_path: str = None):
        if point_path:
            point_cloud = point_path
        point = load_point_cloud(point_cloud)
        
        pts_tensor = self.process_pts(point, self.model.config).unsqueeze(0)
        
        return pts_tensor.to(self.device, dtype=torch.float16)

    def qa(self, data: Dict[str, Any], prompt: str) -> str:
        point_cloud = data.get('point_cloud')
        point_path = data.get('point_cloud_path')
        if point_cloud is None and point_path is None:
            raise ValueError('Point cloud is required for ShapeLLM evaluation')

        if self.model.config.mm_use_pt_start_end:
            qs = self.DEFAULT_PT_START_TOKEN + self.DEFAULT_POINT_TOKEN + self.DEFAULT_PT_END_TOKEN + '\n' + prompt
        else:
            qs = self.DEFAULT_POINT_TOKEN + '\n' + prompt

        conv = self.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = self.tokenizer_point_token(
            full_prompt, self.tokenizer, self.POINT_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(self.device)
        
        pts_tensor = self._prepare_point_cloud(point_cloud, point_path)
        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                points=pts_tensor,
                do_sample=self.temperature > 0 and self.num_beams == 1,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                use_cache=True
            )

        input_token_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        return outputs.strip()


class PointLLM(QAModelInstance):
    def __init__(self, **kwargs):
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = kwargs.get('checkpoint_path')
        if self.model_path is None:
            raise ValueError("PointLLM requires checkpoint_path")
        
        self.conv_mode = kwargs.get('conv_mode', 'vicuna_v1_1')
        self.temperature = kwargs.get('temperature', 0.2)
        self.top_p = kwargs.get('top_p', None)
        self.num_beams = kwargs.get('num_beams', 1)
        self.max_new_tokens = kwargs.get('max_new_tokens', 512)

        from models.dependence.pointllm.model import PointLLMLlamaForCausalLM  
        from models.dependence.pointllm.conversation import conv_templates 
        from models.dependence.pointllm.utils import disable_torch_init  
        from models.dependence.pointllm.data import pc_norm 
        from transformers import AutoTokenizer

        self.conv_templates = conv_templates
        self.pc_norm = pc_norm
        
        disable_torch_init()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = PointLLMLlamaForCausalLM.from_pretrained(self.model_path).to(self.device)
        self.model.eval()
        self.model.initialize_tokenizer_point_backbone_config(self.tokenizer, device=self.device, fix_llm=True)

    def _prepare_point_cloud(self, point_cloud: Union[np.ndarray, torch.Tensor, str]) -> torch.Tensor:
        pc = load_point_cloud(point_cloud)
        if isinstance(pc, torch.Tensor):
            pc = pc.cpu().numpy()
        
        pc = self.pc_norm(pc)
        return torch.from_numpy(pc).float().to(self.device)

    def qa(self, data: Dict[str, Any], prompt: str) -> str:
        point_cloud = _get_point_cloud_input(data)
        if point_cloud is None:
            raise ValueError('Point cloud is required for PointLLM evaluation')

        conv = self.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = self.tokenizer(prompt_text, return_tensors='pt').input_ids.to(self.device)
        point_tensor = self._prepare_point_cloud(point_cloud)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                point_clouds=point_tensor.unsqueeze(0),
                do_sample=self.temperature > 0 and self.num_beams == 1,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                use_cache=True
            )

        response = self.tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()

        return response


class MiniGPT3D(QAModelInstance):
    def __init__(self, **kwargs):
        # CRITICAL: Import package root FIRST to establish sys.modules["minigpt4"] alias
        # This must happen before ANY imports from minigpt4 submodules
        import models.dependence.minigpt3d.minigpt4 as minigpt4_pkg
        
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg_path = kwargs.get('cfg_path')
        if self.cfg_path is None:
            raise ValueError("MiniGPT-3D requires --cfg_path pointing to the config yaml file")
        
        self.max_new_tokens = kwargs.get('max_new_tokens', 150)
        self.min_length = kwargs.get('min_length', 10)
        self.num_beams = kwargs.get('num_beams', 2)
        self.top_p = kwargs.get('top_p', 0.7)
        self.repetition_penalty = kwargs.get('repetition_penalty', 1.0)
        self.length_penalty = kwargs.get('length_penalty', 1.0)
        self.temperature = kwargs.get('temperature', 0.2)
        self.do_sample = kwargs.get('do_sample', False)

        from minigpt4.common.eval_utils import init_model, prepare_texts
        from minigpt4.conversation.conversation import CONV_VISION
        
        self.prepare_texts = prepare_texts
        self.conv_temp = CONV_VISION.copy()
        # self.conv_temp.system = ""

        gpu_id = 0
        if isinstance(self.device, str) and self.device.startswith('cuda'):
            if ':' in self.device:
                gpu_id = int(self.device.split(':')[1])
        
        class Args:
            def __init__(self, cfg_path, gpu_id):
                self.cfg_path = cfg_path
                self.gpu_id = gpu_id
                self.options = None
        
        args = Args(self.cfg_path, gpu_id)
        self.model = init_model(args)
        self.model.eval()

    def _prepare_point_cloud(self, point_cloud: Union[np.ndarray, torch.Tensor, str]) -> torch.Tensor:
        pc = load_point_cloud(point_cloud)
        if isinstance(pc, torch.Tensor):
            pc = pc.cpu().numpy()
        
        return torch.from_numpy(pc).float().unsqueeze(0).to(self.device)

    def qa(self, data: Dict[str, Any], prompt: str) -> str:
        point_cloud = _get_point_cloud_input(data)
        if point_cloud is None:
            raise ValueError('Point cloud is required for MiniGPT-3D evaluation')

        point_tensor = self._prepare_point_cloud(point_cloud)
        texts = self.prepare_texts([prompt], self.conv_temp)

        with torch.inference_mode():
            answers = self.model.generate(
                point_tensor,
                texts,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                min_length=self.min_length,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                length_penalty=self.length_penalty,
                temperature=self.temperature,
                do_sample=self.do_sample
            )

        answer = answers[0].lower().replace('<unk>', '').strip()
        answer = answer.split('###')[0]
        answer = answer.split('Assistant:')[-1].strip()
        
        return answer


class PointAlign(QAModelInstance):
    def __init__(self, **kwargs):
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg_path = kwargs.get(
            'cfg_path',
            '/home/wangxingjian/PointQA_Eval/models/dependence/pointalign/eval_configs/benchmark_evaluation_paper.yaml',
        )
        self.weights_root = kwargs.get('weights_root', '/home/wangxingjian/model/pointalign')
        self.llama_model_path = kwargs.get('llama_model_path') or os.path.join(self.weights_root, 'Phi_2')
        self.bert_base_uncased_path = kwargs.get('bert_base_uncased_path') or os.path.join(
            self.weights_root, 'bert-base-uncased'
        )
        self.pc_encoder_path = kwargs.get('pc_encoder_path') or os.path.join(
            self.weights_root, 'pc_encoder', 'point_model.pth'
        )
        self.pretrain_ckpt = kwargs.get('pretrain_ckpt') or os.path.join(
            self.weights_root, 'pointalign', 'pretrain.pth'
        )
        self.finetune_ckpt = kwargs.get('finetune_ckpt') or os.path.join(
            self.weights_root, 'pointalign', 'finetune.pth'
        )
        self.qformer_pretrained_path = kwargs.get('qformer_pretrained_path')
        if self.qformer_pretrained_path is None:
            raise ValueError(
                "PointAlign requires qformer_pretrained_path. "
                "The released pretrain/finetune checkpoints are delta checkpoints."
            )

        self.max_new_tokens = kwargs.get('max_new_tokens', 150)
        self.min_length = kwargs.get('min_length', 10)
        self.num_beams = kwargs.get('num_beams', 2)
        self.top_p = kwargs.get('top_p', 0.7)
        self.repetition_penalty = kwargs.get('repetition_penalty', 1.0)
        self.length_penalty = kwargs.get('length_penalty', 1.0)
        self.temperature = kwargs.get('temperature', 0.2)
        self.do_sample = kwargs.get('do_sample', False)

        for name, path in [
            ('cfg_path', self.cfg_path),
            ('llama_model_path', self.llama_model_path),
            ('bert_base_uncased_path', self.bert_base_uncased_path),
            ('pc_encoder_path', self.pc_encoder_path),
            ('pretrain_ckpt', self.pretrain_ckpt),
            ('finetune_ckpt', self.finetune_ckpt),
            ('qformer_pretrained_path', self.qformer_pretrained_path),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"PointAlign {name} not found: {path}")

        os.environ['POINTALIGN_WEIGHTS_ROOT'] = self.weights_root

        import models.dependence.pointalign.minigpt4 as minigpt4_pkg
        from minigpt4.common.eval_utils import init_model, prepare_texts
        from minigpt4.common.config import Config
        from minigpt4.common.registry import registry
        from minigpt4.conversation.conversation import CONV_VISION

        self.prepare_texts = prepare_texts
        self.conv_temp = CONV_VISION.copy()

        gpu_id = 0
        if isinstance(self.device, str) and self.device.startswith('cuda') and ':' in self.device:
            gpu_id = int(self.device.split(':')[1])

        class Args:
            def __init__(self, cfg_path, gpu_id):
                self.cfg_path = cfg_path
                self.gpu_id = gpu_id
                self.options = None

        args = Args(self.cfg_path, gpu_id)
        cfg = Config(args)
        model_config = cfg.model_cfg
        model_config.llama_model = self.llama_model_path
        model_config.bert_base_uncased_path = self.bert_base_uncased_path
        model_config.pc_encoder_ckpt_path = self.pc_encoder_path
        model_config.ckpt = self.pretrain_ckpt
        model_config.second_ckpt = self.finetune_ckpt
        model_config.qformer_pretrained_path = self.qformer_pretrained_path
        model_cls = registry.get_model_class(model_config.arch)
        self.model = model_cls.from_config(model_config).to(self.device)
        self.model.eval()

    def _prepare_point_cloud(self, point_cloud: Union[np.ndarray, torch.Tensor, str]) -> torch.Tensor:
        pc = load_point_cloud(point_cloud)
        if isinstance(pc, torch.Tensor):
            pc = pc.cpu().numpy()

        return torch.from_numpy(pc).float().unsqueeze(0).to(self.device)

    def qa(self, data: Dict[str, Any], prompt: str) -> str:
        point_cloud = _get_point_cloud_input(data)
        if point_cloud is None:
            raise ValueError('Point cloud is required for PointAlign evaluation')

        point_tensor = self._prepare_point_cloud(point_cloud)
        texts = self.prepare_texts([prompt], self.conv_temp)

        with torch.inference_mode():
            answers = self.model.generate(
                point_tensor,
                texts,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                min_length=self.min_length,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                length_penalty=self.length_penalty,
                temperature=self.temperature,
                do_sample=self.do_sample
            )

        answer = answers[0].lower().replace('<unk>', '').strip()
        answer = answer.split('###')[0]
        answer = answer.split('Assistant:')[-1].strip()

        return answer


class GreenPLM(QAModelInstance):
    def __init__(self, **kwargs):
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = kwargs.get('model_path', './lava-vicuna_2024_4_Phi-3-mini-4k-instruct')
        self.lora_path = kwargs.get('lora_path')
        self.pretrain_mm_mlp_adapter = kwargs.get('pretrain_mm_mlp_adapter')
        self.pc_ckpt_path = kwargs.get('pc_ckpt_path')
        self.pc_encoder_type = kwargs.get('pc_encoder_type', 'small')
        self.get_pc_tokens_way = kwargs.get('get_pc_tokens_way', 'OM_Pooling')
        self.std = kwargs.get('std', 0.0)

        if self.pretrain_mm_mlp_adapter is None:
            raise ValueError("GreenPLM requires pretrain_mm_mlp_adapter")
        if self.pc_ckpt_path is None:
            raise ValueError("GreenPLM requires pc_ckpt_path")

        self.temperature = kwargs.get('temperature', 0.1)
        self.top_p = kwargs.get('top_p', 0.1)
        self.num_beams = kwargs.get('num_beams', 1)
        self.max_new_tokens = kwargs.get('max_new_tokens', 50)
        self.min_new_tokens = kwargs.get('min_new_tokens', 0)
        self.repetition_penalty = kwargs.get('repetition_penalty', 1.0)

        from models.dependence.greenplm.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from models.dependence.greenplm.llava.conversation import conv_templates
        from models.dependence.greenplm.llava.model.builder import load_pretrained_model
        from models.dependence.greenplm.llava.mm_utils import tokenizer_image_token, get_model_name_from_path

        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.conv_templates = conv_templates
        self.tokenizer_image_token = tokenizer_image_token

        model_name = get_model_name_from_path(self.model_path)
        if 'llava' not in model_name.lower():
            model_name = f"llava-{model_name}"
        self.tokenizer, self.model, self.context_len = load_pretrained_model(
            self.model_path,
            None,
            model_name,
            device_map={"": self.device},
            device=self.device,
        )

        class ModelArgs:
            def __init__(self, **kw):
                for k, v in kw.items():
                    setattr(self, k, v)

        model_args = ModelArgs(
            vision_tower=None,
            pretrain_mm_mlp_adapter=self.pretrain_mm_mlp_adapter,
            encoder_type='pc_encoder',
            std=self.std,
            pc_encoder_type=self.pc_encoder_type,
            pc_feat_dim=192,
            embed_dim=1024,
            group_size=64,
            num_group=512,
            pc_encoder_dim=512,
            patch_dropout=0.0,
            pc_ckpt_path=self.pc_ckpt_path,
            lora_path=self.lora_path,
            model_path=self.model_path,
            get_pc_tokens_way=self.get_pc_tokens_way
        )

        base_model = self.model
        target = base_model.get_model() if hasattr(base_model, 'get_model') else base_model
        target.initialize_other_modules(model_args)

        if self.lora_path:
            from peft import PeftModel  # type: ignore
            self.model = PeftModel.from_pretrained(
                base_model,
                self.lora_path,
                device_map={"": self.device},
                torch_dtype=torch.bfloat16,
                offload_folder=None,
                offload_state_dict=False,
            )
        else:
            self.model = base_model

        # dtype/device
        target = base_model.get_model() if hasattr(base_model, 'get_model') else base_model
        target.to(dtype=torch.bfloat16)
        if hasattr(target, 'vision_tower'):
            target.vision_tower.to(dtype=torch.float)
        self.model.to(self.device)
        self.model.eval()

    def _prepare_point_cloud(self, point_cloud: Union[np.ndarray, torch.Tensor, str]) -> torch.Tensor:
        pc = load_point_cloud(point_cloud)
        if isinstance(pc, torch.Tensor):
            pc = pc.cpu().numpy()
        
        if pc.shape[1] == 3:
            colors = np.zeros_like(pc, dtype=np.float32)
            pc = np.concatenate([pc, colors], axis=1)
        
        return torch.from_numpy(pc).float()

    def qa(self, data: Dict[str, Any], prompt: str) -> str:
        point_cloud = _get_point_cloud_input(data)
        if point_cloud is None:
            raise ValueError('Point cloud is required for GreenPLM evaluation')

        qs = self.DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv_mode = "phi3_instruct"
        conv = self.conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        qs = conv.get_prompt()

        input_ids = self.tokenizer_image_token(
            qs, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        point_tensor = self._prepare_point_cloud(point_cloud).unsqueeze(0).to(self.device, dtype=torch.bfloat16)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=point_tensor,
                do_sample=True if self.temperature > 0 and self.num_beams == 1 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=self.min_new_tokens,
                use_cache=True,
                repetition_penalty=self.repetition_penalty,
            )

        answer = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        answer = answer.replace("<|end|>", "").strip()
        
        return answer


class OneLLM(QAModelInstance):
    def __init__(self, **kwargs):
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        if not str(self.device).startswith('cuda'):
            raise ValueError("OneLLM evaluation currently requires CUDA.")
        self.checkpoint_path = kwargs.get('checkpoint_path')
        if self.checkpoint_path is None:
            raise ValueError("OneLLM requires checkpoint_path")

        self.dtype_name = kwargs.get('dtype', 'fp16')
        self.max_new_tokens = int(kwargs.get('max_new_tokens', 256))
        self.temperature = float(kwargs.get('temperature', 0.1))
        self.top_p = float(kwargs.get('top_p', 0.75))
        self.point_format = kwargs.get('point_format', 'xyzrgb')
        self.no_point_input = bool(kwargs.get('no_point_input', False))

        clip_pretrained_path = kwargs.get('clip_pretrained_path')
        clip_cache_dir = kwargs.get('clip_cache_dir')
        offline = bool(kwargs.get('offline', False))
        if clip_pretrained_path:
            os.environ['ONELLM_OPENCLIP_PRETRAINED'] = clip_pretrained_path
        if clip_cache_dir:
            os.environ['ONELLM_OPENCLIP_CACHE_DIR'] = clip_cache_dir
        if offline:
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'

        from fairscale.nn.model_parallel import initialize as fs_init
        import torch.distributed as dist
        from models.dependence.onellm.util.misc import default_tensor_type, setup_for_distributed
        from models.dependence.onellm.model.meta import MetaModel
        from models.dependence.onellm.data.conversation_lib import conv_templates
        from models.dependence.onellm.data.data_utils import pc_norm

        self.default_tensor_type = default_tensor_type
        self.conv_templates = conv_templates
        self.pc_norm = pc_norm

        master_port = int(kwargs.get('master_port', 23591))
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                rank=0,
                world_size=1,
                init_method=f'tcp://127.0.0.1:{master_port}',
            )
        if not fs_init.model_parallel_is_initialized():
            fs_init.initialize_model_parallel(1)

        gpu_id = 0
        if ':' in str(self.device):
            gpu_id = int(str(self.device).split(':')[-1])
        torch.cuda.set_device(gpu_id)
        setup_for_distributed(True)

        self.target_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}[self.dtype_name]
        dep_root = os.path.join(os.path.dirname(__file__), 'dependence', 'onellm')
        llama_config = os.path.join(dep_root, 'config', 'llama2', '7B.json')
        tokenizer_path = os.path.join(dep_root, 'config', 'llama2', 'tokenizer.model')

        with self.default_tensor_type(dtype=self.target_dtype, device='cuda'):
            self.model = MetaModel('onellm', llama_config, None, tokenizer_path)

        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(self.device, dtype=self.target_dtype)
        self.model.eval()

    def _prepare_point_cloud(self, point_cloud: Union[np.ndarray, torch.Tensor, str]) -> torch.Tensor:
        pc = load_point_cloud(point_cloud)
        if isinstance(pc, np.ndarray):
            pc = torch.from_numpy(pc)
        elif not isinstance(pc, torch.Tensor):
            pc = torch.tensor(pc)

        pc = pc.float()
        pc = self.pc_norm(pc)

        # OneLLM PointPatchEmbed treats channels [:, :, 3:] as xyz.
        if self.point_format == 'xyzrgb' and pc.shape[1] >= 6:
            pc = torch.cat((pc[:, 3:], pc[:, :3]), dim=1)
        return pc.unsqueeze(0)

    def qa(self, data: Dict[str, Any], prompt: str) -> str:
        point_cloud = _get_point_cloud_input(data)
        if point_cloud is None and not self.no_point_input:
            raise ValueError('Point cloud is required for OneLLM evaluation')

        model_points = None
        if not self.no_point_input:
            model_points = self._prepare_point_cloud(point_cloud).to(self.device, dtype=self.target_dtype)

        conv = self.conv_templates['v1'].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        with torch.cuda.amp.autocast(dtype=self.target_dtype):
            response = self.model.generate(
                [prompt_text],
                model_points,
                self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                modal=['point'],
            )[0]
        return response[len(prompt_text):].split('###')[0].strip()


def create_point_qa_model(model_name: str, checkpoint_path: str = None, **kwargs) -> PointQAModel:
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided for point QA models")
    return PointQAModel(model_name=model_name, checkpoint_path=checkpoint_path, **kwargs)
