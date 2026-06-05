import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from minigpt4.common.registry import registry
from minigpt4.models.base_model import disabled_train
from minigpt4.models.minigpt_base import MiniGPTBase
from minigpt4.models.Qformer import BertConfig, BertLMHeadModel
import logging
from peft import LoraConfig, inject_adapter_in_model
import random

logger = logging.getLogger(__name__)


class AlignmentProjector(nn.Module):
    """3-layer MLP projection from LLM hidden dim to Q-Former dim."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden_dim = (input_dim + output_dim) // 2
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.projector(x)


@registry.register_model("minigpt_3d")
class MiniGPT_3D(MiniGPTBase):
    """PointAlign: Feature-Level Alignment Regularization for 3D Vision-Language Models."""

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/minigpt_3d.yaml",
    }

    def __init__(
            self,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            pc_precision="fp16",
            freeze_pc=True,
            llama_model="",
            prompt_template='###Human: {} ###Assistant: ',
            max_txt_len=300,
            end_sym='\n',
            lora_r=64,
            lora_target_modules=['query_key_value', 'dense'],
            lora_alpha=16,
            lora_dropout=0.05,
            chat_template=False,
            use_grad_checkpoint_llm=False,
            max_context_len=3800,
            low_resource=False,
            device_8bit=0,
            QFormer_lora_r=-1,
            train_QFormer_norm=False,
            freeze_Qformer=True,
            QFormer_lora_module=["query", "key", "value"],
            pc_linear_layer=2,
            align_qformer_to_llm=False,
            blip2_pretrained="",
    ):
        super().__init__(
            llama_model=llama_model,
            max_txt_len=max_txt_len,
            max_context_len=max_context_len,
            end_sym=end_sym,
            prompt_template=prompt_template,
            low_resource=low_resource,
            device_8bit=device_8bit,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        self.pc_encoder = self.init_pc_encoder(pc_precision, freeze_pc)

        if pc_linear_layer == 1:
            self.point_2_Qformer_proj = nn.Linear(self.pc_encoder.trans_dim, 1408)
        elif pc_linear_layer == 2:
            self.point_2_Qformer_proj = nn.Sequential(
                nn.Linear(self.pc_encoder.trans_dim, 768),
                nn.GELU(),
                nn.Linear(768, 1408),
            )
        elif pc_linear_layer == 3:
            self.point_2_Qformer_proj = nn.Sequential(
                nn.Linear(self.pc_encoder.trans_dim, 768),
                nn.GELU(),
                nn.Linear(768, 1152),
                nn.GELU(),
                nn.Linear(1152, 1408),
            )

        self.freeze_Qformer = freeze_Qformer
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token=32, vision_width=1408, freeze=self.freeze_Qformer,
            QFormer_lora_r=QFormer_lora_r,
            train_QFormer_norm=train_QFormer_norm,
            QFormer_lora_module=QFormer_lora_module,
        )
        if blip2_pretrained:
            print(f"Load BLIP2 checkpoint: {blip2_pretrained}")
            self.load_from_pretrained(url_or_filename=blip2_pretrained)
        else:
            print("Skip BLIP2 pretrained loading (model.blip2_pretrained is empty).")

        self.llama_proj = nn.Linear(self.Qformer.config.hidden_size, 4096)
        self.llama_proj2 = nn.Linear(4096, self.llama_model.config.hidden_size)

        self.chat_template = chat_template

        if use_grad_checkpoint_llm:
            self.llama_model.gradient_checkpointing_enable()

        self.align_qformer_to_llm = align_qformer_to_llm
        if self.align_qformer_to_llm:
            anchor_dim = self.Qformer.config.hidden_size
            llm_hidden_dim = self.llama_model.config.hidden_size
            self.alignment_projector = AlignmentProjector(
                input_dim=llm_hidden_dim,
                output_dim=anchor_dim,
            )

        self.freeze_pc = freeze_pc
        if self.freeze_pc:
            for name, param in self.pc_encoder.named_parameters():
                param.requires_grad = False
            self.point_encoder = self.pc_encoder.eval()
            self.point_encoder.train = disabled_train

        if self.align_qformer_to_llm:
            for name, param in self.pc_encoder.named_parameters():
                param.requires_grad = False
            for name, param in self.point_2_Qformer_proj.named_parameters():
                param.requires_grad = False
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.query_tokens.requires_grad = False
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            for name, param in self.llama_proj2.named_parameters():
                param.requires_grad = False
            for name, param in self.alignment_projector.named_parameters():
                param.requires_grad = True

    def encode_pc(self, pc, return_qformer_output=False):
        device = pc.device

        with self.maybe_autocast():
            pc_encoder_output = self.pc_encoder(pc)
            mlp_output = self.point_2_Qformer_proj(pc_encoder_output).to(device)

            pc_atts = torch.ones(mlp_output.size()[:-1], dtype=torch.long).to(device)
            query_tokens = self.query_tokens.expand(mlp_output.shape[0], -1, -1)

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=mlp_output,
                encoder_attention_mask=pc_atts,
                return_dict=True,
            )

            qformer_output = query_output.last_hidden_state
            projector_mid = self.llama_proj(qformer_output)
            projector_final = self.llama_proj2(projector_mid)

            atts_llama = torch.ones(projector_final.size()[:-1], dtype=torch.long).to(device)

            if return_qformer_output:
                return projector_final, atts_llama, qformer_output

            return projector_final, atts_llama

    def preparing_embedding(self, samples):
        need_alignment = self.align_qformer_to_llm and self.training

        if 'pc' in samples:
            if need_alignment:
                pc_embeds, pc_atts, qformer_output = self.encode_pc(
                    samples["pc"], return_qformer_output=True
                )
                samples["_qformer_output_cache"] = qformer_output
            else:
                pc_embeds, pc_atts = self.encode_pc(samples["pc"])
        else:
            pc_embeds = pc_atts = None

        if 'conv_q' in samples:
            conv_q, conv_a = samples['conv_q'], samples['conv_a']
            connect_sym = samples['connect_sym'][0]
            conv_q = [q.split(connect_sym) for q in conv_q]
            conv_a = [a.split(connect_sym) for a in conv_a]
            conv_q = [[self.prompt_template.format(item) for item in items] for items in conv_q]
            cond_embeds, cond_atts = self.prompt_wrap(pc_embeds, pc_atts, [q[0] for q in conv_q])
            regress_token_ids, regress_atts, part_targets = self.tokenize_conversation(conv_q, conv_a)
        else:
            if "instruction_input" in samples:
                instruction = samples["instruction_input"]
            elif self.prompt_list:
                instruction = random.choice(self.prompt_list)
            else:
                instruction = None

            if hasattr(self, 'chat_template') and self.chat_template:
                instruction = [self.prompt_template.format(instruct) for instruct in instruction]

            if 'length' in samples:
                bsz, pn, hs = pc_embeds.shape
                pc_embeds = pc_embeds.reshape(len(samples['pc']), -1, pn, hs)
                cond_embeds, cond_atts = self.prompt_wrap(pc_embeds, pc_atts, instruction, samples['length'])
            else:
                cond_embeds, cond_atts = self.prompt_wrap(pc_embeds, pc_atts, instruction)

            self.llama_tokenizer.padding_side = "right"
            text = [t + self.end_sym for t in samples["answer"]]
            regress_tokens = self.llama_tokenizer(
                text, return_tensors="pt", padding="longest",
                truncation=True, max_length=self.max_txt_len, add_special_tokens=False
            ).to(self.device)

            regress_token_ids = regress_tokens.input_ids
            regress_atts = regress_tokens.attention_mask
            part_targets = regress_token_ids.masked_fill(
                regress_token_ids == self.llama_tokenizer.pad_token_id, -100
            )

        regress_embeds = self.embed_tokens(regress_token_ids)
        return cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets

    def forward(self, samples):
        enable_alignment = self.align_qformer_to_llm and self.training

        cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets = \
            self.preparing_embedding(samples)

        qformer_output = samples.pop("_qformer_output_cache", None) if enable_alignment else None

        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(cond_embeds, cond_atts, regress_embeds, regress_atts)

        bos = torch.ones_like(part_targets[:, :1]) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        bos_atts = cond_atts[:, :1]

        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([bos_atts, attention_mask], dim=1)

        targets = torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                             dtype=torch.long).to(self.device).fill_(-100)

        for i, target in enumerate(part_targets):
            targets[i, input_lens[i] + 1:input_lens[i] + len(target) + 1] = target

        with self.maybe_autocast():
            outputs = self.llama_model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                output_hidden_states=enable_alignment,
            )

        ntp_loss = outputs.loss
        total_loss = ntp_loss
        align_loss_value = 0.0

        if enable_alignment and qformer_output is not None:
            try:
                num_query_tokens = 32
                cond_len = cond_embeds.shape[1]
                text_prompt_len = cond_len - num_query_tokens
                pc_token_start = 1 + text_prompt_len
                pc_token_end = 1 + cond_len

                qformer_output_fixed = qformer_output.detach()

                llm_layer_hidden = outputs.hidden_states[15]
                llm_pc_tokens = llm_layer_hidden[:, pc_token_start:pc_token_end, :]

                llm_projected = self.alignment_projector(llm_pc_tokens)

                cosine_sim = F.cosine_similarity(qformer_output_fixed, llm_projected, dim=-1)
                align_loss = (1 - cosine_sim).mean()
                align_loss_value = align_loss.item()

                total_loss = ntp_loss + 0.1 * align_loss

            except Exception as e:
                logger.error(f"Alignment loss calculation failed: {e}")
                import traceback
                traceback.print_exc()

        return {
            "loss": total_loss,
            "ntp_loss": ntp_loss.item() if isinstance(ntp_loss, torch.Tensor) else ntp_loss,
            "align_loss": align_loss_value,
        }

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, freeze, QFormer_lora_r,
                     train_QFormer_norm, QFormer_lora_module):
        encoder_config = BertConfig.from_pretrained(
            os.environ.get("POINTALIGN_BERT_BASE_PATH", "./params_weight/bert-base-uncased")
        )
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        Qformer.cls = None
        Qformer.bert.embeddings.word_embeddings = None
        Qformer.bert.embeddings.position_embeddings = None
        for layer in Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        if freeze:
            for name, param in Qformer.named_parameters():
                param.requires_grad = False
            Qformer.bert = Qformer.bert.eval()
            Qformer.train = disabled_train

        if QFormer_lora_r > 0:
            lora_config = LoraConfig(
                lora_alpha=QFormer_lora_r * 2,
                lora_dropout=0.1,
                r=QFormer_lora_r,
                bias="none",
                target_modules=QFormer_lora_module,
            )
            Qformer = inject_adapter_in_model(lora_config, Qformer)

            if train_QFormer_norm:
                for i, layer in enumerate(Qformer.bert.encoder.layer):
                    layer.attention.output.LayerNorm.weight.requires_grad = True
                    layer.output_query.LayerNorm.weight.requires_grad = True
                    if i % 2 == 0:
                        layer.crossattention.output.LayerNorm.weight.requires_grad = True

        return Qformer, query_tokens

    @classmethod
    def from_config(cls, cfg):
        llama_model = cfg.get("llama_model")
        pc_precision = cfg.get("pc_precision", "fp16")
        freeze_pc = cfg.get("freeze_pc", True)
        prompt_template = cfg.get("prompt_template", '[INST] {} [/INST]')
        max_txt_len = cfg.get("max_txt_len", 300)
        end_sym = cfg.get("end_sym", '\n')
        lora_r = cfg.get("lora_r", 64)
        lora_alpha = cfg.get("lora_alpha", 16)
        QFormer_lora_r = cfg.get("QFormer_lora_r", -1)
        pc_linear_layer = cfg.get("pc_linear_layer", 2)
        freeze_Qformer = cfg.get("freeze_Qformer", True)
        QFormer_lora_module = cfg.get("QFormer_lora_module", ["query", "key", "value"])
        train_QFormer_norm = cfg.get("train_QFormer_norm", False)

        align_qformer_to_llm = cfg.get("align_qformer_to_llm", False)
        blip2_pretrained = cfg.get("blip2_pretrained", "")
        cfg_dir = os.environ.get("POINTALIGN_CFG_DIR")
        if blip2_pretrained and cfg_dir and not os.path.isabs(blip2_pretrained):
            blip2_pretrained = os.path.join(cfg_dir, blip2_pretrained)

        model = cls(
            pc_precision=pc_precision,
            freeze_pc=freeze_pc,
            llama_model=llama_model,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            QFormer_lora_r=QFormer_lora_r,
            pc_linear_layer=pc_linear_layer,
            freeze_Qformer=freeze_Qformer,
            align_qformer_to_llm=align_qformer_to_llm,
            QFormer_lora_module=QFormer_lora_module,
            train_QFormer_norm=train_QFormer_norm,
            blip2_pretrained=blip2_pretrained,
        )

        ckpt_path = cfg.get("ckpt", "")
        if ckpt_path:
            print(f"Load checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
            print(f"  Loaded {len(ckpt['model'])} parameters")
            if msg.missing_keys:
                print(f"  Missing keys: {len(msg.missing_keys)}")
            if msg.unexpected_keys:
                print(f"  Unexpected keys: {len(msg.unexpected_keys)}")

        second_ckpt = cfg.get("second_ckpt", "")
        if second_ckpt:
            print(f"Load second checkpoint: {second_ckpt}")
            ckpt = torch.load(second_ckpt, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
            print(f"  Loaded {len(ckpt['model'])} parameters")
            if msg.missing_keys:
                print(f"  Missing keys: {len(msg.missing_keys)}")
            if msg.unexpected_keys:
                print(f"  Unexpected keys: {len(msg.unexpected_keys)}")

        return model