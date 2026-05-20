import json
import random
from typing import Callable
import os

import diskcache
import numpy as np
import sentence_transformers
import torch


from models import Model

def load_point_cloud(data):
	if isinstance(data, np.ndarray):
		return data
	if isinstance(data, torch.Tensor):
		return data.cpu().numpy()
	if isinstance(data, str) and data.endswith('.npy') and os.path.exists(data):
		return np.load(data)
	raise ValueError(f"Unsupported point cloud input type: {type(data)}, value: {data}")


def resample_point_cloud(pc: np.ndarray, num_points: int, fps_fn=None) -> np.ndarray:
	"""Bring an (N, C) point cloud to exactly ``num_points`` rows.

	- N == num_points: returned as-is
	- N >  num_points: ``fps_fn(pc, num_points)`` if given, else random subsample
	- N <  num_points: pad by sampling rows with replacement
	Mirrors what each official 3D-LLM eval pipeline does to satisfy the model's
	fixed input shape (typically 8192).
	"""
	n = pc.shape[0]
	if n == num_points:
		return pc
	if n > num_points:
		if fps_fn is not None:
			out = fps_fn(pc, num_points)
			if isinstance(out, torch.Tensor):
				out = out.cpu().numpy()
			return out
		idx = np.random.choice(n, num_points, replace=False)
		return pc[idx]
	pad_idx = np.random.choice(n, num_points - n, replace=True)
	return np.concatenate([pc, pc[pad_idx]], axis=0)

	
def make_options(choices, format='letter'):
	assert format in ['numeric', 'letter']
	if format == 'numeric':
		prefix1 = [str(i + 1) for i in range(len(choices))]
	else:
		prefix1 = [chr(ord("a") + i).upper() for i in range(len(choices))]
	prefix2 = [f"({p})" for p in prefix1]
	return prefix1, prefix2, [f'{p} {c}' for p, c in zip(prefix2, choices)]


def check_contain(answer, options):
	contains = [option in answer for option in options]
	if sum(contains) == 1:
		return contains.index(True)
	else:
		return -1


class QAModelInstance:
	def qa(self, data, prompt):
		"(Abstract method) abstract QA method"

	def qa_batch(self, datas, prompts):
		"""Default batched QA: serial fallback. Override in wrappers that support real batching."""
		return [self.qa(d, p) for d, p in zip(datas, prompts)]


class QAModel(Model):
	def __init__(
			self,
			model_name: str,
			prompt_name: str,
			prompt_func: Callable,
			choice_format='letter',
			enable_choice_search: bool = False,
			cache_path: str = None,
	):
		self.model = None
		self.model_name = f'{model_name} ({prompt_name})'
		self.prompt_func = prompt_func
		self.format = choice_format
		self.cache_path = cache_path

		if self.cache_path is None:
			print("[IMPORTANT] model cache is disabled")
		else:
			print(f"[IMPORTANT] model cache is enabled, cache path: {cache_path}")

		self.enable_choice_search = enable_choice_search
		if enable_choice_search:
			# use SBERT to find the closest choice
			model_name = "all-mpnet-base-v2"
			model_root = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
			if model_root:
				local_path = os.path.join(model_root, model_name)
				if os.path.isdir(local_path):
					model_name = local_path
			self.sentence_transformer = sentence_transformers.SentenceTransformer(model_name, device='cpu')

	@torch.no_grad()
	def choice_search(self, free_form_answer, choices):
		query_embedding = self.sentence_transformer.encode([free_form_answer], normalize_embeddings=True)
		choices_embedding = self.sentence_transformer.encode(choices, normalize_embeddings=True)
		top_choice_index = np.argmax(np.dot(choices_embedding, query_embedding.T))
		return choices[top_choice_index]

	def _data_to_str(self, data):
		""" abstract method """

	@torch.no_grad()
	def _qa(self, data, prompt):
		if self.cache_path is None:
			return self.model.qa(data, prompt)
		else:
			with diskcache.Cache(self.cache_path, size_limit=10 * (2 ** 30)) as cache:
				key = json.dumps([self.model_name, self._data_to_str(data), prompt])
				response = cache.get(key, None)
				if response is None:
					response = self.model.qa(data, prompt)
					cache.set(key, response)
				return response

	@torch.no_grad()
	def qa(self, data, question):
		prompt = self.prompt_func(question)
		return self._qa(data, prompt)

	@torch.no_grad()
	def multiple_choice_qa(self, data, question, choices, answer=None):
		# Get VQA model's answer
		prefix1, prefix2, options = make_options(choices, self.format)
		prompt = self.prompt_func(question, choices)
		free_form_answer = self._qa(data, prompt)
		free_form_answer = free_form_answer.strip()

		# Limit the answer to the choices
		if free_form_answer in choices:
			multiple_choice_answer = free_form_answer
		elif free_form_answer in options:
			multiple_choice_answer = choices[options.index(free_form_answer)]
		elif free_form_answer in prefix1:
			multiple_choice_answer = choices[prefix1.index(free_form_answer)]
		elif free_form_answer in prefix2:
			multiple_choice_answer = choices[prefix2.index(free_form_answer)]
		elif self.enable_choice_search:
			multiple_choice_answer = self.choice_search(free_form_answer, choices)
		else:
			multiple_choice_answer = ""
			for to_check in [choices, options, prefix1, prefix2]:
				idx = check_contain(free_form_answer, to_check)
				if idx != -1:
					multiple_choice_answer = choices[idx]
					break

		result = {
			"free_form_answer"      : free_form_answer,
			"multiple_choice_answer": multiple_choice_answer,
			# "choices"               : choices.copy(),
		}
		if answer is not None:
			result["accuracy"] = int(answer == multiple_choice_answer)
		return result

	@torch.no_grad()
	def _qa_batch(self, datas, prompts):
		if self.cache_path is None:
			return self.model.qa_batch(datas, prompts)
		responses = [None] * len(datas)
		miss_idx, miss_data, miss_prompt = [], [], []
		with diskcache.Cache(self.cache_path, size_limit=10 * (2 ** 30)) as cache:
			keys = [json.dumps([self.model_name, self._data_to_str(d), p]) for d, p in zip(datas, prompts)]
			for i, key in enumerate(keys):
				cached = cache.get(key, None)
				if cached is None:
					miss_idx.append(i)
					miss_data.append(datas[i])
					miss_prompt.append(prompts[i])
				else:
					responses[i] = cached
			if miss_idx:
				new_resps = self.model.qa_batch(miss_data, miss_prompt)
				for j, i in enumerate(miss_idx):
					responses[i] = new_resps[j]
					cache.set(keys[i], new_resps[j])
		return responses

	@torch.no_grad()
	def multiple_choice_qa_batch(self, datas, questions, choices_list, answers=None):
		"""Batched multiple-choice QA. Returns a list of result dicts.

		Mirrors :meth:`multiple_choice_qa` exactly but groups model calls so wrappers
		can perform a single batched forward pass.
		"""
		prompts = [self.prompt_func(q, c) for q, c in zip(questions, choices_list)]
		free_form_answers = self._qa_batch(datas, prompts)

		results = []
		if answers is None:
			answers = [None] * len(datas)

		for free_form_answer, choices, answer in zip(free_form_answers, choices_list, answers):
			free_form_answer = (free_form_answer or "").strip()
			prefix1, prefix2, options = make_options(choices, self.format)

			if free_form_answer in choices:
				multiple_choice_answer = free_form_answer
			elif free_form_answer in options:
				multiple_choice_answer = choices[options.index(free_form_answer)]
			elif free_form_answer in prefix1:
				multiple_choice_answer = choices[prefix1.index(free_form_answer)]
			elif free_form_answer in prefix2:
				multiple_choice_answer = choices[prefix2.index(free_form_answer)]
			elif self.enable_choice_search:
				multiple_choice_answer = self.choice_search(free_form_answer, choices)
			else:
				multiple_choice_answer = ""
				for to_check in [choices, options, prefix1, prefix2]:
					idx = check_contain(free_form_answer, to_check)
					if idx != -1:
						multiple_choice_answer = choices[idx]
						break

			result = {
				"free_form_answer"      : free_form_answer,
				"multiple_choice_answer": multiple_choice_answer,
			}
			if answer is not None:
				result["accuracy"] = int(answer == multiple_choice_answer)
			results.append(result)
		return results

	@torch.no_grad()
	def multiple_choice_qa_random_ordering(self, data, question, choices, answer=None, n_trials=3):
		results = {}
		accuracy = 0
		for i in range(n_trials):
			choices_i = choices.copy()
			random.shuffle(choices_i)
			results[i] = self.multiple_choice_qa(data, question, choices_i, answer)
			accuracy += results[i]["accuracy"]
		results["accuracy"] = accuracy / n_trials
		return results
