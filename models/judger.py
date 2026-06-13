"""LLM judge for multiple-choice answer extraction.

Given a question, its labeled options, and a model's free-form response, the
judge returns which option the model selected (letter index) or NONE. We do NOT
give it the ground-truth answer, so it only extracts (no grading bias); the
correct/incorrect decision is made deterministically afterwards by comparing the
extracted option to the ground truth.

Backend: an OpenAI/Gemini-style generateContent proxy (Gemini 3.5 Flash).
API key is read from env JUDGE_API_KEY (never hard-coded).
Calls bypass any http(s)_proxy (the endpoint is reachable directly).
"""
from __future__ import annotations
import os, re, json, string, time, random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence

import requests

DEFAULT_ENDPOINT = "https://runway.devops.rednote.life/openai/google/v1:generateContent"

_SYS = (
    "Identify which multiple-choice option the candidate selected, given the question, "
    "the lettered options, and the candidate's response. Match by the stated letter or by "
    "the meaning/value of the response. Output NONE if the response is empty, refuses, is "
    "ambiguous, names more than one option, or its stated letter disagrees with the content "
    "it describes. Reply with only the option letter or NONE."
)


class Judger:
    def __init__(self, api_key: Optional[str] = None, endpoint: str = DEFAULT_ENDPOINT,
                 max_workers: int = 32, timeout: float = 60.0, retries: int = 4):
        self.api_key = api_key or os.environ.get("JUDGE_API_KEY")
        if not self.api_key:
            raise ValueError("Judger needs an API key (arg api_key or env JUDGE_API_KEY)")
        self.endpoint = endpoint
        self.timeout = timeout
        self.retries = retries
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        # a session that ignores environment proxies
        self.session = requests.Session()
        self.session.trust_env = False

    @staticmethod
    def _prompt(question: str, options: Sequence[str], answer: str) -> str:
        labeled = "\n".join(f"{string.ascii_uppercase[i]}. {o}" for i, o in enumerate(options))
        return (f"Question: {question}\nOptions:\n{labeled}\n"
                f'Candidate response: "{answer}"\n'
                "Which option did the candidate select? Reply with only the letter or NONE.")

    def _call(self, prompt: str) -> str:
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": _SYS}]},
            # thinkingBudget:0 disables thinking (extraction needs none); without it
            # a small maxOutputTokens gets eaten by thinking -> MAX_TOKENS, empty answer.
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 16,
                                  "thinkingConfig": {"thinkingBudget": 0}},
        }
        last = ""
        for attempt in range(self.retries):
            if attempt:  # exponential backoff + jitter (handles 429 / transient)
                time.sleep(min(2 ** attempt, 8) + random.random())
            try:
                r = self.session.post(self.endpoint, headers={
                    "api-key": self.api_key, "Content-Type": "application/json"},
                    data=json.dumps(body), timeout=self.timeout, proxies={"http": None, "https": None})
                if r.status_code != 200:
                    last = f"HTTP {r.status_code}: {r.text[:200]}"; continue
                cand = r.json().get("candidates", [{}])[0]
                text = " ".join(p.get("text", "") for p in cand.get("content", {}).get("parts", [])).strip()
                if text:
                    return text
                last = f"empty text (finishReason={cand.get('finishReason')})"
            except Exception as exc:  # noqa: BLE001
                last = str(exc)
        raise RuntimeError(f"judge call failed after {self.retries} retries: {last}")

    @staticmethod
    def _parse_letter(text: str, n_opts: int) -> Optional[int]:
        if not text:
            return None
        t = text.strip().upper()
        if t.startswith("NONE"):
            return None
        m = re.search(r"[A-Z]", t)
        if not m:
            return None
        idx = ord(m.group(0)) - ord("A")
        return idx if 0 <= idx < n_opts else None

    def extract_one(self, question: str, options: Sequence[str], answer: str) -> Optional[int]:
        """Return the chosen option index, or None."""
        return self._parse_letter(self._call(self._prompt(question, options, answer)), len(options))

    def extract_batch(self, questions: Sequence[str], options_list: Sequence[Sequence[str]],
                      answers: Sequence[str]) -> List[Optional[int]]:
        """Concurrently extract chosen option index for a batch. None = no choice."""
        futs = [self.pool.submit(self.extract_one, q, o, a)
                for q, o, a in zip(questions, options_list, answers)]
        out = []
        for f in futs:
            try:
                out.append(f.result())
            except Exception:  # noqa: BLE001
                out.append(None)
        return out
