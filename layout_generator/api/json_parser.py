"""JSON parser utility for extracting JSON from LLM responses."""

import json
import re
from typing import Optional, Dict, Any


class JSONParser:
    """Extract and parse JSON from LLM responses."""

    @staticmethod
    def parse(response: str) -> Optional[Dict[str, Any]]:
        if isinstance(response, dict):
            return response
        if not isinstance(response, str):
            return None

        # Strategy 1: Find JSON in first code block
        code_blocks = re.findall(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
        for block in code_blocks:
            result = JSONParser._try_parse(block)
            if result is not None:
                return result

        # Strategy 2: Find outermost { ... } using bracket matching
        result = JSONParser._extract_balanced_json(response)
        if result is not None:
            return result

        # Strategy 3: Direct parse
        return JSONParser._try_parse(response)

    @staticmethod
    def _try_parse(text: str) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _extract_balanced_json(text: str) -> Optional[Dict[str, Any]]:
        """Find the first balanced { ... } in text using bracket counting."""
        start = text.find('{')
        if start == -1:
            return None

        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    return JSONParser._try_parse(text[start:i+1])
        return None
