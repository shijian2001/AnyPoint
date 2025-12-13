"""JSON parser utility for extracting JSON from LLM responses."""

import json
import re
from typing import Optional, Dict, Any


class JSONParser:
    """Utility class for parsing and extracting JSON from text responses."""

    @staticmethod
    def parse(response: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON from a response string.

        Args:
            response: String potentially containing JSON

        Returns:
            Parsed JSON dictionary or None if parsing fails
        """
        # If input is already a dict, return it
        if isinstance(response, dict):
            return response
        
        if not isinstance(response, str):
            return None
        
        # Try to match JSON in a code block first
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response
        
        try:
            parsed_data = json.loads(json_str)
            if isinstance(parsed_data, dict):
                return parsed_data
            return None
        except json.JSONDecodeError:
            return None
