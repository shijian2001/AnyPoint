import sys
import os

# Add shapellm to Python path
shapellm_path = os.path.dirname(os.path.abspath(__file__))
if shapellm_path not in sys.path:
    sys.path.insert(0, shapellm_path)

# Add parent directories to path if needed
parent_path = os.path.dirname(shapellm_path)
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
