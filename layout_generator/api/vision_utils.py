import base64
import numpy as np
from PIL import Image
from io import BytesIO
from qwen_vl_utils import process_vision_info
from typing import List, Dict, Any, Tuple


def build_video_message(user_prompt: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Process user prompt containing video inputs and convert to API format.

    Args:
        user_prompt: List of content items with type and data

    Returns:
        Tuple of (processed_content, video_kwargs)
    """
    processed_content = []
    video_kwargs = {}

    for item in user_prompt:
        if item["type"] == "video":
            # Extract video parameters, filtering out non-video keys
            video_params = {k: v for k, v in item.items() if k != "type"}

            video_msg = [{
                "content": [{
                    "type": "video",
                    **video_params
                }]
            }]

            _, video_inputs, video_kwargs = process_vision_info(video_msg, return_video_kwargs=True)

            # Convert video frames to base64
            frames = video_inputs.pop().permute(0, 2, 3, 1).numpy().astype(np.uint8)
            base64_frames = [
                base64.b64encode(
                    (buffer := BytesIO(),
                     Image.fromarray(frame).save(buffer, format="JPEG"),
                     buffer.getvalue())[2]
                ).decode("utf-8")
                for frame in frames
            ]

            processed_content.append({
                "type": "video_url",
                "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}
            })
        else:
            # Handle other content types (text, image, etc.)
            processed_content.append(item)

    return processed_content, video_kwargs


def build_image_message(user_prompt: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Process user prompt containing image inputs and convert to API format.

    Args:
        user_prompt: List of content items with type and data

    Returns:
        Tuple of (processed_content, image_kwargs)

    Note:
        This function is not implemented yet.
    """
    # TODO: Implement image processing logic
    raise NotImplementedError("Image processing not implemented yet")


def build_multimodal_message(user_prompt: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Route multimodal message processing based on content type.

    Args:
        user_prompt: List of content items with type and data

    Returns:
        Tuple of (processed_content, mm_kwargs)
    """
    content_types = {item.get("type") for item in user_prompt}

    if "video" in content_types:
        return build_video_message(user_prompt)
    elif "image" in content_types:
        return build_image_message(user_prompt)
    else:
        # No special processing needed for text-only content
        return user_prompt, {}