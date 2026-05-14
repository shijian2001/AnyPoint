"""Question template library for diverse paraphrasing."""

from typing import Dict, List, Union
import numpy as np

TemplateTree = Union[List[str], Dict[str, "TemplateTree"]]

QUESTION_TEMPLATES: Dict[str, TemplateTree] = {
    "what_distance": {
        "closest": [
            "What is the object that is closest to the {ref}?",
            "Which object is nearest to the {ref}?",
            "Identify the object positioned closest to the {ref}.",
            "Among all objects, which one is the closest to the {ref}?",
            "What object has the shortest distance to the {ref}?",
        ],
        "farthest": [
            "What is the object that is farthest from the {ref}?",
            "Which object is the most distant from the {ref}?",
            "Identify the object positioned farthest from the {ref}.",
            "Among all objects, which one is the farthest from the {ref}?",
            "What object has the greatest distance from the {ref}?",
        ],
    },
    "where_distance": {
        "closest": [
            "Where is the object that is closest to the {ref}?",
            "In what direction is the nearest object relative to the {ref}?",
            "Describe the position of the closest object to the {ref}.",
            "What is the spatial relation of the nearest object to the {ref}?",
        ],
        "farthest": [
            "Where is the object that is farthest from the {ref}?",
            "In what direction is the most distant object relative to the {ref}?",
            "Describe the position of the farthest object from the {ref}.",
            "What is the spatial relation of the farthest object to the {ref}?",
        ],
    },
    "list_attribute_distance": {
        "closest": [
            "List all {attr}s in the components of the object closest to the {ref}.",
            "What are the {attr}s of the nearest object to the {ref}?",
            "Enumerate the {attr}s found in the object that is closest to the {ref}.",
        ],
        "farthest": [
            "List all {attr}s in the components of the object farthest from the {ref}.",
            "What are the {attr}s of the most distant object from the {ref}?",
            "Enumerate the {attr}s found in the object that is farthest from the {ref}.",
        ],
    },
    "count_attribute_distance": {
        "closest": [
            "How many {attr}s are in the components of the object closest to the {ref}?",
            "Count the distinct {attr}s of the nearest object to the {ref}.",
            "How many unique {attr}s does the closest object to the {ref} have?",
        ],
        "farthest": [
            "How many {attr}s are in the components of the object farthest from the {ref}?",
            "Count the distinct {attr}s of the most distant object from the {ref}.",
            "How many unique {attr}s does the farthest object from the {ref} have?",
        ],
    },
    "what_attribute": [
        "What is the {attr} of the {comp} in the {obj}?",
        "Identify the {attr} of the {comp} component of the {obj}.",
        "What {attr} does the {comp} of the {obj} have?",
        "Describe the {attr} property of the {comp} in the {obj}.",
    ],
    "list_attribute": [
        "List all {attr}s in the components of the {obj}.",
        "What are all the {attr}s found in the {obj}'s components?",
        "Enumerate the {attr}s present in the components of the {obj}.",
        "Name all distinct {attr}s in the {obj}.",
    ],
    "count_attribute": [
        "How many distinct {attr}s are in the components of the {obj}?",
        "Count the number of different {attr}s in the {obj}'s components.",
        "How many unique {attr}s does the {obj} have across its components?",
    ],
    "count_object": [
        "How many {obj} are in the scene?",
        "Count the number of {obj} in the scene.",
        "How many instances of {obj} can you find in this scene?",
        "What is the total count of {obj} in the scene?",
    ],
    "frequent_object": {
        "most": [
            "What is the most frequent object in the scene?",
            "Which object appears the most times in the scene?",
            "Identify the object with the highest count in this scene.",
            "What object has the most instances in the scene?",
        ],
        "least": [
            "What is the least frequent object in the scene?",
            "Which object appears the fewest times in the scene?",
            "Identify the object with the lowest count in this scene.",
            "What object has the fewest instances in the scene?",
        ],
    },
    "list_attribute_frequent": {
        "most": [
            "List all {attr}s in the components of the most frequent object in the scene.",
            "What {attr}s does the most common object in the scene have?",
            "Enumerate the {attr}s of the object that appears most often.",
        ],
        "least": [
            "List all {attr}s in the components of the least frequent object in the scene.",
            "What {attr}s does the rarest object in the scene have?",
            "Enumerate the {attr}s of the object that appears least often.",
        ],
    },
    "count_attribute_frequent": {
        "most": [
            "How many {attr}s are in the components of the most frequent object?",
            "Count the distinct {attr}s of the most common object in the scene.",
            "How many unique {attr}s does the most frequent object have?",
        ],
        "least": [
            "How many {attr}s are in the components of the least frequent object?",
            "Count the distinct {attr}s of the rarest object in the scene.",
            "How many unique {attr}s does the least frequent object have?",
        ],
    },
    "what_size": {
        "largest": [
            "What is the largest object in the scene?",
            "Which object has the greatest volume in the scene?",
            "Identify the biggest object in this scene.",
            "What is the most voluminous object here?",
        ],
        "smallest": [
            "What is the smallest object in the scene?",
            "Which object has the least volume in the scene?",
            "Identify the tiniest object in this scene.",
            "What is the most compact object here?",
        ],
    },
    "list_attribute_size": {
        "largest": [
            "List all {attr}s in the components of the largest object in the scene.",
            "What {attr}s does the biggest object in this scene have?",
            "Enumerate the {attr}s of the largest object.",
        ],
        "smallest": [
            "List all {attr}s in the components of the smallest object in the scene.",
            "What {attr}s does the tiniest object in this scene have?",
            "Enumerate the {attr}s of the smallest object.",
        ],
    },
    "count_attribute_size": {
        "largest": [
            "How many {attr}s are in the components of the largest object?",
            "Count the distinct {attr}s of the biggest object in the scene.",
            "How many unique {attr}s does the largest object have?",
        ],
        "smallest": [
            "How many {attr}s are in the components of the smallest object?",
            "Count the distinct {attr}s of the tiniest object in the scene.",
            "How many unique {attr}s does the smallest object have?",
        ],
    },
    "where_size": {
        "with_reference": {
            "largest": [
                "Where is the largest object relative to the {ref}?",
                "What is the spatial position of the biggest object with respect to the {ref}?",
                "Describe where the largest object is in relation to the {ref}.",
            ],
            "smallest": [
                "Where is the smallest object relative to the {ref}?",
                "What is the spatial position of the tiniest object with respect to the {ref}?",
                "Describe where the smallest object is in relation to the {ref}.",
            ],
        },
        "reference_to_target": {
            "largest": [
                "Where is the {ref} relative to the largest object?",
                "Describe the position of the {ref} with respect to the biggest object.",
            ],
            "smallest": [
                "Where is the {ref} relative to the smallest object?",
                "Describe the position of the {ref} with respect to the tiniest object.",
            ],
        },
    },
    "what_relation": [
        "What is the object that is {rel} the {ref}?",
        "Which object is {rel} the {ref}?",
        "Identify the object {rel} the {ref}.",
        "What object is positioned {rel} the {ref}?",
    ],
    "multi_hop_relation": [
        "What is the object {rel2} the object {rel1} the {anchor}?",
        "Which object is {rel2} the object that is {rel1} the {anchor}?",
        "Identify the object that is {rel2} the one {rel1} the {anchor}.",
    ],
}


def sample_template(rng: np.random.RandomState, generator_type: str, **kwargs) -> str:
    """Sample a random question template for the given generator type and config."""
    templates = QUESTION_TEMPLATES[generator_type]

    # Traverse nested dict using known config keys
    lookup_keys = ["distance_type", "frequency_type", "reference_mode", "size_type"]
    for key in lookup_keys:
        if isinstance(templates, dict) and key in kwargs:
            sub_key = kwargs[key]
            if sub_key in templates:
                templates = templates[sub_key]

    if isinstance(templates, list):
        return templates[rng.randint(len(templates))]

    # Fallback: pick first available list
    while isinstance(templates, dict):
        templates = next(iter(templates.values()))
    if isinstance(templates, list):
        return templates[rng.randint(len(templates))]

    return templates
