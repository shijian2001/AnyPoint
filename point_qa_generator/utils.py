import numpy as np
from typing import List
from layout_generator.constants import VALID_RELATIONS

# =============================================================================
# CONSTANTS
# =============================================================================

# Attribute types used in generators
ATTRIBUTES = ["material", "color", "shape", "texture"]

# Number generator configurations: (num_types, count_distribution)
# Each distribution ensures unique most/least frequencies for number-based questions
NUMBER_GENERATOR_CONFIGS = {
    3: [(2, [1, 2])],
    4: [(2, [1, 3])],
    5: [(2, [1, 4]), (2, [2, 3])],
    6: [(2, [1, 5]), (2, [2, 4]), (3, [1, 2, 3])],
    7: [(2, [2, 5]), (2, [3, 4]), (3, [1, 2, 4])],
    8: [(2, [3, 5]), (3, [1, 2, 5]), (3, [1, 3, 4])],
    9: [(2, [4, 5]), (3, [1, 2, 6]), (3, [1, 3, 5]), (3, [2, 3, 4]), (4, [1, 2, 2, 4])]
}


# =============================================================================
# SPATIAL RELATION FUNCTIONS
# =============================================================================

def calculate_relation_from_positions(target_pos: np.ndarray, ref_pos: np.ndarray) -> str:
    """Calculate spatial relation from 3D positions using dominant axis method.
    
    Uses the axis with largest absolute difference to determine relation:
    - X-axis dominant: "to the left of" / "to the right of"
    - Y-axis dominant: "above" / "below"
    - Z-axis dominant: "in front of" / "behind"
    
    Args:
        target_pos: Target object position [x, y, z]
        ref_pos: Reference object position [x, y, z]
        
    Returns:
        Spatial relation string from VALID_RELATIONS
    """
    delta = np.array(target_pos) - np.array(ref_pos)
    abs_delta = np.abs(delta)
    
    # Find dominant axis
    dominant_axis = np.argmax(abs_delta)
    
    # Map to relation based on dominant axis and direction
    if dominant_axis == 0:  # X-axis
        return "to the right of" if delta[0] > 0 else "to the left of"
    elif dominant_axis == 1:  # Y-axis
        return "above" if delta[1] > 0 else "below"
    else:  # Z-axis
        return "in front of" if delta[2] > 0 else "behind"


# =============================================================================
# POINT CLOUD TRANSFORMATION FUNCTIONS
# =============================================================================

def rotate_x_axis(point_cloud: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate point cloud around X-axis.
    
    Args:
        point_cloud: Input point cloud array (N, 3+) where first 3 columns are coordinates
        angle_degrees: Rotation angle in degrees
        
    Returns:
        Rotated point cloud with same shape as input
    """
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle_radians), -np.sin(angle_radians)],
        [0, np.sin(angle_radians), np.cos(angle_radians)]
    ])
    
    coordinates = point_cloud[:, :3]
    colors = point_cloud[:, 3:] if point_cloud.shape[1] > 3 else np.zeros((point_cloud.shape[0], 3))
    rotated_coordinates = np.dot(coordinates, rotation_matrix.T)
    
    return np.hstack((rotated_coordinates, colors))


def center_and_scale_point_cloud(point_cloud: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
    """Center and scale point cloud to fit within unit sphere.
    
    Args:
        point_cloud: Input point cloud array (N, 3+) where first 3 columns are coordinates
        scale_factor: Scaling factor to apply after normalization
        
    Returns:
        Centered and scaled point cloud with same shape as input
    """
    coordinates = point_cloud[:, :3]
    centroid = np.mean(coordinates, axis=0)
    centered_coordinates = coordinates - centroid
    
    max_distance = np.max(np.linalg.norm(centered_coordinates, axis=1))
    if max_distance > 0:
        scaled_coordinates = centered_coordinates / max_distance * scale_factor
    else:
        scaled_coordinates = centered_coordinates
        
    return np.hstack((scaled_coordinates, point_cloud[:, 3:]))


def translate_point_cloud(point_cloud: np.ndarray, x_offset: float, y_offset: float) -> np.ndarray:
    """Translate point cloud by x and y offsets.
    
    Args:
        point_cloud: Input point cloud array (N, 3+) where first 3 columns are coordinates
        x_offset: Translation in x direction
        y_offset: Translation in y direction
        
    Returns:
        Translated point cloud with same shape as input
    """
    coordinates = point_cloud[:, :3]
    offsets = np.array([x_offset, y_offset, 0])
    translated_coordinates = coordinates + offsets
    
    return np.hstack((translated_coordinates, point_cloud[:, 3:]))