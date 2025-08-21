import numpy as np
from typing import List

# =============================================================================
# CONSTANTS
# =============================================================================

# Attribute types used in generators
ATTRIBUTES = ["material", "color", "shape", "texture"]

# 3x3 Grid system constants
GRID_POSITIONS = {
    0: "top left", 1: "top center", 2: "top right",
    3: "middle left", 4: "center", 5: "middle right", 
    6: "bottom left", 7: "bottom center", 8: "bottom right"
}

GRID_COORDINATES = {
    0: [-2.0, 2.0], 1: [0.0, 2.0], 2: [2.0, 2.0],
    3: [-2.0, 0.0], 4: [0.0, 0.0], 5: [2.0, 0.0],
    6: [-2.0, -2.0], 7: [0.0, -2.0], 8: [2.0, -2.0]
}

# 3x3 Grid relative distance mapping
# Each key represents a reference grid position (0-8)
# Each value is a list of lists, where inner lists contain grids at the same distance level
RELATIVE_DISTANCE = {
    '0': [[1, 3], [4], [2, 6], [5, 7], [8]],
    '1': [[0, 4, 2], [3, 5], [7], [6, 8]],
    '2': [[1, 5], [4], [0, 8], [3, 7], [6]],
    '3': [[0, 4, 6], [1, 7], [5], [2, 8]],
    '4': [[1, 3, 5, 7], [0, 2, 6, 8]],
    '5': [[2, 4, 8], [1, 7], [3], [0, 6]],
    '6': [[3, 7], [4], [0, 8], [1, 5], [2]],
    '7': [[4, 6, 8], [3, 5], [1], [2, 0]],
    '8': [[5, 7], [4], [2, 6], [1, 3], [0]],
}

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
# DISTANCE UTILITY FUNCTIONS
# =============================================================================

def get_relative_distance_level(ref_grid: int, target_grid: int) -> int:
    """Get the distance level between reference and target grids.
    
    Args:
        ref_grid: Reference grid position (0-8)
        target_grid: Target grid position (0-8)
        
    Returns:
        Distance level (0 = closest, higher = farther), or -1 if not found
    """
    for idx, level in enumerate(RELATIVE_DISTANCE[str(ref_grid)]):
        if target_grid in level:
            return idx
    return -1


def get_max_distance_level(ref_grid: int) -> int:
    """Get the maximum distance level from reference grid.
    
    Args:
        ref_grid: Reference grid position (0-8)
        
    Returns:
        Maximum distance level from this grid
    """
    return len(RELATIVE_DISTANCE[str(ref_grid)]) - 1


def get_farther_grids(ref_grid: int, target_grid: int) -> List[int]:
    """Get all grids farther than target from reference.
    
    Args:
        ref_grid: Reference grid position (0-8)
        target_grid: Target grid position (0-8)
        
    Returns:
        List of grid positions farther than target from reference
    """
    ref_level = get_relative_distance_level(ref_grid, target_grid)
    farther_grids = []
    for level in RELATIVE_DISTANCE[str(ref_grid)][ref_level + 1:]:
        farther_grids.extend(level)
    return farther_grids


def get_closer_grids(ref_grid: int, target_grid: int) -> List[int]:
    """Get all grids closer than target from reference.
    
    Args:
        ref_grid: Reference grid position (0-8)
        target_grid: Target grid position (0-8)
        
    Returns:
        List of grid positions closer than target from reference
    """
    ref_level = get_relative_distance_level(ref_grid, target_grid)
    closer_grids = []
    for level in RELATIVE_DISTANCE[str(ref_grid)][:ref_level]:
        closer_grids.extend(level)
    return closer_grids


def get_relative_position(target_grid: int, ref_grid: int) -> str:
    """Get relative position description of target with respect to reference.
    
    Grid layout (0-8):
    0 1 2
    3 4 5  
    6 7 8
    
    Args:
        target_grid: Target grid position (0-8)
        ref_grid: Reference grid position (0-8)
        
    Returns:
        String description of relative position
    """
    target_row, target_col = divmod(target_grid, 3)
    ref_row, ref_col = divmod(ref_grid, 3)
    
    row_diff = target_row - ref_row
    col_diff = target_col - ref_col
    
    # Determine relative position
    if row_diff == 0 and col_diff == 0:
        return "at the same position as"
    elif row_diff == 0:  # Same row
        return "to the right of" if col_diff > 0 else "to the left of"
    elif col_diff == 0:  # Same column
        return "below" if row_diff > 0 else "above"
    else:  # Diagonal
        vertical = "below" if row_diff > 0 else "above"
        horizontal = "to the right of" if col_diff > 0 else "to the left of"
        return f"{vertical} and {horizontal}"


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