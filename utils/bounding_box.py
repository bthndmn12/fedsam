import numpy as np

def get_bounding_box(ground_truth_map):
    """Get a bounding box from a ground truth map
    Args:
        ground_truth_map (np.array): Ground truth map
        Returns:
            bbox (list): Bounding box coordinates"""
    # Check if the ground truth map is empty
    if np.sum(ground_truth_map) == 0:
        return [0, 0, ground_truth_map.shape[1], ground_truth_map.shape[0]]

    # Get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    # Add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    return bbox