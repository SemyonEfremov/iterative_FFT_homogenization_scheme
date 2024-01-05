import numpy as np

def kron_delta(x: int, y: int) -> int:
    if x == y:
        return 1
    else:
        return 0

def generate_test_structure(spatial_dimensions):
    structure = np.zeros(tuple(spatial_dimensions))
    if len(spatial_dimensions) == 2:
        for i in range(spatial_dimensions[0]):
            for j in range(spatial_dimensions[1]):
                if ((i / (spatial_dimensions[0] - 1) - 0.5)**2
                    + (j / (spatial_dimensions[1] - 1) - 0.5)**2) <= 0.2:
                    structure[i,j] = 1
    else:
        for i in range(spatial_dimensions[0]):
            for j in range(spatial_dimensions[1]):
                for l in range(spatial_dimensions[2]):
                    if ((i / (spatial_dimensions[0] - 1) - 0.5)**2
                        + (j / (spatial_dimensions[1] - 1) - 0.5)**2
                        + (l / (spatial_dimensions[2] - 1) - 0.5)**2) <= 0.2:
                        structure[i,j] = 1
    return structure
