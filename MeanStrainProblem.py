import numpy as np
import Tensor
import Problem

class MeanStrainProblem(Problem.Problem):

    def __init__(self, order_values: np.ndarray, spatial_dimensions: np.ndarray) -> None:
        self._strain = Tensor.Tensor(order_values, spatial_dimensions)
        self._stress = Tensor.Tensor(order_values, spatial_dimensions)
        self._elasticity = Tensor.Tensor(order_values, spatial_dimensions)