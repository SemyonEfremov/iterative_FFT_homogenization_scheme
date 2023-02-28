import numpy as np
import cupy as cp
import Tensor
import SupplementalMethods as sup
from abc import ABC, abstractmethod

class Problem(ABC):

    @abstractmethod
    def execute_iteration(self, *args):
        pass

    @abstractmethod
    def execute_method(self, *args):
        pass

    @abstractmethod
    def reinitialize_state(self, *args):
        pass

    @staticmethod
    def compute_lame_parameters(structure: np.ndarray,
                                min_young_modulus: float,
                                max_young_modulus: float,
                                poisson_ratio: float) -> np.ndarray:
        dimensions = list(structure.shape)
        dimensions.insert(0, 2)
        lame_parameters = np.zeros(tuple(dimensions))
        if (poisson_ratio != -1) and (poisson_ratio != 0.5):
            multiplier = poisson_ratio\
                / ((1 + poisson_ratio)*(1 - 2*poisson_ratio))
        else:
            multiplier = 0
        young_mod_distribution_gpu = (max_young_modulus
            - min_young_modulus)*cp.array(structure) + min_young_modulus
        lame_parameters[0] = cp.asnumpy(multiplier*young_mod_distribution_gpu)
        if poisson_ratio != -1:
            multiplier = 1 / (2*(1 + poisson_ratio))
        else:
            multiplier = 0
        lame_parameters[1] = cp.asnumpy(multiplier*young_mod_distribution_gpu)
        return lame_parameters

    @staticmethod
    def compute_isotropic_elasticity(lame_parameters: np.ndarray,
                                     tensor_size: int) -> np.ndarray:
        if (lame_parameters.size > 2):
            size = np.concatenate((np.array([tensor_size, tensor_size]),
                                   lame_parameters[0].shape))
            size = tuple(size)
        else:
            size = (tensor_size, tensor_size)
        elasticity_tensor = np.zeros(size)
        for row_idx in range(tensor_size):
            for col_idx in range(tensor_size):
                elasticity_element_gpu = sup.kron_delta(row_idx, col_idx)*\
                    cp.array(lame_parameters[1])
                if (((col_idx / tensor_size) <= 0.5) and
                    ((row_idx / tensor_size) <= 0.5)):
                    elasticity_element_gpu += cp.array(lame_parameters[0]) +\
                        sup.kron_delta(row_idx, col_idx)*\
                        cp.array(lame_parameters[1])
                elasticity_tensor[row_idx, col_idx] =\
                    cp.asnumpy(elasticity_element_gpu)
        return elasticity_tensor

    @staticmethod
    def compute_in_product(tensor_left: Tensor.Tensor,
                           tensor_right: Tensor.Tensor) -> np.ndarray:
        values_left = tensor_left.get_values()
        values_right = tensor_right.get_values()
        order_left = tensor_left.get_order()
        order_right = tensor_right.get_order()
        product_values = np.zeros(np.concatenate(order_left[0], order_right[1],\
            tensor_left.get_dimensions()))
        product_values_gpu = cp.zeros(values_left.shape)
        for row_idx in range(order_left[0]):
            for column_idx in range(order_right[1]):
                for sum_idx in range(order_right[0]):
                    product_values_gpu =\
                        cp.array(values_left[row_idx,sum_idx])
                    product_values_gpu *=\
                        cp.array(values_right[sum_idx,column_idx])
                    product_values_gpu +=\
                        cp.array(product_values[row_idx,column_idx])
                    product_values[row_idx,column_idx] =\
                        cp.asnumpy(product_values_gpu)
        return product_values