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

    # Calculates a standart notation 4-digit index ("quad_index")
    # for an element of a 4th order tensor
    # based on a given 2-digit Voigt index ("double_index") in 3D space
    @staticmethod
    def calculate_t_index_2d(double_index: list[int]) -> list[int]:
        quad_index = []
        for index in double_index:
            if index < 2:
                quad_index.append(index); quad_index.append(index)
            elif index == 2:
                quad_index.append(0); quad_index.append(1)
        return quad_index
    
    # Calculates a standart notation 4-digit index ("quad_index")
    # for an element of a 4th order tensor
    # based on a given 2-digit Voigt index ("double_index") in 3D space
    @staticmethod
    def calculate_t_index_3d(double_index: list[int]) -> list[int]:
        quad_index = []
        for index in double_index:
            if index < 3:
                quad_index.append(index); quad_index.append(index)
            elif index == 3:
                quad_index.append(1); quad_index.append(2)
            elif index == 4:
                quad_index.append(0); quad_index.append(2)
            elif index == 5:
                quad_index.append(0); quad_index.append(1)
        return quad_index
    
    @staticmethod
    def calculate_permutations(quad_index: list[int]) -> list[list[int]]:
        permutations = []
        permutations.append([quad_index[0], quad_index[2], quad_index[1],
            quad_index[3]])
        permutations.append([quad_index[0], quad_index[3], quad_index[1],
            quad_index[2]])
        permutations.append([quad_index[1], quad_index[2], quad_index[0],
            quad_index[3]])
        permutations.append([quad_index[1], quad_index[3], quad_index[0],
            quad_index[2]])
        return permutations

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
        product_values = np.zeros(np.concatenate(order_left[0], order_right[1],
            tensor_left.get_dimensions()))
        product_values_gpu = cp.zeros(values_left.shape)
        for row_idx in range(order_left[0]):
            for column_idx in range(order_right[1]):
                for sum_idx in range(order_right[0]):
                    product_values_gpu \
                        = cp.array(values_left[row_idx,sum_idx])
                    product_values_gpu \
                        *= cp.array(values_right[sum_idx,column_idx])
                    product_values_gpu \
                        += cp.array(product_values[row_idx,column_idx])
                    product_values[row_idx,column_idx] \
                        = cp.asnumpy(product_values_gpu)
        return product_values

    # Computes continuous Green tensor values for mean strain based problem.
    # "freq_vector" is a frequency vector corresponding to the computational
    # domain.
    # "lame_reference" are constant lame parameters corresponding to chosen
    # reference media.
    @staticmethod
    def compute_green_strain(freq_vector: list[np.ndarray],
                             lame_reference: np.ndarray,
                             tensor_size: int) -> np.ndarray:
        tensor_shape = [tensor_size, tensor_size]
        for freq_element in freq_vector:
            tensor_shape.append(freq_element.size)
        green_values = np.zeros(tuple(tensor_shape))
        print(len(freq_vector))
        
        if len(freq_vector) == 2:
            freq_mesh \
                = np.meshgrid(freq_vector[0], freq_vector[1])
            freq_mesh[0] = freq_mesh[0].T
            freq_mesh[1] = freq_mesh[1].T
            freq_abs = cp.asnumpy(cp.array(freq_mesh[0])**2
                + cp.array(freq_mesh[1])**2)
            freq_abs[0,0] = 1
            for row_idx in range(tensor_size):
                for col_idx in range(tensor_size):
                    std_idx = Problem.calculate_t_index_2d([row_idx, col_idx])
                    #print(std_idx)
                    
                    green_values_gpu = cp.array(freq_mesh[std_idx[0]])
                    for idx in std_idx[1:]:
                        green_values_gpu *= cp.array(freq_mesh[idx])
                    green_values_gpu /= cp.array(freq_abs)
                    green_values_gpu *= - (lame_reference[0]
                        + lame_reference[1]) / (lame_reference[1]
                        *(lame_reference[0] + 2*lame_reference[1]))
                    #green_values_gpu += (0.25 / lame_reference[1])\
                    #    * cp.array(freq_mesh[std_idx[0]])
                    green_values[row_idx, col_idx] \
                        = cp.asnumpy(green_values_gpu)

                    ind_perms = Problem.calculate_permutations(std_idx)
                    for perm in ind_perms:
                        if sup.kron_delta(perm[0], perm[1]):
                            green_values_gpu = cp.array(freq_mesh[perm[2]])
                            green_values_gpu *= cp.array(freq_mesh[perm[3]])
                            green_values_gpu *= 0.25 / lame_reference[1]
                            green_values_gpu \
                                += cp.array(green_values[row_idx, col_idx])
                            green_values[row_idx, col_idx] \
                                = cp.asnumpy(green_values_gpu)
                    green_values_gpu /= cp.array(freq_abs)
                    
                    green_values[row_idx, col_idx] \
                        = cp.asnumpy(green_values_gpu)
        else:
            freq_mesh \
                = np.meshgrid(freq_vector[1], freq_vector[0], freq_vector[2])
            freq_mesh[0], freq_mesh[1] = freq_mesh[1], freq_mesh[0]
            freq_abs = cp.asnumpy(cp.array(freq_mesh[0])**2
                + cp.array(freq_mesh[1])**2 + cp.array(freq_mesh[2])**2)
            freq_abs[0,0,0] = 1
            for row_idx in range(tensor_size):
                for col_idx in range(tensor_size):
                    std_idx = Problem.calculate_t_index_3d([row_idx, col_idx])
                    #print(std_idx)
                    
                    green_values_gpu = cp.array(freq_mesh[std_idx[0]])
                    for idx in std_idx[1:]:
                        green_values_gpu *= cp.array(freq_mesh[idx])
                    green_values_gpu /= cp.array(freq_abs)
                    green_values_gpu *= - (lame_reference[0]
                        + lame_reference[1]) / (lame_reference[1]
                        *(lame_reference[0] + 2*lame_reference[1]))
                    #green_values_gpu += (0.25 / lame_reference[1])\
                    #    * cp.array(freq_mesh[std_idx[0]])
                    green_values[row_idx, col_idx] \
                        = cp.asnumpy(green_values_gpu)
                    
                    ind_perms = Problem.calculate_permutations(std_idx)
                    for perm in ind_perms:
                        if sup.kron_delta(perm[0], perm[1]):
                            green_values_gpu = cp.array(freq_mesh[perm[2]])
                            green_values_gpu *= cp.array(freq_mesh[perm[3]])
                            green_values_gpu *= 0.25 / lame_reference[1]
                            green_values_gpu \
                                += cp.array(green_values[row_idx, col_idx])
                            green_values[row_idx, col_idx] \
                                = cp.asnumpy(green_values_gpu)
                    green_values_gpu /= cp.array(freq_abs)
                    
                    green_values[row_idx, col_idx] \
                        = cp.asnumpy(green_values_gpu)
        return green_values

    # Computes continuous Green tensor values for mean stress based problem.
    # "freq_vector" is a frequency vector corresponding to the computational
    # domain.
    # "lame_reference" are constant lame parameters corresponding to chosen
    # reference media.
    @staticmethod
    def compute_green_stress(freq_vector: list,
                             lame_reference: np.ndarray,
                             tensor_size: int) -> np.ndarray:
        tensor_shape = [tensor_size, tensor_size]
        for freq_element in freq_vector:
            tensor_shape.append(freq_element.size)
        green_values = np.zeros(tuple(tensor_shape))
        for row_idx in range(tensor_size):
            for col_idx in range(tensor_size):
                green_values_gpu = cp.array(freq_vector[0])
                green_values[row_idx, col_idx] = cp.asnumpy(green_values_gpu)
        return green_values

    # Computes discrete Green function values for mean strain based problem.
    # "freq_vector" is a frequency vector corresponding to the computational
    # domain.
    # "lame_reference" are constant lame parameters corresponding to chosen
    # reference media.
    @staticmethod
    def compute_green_discrete(freq_vector: list,
                               lame_reference: np.ndarray,
                               tensor_size: int) -> np.ndarray:
        tensor_shape = [tensor_size]
        for freq_element in freq_vector:
            tensor_shape.append(freq_element.size)
        green_values = np.zeros(tuple(tensor_shape))
        for elem_idx in range(tensor_size):
                green_values_gpu = cp.array(freq_vector[0])
                green_values[elem_idx] = cp.asnumpy(green_values_gpu)
        return green_values