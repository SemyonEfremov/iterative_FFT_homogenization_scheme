import numpy as np
import cupy as cp

class Tensor(object):

    def __init__(self, order_values: np.ndarray, spatial_dimensions: np.ndarray) -> None:
        self._order = order_values
        self._dimensions = spatial_dimensions
        self._tensor_values = np.zeros((np.concatenate((self._order, self._dimensions), axis = 0)))

    def set_mean_value(self, mean_value: np.ndarray) -> None:
        pass

    def calculate_mean_value(self) -> np.ndarray:
        return np.zeros(1)

    @staticmethod
    def calculate_in_product(t_values_a: Tensor, t_values_b: Tensor)
        pass

    def compute_ffourier(self) -> np.ndarray:
        return np.zeros(1)

    def compute_bfourier(self) -> np.ndarray:
        return np.zeros(1)

    def get_info(self) -> tuple:
        return self._tensor_values, self._order, self._dimensions

if __name__ == "__main__":
    t_order = np.array([6,1])
    t_dimensions = np.array([2, 2, 2])
    print(t_order, t_dimensions)
    t_tensor = Tensor(t_order, t_dimensions)
    temp = t_tensor._tensor_values
    print(temp[:,:,0,0,0])
    temp[0,0,0,0,0] = 1
    print(t_tensor._tensor_values[:,:,0,0,0])