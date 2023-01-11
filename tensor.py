import numpy as np
import cupy as cp

class Tensor(object):

    def __init__(self, order_values: np.ndarray, spatial_dimensions: np.ndarray) -> None:
        self.order = order_values
        self.dimensions = spatial_dimensions
        self.tensor_values = np.zeros((np.concatenate((self.order, self.dimensions), axis = 0)))

    def set_mean_value(self, mean_value: np.ndarray) -> None:
        current_mean_value = self.calculate_mean_value()
        mean_value_delta = current_mean_value - mean_value
        tensor_values_gpu = cp.array(self.tensor_values) - cp.array(mean_value_delta)
        self.tensor_values = cp.asnumpy(tensor_values_gpu)
        del tensor_values_gpu

    def calculate_mean_value(self) -> np.ndarray:
        along_axes = tuple(range(-1, -(len(self.dimensions) + 1), -1))
        mean_value_gpu = cp.mean(cp.array(self.tensor_values), along_axes)
        mean_value_cpu = cp.asnumpy(mean_value_gpu)
        del mean_value_gpu
        return mean_value_cpu

    def compute_ffourier(self) -> np.ndarray:
        tensor_values_gpu = cp.array(self.tensor_values)
        return np.zeros(1)

    def compute_bfourier(self) -> np.ndarray:
        return np.zeros(1)

    def get_info(self) -> tuple:
        return self.tensor_values, self.order, self.dimensions

if __name__ == "__main__":
    t_order = np.array([6,1])
    t_dimensions = np.array([2, 2, 2])
    print(t_order, t_dimensions)
    t_tensor = Tensor(t_order, t_dimensions)
    temp = t_tensor.tensor_values
    print(temp[:,:,0,0,0])
    temp[0,0,0,0,0] = 1
    print(t_tensor.tensor_values[:,:,0,0,0])