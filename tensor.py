import numpy as np
import cupy as cp
import os, psutil

class Tensor(object):

    # in case of a vetor we need to check whether order values.shape is (1 x m),
    # not (m)
    def __init__(self, order_values: np.ndarray, spatial_dimensions: np.ndarray) -> None:
        self.order = order_values
        self.dimensions = spatial_dimensions
        self.tensor_values = np.zeros((np.concatenate((self.order, self.dimensions), axis = 0)))

    # need ot compare input.shape to self.order
    def set_mean_value(self, mean_value: np.ndarray) -> None:
        current_mean_value = self.calculate_mean_value()
        tensor_values_gpu = cp.zeros(tuple(self.dimensions))
        mean_value_delta = current_mean_value - mean_value
        for row_idx in range(self.order[0]):
            for column_idx in range(self.order[1]):
                tensor_values_gpu = cp.array(self.tensor_values[row_idx,column_idx]) - cp.array(mean_value_delta[row_idx,column_idx])
                self.tensor_values[row_idx,column_idx] = cp.asnumpy(tensor_values_gpu)
        del tensor_values_gpu

    def calculate_mean_value(self) -> np.ndarray:
        mean_value_cpu = np.zeros(tuple(self.order))
        mean_value_gpu = cp.zeros(tuple(self.order))
        along_axes = tuple(range(-1, -(len(self.dimensions) + 1), -1))
        for row_idx in range(self.order[0]):
            for column_idx in range(self.order[1]):
                mean_value_gpu = cp.mean(cp.array(self.tensor_values[row_idx,column_idx]), axis = along_axes)
                mean_value_cpu[row_idx,column_idx] = cp.asnumpy(mean_value_gpu)
        del mean_value_gpu
        return mean_value_cpu

    def compute_ffourier(self) -> np.ndarray:
        along_axes = tuple(range(-1, -(len(self.dimensions) + 1), -1))
        tensor_values_gpu = cp.array(self.tensor_values)
        tensor_values_gpu = cp.fft.fftn(tensor_values_gpu, axes = along_axes)
        tensor_values_fourier = cp.asnumpy(tensor_values_gpu)
        del tensor_values_gpu
        return tensor_values_fourier

    def compute_bfourier(self) -> np.ndarray:
        along_axes = tuple(range(-1, -(len(self.dimensions) + 1), -1))
        tensor_values_gpu = cp.array(self.tensor_values)
        tensor_values_gpu = cp.fft.ifftn(tensor_values_gpu, axes = along_axes)
        tensor_values_spatial = cp.asnumpy(tensor_values_gpu)
        del tensor_values_gpu
        return tensor_values_spatial

    def get_values(self) -> np.ndarray:
        return np.copy(self.tensor_values)
    
    def set_values(self, values: np.ndarray) -> None:
        self.tensor_values = np.copy(values)

    def get_info(self) -> tuple:
        return self.tensor_values, self.order, self.dimensions

    def check_attributes(self) -> None:
        pass

if __name__ == "__main__":
    t_order = np.array([6,1])
    t_dimensions = np.array([300,300,300])
    t_tensor = Tensor(t_order, t_dimensions)
    values = t_tensor.get_values()
    print(values[:,:,0,0,0])
    print(t_tensor.calculate_mean_value())
    new_mean = np.array([1,3,5,7,9,11]).reshape(6,1)
    print(new_mean)
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss)  # in bytes
    t_tensor.set_mean_value(new_mean)
    values = t_tensor.get_values()
    print(values[:,:,5,2,6])
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss)  # in bytes