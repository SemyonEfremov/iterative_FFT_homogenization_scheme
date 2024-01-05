import numpy as np
import cupy as cp
import os, psutil

class Tensor(object):

    # in case of a vetor we need to check whether order values.shape is (1 x m),
    # not (m)
    # maybe change order and dimensions from ndarray to immutable tuple
    def __init__(self,
                 order_values: np.ndarray,
                 spatial_dimensions: np.ndarray) -> None:
        self.order = order_values
        self.dimensions = spatial_dimensions
        self.tensor_values = np.zeros((np.concatenate((self.order,
                                                self.dimensions), axis = 0)))

    # need ot compare input.shape to self.order
    def set_mean_value(self, mean_value: np.ndarray) -> None:
        current_mean_value = self.calculate_mean_value()
        print(current_mean_value.shape)
        tensor_values_gpu = cp.zeros(tuple(self.dimensions))
        mean_value_delta = current_mean_value - mean_value
        print(mean_value_delta.shape)
        for row_idx in range(self.order[0]):
            for column_idx in range(self.order[1]):
                tensor_values_gpu \
                    = cp.array(self.tensor_values[row_idx,column_idx]) \
                    - cp.array(mean_value_delta[row_idx,column_idx])
                self.tensor_values[row_idx,column_idx] \
                    = cp.asnumpy(tensor_values_gpu)
        del tensor_values_gpu

    def calculate_mean_value(self) -> np.ndarray:
        mean_value_cpu = np.zeros(tuple(self.order))
        mean_value_gpu = cp.zeros(tuple(self.order))
        along_axes = tuple(range(-1, -(len(self.dimensions) + 1), -1))
        for row_idx in range(self.order[0]):
            for column_idx in range(self.order[1]):
                mean_value_gpu \
                    = cp.mean(cp.array(self.tensor_values[row_idx,column_idx]),
                                                            axis = along_axes)
                mean_value_cpu[row_idx,column_idx] = cp.asnumpy(mean_value_gpu)
        del mean_value_gpu
        return mean_value_cpu

    @staticmethod
    def compute_ddot_prod(t_1: np.ndarray,
                          t_2: np.ndarray)->np.ndarray:
        if t_1.shape[1] != t_2.shape[0]:
            print('ERROR: Impossible to compute double dot product \
                  (inconsistent tensors dimensions)')
            return np.array([])
        print((t_1.shape[0],)+t_2.shape[1:])
        ddot_values = np.zeros((t_1.shape[0],)+t_2.shape[1:])
        print(ddot_values.shape)
        ddot_values_gpu = cp.zeros(tuple(t_1.shape[2:]))
        print(ddot_values_gpu.shape)
        for row_idx in range(t_1.shape[0]):
            for col_idx in range(t_2.shape[1]):
                for sum_idx in range(t_1.shape[1]):
                    ddot_values_gpu = cp.array(t_1[row_idx,sum_idx])
                    ddot_values_gpu *= cp.array(t_2[sum_idx,col_idx])
                    ddot_values_gpu += cp.array(ddot_values[row_idx,col_idx])
                    ddot_values[row_idx,col_idx] = cp.asnumpy(ddot_values_gpu)
        del ddot_values_gpu
        return ddot_values

    def compute_ffourier(self) -> np.ndarray:
        along_axes = tuple(range(-1, -(len(self.dimensions) + 1), -1))
        tensor_values_gpu = cp.zeros(tuple(self.dimensions))
        tensor_values_fourier = np.zeros(self.tensor_values.shape)
        for row_idx in range(self.order[0]):
            for column_idx in range(self.order[1]):
                tensor_values_gpu \
                    = cp.array(self.tensor_values[row_idx,column_idx])
                tensor_values_gpu \
                    = cp.fft.fftn(tensor_values_gpu, axes = along_axes)
                tensor_values_fourier[row_idx,column_idx] \
                    = cp.asnumpy(tensor_values_gpu)
        del tensor_values_gpu
        return tensor_values_fourier

    def compute_bfourier(self) -> np.ndarray:
        along_axes = tuple(range(-1, -(len(self.dimensions) + 1), -1))
        tensor_values_gpu = cp.zeros(tuple(self.dimensions))
        tensor_values_spatial = np.zeros(self.tensor_values.shape)
        for row_idx in range(self.order[0]):
            for column_idx in range(self.order[1]):
                tensor_values_gpu \
                    = cp.array(self.tensor_values[row_idx,column_idx])
                tensor_values_gpu \
                    = cp.fft.ifftn(tensor_values_gpu, axes = along_axes)
                tensor_values_spatial[row_idx,column_idx] \
                    = cp.asnumpy(tensor_values_gpu)
        del tensor_values_gpu
        return tensor_values_spatial

    def compute_frob_norm(self) -> np.ndarray:
        norm_values_gpu = cp.zeros(tuple(self.dimensions))
        for row_idx in range(self.order[0]):
            for col_idx in range(self.order[1]):
                norm_values_gpu \
                    += cp.array(self.tensor_values[row_idx,col_idx])**2
        norm_values = cp.asnumpy(norm_values_gpu)
        return norm_values

    def get_values(self) -> np.ndarray:
        return np.copy(self.tensor_values)
    
    def set_values(self, values: np.ndarray) -> None:
        self.tensor_values = np.copy(values)

    def get_order(self) -> np.ndarray:
        return np.copy(self.order)

    def get_dimensions(self) -> np.ndarray:
        return np.copy(self.dimensions)

    def get_tensor_shape(self) -> tuple:
        return self.tensor_values.shape

    def get_info(self) -> tuple:
        return (self.tensor_values, self.order, self.dimensions)

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