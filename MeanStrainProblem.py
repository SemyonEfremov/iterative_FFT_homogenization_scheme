import numpy as np
import cupy as cp
import Tensor
from Problem import Problem
from ProblemParameters import ProblemParameters

class MeanStrainProblem(Problem):

    def __init__(self, problem_parameters: ProblemParameters) -> None:
        self.parameters = problem_parameters
        flat_order = np.array([self.parameters.tensor_size, 1])
        square_order = np.array([self.parameters.tensor_size,
                                 self.parameters.tensor_size])
        self.strain = Tensor.Tensor(flat_order,
                                    self.parameters.spatial_dimensions)
        self.strain.set_mean_value(self.parameters.mean_strain)
        self.stress = Tensor.Tensor(flat_order,
                                    self.parameters.spatial_dimensions)
        self.elasticity = Tensor.Tensor(square_order,
                                        self.parameters.spatial_dimensions)
        self.lame_parameters =\
            self.compute_lame_parameters(self.parameters.structure,
                                         self.parameters.min_young_modulus,
                                         self.parameters.max_young_modulus,
                                         self.parameters.poisson_ratio)
        elasticity_values =\
            self.compute_isotropic_elasticity(self.lame_parameters,
                                              self.parameters.tensor_size)
        self.elasticity.set_values(elasticity_values)
        self.domain_size = self.parameters.domain_size
        self.freq_vector = []
        for i in range(self.parameters.spatial_dimensions.size):
            freq_gpu = cp.fft.fftfreq(self.parameters.spatial_dimensions[i],
                (self.parameters.domain_size[i] /
                self.parameters.spatial_dimensions[i]))
            self.freq_vector.append(cp.asnumpy(freq_gpu))

    def execute_iteration(self, problem_parameters: ProblemParameters) -> None:
        pass

    def execute_method(self, problem_parameters: ProblemParameters) -> None:
        pass

    def reinitialize_state(self, problem_parameters: ProblemParameters) -> None:
        
        self.parameters = problem_parameters
        flat_order = np.array([self.parameters.tensor_size, 1])
        square_order = np.array([self.parameters.tensor_size,
                                 self.parameters.tensor_size])
        self.strain = Tensor.Tensor(flat_order,
                                    self.parameters.spatial_dimensions)
        self.strain.set_mean_value(self.parameters.mean_strain)
        self.stress = Tensor.Tensor(flat_order,
                                    self.parameters.spatial_dimensions)
        self.elasticity = Tensor.Tensor(square_order,
                                        self.parameters.spatial_dimensions)
        self.lame_parameters =\
            self.compute_lame_parameters(self.parameters.structure,
                                         self.parameters.min_young_modulus,
                                         self.parameters.max_young_modulus,
                                         self.parameters.poisson_ratio)
        elasticity_values =\
            self.compute_isotropic_elasticity(self.lame_parameters,
                                              self.parameters.tensor_size)
        self.elasticity.set_values(elasticity_values)
        self.domain_size = self.parameters.domain_size
        self.freq_vector = []
        for i in range(self.parameters.spatial_dimensions.size):
            freq_gpu = cp.fft.fftfreq(self.parameters.spatial_dimensions[i],
                (self.parameters.domain_size[i] /
                self.parameters.spatial_dimensions[i]))
            self.freq_vector.append(cp.asnumpy(freq_gpu))