import numpy as np
import cupy as cp
import Problem
from MeanStrainProblem import MeanStrainProblem
from ProblemParameters import ProblemParameters
from Tensor import Tensor

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

if __name__ == "__main__":
    mean_strain = np.array([1, 0, 0, 0, 0, 0])
    tensor_size = 6
    spatial_dimensions = np.array([10, 13, 5])
    domain_size = np.array([1, 1, 1])
    structure = generate_test_structure(spatial_dimensions)
    #print(structure)
    problem_param = ProblemParameters()
    problem_param.initialize_values(mean_strain, mean_strain, structure, 1.0,
                                    0.01, 0.3, np.array([1, 1]), tensor_size,
                                    spatial_dimensions, domain_size)
    #print(problem_param.structure)
    test_program = MeanStrainProblem(problem_param)
    elasticity = test_program.elasticity.get_values()
    #print(test_program.lame_parameters[:,0,0])
    #print(elasticity[:,:,0,0])
    #print(test_program.lame_parameters[:,5,5])
    #print(elasticity[:,:,5,5])
    #print(elasticity[0,0,:,:])
    #print(elasticity[0,1,:,:])
    print(test_program.freq_vector[0])
    print(test_program.freq_vector[1])
    print(test_program.freq_vector[2])
    test_mesh = Problem.Problem.compute_green_strain(test_program.freq_vector,
                                                     np.array([1.0, 2.0]),
                                                     tensor_size)
    test_tensor = Tensor(np.array([tensor_size, tensor_size]),
                                  spatial_dimensions)
    print(test_mesh[0,1,0,:])
    print(test_mesh[0,1,:,0])
    print(Problem.Problem.calculate_t_index_2d([1, 1]))