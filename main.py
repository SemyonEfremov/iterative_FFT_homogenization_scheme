import numpy as np
import cupy as cp
import Problem
from MeanStrainProblem import MeanStrainProblem
from ModelParameters import ModelParameters
from ModelParameters2D import ModelParameters2D
from ModelParameters3D import ModelParameters3D
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
    model_param = ModelParameters2D()
    model_param_1 = ModelParameters2D()
    model_param_2 = ModelParameters3D()
    print(model_param is model_param_1)
    print(model_param is model_param_2)
    print(model_param_2.domain_size)
    print(model_param.domain_size)
    #model_param.initialize_values(mean_strain, mean_strain, structure, 1.0,
    #                                0.01, 0.3, np.array([1, 1]), tensor_size,
    #                                spatial_dimensions, domain_size)
    #print(problem_param.structure)
    test_program = MeanStrainProblem(model_param)
    #elasticity = test_program.elasticity.get_values()
    #test_mesh = test_program.green_tensor.get_values()
    #print(test_mesh.shape)
    #print(test_program.parameters.method_precision)
    #print(len(test_program.elasticity.get_values().shape))
    #print(test_program.strain.get_tensor_shape())
    #stress = Tensor.compute_ddot_prod(test_program.elasticity.get_values(),
    #                                  test_program.strain.get_values())
    #print(stress.shape)
    print(test_program.elasticity.get_values()[:,:,0,0])
    print(test_program.strain.get_values()[:,:,0,0])
    print(test_program.stress.get_values()[:,:,0,0])
    #print(stress[:,:,0,0])
    #print(test_mesh[0,1,0,:])
    #print(test_mesh[0,1,:,0])
    #print(Problem.Problem.calculate_t_index_2d([1, 1]))