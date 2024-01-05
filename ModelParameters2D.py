import numpy as np
from ModelParameters import ModelParameters
import SupplementalMethods

class ModelParameters2D(ModelParameters):
    
    def __new__(cls):
        if hasattr(cls, 'instance'):
            return cls.instance
        cls.instance = super().__new__(cls)
        if cls.__name__ == type(cls.instance).__name__:
            cls.instance.mean_strain = np.array([1.0, 0.0, 0.0])
            cls.instance.mean_stress = np.array([1.0, 0.0, 0.0])
            cls.instance.max_young_modulus = 1.0
            cls.instance.min_young_modulus = 0.1
            cls.instance.poisson_ratio = 0.3
            cls.instance.reference_lame = np.array([1.0, 1.0])
            cls.instance.tensor_size = 3
            cls.instance.spatial_dimensions = np.array([100, 100])
            cls.instance.structure =\
                SupplementalMethods.generate_test_structure(
                cls.instance.spatial_dimensions)
            cls.instance.domain_size = np.array([1.0, 1.0])
            cls.method_precision = 1e-3
        return cls.instance