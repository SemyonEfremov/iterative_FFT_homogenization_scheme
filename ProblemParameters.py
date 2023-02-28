import numpy as np

class ProblemParameters(object):
    
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ProblemParameters, cls).__new__(cls)
        return cls.instance

    def initialize_values(self, mean_strain: np.ndarray,
                          mean_stress: np.ndarray,
                          structure: np.ndarray,
                          max_young_modulus: float,
                          min_young_modulus: float,
                          poisson_ratio: float,
                          reference_lame: np.ndarray,
                          tensor_size: int,
                          spatial_dimensions: np.ndarray,
                          domain_size: np.ndarray) -> None:
        self.mean_strain = mean_strain
        self.mean_stress = mean_stress
        self.structure = structure
        self.max_young_modulus = max_young_modulus
        self.min_young_modulus = min_young_modulus
        self.poisson_ratio = poisson_ratio
        self.reference_lame = reference_lame
        self.tensor_size = tensor_size
        self.spatial_dimensions = spatial_dimensions
        self.domain_size = domain_size