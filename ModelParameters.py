import numpy as np

class ModelParameters(object):
    
    def __new__(cls):
        if hasattr(ModelParameters, 'instance'):
            return ModelParameters.instance
        for child in ModelParameters.__subclasses__():
            if hasattr(child, 'instance'):
                return child.instance
        cls.instance = super(ModelParameters, cls).__new__(cls)
        cls.instance.mean_strain = np.array([])
        cls.instance.mean_stress = np.array([])
        cls.instance.structure = np.array([])
        cls.instance.max_young_modulus = 0.0
        cls.instance.min_young_modulus = 0.0
        cls.instance.poisson_ratio = 0.0
        cls.instance.reference_lame = np.array([])
        cls.instance.tensor_size = 0
        cls.instance.spatial_dimensions = np.array([])
        cls.instance.domain_size = np.array([])
        cls.method_precision = 1.0
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