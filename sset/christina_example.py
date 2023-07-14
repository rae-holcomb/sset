class TSGenerator(ABC):
    """Base Class"""
    def __init__(self, params:dict[stats.distributions]):
        for key, value in self.kwargs.items():
            setattr(self, key, value)
        raise NotImplementedError

    def __iter__(self, key):
        return self.params_list[key]

    def __repr__(self):
        return "TSGenerator"

    def sample(self, time):
        raise NotImplementedError
    
    def update(self, params):
        # iterate through and update
        raise NotImplementedError

    def sample_parameter(self, param_name):
        raise NotImplementedError

    def plot_parameters(self):
        raise NotImplementedError


class SineTSGenerator(TSGenerator):

    def __init__(self, name:str, A:float|stats.distribution=1, offset|stats.distribution:float=0):
        """
        Sine?

        Parameters
        ----------

        A: stats.Normal
            
        """
        self.A = convert_to_distribution(A)
        self.offset = convert_to_distribution(offset)

    def sample(self, time)->np.ndarray:
        """....size time"""
        return np.sin(self.A.sample() * time) + self.offset.sample()

    def __repr__(self):
        return "SineTSGenerator"


def convert_to_distribution(input:float|stats.distribution):
    # logic to recast float to dist if necessary
    return output

SineTSGenerator(A=1, offset=stats.distribution(0, 0, 1)).sample() 