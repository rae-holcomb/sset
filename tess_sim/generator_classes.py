# %matplotlib widget
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
# import pandas as pd

import butterpy as bp
import ellc
import typing
from scipy import stats
import trc_funcs as trc
# import PRF

# import eleanor
# import astropy.wcs as wcs
# from astropy import units as u
# from astropy.coordinates import SkyCoord
# from astroquery.mast import Tesscut
# from astropy.stats import sigma_clip
# import scipy.signal
# import astropy.table

from dataclasses import dataclass

class MixtureModel(stats.rv_continuous):
    def __init__(self, submodels, *args, **kwargs):
        """Add."""
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        self.dist = 'MixtureModel'
        # self.kwds = ''

    def __repr__():
        return 'MixtureModel'        

    def _pdf(self, x):
        """Add"""
        pdf = self.submodels[0].pdf(x)
        for submodel in self.submodels[1:]:
            pdf += submodel.pdf(x)
        pdf /= len(self.submodels)
        return pdf

    def rvs(self, size=None):
        """Add"""
        submodel_choices = np.random.randint(len(self.submodels), size=size)
        submodel_samples = [submodel.rvs(size=size) for submodel in self.submodels]
        rvs = np.choose(submodel_choices, submodel_samples)
        return rvs

# @dataclass
class TSGenerator():
    """A generic class that generates a signal in a light curve."""

    def __init__(self, name, params:typing.Dict[str,stats.rv_continuous]={}) -> None:
        # define each arg distribution in the definition
        # func needs to be in the format func(time, **kwargs)
        # name must be a string
        self.name = name
        self.params = params.copy()

        # loop through and convert floats to distributions as needed
        for key, value in params.items():
            self.params[key] = trc.convert_to_distribution(value)

        # # loop through and assign the parameters to class variables
        # for key, value in params.items():
        #     setattr(self, key, trc.convert_to_distribution(value))
        pass

    # def __post_init__(self) -> None:
    #     print('post init')
    #     pass
    #     """Loop through and convert all attributes to distributions.
    #     This isn't getting called for some reason and I can't figure out why."""
    #     print('__post_init__')
    #     for key, value in self.params.items():
    #         self.setattr(self, key, trc.convert_to_distribution(value))
    #     # pass

    def __str__(self) -> str:
        # print out a summary of the args and their distributions
        output = str(self.name) + '\n'
        # loop through and add distribution parameters
        # for key, value in self.__dict__.items():
        for key, value in self.params.items():
            if key == 'name':
                continue
            elif isinstance(value, (int, float)):
                output += '\t' + key + ': ' + str(value) + '\n'
            elif isinstance(value, MixtureModel):
                output += '\t' + key + ': ' + 'MixtureModel' + '\n'
                output += '\t\t' + str([str(model.dist).split(" ")[0] + '>' for model in value.submodels]) + '\n'
                output += '\t\t' + str([model.kwds for model in value.submodels])  + '\n'
            elif isinstance(value.dist, stats.rv_continuous):
                output += '\t' + key + ': ' + str(value.dist).split(" ")[0] + '>, ' + str(value.kwds) + '\n'
            else: 
                output += '\t' + key + ': ' + 'Unrecognized distribution type' + '\n'
        
        return output

    def __repr__(self):
        return "TSGenerator"
        
    def __iter__(self, key):
        return self.__dict__[key]

    def update_param(self, name:str, distr: stats.rv_continuous) -> None:
        """Update or add a distribution for a particular parameter. Must be a [input format here]."""
        setattr(self, name, distr)
        pass

    def delete_param(self, name:str) -> None:
        """Remove a parameter."""
        delattr(self, name)
        pass

    def plot_parameter(self, name:str) -> None:
        raise NotImplementedError

    def sample(self) -> typing.Dict:
        """Pulls a value from each parameter distribution."""
        # iterate over the parameters and pull one value from each distribution
        param_values = self.params.copy()
        # del param_values['name']
        # del param_values['params']
        for key, distr in param_values.items():
            param_values[key] = distr.rvs()
        return param_values

    def generate_signal(self, time, **kwargs) -> typing.Tuple[np.ndarray, typing.Dict] :
        """To be supplied by the user. Must take time as a positional argument, and return a tuple containing the flux array as the first argument and a dictionary with the selected parameter values as the second."""
        raise NotImplementedError

        # # make copy of the kwargs for this specific instantiation
        # params = self.kwargs.copy()

        # # draw parameter values from the distributions
        # for key, value in self.distributions.items():
        #     params[key] = value.rvs(1)[0]
        
        # print(params)
        # print(len(time))

        # # calculate the flux
        # flux = self.func(time, **params)
        # return flux, params


# class FunctionSelector:
class FunctionSelector:
    """Used to select what functions get inputed.
    TO DO: restrict function input types."""
    def __init__(self, generators: typing.List[typing.Tuple[TSGenerator, float]] = None):
        self.generators = {}
        self.weights = {}
        
        if generators is not None:
            for item in generators:
                self.generators[item[0].name] = item[0]
                self.weights[item[0].name] = item[1]
        # self.weights = {}

    def __str__(self) -> str:
        output = ''
        for key in self.generators.keys():
            output += 'Name: ' + key + ', Weight: ' + str(self.weights[key]) + '\n'
        return output
    
    def print_verbose() -> str:
        """Prints out a long version, with all the functions and their kwargs listed."""
        raise NotImplementedError
        pass

    def add_generator(self, name:str, generator:TSGenerator, weight:float) -> None:
        """Adds a function to the options which can be selected.
        TO DO: restrict function input types."""
        self.generators[name] = (generator, weight)
        # self.weights[name] = weight

    def delete_generator(self, name:str) -> None:
        """Deletes a function from the selection options."""
        del self.generators[name]
        # del self.weights[name]

    def update_weight(self, name:str, weight:float) -> None:
        """Updates the weight on a given generator"""
        self.generators[name] = (self.generators[name][0], weight)

    def select_generator(self) -> TSGenerator:
        # (OLD) selected_generator = random.choices(list(zip(*self.funcs.values()))[0], weights=list(zip(*self.funcs.values()))[1])
        keys = list(self.weights.keys())
        selected_key = random.choices(keys, weights=[self.weights[key] for key in keys], k=10)
        return selected_key
    
    def instantiate_function(self, time:np.ndarray) -> [np.ndarray, typing.Dict]:
        """Add."""
        # pick what type of function
        selected_key = self.select_generator()
        # print(selected_key)

        # apply the function to the time array and return the flux and the selected args
        # print(type(selected_key))
        # func = self.funcs[selected_key][0]
        flux, params = self.generators[selected_key].generate_signal(time)
        return flux, params


# subclass from TSGenerator
class SineTSGenerator(TSGenerator):
    def __init__(self, name, params:typing.Dict[str,stats.rv_continuous]={'A':1, 'B':1, 'C':0, 'D':0}) -> None:
        # define each arg distribution in the definition
        # func needs to be in the format func(time, **kwargs)
        # name must be a string
        self.name = name
        self.params = params

        # loop through and assign the parameters to class variables
        for key, value in params.items():
            setattr(self, key, trc.convert_to_distribution(value))
        pass


    # def __post_init__(self, A:stats.rv_continuous=1, B:stats.rv_continuous=1, C:stats.rv_continuous=0, D:stats.rv_continuous=0) -> None:
    #     """
    #     Sine function of the form y = A * sin(B(x+C) + D).

    #     Parameters
    #     ----------

    #     A: stats.Normal
    #     B:
    #     C:
    #     D:  
    #     """
    #     self.A = trc.convert_to_distribution(A)
    #     self.B = trc.convert_to_distribution(B)
    #     self.C = trc.convert_to_distribution(C)
    #     self.D = trc.convert_to_distribution(D)


    def generate_signal(self, time) -> np.ndarray:
        """....size time"""
        return self.A.rvs() * np.sin(self.B.rvs()*(time+self.C.rvs()) + self.D.rvs())
        # return np.sin(self.A.sample() * time) + self.offset.sample()

    def __repr__(self):
        return "SineTSGenerator"


class EclipsingBinaryTSGenerator(TSGenerator):
    def __init__(self, name, params:typing.Dict[str,stats.rv_continuous]={}) -> None:
        """
        Eclipsing Binary using the ellc package. Has several parameter distributions set by default, as noted below. Only use the params input to modify from this default.

        Parameters
        ----------

        A: stats.Normal
            
        """
        self.name = name
        self.params = {
            'radius_1': ,
            'radius_2': ,
            'incl':  ,
            'ecc': ,
            'om': ,
            'period': ,
            'sbratio': ,
            't_zero': 
        }
        
        
        params.copy()

        # loop through and assign the parameters to class variables
        for key, value in params.items():
            self.params[key] = trc.convert_to_distribution(value)
            setattr(self, key, trc.convert_to_distribution(value))
        pass

    def __repr__(self):
        return "EclipsingBinaryTSGenerator"
        
    def generate_signal(self, time, param_values=None) -> np.ndarray:
        """Add."""
        if param_values is None:
            # sample the distributions
            param_values = self.sample(self.params)

        # calculate f_c, f_s, and any other calculable values
        ecc = param_values['ecc']
        om = param_values['om']
        param_values['f_c'] = np.sqrt(ecc)*np.cos(om)
        param_values['f_s'] = np.sqrt(np.abs(ecc - param_values['f_c']**2))
        param_values['t_zero'] = param_values['t_zero'] * param_values['period']

        # delete non-kwarg parameters
        del param_values['ecc']
        del param_values['om']

        # calculate the flux
        flux = ellc.lc(time, **param_values)
        return flux, param_values


   