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

        # # loop through and convert floats to distributions as needed
        # for key, value in params.items():
        #     self.params[key] = trc.convert_to_distribution(value)

        # # loop through and assign the parameters to class variables
        # for key, value in params.items():
        #     setattr(self, key, trc.convert_to_distribution(value))
        pass

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

    def sample(self) -> typing.Dict[str, float]:
        """Pulls a value from each parameter distribution."""
        # iterate over the parameters and pull one value from each distribution
        param_values = self.params.copy()
        # del param_values['name']
        # del param_values['params']
        for key, distr in param_values.items():
            if isinstance(distr, (float, int)):
                param_values[key] = distr
            else:
                param_values[key] = distr.rvs()
        return param_values

    def generate_signal(self, time:np.ndarray, **kwargs) -> typing.Tuple[np.ndarray, typing.Dict[str,float]] :
        """This function draws a value from each of the parameter distributions and then calls self.functional_form() with those parameter values to generate a signal for the given time array. Must take time as a positional argument, and return a tuple containing the flux array as the first argument and a dictionary with the selected parameter values as the second."""
        param_values = self.sample()
        return self.functional_form(time, param_values)

    def functional_form(self, time:np.ndarray, params:typing.Dict[str,stats.rv_continuous]) -> typing.Tuple[np.ndarray, typing.Dict[str,float]] :
        """To be supplied by the user. This function provides the functional form to generate a signal given a time array. It must take time and a dictionary of parameters as positional arguments, and return a tuple containing the flux array as the first argument and a dictionary with the selected parameter values as the second."""
        raise NotImplementedError


# class FunctionSelector:
class FunctionSelector():
    """Used to select what functions get inputed.
    TO DO: restrict function input types.
    The generators list should be [obj:TSGenerator, weight:float]"""
    def __init__(self, generators: typing.List[typing.Tuple[TSGenerator, float]] = None):
    # def __init__(self, generators: list[tuple[TSGenerator, float]] = None):
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
    
    def print_verbose(self) -> str:
        """Prints out a long version, with all the functions and their kwargs listed."""
        raise NotImplementedError

    def add_generator(self, generator:TSGenerator, weight:float) -> None:
        """Adds a function to the options which can be selected.
        TO DO: restrict function input types."""
        # self.generators[name] = (generator, weight)
        self.generators[generator.name] = generator
        self.weights[generator.name] = weight

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
        selected_key = random.choices(keys, weights=[self.weights[key] for key in keys], k=1)[0]
        return selected_key
    
    def instantiate_function(self, time:np.ndarray, **kwargs) -> typing.Tuple[np.ndarray, typing.Dict]:
        """Add."""
        # pick what type of function
        selected_key = self.select_generator()
        # print(selected_key)

        # apply the function to the time array and return the flux and the selected args
        # print(type(selected_key))
        # func = self.funcs[selected_key][0]
        flux, params = self.generators[selected_key].generate_signal(time, **kwargs)
        return selected_key, flux, params


# subclass from TSGenerator
class SineTSGenerator(TSGenerator):
    def __init__(self, name, params:typing.Dict[str,stats.rv_continuous]={'A':1, 'B':1, 'C':0, 'D':1}) -> None:
        # define each arg distribution in the definition
        # func needs to be in the format func(time, **kwargs)
        # name must be a string
        self.name = name
        self.params = params

        # loop through and assign the parameters to class variables
        for key, value in params.items():
            setattr(self, key, trc.convert_to_distribution(value))
        pass

    def functional_form(self, time:np.ndarray, param_values:typing.Dict[str,stats.rv_continuous]) -> typing.Tuple[np.ndarray, typing.Dict[str,float]]:
        """....size time"""
        # param_values = self.sample()
        flux = param_values['A'] * np.sin(param_values['B']*(time+param_values['C'])) + param_values['D']
        return flux, param_values

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
            'radius_1': stats.uniform(loc=.05, scale=.2),
            'radius_2': stats.uniform(loc=.05, scale=.2),
            'incl': stats.uniform(loc=80, scale=10),
            'ecc': MixtureModel([stats.truncexpon(b=5, scale=.1), stats.truncnorm(loc=0, scale=.3, a=0, b=5/3)]),
            'om': MixtureModel([stats.uniform(loc=0, scale=0.0), stats.uniform(loc=np.pi, scale=0.0)]),  # choose positive or negative phase
            'period': stats.lognorm(s=1., loc=.3, scale=4),
            'sbratio': MixtureModel([stats.uniform(loc=1.2, scale=.8), stats.uniform(loc=1.2, scale=.8), stats.truncnorm(loc=2, scale=.12, a=-5, b=0)]),
            't_zero': stats.uniform(loc=0, scale=1)   # uniform in phase space
        }
        
        # loop through and assign the parameters to class variables
        for key, value in params.items():
            self.params[key] = trc.convert_to_distribution(value)
            # setattr(self, key, trc.convert_to_distribution(value))
        pass

    def __repr__(self):
        return "EclipsingBinaryTSGenerator"
        
    def functional_form(self, time:np.ndarray, param_values:typing.Dict[str,stats.rv_continuous]=None) -> np.ndarray:
        """Add."""
        # if param_values is None:
        #     # sample the distributions
        #     param_values = self.sample()

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


# NOTE TO SELF
# Need to adjust the Butterpy and EB classes so that the param dictionary
# overwrites only the specific variables fed into the class


class ButterpyTSGenerator(TSGenerator):
    def __init__(self, name, params:typing.Dict[str,stats.rv_continuous]={}) -> None:
        """
        Rotating stars using the Butterpy package. Has several parameter distributions set by default, as noted below. Only use the params input to modify from this default.

        NOTE: Currently, this only works in python 3.8.

        Parameters
        ----------

        A: stats.Normal
            
        """
        self.name = name
        self.params = {
            'butterfly': True, # make two separate classes for True and False for this
            'activity_rate': 1, #stats.uniform(loc=.05, scale=.2),
            'cycle_length': 1, #stats.uniform(loc=80, scale=10),
            'cycle_overlap': 1, #MixtureModel([stats.truncexpon(b=5, scale=.1), stats.truncnorm(loc=0, scale=.3, a=0, b=5/3)]),
            'decay_time': 1, #MixtureModel([stats.uniform(loc=0, scale=0.0), stats.uniform(loc=np.pi, scale=0.0)]), 
            'max_ave_lat': 1, #stats.lognorm(s=1., loc=.3, scale=4),
            'min_ave_lat': 1, #MixtureModel([stats.uniform(loc=1.2, scale=.8), stats.uniform(loc=1.2, scale=.8), stats.truncnorm(loc=2, scale=.12, a=-5, b=0)]),
            'alpha_med': 1, #stats.uniform(loc=0, scale=1),   
            'period': 1, #stats.uniform(loc=0, scale=1),   
            'incl': stats.uniform(loc=0, scale=1), #later transform with sin^2i,   
            'decay_timescale': 1, #stats.uniform(loc=0, scale=1),   
            'diffrot_shear': 1, #stats.uniform(loc=0, scale=1)   
        }
        
        # loop through and assign the parameters to class variables
        for key, value in params.items():
            self.params[key] = trc.convert_to_distribution(value)
            # setattr(self, key, trc.convert_to_distribution(value))
        pass

    def __repr__(self):
        return "EclipsingBinaryTSGenerator"
        
    def generate_signal(self, time:np.ndarray, param_values:typing.Dict[str,stats.rv_continuous]=None) -> np.ndarray:
        """Add."""
        if param_values is None:
            # sample the distributions
            param_values = self.sample()

        # calculate f_c, f_s, and any other calculable values
        incl = trc.convert_to_incl(param_values['incl'])
        
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

   