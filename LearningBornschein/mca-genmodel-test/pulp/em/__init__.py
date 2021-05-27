#
#  Author:   Jorg Bornschein <bornschein@fias.uni-frankfurt.de)
#  Lincense: Academic Free License (AFL) v3.0
#

"""

Expectation Maximization Framework 
==================================


Usage example::

    import pulp.em as em
    from pulp.em.LinCA import LinCA
    from pulp.utils.dlog import dlog

    # Prepare Model 
    model = LinCA()
    anneal = em.LinearAnneal([ (200, 2, 50) ])
    
    # Prepare training data

    # Instantiate and run EM
    em = em.EM()
    em.model = model
    em.anneal = anneal
    em.set_data(y)
    em.run()

"""

from __future__ import division

import numpy as np
from mpi4py import MPI

try:
    from abc import ABCMeta, abstractmethod
except ImportError, e:
    from pulp.utils.py25_compatibility import _py25_ABCMeta as ABCMeta
    from pulp.utils.py25_compatibility import _py25_abstractmethod as abstractmethod

import pulp.utils.tracing as tracing
import pulp.utils.parallel as parallel

from pulp.utils.datalog import dlog


#=============================================================================
# General EM Model Base Class

class Model():
    """ Model Base Class.

    Includes knowledge about parameters, data generation, model specific 
    functions, E and M step.

    Specific models will be subclasses of this abstract base class.
    """
    __metaclass__ = ABCMeta

    def __init__(self, comm=MPI.COMM_WORLD):
        self.comm = comm
        self.noise_policy = {}

    @abstractmethod
    def generate_data(self, model_params, N):
        """ Generate datapoints according to the model.

        Given the model parameters *model_params* return a dataset 
        of *N* datapoints.
        """
        return data # as dictionary

    @abstractmethod
    def step(self, anneal, model_params, my_data):
        """
        """
        pass

    @abstractmethod
    def standard_init(self, data):
        """ Initialize a set of model parameters in some 
            sane way.

            Return value is model_parameter dictionary
        """
        pass


    @tracing.traced
    def noisify_params(self, model_params, anneal):
        """ Noisify model params.

        Noisify the given model parameters according to self.noise_policy
        and the annealing object provided. The noise_policy of some model
        parameter PARAM will only be applied if the annealing object 
        provides a noise strength via PARAM_noise.

        """
        H, D = self.H, self.D
        normal = np.random.normal
        comm = self.comm

        for param, policy in self.noise_policy.items():
            pvalue = model_params[param]
            if anneal[param+"_noise"] != 0.0:
                if np.isscalar(pvalue):         # Param to be noisified is scalar
                    new_pvalue = 0
                    if comm.rank == 0:
                        scale = anneal[param+"_noise"]
                        new_pvalue = pvalue + normal(scale=scale)
                        if new_pvalue < policy[0]:
                            new_value = policy[0]
                        if new_pvalue >= policy[1]:
                            new_value = policy[1]
                        if policy[2]:
                            new_pvalue = np.abs(new_pvalue)
                    pvalue = comm.bcast(new_pvalue)
                else:                                  # Param to be noisified is an ndarray
                    if comm.rank == 0:
                        scale = anneal[param+"_noise"]
                        shape = pvalue.shape
                        new_pvalue = pvalue + normal(scale=scale, size=shape)
                        low_bound, up_bound, absify = policy
                        new_pvalue = np.maximum(low_bound, new_pvalue)
                        new_pvalue = np.minimum( up_bound, new_pvalue)
                        if absify:
                            new_pvalue = np.abs(new_pvalue)
                        pvalue = new_pvalue
                    comm.Bcast([pvalue, MPI.DOUBLE])

            model_params[param] = pvalue 

        return model_params

    def gain(self, old_params, new_params):
        return 0.

#=============================================================================#
# EM Class

class EM():
    """ This class drives the EM algorithm. 

    It uses instances of the following classes:
     *model*   Instance of a Model class.
     *anneal*  Instance of an Annealing class.
     *data*    Training data in a dictionary. The required content is model,
        but usually data['y'] should contain the trainig data.
     *lparams* Initial set of model parameters in a dictionary. Exact content
        is model dependent.

    """
    def __init__(self, model=None, anneal=None, data=None, 
                  lparams=None, mpi_comm=None):
        self.model = model;
        self.anneal = anneal
        self.data = data
        self.lparams = lparams
        self.mpi_comm = mpi_comm

    def step(self):
        """ Execute a single EM-Step """
        model = self.model
        anneal = self.anneal
        my_data = self.data
        model_params = self.lparams

        # Do an complete EM-step
        new_model_params = model.step(anneal, model_params, my_data)

    def run(self,iter_noise, verbose=False):
        """ Run a complete cooling-cycle 

        When *verbose* is True a progress message is printed for every step
        via dlog.progress(...)
        """
        model = self.model
        anneal = self.anneal
        my_data = self.data
        model_params = self.lparams
        temp1 = my_data['y'].shape[0]
        temp2 = my_data['y'].shape[1]
        
        repeats = 20
        #my_data changed
        new_data =np.zeros((repeats*temp1,temp2))
        for i in range(repeats):
            new_data[temp1*i:temp1*(i+1),:] = np.array([my_data['y'] + np.random.normal(scale = iter_noise, size=(temp1, temp2))])

        my_data['y'] = new_data
        print my_data['y'].shape

        while not anneal.finished:

            # Progress message
            if verbose:
                dlog.progress("EM step %d of %d" % (anneal['step']+1, anneal['max_step']), anneal['position'])

            # Do E and M step
            new_model_params = model.step(anneal, model_params, my_data)
            
            # Calculate the gain so that dynamic annealing schemes can be implemented
            gain = model.gain(model_params, new_model_params)

            anneal.next(gain)
            if anneal.accept:
                model_params = new_model_params

        self.lparams = model_params

