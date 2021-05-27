#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 12:27:59 2019

@author: achattoraj
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 15:40:42 2019

@author: achattoraj
"""


"""
Created on Wed Oct  3 09:57:45 2018

@author: achattoraj
"""

from brian2 import *
import numpy as np


@implementation('numpy', discard_units=True)
@check_units(template_match=1, R_other=1, R_self=1, prior=1, sigma2=1, result=1)
def posterior_odds(template_match, R_other, R_self, prior, sigma2):
    return (1 / (1 + (1 / prior - 1) * np.exp(-(template_match + R_other + R_self / 2) / sigma2))) 


# Note that tau is technically in 's' and P is in 'spikes per bin', which is not identical to
# 'spikes per second', so a conversion is needed, which is handled by using a unitless 'adjusted'
# tau.
@implementation('numpy', discard_units=True)
@check_units(FR=hertz, threshold=volt, baseline=volt, tau=second, result=volt)
def I_of_FR(FR, threshold, baseline, tau):
    return ((threshold - baseline) / (1 - np.exp(-1 / (FR * tau)))) 


# 'xi' in brian is dw/dt of a weiner process. the 'tau ** -0.5' bit is in their docs and somehow
# makes the units of a stochastic diff-eq work.
dv_dt = "dv/dt = (I + baseline - v) / tau + membrane_noise * xi * tau ** -0.5 : volt (unless refractory)"

# The model neurons use dv_dt for each neuron's membrane potential update with:
# - I: the net input current (a function of probability, defined above)
# - P: the probability (output of the posterior_odds function)
# - template_match: the match between the image and this neuron's projective field. It is set in
#   the LIFSamplingModel with a call to condition()
# - psp: post synaptic potential. Mathematically it corresponds to conditioning on other neurons in
#   Gibbs sampling (i.e. the sum of x_i * R_ij)
# - R_self: R_kk in the equations
# - is_active: a boolean needed for book-keeping so that psp_post is only incremented if this
#   neuron hasn't fired in a while.
model_eqs = '''
%s
I = I_of_FR(FR, threshold, baseline, tau) : volt 
FR = P * sample_hz + stimulate_timer(t, i) : hertz 
P = posterior_odds(template_match(t, i), psp, R_self, prior, sigma2) : 1 
psp : 1
is_active : 1
R_self : 1
''' % dv_dt

class LIFSamplingModel_stimulate(object):

    def __init__(self, N, G, membrane_noise=0, prior=0.013, pixel_noise=1, baseline_mV=-70,
                 threshold_mV=-55, psp_length_ms=10, tau_ms=20, refractory_ms=0, verbose=False,
                 trial_duration_ms=1000, photo_noise=0.0, photo_noise_refresh_ms=10, rec_noise=0, photostim_rate_increase=0,stimulation_noise_sig=0,photostim_cell=0,stimulation_duration_ms=500):
        """Create a model with N neurons that have projective fields in the columns of G

        Required Arguments:
        - N: number of neurons
        - G: pixels x N matrix with each neuron's projective fields in the columns

        Optional Arguments:
        - membrane_noise: variance of random drift added to membrane potential (default 0)
        - prior: bernoulli prior probability for all neurons (default 0.13)
        - pixel_noise: pixel membrane_noise of the generative model, i.e. the variance of each pixel.
          (default 1)
        - baseline_mV: millivolt resting potential for all neurons (default -70)
        - threshold_mV: millivolt threshold potential for all neurons (default -55)
        - psp_length_ms: number of milliseconds each square-wave PSP lasts. This is also the width
          of a sampling bin. (default 10)
        - tau_ms: time constant of neurons' membrane potential leakiness (default 20)
        - refractory_ms: length of refractory period in milliseconds (default 5)
        - photo_noise: magnitude of input noise added to the image per 'frame' (default 0)
        - photo_noise_refresh_ms: length of one 'frame' (deafult 10)
        - trial_duration_ms: length of each each trial in milliseconds (default 5000)
        - rec_noise: amount of recurrent synapse noise
        """
        assert(G.shape[1] == N)
        self.__dict__.update(locals())

        self.R = -np.dot(G.T, G)

        # Populate 'self' with variables global to the brian model
        self.baseline = baseline_mV * mV
        self.threshold = threshold_mV * mV
        self.sigma2 = pixel_noise * pixel_noise
        self.prior = prior
        self.psp_length = float(psp_length_ms/1000.0) * second
        self.tau = float(tau_ms/1000.0) * second
        self.sample_hz = 1 / self.psp_length  # Note that psp_length is sampling bin size
        self.membrane_noise = membrane_noise * mV
        self.rec_noise = rec_noise
        self.is_compiled = False
        self.photostim_rate_increase = photostim_rate_increase * hertz
        self.photo_stim_cell = photostim_cell
        self.stimulation_noise_sig = stimulation_noise_sig
        self.stimulation_duration_ms = stimulation_duration_ms

    def compile(self):
        if not self.is_compiled:
            # Tell brian2 to lookup any unkown symbol 'xyz' in self.xyz
            local_namespace = self.__dict__

            if self.verbose:
                print("Creating population of %d neurons" % self.N)

            # Build network
            self.population = NeuronGroup(self.N,
                                          model_eqs,
                                          method='euler',
                                          threshold="v>%f*mV" % self.threshold_mV,
                                          reset="v=%f*mV" % self.baseline_mV,
                                          refractory=self.refractory_ms * ms,
                                          namespace=local_namespace)
            self.population.R_self = self.R.diagonal()
            # TODO - random initialization of v
            self.population.v = np.random.uniform(self.baseline,self.threshold,self.N) * volt
            self.population.psp = 0
            self.population.is_active = 0
            # Note: template_match is required by the model but not set until a call to condition()

#          
            if self.verbose:
                print("Creating synapses")

            # Create weighted synapses from neuron i to (the PSP) of neuron j. Note regarding the dict
             #in on_pre: brian executes the values in alphabetical order of the keys.
            
#           
            self.population_synapses = Synapses(self.population, self.population, """temp : 1 
                                                                                   w : 1""",
                                                
                                                on_pre={'a': 'temp = (exp( rec_noise * randn() - rec_noise**2/2) ) * (is_active_pre==0) + temp * (is_active_pre>0) ',     
#                                                        'b': 'temp = temp*(temp<5) + 1*(1-(temp<5))',
                                                        'c': 'psp_post += w * temp ',
                                                        'd': 'is_active_pre += 1',
                                                        'e': 'psp_post -= w * temp ',
                                                        'f': 'is_active_pre -= 1'},
                                                        namespace=local_namespace) 
#                                                namespace=local_namespace)

            self.population_synapses.connect(condition='i!=j')
            self.population_synapses.w = self.R[np.nonzero(1 - np.eye(self.N))] 
            self.population_synapses.temp = np.ones(len(self.population_synapses.w))
            self.population_synapses.e.delay = float(self.psp_length_ms/1000.0) * second
            self.population_synapses.f.delay = float(self.psp_length_ms/1000.0) * second
            
            # When neuron i fires, it starts 'timer' i counting.
#            
            if self.verbose:
                print("Initialization done.")

            self.is_compiled = True

    def condition(self, image):
        # Compute template_match for each neuron; this gives each an input current.
        im = image.flatten()
        assert(self.G.shape[0] == len(im))
        noise_bins = int(np.ceil(self.trial_duration_ms / self.photo_noise_refresh_ms))
        stim_bins = int(noise_bins * self.stimulation_duration_ms / self.trial_duration_ms)
        noise = self.photo_noise * np.random.randn(noise_bins, len(im))
        stimulation_noise = self.photostim_rate_increase + self.stimulation_noise_sig * np.random.randn(stim_bins) * hertz
        # noisy_input_drive is size (noise_bins, neurons) as required by brian2 TimedArray
        extra_rate = np.zeros((noise_bins,self.N)) * hertz
        extra_rate[0:stim_bins, self.photostim_cell] = stimulation_noise
        noisy_input_drive = np.dot(im.T + noise, self.G) 
        for i in range(noise_bins):
            tmp = ( np.exp(self.rec_noise * np.random.randn(self.N)- self.rec_noise**2/2))
#            tmp = (tmp<5)*tmp + (1-(tmp<5)) * 1
            noisy_input_drive[i,:] = noisy_input_drive[i,:] * tmp 
        self.template_match = TimedArray(noisy_input_drive, dt=self.photo_noise_refresh_ms * ms)
        self.stimulate_timer = TimedArray(extra_rate,dt=self.photo_noise_refresh_ms * ms)
    def simulate(self, monitor=[], timer_monitor=[]):
        self.compile()
        self.spikemon = SpikeMonitor(self.population)
        net = Network(self.population,
                      self.population_synapses,
                      self.spikemon)
        if monitor:
            self.monitor = StateMonitor(self.population, monitor, record=True)
            net.add(self.monitor)

        net.run(self.trial_duration_ms * ms)
        return self.spikemon


#if __name__ == '__main__':
#    import matplotlib.pyplot as plt
#    N_ = 10  # number of neurons
#    w_ = 8   # width of projective field patch (pixels)
#
#    # create oriented-gabor projective fields
#    G_ = np.zeros((w_**2, N_))
#    for i in range(N_):
#        angle = i * pi / N_
#        x = np.linspace(-2, 2, w_)
#        [xx, yy] = np.meshgrid(x, x)
#        xx, yy = xx.flatten(), yy.flatten()
#        s, c = np.sin(angle), np.cos(angle)
#        rx = c * xx - s * yy
#        window = np.exp(-(xx * xx + yy * yy))
#        grating = (cos(rx * 3 * pi / 4) + 1) / 2
#        G_[:, i] = window * grating
#    lif = LIFSamplingModel(N_, G_, photo_noise=0.1, pixel_noise=1, trial_duration_ms=200)
#    lif.condition(np.random.random((w_, w_)))
#    spikes = lif.simulate(monitor=["v", "P", "I", "psp", "is_active"], timer_monitor=["a", "fired"])  # noqa:E501
#
#    plt.subplot(121)
#    for i in range(N_):
#        plt.plot(lif.monitor.t / ms, lif.monitor.v[i])
#    plt.subplot(122)
#    for i in range(N_):
#        plt.plot(lif.monitor.t / ms, lif.monitor.P[i])
#    plt.show()
