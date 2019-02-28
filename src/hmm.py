"""
Hidden Markov Models.
"""
import numpy as np
from hmmlearn.base import _BaseHMM

class HMM(_BaseHMM):
    
    def __init__(
        self, n_components=7, startprob_prior=1.0, transmat_prior=None
    ):
        pass