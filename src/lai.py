"""
Local ancestry inference.
"""
import numpy as np

def single_window_inference(models, window_size, H, save_path=None):
    """
    TODO: doc
    
    Args:
    ======================================================
    (list of rbm.RBM) models: 
    * Population-specific restricted Boltzmann machines for haplotype imputation.
    ====================================
    (int) window_size:
    * Number of adjacent positions to consider.
    ====================================
    (np.array) H: 
    * An array of shape (num_haplotypes, haplotype_len) containing haplotypes to estimate
      local ancestry for.
    * NOTE: for better (more valid) results, none of the models should have been trained
            any of this data.
    """
    pass






















if __name__ = '__main__':
    pass