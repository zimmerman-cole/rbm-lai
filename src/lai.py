"""
Local ancestry inference.
"""
import numpy as np
import itertools
from collections import namedtuple

import matplotlib.pyplot as plt

def compute_window_accuracies(imputations, H_valid, window_size, save_path=None, hooks=list()):
    """
    TODO: doc
    
    Args:
    ======================================================
    (dict) imputations: 
    * Imputations made by each population-specific RBM on the (validation) data.
    * Entries (key : value) should include:
       * 'AFR' : (np.array) imputes_AFR - array of shape (num_haps, haplotype_len).
       * Same entries for other populations 'EUR', 'EAS', 'AMR'
       * 'model_info': list of strings - containing information (training, etc.) on each
                                         RBM model.
       * 'H_valid_info': (str) info - information on which rows of the validation data
                                      correspond to individuals from which population.
    ====================================
    (int) window_size:
    * Number of adjacent positions to consider.
    ====================================
    (np.array) H_valid: 
    * An array of shape (num_haps, haplotype_len) containing the target haplotypes.
    """
    pops = set(list(imputations.keys()))
    pops = list(pops.difference({'model_info', 'H_valid_info'}))

    n, m = imputations[pops[0]].shape

    accuracies = {p: np.zeros((n, 0)) for p in pops}
    # For all haplotypes: compute window accuracy for all positions, for all 
    #   *individual* models
    for col in range(m):
        for p in pops:
            w_b, w_e = max(0, col-window_size), min(m, col+window_size+1)
            w_size = w_e - w_b
            accs = np.where(H_valid[:, w_b:w_e] == imputations[p][:, w_b:w_e], 1, 0).sum(axis=1)
            accs = accs / w_size

            accuracies[p] = np.hstack([accuracies[p], accs.reshape(-1, 1)])
            
        for h in hooks:
            h(col, accuracies)
            
    if save_path is not None:
        out = dict()
        out['model_info'] = imputations['model_info']
        out['H_valid_info'] = imputations['H_valid_info']
        out['accuracies'] = accuracies
        try:
            f = open(save_path, 'wb')
            pkl.dump(out, f)
        except:
            print('Saving accuracies failed.')
        finally:
            f.close()
            
    return accuracies


def assign_ancestries(accuracies, hooks=[]):
    
    pops = list(accuracies.keys())
    n, m = list(accuracies.values())[0].shape
    
    out = list()

    for i, g_row in enumerate(range(0, n, 2)):
        out.append(list())
        
        for col in range(m):
            p_accs = dict()
            for (p1, p2) in itertools.combinations_with_replacement(pops, 2):
            
                acc1 = accuracies[p1][g_row][col] * accuracies[p2][g_row+1][col]
                acc2 = accuracies[p1][g_row+1][col] * accuracies[p2][g_row][col]
                
                p_accs[frozenset({p1, p2})] = max(acc1, acc2)
            
            best_pair = max(p_accs.items(), key=lambda item: item[1])[0]
            out[i].append(best_pair)
            
        for hook in hooks:
            hook(i, g_row)

    return out

if __name__ == '__main__':
    pass