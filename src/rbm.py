import time
from collections import namedtuple
from tqdm import tqdm
import pickle as pkl

import numpy as np
from sklearn.neural_network import BernoulliRBM


class RBM(BernoulliRBM):
    
    _fields_ = ['n_components', 'n_iter', 'learning_rate', 'batch_size', 'valid_accs']
    
    def __init__(
        self, population='?', n_components=256, learning_rate=0.1, batch_size=10, n_iter=10,
        verbose=0, random_state=None
    ):
        super(RBM, self).__init__(
            n_components, learning_rate, batch_size, n_iter, verbose, random_state
        )
        self.population = population
        
        self.trained = False
        
    def fit(self, X, y=None):
        super(RBM, self).fit(X, y)
        self.trained = True
        
    @property
    def a(self):
        return self.intercept_visible_
    
    @property
    def b(self):
        return self.intercept_hidden_
    
    @property
    def W(self):
        return self.components_
        
    def free_energy(self, v, eps=1e-5):
        """
        TODO: doc
        """
        v = v.reshape(-1, len(self.a))
        n = v.shape[0]
        
        term1 = -np.inner(v, self.a)

        x = self.b + np.dot(v, self.W.T)
        p = RBM.sigmoid(x)
        p = np.clip(p, eps, 1-eps)

        term2 = -(x * p).sum(axis=1)

        term3 = np.sum(p * np.log(p) + (1. - p) * np.log(1. - p), axis=1)

        return term1 + term2 + term3
    
    def impute(self, v, position):
        """
        Args:
        ==========
        (np.array) v: Array of haplotypes of shape (num_haps, hap_len).
        (int) position: Position (column number) to impute given all the other columns
        
        Returns:
        ==========
        (np.array) out: Array of imputed values in {0, 1} of shape (num_haps, )
        """
        if len(v.shape) == 1:
            v = v.reshape(1, -1)
            n = 1
        else:
            n = v.shape[0]
        
        v0 = np.array(v)
        v0[:, position] = 0.
        Fv_0 = self.free_energy(v0)
        
        v1 = np.array(v)
        v1[:, position] = 1.
        Fv_1 = self.free_energy(v1)
        
        return np.where(Fv_0 < Fv_1, 0, 1)
    
    @classmethod
    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))
    
    @classmethod
    def hyperparameter_search(
        cls, H, param_list, num_folds=5, verbose=False
    ):
        """
        TODO: doc
        """
        results = list()
        fold_size = int(np.ceil(H.shape[0] / num_folds))
        
        result_t = namedtuple('Result', RBM._fields_)
    
        for param_dict in param_list:
            if 'n_iter' not in param_dict:
                param_dict['n_iter'] = 10

            if verbose:
                print('='*40)
                print('Trying parameters:')
                print(str(RBM(**param_dict)))
                print('-'*30)
      
            accs = []
            iter_accs = [list() for _ in range(param_dict['n_iter'])]
            full_time = time.time()
            for fold_num, v_start in enumerate(np.linspace(0, H.shape[0], num_folds, dtype=int)):
                if verbose:
                    print('-'*10)
                    print('Fold %d' % fold_num)
                fold_time = time.time()
    
                v_end = v_start + fold_size
                H_train = np.vstack([H[:v_start], H[v_end:]])
                H_valid = H[v_start:v_end]
                
                if H_valid.shape[0] == 0:
                    continue
                
                if verbose:
                    c_params = dict(param_dict)
                    c_params['n_iter'] = 1
                    rbm = RBM(**c_params)
                    for i in range(param_dict['n_iter']):
                        rbm.fit(H_train)
                        
                        v_acc = rbm.compute_imputation_accuracy(H_valid)
                        print('[Fold %d, Iter %d]: avg acc=%.3f' % (fold_num, i+1, v_acc))
                        iter_accs[i].append(v_acc)
                        
                    accs.append(v_acc)
                else:
                    rbm = RBM(**param_dict)
                    rbm.fit(H_train)
                
                    acc = rbm.compute_imputation_accuracy(H_valid)
                    accs.append(acc)
                
                if verbose:
                    print('Fold took %.2f seconds' % (time.time() - fold_time))
            
            param_dict.update({'valid_accs': accs, 'iter_accs': iter_accs})
            result = result_t(**param_dict)
            results.append(result)
            
            if verbose:
                print('-'*30)
                print('Param evaluation took %.2f seconds' % (time.time() - full_time))
                avg_acc, std_acc = np.mean(result['valid_accs']), np.std(result['valid_accs'])
                print('Accuracy: (avg, std) = (%.3f, %.3f)' % (avg_acc, std_acc))
    
        return results
    
    def compute_imputation_accuracy(self, H, col_idx=None):
        if col_idx is None:
            col_idx = np.random.choice(H.shape[1], replace=False, size=1000).astype(int)

        accs = []
        for col in tqdm(col_idx):
            predics = self.impute(H, position=col)
            num_correct = len(np.argwhere(predics == H[:, col]))
            accs.append(num_correct / H.shape[0])

        return np.mean(accs)
    
    def _make_pkl_filename(self):
        filename = str(self)
        to_replace = {
            ('batch_size=', 'bs'), ('learning_rate=', 'lr'), ('n_components=', 'nc'),
            ('n_iter=', 'ni'), ('population=', 'p'), ('verbose=', 'v'), ('\n', ''), 
            (' ', ''), (',', '_'), ("'", '')
        }
        
        for (s_in, s_out) in to_replace:
            filename = filename.replace(s_in, s_out)
            
        # Remove random_state info
        filename = filename[:filename.index('random_state=')] + filename[filename.index('v'):]
        return filename[4:-1] + '.p'
    
    def save(self):
        if not self.trained:
            print(('NOT SAVING %s.' % str(self)) + ' (IT HAS NOT BEEN TRAINED.)')
            return 
        filepath = 'data/' + self._make_pkl_filename()
    
        f = open(filepath, 'wb')
        pkl.dump(self, f)
        f.close()
    
    
    
    
    
if __name__ == '__main__':
    pass