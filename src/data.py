import numpy as np

def load_H1(clean=True):
    H_path = 'data/H1/H.txt'
    
    if clean:
        cols = np.loadtxt('data/H1/clean/clean_columns.txt').astype(int)
    else:
        cols = None  # all
        
    H = np.loadtxt(H_path, usecols=cols)
    
    return H


def load_H2():
    return np.loadtxt('data/H2/H.txt')


def load_data(clean=True):
    H1 = load_H1(clean)
    H2 = load_H2()
    
    return np.hstack([H1, H2])

def load_population_data():
    pops = ['AFR', 'EUR', 'EAS', 'AMR']
    out = dict()
    for p in pops:
        out[p] = np.loadtxt('data/H_%s.txt' % p)
        
    return out