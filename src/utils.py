import os
import pickle as pkl

from IPython.display import display, clear_output
from ipywidgets import IntProgress

def load_model(pop='all', min_units=0, min_iter=0):
    to_repl = {
        'bs': 'bs=',
        '_lr': '_lr=',
        '_nc': '_nc=',
        '_ni': '_ni=',
        '_note': '_note=',
        '_p': '_pop='
    }
    
    files = os.listdir('./saved_models')
    for i, fname in enumerate(files):
        for (s_in, s_out) in to_repl.items():
            fname = fname.replace(s_in, s_out)

        specs = fname.split('_')
        d_out = dict()
        for s in specs:
            k, v = s.split('=')
            d_out[k] = v
            
        if (pop != 'all') and (d_out['pop'] != pop):
            print()
            continue
            
        if int(d_out['nc']) < min_units:
            continue
            
        print('[%d]' % i, d_out)
        
    choice = int(input('Which file number?\n'))
    if choice == '':
        return None
    
    filename = files[choice]
    
    f = open('./saved_models/%s' % filename, 'rb')
    model = pkl.load(f)
    f.close()
    return model

def progress_bar(mx, generator):
    prog = IntProgress(value=0, max=mx)
    display(prog)
    
    for g in generator:
        yield g
        prog.value += 1
        
    prog.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    pass