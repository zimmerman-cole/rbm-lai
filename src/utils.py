import os
import pickle as pkl

def load_model(pop='all'):
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
            
        if d_out['pop'] != 'all' and d_out['pop'] != pop:
            continue
            
        print('[%d]' % i, d_out)
        
    choice = int(input('Which file number?'))
    if choice == '':
        return None
    
    filename = files[choice]
    
    f = open('./saved_models/%s' % filename, 'rb')
    model = pkl.load(f)
    f.close()
    return model