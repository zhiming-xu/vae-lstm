# !/usr/bin/env python3
import gluonnlp as nlp

DATA_DIR = 'data/'
FILES = ['train_source.txt', 'train_target.txt']

def _load_dataset(src_name, tgt_name, length=50000):
    '''
    src_name: filename of source sentence, tgt_name: filename of target sentence
    '''
    with open(src_name, 'r') as f_src, open(tgt_name, 'r') as f_tgt:
        src_list, tgt_list = [line for line in f_src], [line for line in f_tgt]    
    return [pair for pair in zip(src_list, tgt_list)][:length]

def get_dataset_str(folder='mscoco'):
    '''
    the only interface this file exposes for other parts to get a dataset in string form
    '''
    paths = [DATA_DIR + folder + '/' + filename for filename in FILES]
    dataset_str = _load_dataset(*paths)
    train_dataset_str, valid_dataset_str = nlp.data.train_valid_split(dataset_str, valid_ratio=.1)
    return train_dataset_str, valid_dataset_str