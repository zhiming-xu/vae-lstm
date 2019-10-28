# !/usr/bin/env python3
import gluonnlp as nlp
import multiprocessing as mp
import time, logging, itertools

logging.basicConfig(level=logging.INFO)

def _clip_length(sample, clipper, tokenizer):
    '''
    clip the source and target sentence with `clipper`
    '''
    src, tgt = sample
    return clipper(tokenizer(src)), clipper(tokenizer(tgt))

def _dataset_tokenizer(dataset, length):
    '''
    tokenize the source and target sentences, clip them to len `length`
    '''
    start = time.time()
    clipper = nlp.data.ClipSequence(length=length)
    tokenizer = nlp.data.SpacyTokenizer('en')
    with mp.Pool() as pool:
        dataset_tk = pool.starmap(_clip_length, itertools.product(dataset, [clipper], [tokenizer]))
    end = time.time()
    logging.info('Finish tokenizing after %.2fs, #Sentence pairs=%d' % \
                (end - start, len(dataset))) 
    return dataset_tk

def _create_vocab(dataset, max_size):
    seqs = [sample[0] + sample[1] for sample in dataset]
    counter = nlp.data.count_tokens(list(itertools.chain.from_iterable(seqs)))
    vocab = nlp.Vocab(counter=counter, max_size=max_size)
    return vocab