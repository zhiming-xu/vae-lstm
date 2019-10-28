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
    tokenize the source and target sentences, clip them to len `length`, dataset is of form
    [[src_sentence, tgt_sentence], ...]
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
    '''
    create vocabulary from dataset, whose row is [tokenized_src, tokenized_tgt]
    '''
    seqs = [sample[0] + ' ' + sample[1] for sample in dataset]
    counter = nlp.data.count_tokens(list(itertools.chain.from_iterable(seqs)))
    vocab = nlp.Vocab(counter=counter, max_size=max_size)
    return vocab

def _token_to_index(dataset, vocab, emb_src='glove.42B.300d'):
    '''
    convert dataset whose row is [tokenized_src, tokenized_tgt] to [indexed_src, indexed_tgt]
    '''
    vocab.set_embedding(nlp.embedding.GloVe(source=emb_src))
    tk2idx = lambda sample, vocab: [vocab[sample[0]], vocab[sample[1]]]
    with mp.Pool() as pool:
        dataset_idx = pool.starmap(tk2idx, itertools.product(dataset, [vocab]))
    return dataset_idx

def get_dataset(dataset_str, clip_length=25, vocab_size=50000, emb_src='glove.42B.300d'):
    '''
    wrapper of the above functions, return the indexed dataset, and the vacabulary,
    which is used for model's embedding layer
    '''
    dataset_tk = _dataset_tokenizer(dataset_str, clip_length)
    vocab = _create_vocab(dataset_tk, max_size=vocab_size)
    dataset_idx = _token_to_index(dataset_tk, vocab, emb_src=emb_src)
    return dataset_idx, vocab