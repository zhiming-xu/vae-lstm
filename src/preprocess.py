# !/usr/bin/env python3
# this file defines helper functions for tokenzing sentences, build vocabulary,
# and batchify indexed dataset
from mxnet import gluon
import gluonnlp as nlp
import multiprocessing as mp
import time, logging, itertools

logging.basicConfig(filename=__name__, level=logging.INFO)

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
        dataset_tk = gluon.data.SimpleDataset(
            pool.starmap(_clip_length, itertools.product(dataset, [clipper], [tokenizer]))
        )
    end = time.time()
    logging.info('Finish tokenizing after %.2fs, #Sentence pairs=%d' % \
                (end - start, len(dataset))) 
    return dataset_tk

def _create_vocab(dataset, max_size):
    '''
    create vocabulary from dataset, whose row is [tokenized_src, tokenized_tgt]
    '''
    seqs = [sample[0] + sample[1] for sample in dataset]
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

def _get_idx_dataset(dataset_str, clip_length=25, vocab_size=50000, emb_src='glove.42B.300d'):
    '''
    wrapper of the above functions, return the indexed dataset, and the vacabulary,
    which is used for model's embedding layer
    '''
    dataset_tk = _dataset_tokenizer(dataset_str, clip_length)
    vocab = _create_vocab(dataset_tk, max_size=vocab_size)
    dataset_idx = _token_to_index(dataset_tk, vocab, emb_src=emb_src)
    return dataset_idx, vocab

def _get_length(sample):
    '''
    used for fixed length sampler, will return the larger length of either source or target
    sentence
    '''
    max_length = max(len(sample[0]), sample[1])
    return max_length, max_length

def _get_sampler(dataset_idx, batch_size=64, num_buckets=10, ratio=.5):
    '''
    for training set, we use this function to return the sampler
    '''
    with mp.Pool() as pool:
        lengths = pool.map(_get_length, dataset_idx)
    sampler = nlp.data.sampler.FixedBucketSampler(lengths=lengths, batch_size=batch_size, \
                                                  num_buckets=num_buckets, ratio=ratio)
    logging.debug(sampler.stats())
    return sampler

def _get_batch_dataloader(dataset_idx, batch_size=64, sampler=None):
    '''
    batchify the dataset whose row are [src_idx, tgt_idx], batch size is set to batch_size
    '''
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0, pad_val=0, ret_length=True),
        nlp.data.batchify.Pad(axis=0, pad_val=0, ret_length=True)
    )

    if sampler:
        dataset_batch = gluon.data.DataLoader(
            dataset_idx, batch_size=batch_size, sampler=sampler, batchify_fn=batchify_fn
        )
    else:
        dataset_batch = gluon.data.DataLoader(
            dataset_idx, batch_size=batch_size, shuffle=False, batchify_fn=batchify_fn
        )
    return dataset_batch

def get_dataset(train_dataset_str, valid_dataset_str, clip_length=25, vocab_size=50000, \
                emb_src='glove.42B.300d', batch_size=64, num_buckets=10, ratio=.5):
    '''
    only interface this file needs to expose for other parts to get dataloader
    '''
    train_dataset_idx = _get_idx_dataset(train_dataset_str, clip_length=clip_length, \
                                         vocab_size=vocab_size, emb_src=emb_src)
    valid_dataset_idx = _get_idx_dataset(valid_dataset_str, clip_length=clip_length, \
                                         vocab_size=vocab_size, emb_src=emb_src)
    train_sampler = _get_sampler(train_dataset_idx, batch_size=batch_size, \
                                 num_buckets=num_buckets, ratio=ratio)
    train_dataloder = _get_batch_dataloader(train_dataset_idx, batch_size=batch_size, \
                                            sampler=train_sampler)
    valid_dataloader = _get_batch_dataloader(valid_dataset_idx, batch_size=batch_size)
    return train_dataloder, valid_dataloader
    