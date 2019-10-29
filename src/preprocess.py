# !/usr/bin/env python3
# this file defines helper functions for tokenzing sentences, build vocabulary,
# and batchify indexed dataset
from mxnet import gluon
import gluonnlp as nlp
# import multiprocessing as mp
import time, logging, itertools

logging.basicConfig(level=logging.INFO)

def _clip_length(sample, clipper, tokenizer):
    '''
    clip the source and target sentence with `clipper`
    '''
    src, tgt = sample
    # the last char is always a linebreak, just ignore it
    return clipper(tokenizer(src)[:-1]), clipper(tokenizer(tgt)[:-1])

def _tokenize_dataset(dataset, length=25):
    '''
    tokenize the source and target sentences, clip them to len `length`, dataset is of form
    [[src_sentence, tgt_sentence], ...]
    '''
    start = time.time()
    clipper = nlp.data.ClipSequence(length=length)
    tokenizer = nlp.data.SpacyTokenizer('en')
    ''' to debug with vscode, mp needs to be turned off
    with mp.Pool() as pool:
        dataset_tk = gluon.data.SimpleDataset(
            pool.starmap(_clip_length, itertools.product(dataset, [clipper], [tokenizer]))
        )
    '''
    dataset_tk = [_clip_length(sample, clipper, tokenizer) for sample in dataset]
    end = time.time()
    logging.info('Finish tokenizing after %.2fs, #Sentence pairs=%d' % \
                (end - start, len(dataset))) 
    return dataset_tk

def _create_vocab(dataset_tk, max_size):
    '''
    create vocabulary from (train) dataset, whose row is [tokenized_src, tokenized_tgt]
    '''
    seqs = [sample[0] + sample[1] for sample in dataset_tk]
    counter = nlp.data.count_tokens(list(itertools.chain.from_iterable(seqs)))
    vocab = nlp.Vocab(counter=counter, max_size=max_size)
    return vocab

def _tk2idx(sample, vocab):
    '''
    token to index for one sample
    '''
    return vocab[sample[0]], vocab[sample[1]]

def _token_to_index(dataset_tk, vocab):
    '''
    convert dataset whose row is [tokenized_src, tokenized_tgt] to [indexed_src, indexed_tgt]
    '''
    dataset_idx = [_tk2idx(sample, vocab) for sample in dataset_tk]
    return dataset_idx

def _get_length(sample):
    '''
    used for fixed length sampler, will return the larger length of either source or target
    sentence
    '''
    max_length = max(len(sample[0]), len(sample[1]))
    return max_length

def _get_sampler(dataset_idx, batch_size=64, num_buckets=10, ratio=.5):
    '''
    for training set, we use this function to return the sampler
    '''
    lengths = [_get_length(sample) for sample in dataset_idx]
    sampler = nlp.data.sampler.FixedBucketSampler(lengths=lengths, batch_size=batch_size, \
                                                  num_buckets=num_buckets, ratio=ratio)
    logging.info(sampler.stats())
    return sampler

def _get_batch_dataloader(dataset_idx, batch_size=None, sampler=None):
    '''
    batchify the dataset whose row are [src_idx, tgt_idx], batch size is set to batch_size
    '''
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0, pad_val=0),
        nlp.data.batchify.Pad(axis=0, pad_val=0)
    )

    if sampler:
        dataloader = gluon.data.DataLoader(
            dataset_idx, batch_sampler=sampler, batchify_fn=batchify_fn
        )
    else:
        dataloader = gluon.data.DataLoader(
            dataset_idx, batch_size=batch_size, shuffle=False, batchify_fn=batchify_fn
        )
    return dataloader

def get_dataloader(train_dataset_str, valid_dataset_str, clip_length=25, vocab_size=50000, \
                   batch_size=64, num_buckets=10, ratio=.5):
    '''
    only interface this file needs to expose for other parts to get dataloader
    '''
    logging.info('Begin to tokenize train set')
    train_dataset_tk = _tokenize_dataset(train_dataset_str, length=clip_length)
    logging.info('Begin to tokenize valid set')
    valid_dataset_tk = _tokenize_dataset(valid_dataset_str, length=clip_length)
    logging.info('Begin to build vocabulary')
    vocab = _create_vocab(train_dataset_tk, max_size=vocab_size)
    train_dataset_idx = _token_to_index(train_dataset_tk, vocab)
    valid_dataset_idx = _token_to_index(valid_dataset_tk, vocab)
    train_sampler = _get_sampler(train_dataset_idx, batch_size=batch_size, \
                                 num_buckets=num_buckets, ratio=ratio)
    train_dataloader = _get_batch_dataloader(train_dataset_idx, sampler=train_sampler)
    valid_dataloader = _get_batch_dataloader(valid_dataset_idx, batch_size=batch_size)
    return train_dataloader, valid_dataloader, vocab