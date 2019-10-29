# !/usr/bin/env python3
from util import get_dataset_str
from preprocess import get_dataloader
from model import VAE_LSTM

if __name__ == '__main__':
    train_dataset_str, valid_dataset_str = get_dataset_str()
    train_ld, valid_ld, vocab = get_dataloader(train_dataset_str, valid_dataset_str, \
                                clip_length=15, vocab_size=50000, batch_size=64)
    vocab.set_embedding()

    model = VAE_LSTM(emb_size=300)