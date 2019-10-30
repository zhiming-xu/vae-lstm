# !/usr/bin/env python3
from util import get_dataset_str
from preprocess import get_dataloader
from model import VAE_LSTM, model_ctx
from train import train_valid
import mxnet as mx
import gluonnlp as nlp
from mxnet import gluon

if __name__ == '__main__':
    train_dataset_str, valid_dataset_str = get_dataset_str()
    train_ld, valid_ld, vocab = get_dataloader(train_dataset_str, valid_dataset_str, \
                                clip_length=15, vocab_size=50000, batch_size=64)
    vocab.set_embedding(nlp.embedding.GloVe(source='glove.42B.300d'))
    model = VAE_LSTM(emb_size=300, vocab_size=len(vocab), hidden_size=256, num_layers=1)
    model.initialize(init=mx.initializer.Xavier(magnitude=.7), ctx=model_ctx)
    model.embedding_layer.weight.set_data(vocab.embedding.idx_to_vec)
    model.embedding_layer.collect_params().setattr('grad_req', 'null')
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': 1e-3})
    train_valid(train_ld, valid_ld, model, trainer, num_epoch=10, ctx=model_ctx)
    model.save_parameters('vae-lstm.params')