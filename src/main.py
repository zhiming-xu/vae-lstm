# !/usr/bin/env python3
from util import get_dataset_str
from preprocess import get_dataloader
from model import VAE_LSTM, model_ctx
from train import train_valid
import mxnet as mx
import gluonnlp as nlp
from mxnet import nd, gluon

def generate(model, original_sts, sample, vocab, max_len, ctx):
    '''
    use the model to generate a paraphrase sentence
    '''
    original_tk = vocab[original_sts.lower().split(' ')]
    original_tk = nlp.data.PadSequence(length=max_len, pad_val=0)(original_tk)
    original_tk.expand_dims(axis=0) # add the N dimension
    original_emb = model.embedding_layer(original_tk).swapaxes(0, 1)    # NTC to TNC
    start_state = model.encoder.original_encoder.begin_state(batch_size=1, ctx=ctx)
    _, last_state = model.encoder.original_encoder(original_emb, start_state)
    decoded = nd.array([vocab.bos], ctx=ctx).expand_dims(axis=0)
    predict_tokens = []
    for _ in range(max_len):
        output, last_state = model.decoder(last_state, decoded, sample)
        decoded = output.argmax(axis=2)
        pred = decoded.squeeze(axis=0).astype('int32').asscalar()
        if pred == vocab.eos:
            break
        predict_tokens.append(pred)
    return ' '.join(vocab.to_tokens(predict_tokens))


if __name__ == '__main__':
    test = True
    train_dataset_str, valid_dataset_str = get_dataset_str(length=10000)
    train_ld, valid_ld, vocab = get_dataloader(train_dataset_str, valid_dataset_str, \
                                clip_length=25, vocab_size=50000, batch_size=64)
    vocab.set_embedding(nlp.embedding.GloVe(source='glove.42B.300d'))
    model = VAE_LSTM(emb_size=300, vocab_size=len(vocab), hidden_size=256, num_layers=2)
    if test:
        model.load_parameters('vae-lstm.params', ctx=model_ctx)
        sample = nd.normal(loc=0, scale=1, shape=(1, 256), ctx=model_ctx)
        original_sts = 'a brown cat be sitting on the mat'
        generate(model, original_sts, sample, vocab, 25, model_ctx)
    else:
        model.initialize(init=mx.initializer.Xavier(magnitude=1), ctx=model_ctx)
        model.embedding_layer.weight.set_data(vocab.embedding.idx_to_vec)
        model.embedding_layer.collect_params().setattr('grad_req', 'null')
        trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': 3e-4})
        train_valid(train_ld, valid_ld, model, trainer, num_epoch=50, ctx=model_ctx)
        model.save_parameters('vae-lstm.params')
