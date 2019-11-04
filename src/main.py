# !/usr/bin/env python3
from util import get_dataset_str
from preprocess import get_dataloader
from model import VAE_LSTM, model_ctx
from train import train_valid
import mxnet as mx
import gluonnlp as nlp
from mxnet import nd, gluon
import json, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--inference', action='store_true', help='Do inference only')

args = parser.parse_args()

def generate(model, original_sts, sample, vocab, max_len, ctx):
    '''
    use the model to generate a paraphrase sentence
    '''
    original_tk = vocab[original_sts.lower().split(' ')]
    # original_tk = nlp.data.PadSequence(length=max_len, pad_val=0)(original_tk)
    original_tk = nlp.data.ClipSequence(max_len)(original_tk)
    original_tk = nd.array(original_tk, ctx=model_ctx).expand_dims(axis=0) # add N
    original_emb = model.embedding_layer(original_tk).swapaxes(0, 1)    # NTC to TNC
    start_state = model.encoder.original_encoder.begin_state(batch_size=1, ctx=ctx)
    _, last_state = model.encoder.original_encoder(original_emb, start_state)
    decoded = nd.array([vocab['bos']], ctx=ctx).expand_dims(axis=0)
    decoded = model.embedding_layer(decoded).swapaxes(0, 1) # idx to emb, NTC to TNC
    predict_tokens = []
    for _ in range(max_len):
        output, last_state = model.decoder(last_state, decoded, sample)
        decoded = output.argmax(axis=2)
        pred = int(decoded.squeeze(axis=0).asscalar())
        decoded = model.embedding_layer(decoded).swapaxes(0, 1)
        if pred == vocab['eos']:
            break
        predict_tokens.append(pred)
    return ' '.join(vocab.to_tokens(predict_tokens))

def generate_v2(model, original_sts, paraphrase_sts, sample, vocab, max_len, ctx):
    '''
    use the model to generate a paraphrase sentence, with *both* original and
    paraphrase input
    '''
    original_tk = vocab[original_sts.lower().split(' ')]
    # original_tk = nlp.data.PadSequence(length=max_len, pad_val=0)(original_tk)
    original_tk = nlp.data.ClipSequence(max_len)(original_tk)
    original_tk = nd.array(original_tk, ctx=model_ctx).expand_dims(axis=0) # add N
    paraphrase_tk = vocab[paraphrase_sts.lower().split(' ')]
    paraphrase_tk = nd.array(paraphrase_tk, ctx=model_ctx).expand_dims(axis=0)
    model(original_tk, paraphrase_tk)
    output = model.output.squeeze(axis=0)   # output is of shape T[vocab_size]
    idx_list = nd.argmax(output, axis=-1).astype('int32').asnumpy().tolist()
    return ' '.join(vocab.to_tokens(idx_list))

if __name__ == '__main__':
    if args.inference:
        with open('data/vocab.json', 'r') as f:
            vocab = nlp.Vocab.from_json(json.load(f))
        model = VAE_LSTM(emb_size=300, vocab_size=len(vocab), hidden_size=256, num_layers=2)
        model.load_parameters('vae-lstm.params', ctx=model_ctx)
        sample = nd.normal(loc=0, scale=1, shape=(1, 256), ctx=model_ctx)
        original_sts = 'a very clean and well decorated empty bathroom'
        paraphrase_sts = 'a bathroom with blue paint on the walls above it'
        print('Result:', generate_v2(model, original_sts, paraphrase_sts, sample, \
                                     vocab, max_len=25, ctx=model_ctx))
    else:
        # load train, valid dataset
        train_dataset_str, valid_dataset_str = get_dataset_str(length=7000)
        train_ld, valid_ld, vocab = get_dataloader(train_dataset_str, valid_dataset_str, \
                                    clip_length=25, vocab_size=50000, batch_size=64)
        # save the vocabulary for use when generating
        vocab_js = vocab.to_json()
        with open('data/vocab.json', 'w') as f:
            json.dump(vocab_js, f)
        # set embedding
        vocab.set_embedding(nlp.embedding.GloVe(source='glove.42B.300d'))
        # create model
        model = VAE_LSTM(emb_size=300, vocab_size=len(vocab), hidden_size=256, num_layers=2)
        model.initialize(init=mx.initializer.Xavier(magnitude=1), ctx=model_ctx)
        model.embedding_layer.weight.set_data(vocab.embedding.idx_to_vec)
        model.embedding_layer.collect_params().setattr('grad_req', 'null')
        # trainer and training
        trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': 3e-4})
        train_valid(train_ld, valid_ld, model, trainer, num_epoch=100, ctx=model_ctx)
        model.save_parameters('vae-lstm.params')
