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
parser.add_argument('--gen', action='store_true', help='generate only')
parser.add_argument('--from_param', action='store_true', help='start from existent params')
parser.add_argument('--param', type=str, help='pretrained parameters')
parser.add_argument('--org_sts', type=str, help='original sentence for generation')
parser.add_argument('--prp_sts', type=str, help='paraphrase sentence for generation')
parser.add_argument('--dataset', type=str, default='mscoco', help='paraphrase dataset used')
parser.add_argument('--nsample', type=int, default=None, help='# of training samples used')
parser.add_argument('--nepoch', type=int, default=100, help='# of training epoch')
parser.add_argument('--batch_size', type=int, default=64, help='train batch size')
parser.add_argument('--seq_len', type=int, default=25, help='max sequence length after clipping')
# parser.add_argument('--ctx', type=str, default='gpu', help='use cpu/gpu, gpu by default')

args = parser.parse_args()

def generate(model, original_sts, paraphrase_sts, sample, vocab, ctx):
    '''FIXME this way of generation does not work now
    use the model to generate a paraphrase sentence, max_len is the max length of
    generated sentence
    '''
    original_idx = vocab[original_sts.lower().split(' ')]
    original_idx = nd.array(original_idx, ctx=model_ctx).expand_dims(axis=0) # add N
    paraphrase_idx = vocab[paraphrase_sts.lower().split(' ')]
    paraphrase_idx = nd.array(paraphrase_idx, ctx=model_ctx).expand_dims(axis=0)
    pred = model.predict(original_idx, paraphrase_idx, sample)
    # eliminate all tokens after `eos` in predicted sentence
    try:
        pred = pred[:pred.index(vocab['<eos>'])]
    except ValueError:
        pass
    return ' '.join(vocab.to_tokens(pred))

def generate_v2(model, original_sts, paraphrase_sts, vocab, ctx):
    '''
    use the model to generate a paraphrase sentence, with *both* original and
    paraphrase input
    '''
    original_idx = vocab[original_sts.lower().split(' ')]
    original_idx = nd.array(original_idx, ctx=model_ctx).expand_dims(axis=0) # add N
    paraphrase_idx = vocab[paraphrase_sts.lower().split(' ')]
    paraphrase_idx = nd.array(paraphrase_idx, ctx=model_ctx).expand_dims(axis=0)
    model(original_idx, paraphrase_idx)
    output = model.output.squeeze(axis=0)   # output is of shape T[vocab_size]
    idx_list = nd.argmax(output, axis=-1).astype('int32').asnumpy().tolist()
    return ' '.join(vocab.to_tokens(idx_list))

if __name__ == '__main__':
    if args.gen:
        with open('data/vocab.json', 'r') as f:
            vocab = nlp.Vocab.from_json(json.load(f))
            
        model = VAE_LSTM(emb_size=300, vocab_size=len(vocab), hidden_size=256, num_layers=2)
        model.load_parameters(args.param, ctx=model_ctx)
        sample = nd.normal(loc=0, scale=1, shape=(1, 256), ctx=model_ctx)
        original_sts, paraphrase_sts = args.org_sts, args.prp_sts
        print('\033[33mOriginal: \033[34m%s\033[0m' % original_sts)
        print('\033[33mParaphrase: \033[34m%s\033[0m' % paraphrase_sts)
        print('\033[31mResult 1: \033[35m%s\033[0m' % generate(model, original_sts, \
              paraphrase_sts, sample, vocab, ctx=model_ctx))
        print('\033[31mResult 2: \033[35m%s\033[0m' % generate_v2(model, original_sts, \
              paraphrase_sts, vocab, ctx=model_ctx))
    else:
        # load train, valid dataset
        train_dataset_str, valid_dataset_str = get_dataset_str(folder=args.dataset, \
                                                               length=args.nsample)
        train_ld, valid_ld, vocab = get_dataloader(train_dataset_str, valid_dataset_str, \
                                                   clip_length=args.seq_len, \
                                                   vocab_size=50000, batch_size=64)
        # save the vocabulary for use when generating
        vocab_js = vocab.to_json()
        with open('data/vocab.json', 'w') as f:
            json.dump(vocab_js, f)
        # set embedding
        vocab.set_embedding(nlp.embedding.GloVe(source='glove.6B.300d'))
        # create model
        model = VAE_LSTM(emb_size=300, vocab_size=len(vocab), hidden_size=256, num_layers=2)
        if args.from_param:
            # start from previous params
            model.load_parameters(args.param, ctx=model_ctx)
        else:
            # new start
            model.initialize(init=mx.initializer.Xavier(magnitude=1), ctx=model_ctx)
        # embedding layer for idx2vec, intentionally set in both decoder&encoder
        model.encoder.embedding_layer.weight.set_data(vocab.embedding.idx_to_vec)
        model.decoder.embedding_layer.weight.set_data(vocab.embedding.idx_to_vec)
        model.encoder.embedding_layer.collect_params().setattr('grad_req', 'null')
        model.decoder.embedding_layer.collect_params().setattr('grad_req', 'null')
        # trainer and training
        trainer = gluon.Trainer(model.collect_params(), 'adam', \
                               {'learning_rate': 3e-4, 'clip_gradient': 2.5, \
                                'wd': 2e-5})
        train_valid(train_ld, valid_ld, model, trainer, num_epoch=args.nepoch, ctx=model_ctx)
        model.save_parameters('data/vae-lstm.params')
