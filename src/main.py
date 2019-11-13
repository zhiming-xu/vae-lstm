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
parser.add_argument('--param', type=str, default=None, help='start from pretrained params')
parser.add_argument('--org_sts', type=str, help='original sentence for generation')
parser.add_argument('--prp_sts', type=str, help='paraphrase sentence for generation')
parser.add_argument('--dataset', type=str, default='mscoco', help='paraphrase dataset used')
parser.add_argument('--nsample', type=int, default=None, help='# of training samples used')
parser.add_argument('--nepoch', type=int, default=100, help='# of training epoch')
parser.add_argument('--batch_size', type=int, default=64, help='train batch size')
parser.add_argument('--seq_len', type=int, default=25, help='max sequence length after clipping')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--clip_grad', type=float, default=2., help='clipped gradient')
parser.add_argument('--ckpt_interval', type=int, default=10, help='save params every \
                    `ckpt_interval` epochs')
parser.add_argument('--fix_emb', action='store_true', help='fix word embedding')

args = parser.parse_args()

def generate(model, original_sts, sample, vocab, ctx):
    '''
    use the model to generate a paraphrase sentence, max_len is the max length of
    generated sentence
    '''
    original_idx = vocab[original_sts.lower().split(' ')]
    original_idx = nd.array(original_idx, ctx=model_ctx).expand_dims(axis=0) # add N
    # paraphrase_idx = vocab[paraphrase_sts.lower().split(' ')]
    # paraphrase_idx = nd.array(paraphrase_idx, ctx=model_ctx).expand_dims(axis=0)
    pred = model.predict(original_idx, sample, bos=vocab['<bos>'], eos=vocab['<eos>'])
    return ' '.join(vocab.to_tokens(pred))

if __name__ == '__main__':
    if args.gen:
        with open('data/'+args.dataset+'/vocab.json', 'r') as f:
            vocab = nlp.Vocab.from_json(json.load(f))
            
        model = VAE_LSTM(emb_size=300, vocab_size=len(vocab), hidden_size=256, num_layers=2)
        model.load_parameters(args.param, ctx=model_ctx)
        sample = nd.normal(loc=0, scale=1, shape=(1, 256), ctx=model_ctx)
        print('\033[33mOriginal: \033[34m%s\033[0m' % args.org_sts)
        print('\033[31mResult: \033[35m%s\033[0m' % generate(model, args.org_sts, \
                                                    sample, vocab, ctx=model_ctx))
        # print('\033[31mResult 2: \033[35m%s\033[0m' % generate_v2(model, original_sts, \
        #      paraphrase_sts, vocab, ctx=model_ctx))
    else:
        # load train, valid dataset
        train_dataset_str, valid_dataset_str = get_dataset_str(folder=args.dataset, \
                                                               length=args.nsample)
        # start from existing parameters
        if args.param:
            with open('data/'+args.dataset+'/vocab.json', 'r') as f:
                vocab = nlp.Vocab.from_json(json.load(f))
            # use this loaded vocab
            train_ld, valid_ld = get_dataloader(train_dataset_str, valid_dataset_str, \
                                                clip_length=args.seq_len, vocab=vocab, \
                                                batch_size=args.batch_size)
            model = VAE_LSTM(emb_size=300, vocab_size=len(vocab), hidden_size=256, num_layers=2)
            model.load_parameters(args.param, ctx=model_ctx)
        # new start, randomly initialize model
        else:
            train_ld, valid_ld, vocab = get_dataloader(train_dataset_str, valid_dataset_str, \
                                                       clip_length=args.seq_len, vocab_size=50000, \
                                                       batch_size=args.batch_size)
            vocab_js = vocab.to_json()
            with open('data/'+args.dataset+'/vocab.json', 'w') as f:
                json.dump(vocab_js, f)
            model = VAE_LSTM(emb_size=300, vocab_size=len(vocab), hidden_size=256, num_layers=2)
            # new start
            model.initialize(init=mx.initializer.Xavier(magnitude=.7), ctx=model_ctx)
        # set embedding
        vocab.set_embedding(nlp.embedding.GloVe(source='glove.6B.300d'))
        # embedding layer for idx2vec, intentionally set in both decoder & encoder
        model.encoder.embedding_layer.weight.set_data(vocab.embedding.idx_to_vec)
        model.decoder.embedding_layer.weight.set_data(vocab.embedding.idx_to_vec)
        if args.fix_emb:
            model.encoder.embedding_layer.collect_params().setattr('grad_req', 'null')
            model.decoder.embedding_layer.collect_params().setattr('grad_req', 'null')
        # trainer
        trainer = gluon.Trainer(model.collect_params(), 'adam', \
                               {'learning_rate': args.lr, 'clip_gradient': args.clip_grad, \
                                'wd': 2e-5})
        # train and valid
        train_valid(train_ld, valid_ld, model, trainer, num_epoch=args.nepoch, ctx=model_ctx, \
                    ckpt_interval=args.ckpt_interval)
        model.save_parameters('params/vae-lstm.params')
