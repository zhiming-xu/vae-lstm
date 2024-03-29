# !/usr/bin/env python3
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn, rnn, loss

try:
    nd.ones(shape=(2, 2), ctx=mx.gpu())
    model_ctx = mx.cpu()
except:
    model_ctx = mx.cpu()

class VAEEncoder(nn.Block):
    '''
    encoder part of the VAE model, consisting of two LSTMs, one for encode original, the other
    for encode original AND paraphase together. output of the latter is passed to a de facto
    LSTM to generate mu and lv
    '''
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers=2, dropout=.3, \
                 bidir=True, latent_size=64, **kwargs):
        '''
        init this class, create relevant rnns
        '''
        super(VAEEncoder, self).__init__(**kwargs)
        with self.name_scope():
            self.hidden_size = hidden_size
            self.hidden_factor = (2 if bidir else 1) * num_layers
            self.embedding_layer = nn.Embedding(vocab_size, emb_size)
            self.original_encoder = rnn.LSTM(hidden_size=hidden_size, num_layers=num_layers, \
                                             dropout=dropout, bidirectional=bidir, \
                                             prefix='VAEEncoder_org_encoder')
            self.paraphrase_encoder = rnn.LSTM(hidden_size=hidden_size, num_layers=num_layers, \
                                               dropout=dropout, bidirectional=bidir, \
                                               prefix='VAEEncoder_prp_encoder')
            # dense layers calculating mu and lv to sample, since the length of the input is
            # flexible, we need to use RNN
            self.output_mu = nn.Dense(units=latent_size)
            self.output_sg = nn.Dense(units=latent_size)

    def forward(self, original_idx, paraphrase_idx):
        '''
        forward pass, inputs are token_idx of original sentences and paraphrase sentences, layout NT
        '''
        original_emb = self.embedding_layer(original_idx).swapaxes(0, 1)
        # remove the <bos> and <eos> token in paraphrase here
        paraphrase_emb = self.embedding_layer(paraphrase_idx[:, 1:-1]).swapaxes(0, 1)
        # to let lstm return final state and memory cell, we need to pass `start_state`
        start_state = self.original_encoder.begin_state(batch_size=original_emb.shape[1], ctx=model_ctx)
        # original_encoder_state is a list: [hidden_output, memory cell] of the last time step,
        # pass them as starting state of paraphrase encoder, just like in Seq2Seq
        _, original_last_states = self.original_encoder(original_emb, start_state)
        # we will take the hidden_output of the last step of our second lstm
        _, (paraphrase_hidden_state, _) = self.paraphrase_encoder(paraphrase_emb, original_last_states)
        # this is the \phi of VAE encoder, i.e., mean and logv
        context = paraphrase_hidden_state.reshape(shape=(-1, self.hidden_factor * self.hidden_size))
        mean = self.output_mu(context) # \mu, mean of sampled distribution
        logv = self.output_sg(context) # \sg, std dev of sampler distribution based on
        return mean, logv, original_last_states
    
    def encode(self, original_idx):
        '''
        this function is used when generating, return the last state of lstm when doing original
        sentence embedding
        '''
        original_emb = self.embedding_layer(original_idx).swapaxes(0, 1)
        # batch_size -> T[N]C
        start_state = self.original_encoder.begin_state(batch_size=original_idx.shape[0], ctx=model_ctx)
        _, original_last_states = self.original_encoder(original_emb, start_state)
        return original_last_states
        
class VAEDecoder(nn.Block):
    '''
    decoder part of the VAE model
    '''
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers=2, dropout=.3, bidir=True, **kwargs):
        '''
        init this class, create relevant rnns, note: we will share the original sentence encoder
        between VAE encoder and VAE decoder
        '''
        super(VAEDecoder, self).__init__(**kwargs)
        with self.name_scope():
            self.embedding_layer = nn.Embedding(vocab_size, emb_size)
            self.paraphrase_decoder = rnn.LSTM(hidden_size=hidden_size, num_layers=num_layers, \
                                               dropout=dropout, bidirectional=bidir, \
                                               prefix='VAEDecoder_prp_decoder')
            # the `output_size` should be set eqaul to the vocab size (a probablity distribution
            # over all words in vocabulary)
            self.dense_output = nn.Dense(vocab_size, activation='tanh')

    def forward(self, last_state, last_idx, latent_input):
        '''
        forward pass, inputs are last states (a list of last hidden output and last memory cell)
        paraphrase sentence embedding and latent output of encoder, i.e., z (mu sg when training,
        sampled from N(0, 1) when testing)
        for the first step, `last_state` is the last state of the original sentence encoder
        '''
        # from token_idx to embedding
        last_emb = self.embedding_layer(last_idx).swapaxes(0, 1)
        # latent_input is of shape (batch_size, hidden_size), we need to add the time dimension
        # and repeat itself T times to concat to paraphrase embedding, layout TN[hiddent_size]
        latent_input = latent_input.expand_dims(axis=0).repeat(repeats=last_emb.shape[0], axis=0)
        # layout is TNC, so concat along the last (channel) dimension, layout TN[emb_size+hidden_size]
        decoder_input = nd.concat(last_emb, latent_input, dim=-1)
        # decoder output is of shape TNC
        decoder_output, decoder_state = self.paraphrase_decoder(decoder_input, last_state)
        vocab_output = self.dense_output(decoder_output.swapaxes(0, 1))
        return vocab_output, decoder_state

class VAE_LSTM(nn.Block):
    '''
    wrapper of all part of this model
    '''
    def __init__(self, emb_size, vocab_size, hidden_size=256, num_layers=2, dropout=.2, \
                 bidir=True, latent_size=64, **kwargs):
        super(VAE_LSTM, self).__init__(**kwargs)
        with self.name_scope():
            self.latent_size = latent_size
            # i have confirmed the calculation of kl divergence is right
            self.kl_div = lambda mean, logv: 0.5 * nd.sum(1 + logv - mean.square() - logv.exp())
            # self.kl_div = lambda mu, sg: (-0.5 * nd.sum(sg - mu*mu - nd.exp(sg) + 1, 1)).mean().squeeze()
            self.ce_loss = loss.SoftmaxCELoss()
            self.encoder = VAEEncoder(vocab_size=vocab_size, emb_size=emb_size, hidden_size=hidden_size, \
                                      num_layers=num_layers, dropout=dropout, bidir=bidir)
            self.decoder = VAEDecoder(vocab_size=vocab_size, emb_size=emb_size, hidden_size=hidden_size, \
                                      num_layers=num_layers, dropout=dropout, bidir=bidir)

    def forward(self, original_idx, paraphrase_idx):
        '''
        forward pass of the whole model, original/paraphrase_idx are both of layout
        NT, to be added "C" by embedding layer
        '''
        # ENCODER part
        mean, logv, last_state = self.encoder(original_idx, paraphrase_idx)
        # sample from Gaussian distribution N(0, 1), of the shape (batch_size, hidden_size)
        z = nd.normal(loc=0, scale=1, shape=(original_idx.shape[0], self.latent_size), ctx=model_ctx)
        latent_input = mean + z * nd.exp(0.5 * logv)  # exp() is to make the std dev positive
        
        # DECODER part
        # the KL Div should be calculated between the sample from N(0, 1), and the distribution after
        # Parameterization Trick, negation since we want it to be small
        kl_loss = -self.kl_div(mean, logv)
        # first paraphrase_input should be the <bos> token
        last_idx = paraphrase_idx[:, 0:1]
        ce_loss = 0
        # decode the sample
        for pos in range(paraphrase_idx.shape[-1]-1):
            vocab_output, last_state = self.decoder(last_state, last_idx, latent_input)
            # only compare the label we predict, note the first is bos and will be ignored
            ce_loss = ce_loss + self.ce_loss(vocab_output, paraphrase_idx[:, pos+1:pos+2])
            last_idx = vocab_output.argmax(axis=-1, keepdims=True)
        return kl_loss, ce_loss

    def predict(self, original_idx, normal_distr, bos, eos, max_len=25):
        '''
        this method is for predicting a paraphrase sentence
        '''
        # 2 is for <bos>, might set as a param later
        last_idx = nd.array([bos], ctx=model_ctx).expand_dims(axis=0)
        last_state = self.encoder.encode(original_idx)
        # predict a token list of at most max_len tokens
        pred_tks = []
        for _ in range(max_len):
            pred, last_state = self.decoder(last_state, last_idx, normal_distr)
            pred_tk = int(pred.argmax(axis=-1).squeeze().astype('int32').asscalar())
            if pred_tk == eos:
                pred_tks.append(pred_tk)
                break
            pred_tks.append(pred_tk)
        return pred_tks