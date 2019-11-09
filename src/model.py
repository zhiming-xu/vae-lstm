# !/usr/bin/env python3
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn, rnn, loss

try:
    nd.ones(shape=(2, 2), ctx=mx.gpu())
    model_ctx = mx.gpu()
except:
    model_ctx = mx.cpu()

class VAEEncoder(nn.Block):
    '''
    encoder part of the VAE model, consisting of two LSTMs, one for encode original, the other
    for encode original AND paraphase together. output of the latter is passed to a de facto
    LSTM to generate mu and lv
    '''
    def __init__(self, hidden_size, num_layers=3, dropout=.3, bidir=False, **kwargs):
        '''
        init this class, create relevant rnns
        '''
        super(VAEEncoder, self).__init__(**kwargs)
        with self.name_scope():
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.original_encoder = rnn.LSTM(hidden_size=hidden_size, num_layers=num_layers, \
                                             dropout=dropout, bidirectional=bidir, \
                                             prefix='original_sentence_encoder_VAEEncoder')
            self.paraphrase_encoder = rnn.LSTM(hidden_size=hidden_size, num_layers=num_layers, \
                                               dropout=dropout, bidirectional=bidir, \
                                               prefix='paraphrase_sentence_encoder_VAEEncoder')
            # dense layers calculating mu and lv to sample, since the length of the input is
            # flexible, we need to use RNN
            self.output_mu = rnn.LSTM(hidden_size=hidden_size, dropout=dropout)
            self.output_sg = rnn.LSTM(hidden_size=hidden_size, dropout=dropout)

    def forward(self, original_input, paraphrase_input):
        '''
        forward pass, inputs are embeddings of original sentences and paraphrase sentences, layout TNC
        '''
        # to let lstm return final state and memory cell, we need to pass `start_state`
        start_state = self.original_encoder.begin_state(batch_size=original_input.shape[1], ctx=model_ctx)
        # original_encoder_state is a list: [hidden_output, memory cell] of the last time step,
        # pass them as starting state of paraphrase encoder, just like in Seq2Seq
        _, original_last_state = self.original_encoder(original_input, start_state)
        paraphrase_encoded, _ = self.paraphrase_encoder(paraphrase_input, original_last_state)
        # this is the \phi of VAE encoder, i.e., \mu and "\sigma", FIXME: use the last output now
        # thus their shapes are of (batch_size, hidden_size)
        mu = self.output_mu(paraphrase_encoded)[-1] # \mu, mean of sampled distribution
        sg = self.output_sg(paraphrase_encoded)[-1] # \sg, std dev of sampler distribution based on
        return mu, sg, original_last_state
    
    def encode(self, original_input):
        '''
        this function is used when generating, return the last state of lstm when doing original
        sentence embedding
        '''
        # batch_size -> T[N]C
        start_state = self.original_encoder.begin_state(batch_size=original_input.shape[1], ctx=model_ctx)
        _, original_last_state = self.original_encoder(original_input, start_state)
        return original_last_state
        
class VAEDecoder(nn.Block):
    '''
    decoder part of the VAE model
    '''
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers=3, dropout=.3, bidir=False, **kwargs):
        '''
        init this class, create relevant rnns, note: we will share the original sentence encoder
        between VAE encoder and VAE decoder
        '''
        super(VAEDecoder, self).__init__(**kwargs)
        with self.name_scope():
            self.embedding_layer = nn.Embedding(vocab_size, emb_size)
            self.paraphrase_decoder = rnn.LSTM(hidden_size=hidden_size, num_layers=num_layers, \
                                               dropout=dropout, bidirectional=bidir, \
                                               prefix='paraphrase_sentence_decoder_VAEDecoder')
            # the `output_size` should be set eqaul to the vocab size (a probablity distribution
            # over all words in vocabulary)
            self.dense_output = nn.Dense(vocab_size, activation='relu', flatten=False)

    def forward(self, last_state, last_idx, latent_input):
        '''
        forward pass, inputs are last states (a list of last hidden output and last memory cell)
        paraphrase sentence embedding and latent output of encoder, i.e., z (mu sg when training,
        sampled from N(0, 1) when testing)
        for the first step, `last_state` is the last state of the original sentence encoder
        '''
        # from token_idx to embedding
        last_emb = self.embedding_layer(last_idx)
        # latent_input is of shape (batch_size, hidden_size), we need to add the time dimension
        # and repeat itself T times to concat to paraphrase embedding, layout TN[hiddent_size]
        latent_input = latent_input.expand_dims(axis=0).repeat(repeats=last_emb.shape[0], axis=0)
        # layout is TNC, so concat along the last (channel) dimension, layout TN[emb_size+hidden_size]
        decoder_input = nd.concat(last_emb, latent_input, dim=-1)
        # decoder output is of shape TN[hidden_size]
        decoder_output, decoder_state = self.paraphrase_decoder(decoder_input, last_state)
        # since we calculate KL-loss with layout TNC, we will keep it this way
        decoder_output = self.dense_output(decoder_output)
        return decoder_output, decoder_state

class VAE_LSTM(nn.Block):
    '''
    wrapper of all part of this model
    '''
    def __init__(self, emb_size, vocab_size, hidden_size, num_layers, dropout=.3, bidir=False, **kwargs):
        super(VAE_LSTM, self).__init__(**kwargs)
        with self.name_scope():
            self.embedding_layer = nn.Embedding(vocab_size, emb_size)
            self.hidden_size = hidden_size
            self.emb_size = emb_size
            self.num_layers = num_layers
            self.kl_div = lambda mu, sg: 0.5 * nd.sum(1 + sg - nd.square(mu) - nd.exp(sg), axis=-1)
            self.log_loss = loss.SoftmaxCELoss()
            self.encoder = VAEEncoder(hidden_size=hidden_size, num_layers=num_layers, \
                                      dropout=dropout, bidir=bidir)
            self.decoder = VAEDecoder(vocab_size=vocab_size, emb_size=emb_size, hidden_size=hidden_size, \
                                      num_layers=num_layers, dropout=dropout, bidir=bidir)

    def forward(self, original_idx, paraphrase_idx):
        # from idx to sentence embedding
        original_emb = self.embedding_layer(original_idx).swapaxes(0, 1) # from NTC to TNC
        # we do not need to include bos and eos token in embedding
        paraphrase_emb = self.embedding_layer(paraphrase_idx[:, 1:-1]).swapaxes(0, 1) # same as above
        # encoder part
        mu, sg, last_state = self.encoder(original_emb, paraphrase_emb)
        # sample from Gaussian distribution N(0, 1), of the sample shape to mu/lv
        eps = nd.normal(loc=0, scale=1, shape=(original_emb.shape[1], self.hidden_size), ctx=model_ctx)
        latent_input = mu + nd.exp(0.5 * sg) * eps  # exp is to make the std dev positive
        # the KL Div should be calculated between the sample from N(0, 1), and the distribution after
        # Parameterization Trick, negation since we want it to be small
        kl_loss = -self.kl_div(mu, sg)
        # first paraphrase_input should be the bos token
        last_idx = paraphrase_idx[:, 0:1]
        log_loss = 0
        # decode the sample
        for pos in range(paraphrase_idx.shape[0]-1):
            y, last_state = self.decoder(last_state, last_idx, latent_input)
            last_idx = y.argmax(axis=-1)
            log_loss += self.log_loss(y.swapaxes(0, 1), paraphrase_idx[:, pos+1:pos+2])
        # y is the decoded full sentence, of layout TNC, need to change to NTC
        loss = log_loss + kl_loss
        return loss

    def predict(self, original_idx, paraphrase_idx, normal_distr, max_len=25):
        '''FIXME: this might be wrong
        this method is for predicting a paraphrase sentence
        '''
        original_emb = self.embedding_layer(original_idx).swapaxes(0, 1)
        last_input = nd.zeros(shape=(1, 1, self.emb_size), ctx=model_ctx)
        last_state = self.encoder.encode(original_emb)
        # we will just pred `max_len` tokens, and address <eos> token outside this method
        pred_tk = []
        for _ in range(max_len):
            pred, last_state = self.decoder(last_state, last_input, normal_distr)
            pred_tk.append(int(pred.argmax(axis=-1).squeeze().astype('int32').asscalar()))
        return pred_tk