# !/usr/bin/env python3
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn, rnn

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
        forward pass, inputs are embeddings of original sentences and paraphrase sentences
        '''
        # to let lstm return final state and memory cell, we need to pass `start_state`
        start_state = self.original_encoder.begin_state(batch_size= \
                      original_input.shape[1], ctx=model_ctx)
        # original_encoded is the output of each time step, of shape TN[hidden_size] (omit for now)
        # original_encoder_state is a list: [hidden_output, memory cell] of the last time step
        _, original_encoder_state = self.original_encoder(original_input, start_state)
        '''TODO:  this part might be wrong
        # concat the hidden representation of original sentence and embedding of paraphrase
        # sentence, the result is of shape TN[hidden_size+emb_size]
        ori_para_concated = nd.concat(original_encoded, paraphrase_input, dim=-1)
        '''
        paraphrase_encoded, _ = self.paraphrase_encoder(paraphrase_input, original_encoder_state)
        # this is the \phi of VAE encoder, i.e., \mu and "\sigma", FIXME: use the last output now
        # thus their shapes are of (batch_size, hidden_size)
        mu = self.output_mu(paraphrase_encoded)[-1] # \mu, mean of sampled distribution
        sg = self.output_sg(paraphrase_encoded)[-1] # \sg, std dev of sampler distribution based on
        return mu, sg
        
class VAEDecoder(nn.Block):
    '''
    decoder part of the VAE model
    '''
    def __init__(self, output_size, hidden_size, num_layers=3, dropout=.3, \
                 bidir=False, **kwargs):
        '''
        init this class, create relevant rnns
        '''
        super(VAEDecoder, self).__init__(**kwargs)
        with self.name_scope():
            self.original_encoder = rnn.LSTM(hidden_size=hidden_size, num_layers=num_layers, \
                                             dropout=dropout, bidirectional=bidir, \
                                             prefix='original_sentence_encoder_VAEDecoder')
            self.paraphrase_decoder = rnn.LSTM(hidden_size=hidden_size, num_layers=num_layers, \
                                               dropout=dropout, bidirectional=bidir, \
                                               prefix='paraphrase_sentence_decoder_VAEDecoder')
            self.dense_output = nn.Dense(units=output_size, activation='sigmoid', flatten=False)

    def forward(self, original_input, paraphrase_input, latent_input):
        '''
        forward pass, inputs are embeddings of original sentences, paraphrase sentences and 
        latent output of encoder, i.e., z calculated from mu and lv
        '''
        # begin{same as the encoder}
        start_state = self.original_encoder.begin_state(batch_size= \
                      original_input.shape[1], ctx=model_ctx)
        _, original_encoded_state = self.original_encoder(original_input, start_state)
        # end{same as the encoder}
        # latent_input is of shape (batch_size, hidden_size), we need to add the time dimension
        # and repeat itself T times to concat to paraphrase embedding
        latent_input = latent_input.expand_dims(axis=0).repeat(repeats= \
                       paraphrase_input.shape[0], axis=0)
        # layout is TNC, so concat along the last, i.e., channel dimension
        decoder_input = nd.concat(paraphrase_input, latent_input, dim=-1)
        # decoder output is of shape TN[hidden_size]
        decoder_output, _ = self.paraphrase_decoder(decoder_input, original_encoded_state)
        # since we calculate KL-loss with layout TNC, we will keep it this way
        decoder_output = self.dense_output(decoder_output)
        return decoder_output

class VAE_LSTM(nn.Block):
    '''
    wrapper of all this model
    '''
    def __init__(self, emb_size, vocab_size, hidden_size, num_layers, dropout=.3, bidir=False, **kwargs):
        super(VAE_LSTM, self).__init__(**kwargs)
        with self.name_scope():
            self.soft_zero = 1e-6
            self.embedding_layer = nn.Embedding(vocab_size, emb_size)
            self.hidden_size = hidden_size
            self.encoder = VAEEncoder(hidden_size=hidden_size, num_layers=num_layers, \
                                      dropout=dropout, bidir=bidir)
            self.decoder = VAEDecoder(output_size=emb_size, hidden_size=hidden_size, \
                                      num_layers=num_layers, dropout=dropout, bidir=bidir)

    def forward(self, original_input, paraphrase_input):
        # from idx to sentence embedding
        original_input = self.embedding_layer(original_input).swapaxes(0, 1) # from NTC to TNC
        paraphrase_input = self.embedding_layer(paraphrase_input).swapaxes(0, 1) # same as above
        # encoder part
        mu, sg = self.encoder(original_input, paraphrase_input)
        # sample from Gaussian distribution N(0, 1), of the sample shape to mu/lv
        eps = nd.normal(loc=0, scale=1, shape=(original_input.shape[1], \
                                               self.hidden_size), ctx=model_ctx)
        latent_input = mu + nd.exp(0.5 * sg) * eps  # exp is to make the std dev non-negative
        # decode the sample
        y = self.decoder(original_input, paraphrase_input, latent_input)
        self.output = y
        # FIXME: the loss might not be calculated this way, since paraphrase_input is not a
        # probablity distribution
        KL = 0.5 * nd.sum(1 + sg - mu * mu - nd.exp(sg), axis=1)
        logloss = nd.sum(paraphrase_input * nd.log(y + self.soft_zero) + (1 - paraphrase_input) * \
                  nd.log(1 - y + self.soft_zero), axis=(0, 2))
        loss = - logloss - KL
        return loss