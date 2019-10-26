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
    encoder part of the VAE model
    '''
    def __init__(self, hidden_size, num_layers=3, layout='TNC', dropout=.3, bidir=False, **kwargs):
        '''
        init this class, create relevant rnns
        '''
        super(VAEEncoder, self).__init__(**kwargs)
        with self.name_scope():
            self.layout = layout
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.original_encoder = rnn.LSTM(hidden_size=hidden_size, num_layers=num_layers, \
                                             layout=layout, dropout=dropout, bidirectional=bidir)
            self.paraphrase_encoder = rnn.LSTM(hidden_size=hidden_size, num_layers=num_layers, \
                                               layout=layout, dropout=dropout, bidirectional=bidir)
            # we will only use the last output of this lstm
            self.output_mu = nn.Dense(units=hidden_size, activation='relu', dropout=dropout)
            self.output_lv = nn.Dense(units=hidden_size, activation='relu', dropout=dropout)

    def forward(self, original_input, paraphrase_input):
        '''
        forward pass, inputs are embeddings of original sentences and paraphrase sentences
        '''
        zero_hc = nd.zeros(shape=(self.num_layers, original_input.shape[self.layout.find('N')], \
                                  self.hidden_size))
        original_encoded, original_encoder_hc = self.original_encoder(original_input, \
                                                                     [zero_hc, zero_hc])
        ori_para_concated = nd.concat(original_encoded, paraphrase_input, dim=self.layout.find('C'))
        paraphrase_encoded, _ = self.paraphrase_encoder(ori_para_concated, original_encoder_hc)
        # this is the \phi of VAE encoder, i.e., \mu and \sigma
        mu = self.output_mu(paraphrase_encoded)
        lv = self.output_lv(paraphrase_encoded)
        return mu, lv
        
class VAEDecoder(nn.Block):
    '''
    decoder part of the VAE model
    '''
    def __init__(self, emb_size, hidden_size, num_layers=3, layout='TNC', dropout=.3, \
                 bidir=False, **kwargs):
        '''
        init this class, create relevant rnns
        '''
        super(VAEDecoder, self).__init__(**kwargs)
        with self.name_scope():
            self.layout = layout
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.original_encoder = rnn.LSTM(hidden_size=hidden_size, num_layers=num_layers, \
                                    layout=layout, dropout=dropout, bidirectional=bidir)
            self.paraphrase_decoder = rnn.LSTM(hidden_size=emb_size, num_layers=num_layers, \
                                      layout=layout, dropout=dropout, bidirectional=bidir)

    def forward(self, original_input, paraphrase_input, latent_input):
        '''
        forward pass, inputs are embeddings of original sentences, paraphrase sentences and 
        latent output of encoder, i.e., z calculated from mu and lv
        '''
        start_state = self.original_encoder.begin_state(batch_size= \
                      original_input.shape[self.layout.find('N')])
        _, original_encoded_hc = self.original_encoder(original_input, start_state)
        latent_input = latent_input.broadcast_to(shape=paraphrase_input.shape)
        decoder_input = nd.concat(paraphrase_input, latent_input, dim=self.layout.find('C'))
        decoder_output, _ = self.paraphrase_decoder(decoder_input, original_encoded_hc)
        return decoder_output

class VAE_LSTM(nn.Block):
    '''
    wrapper of all this model
    '''
    def __init__(self, emb_size, hidden_size, num_layers, layout='TNC', dropout=.3, \
                 bidir=False, **kwargs):
        super(VAE_LSTM, self).__init__(**kwargs)
        with self.name_scope():
            self.soft_zero = 1e-8
            self.layout = layout
            self.hidden_size = hidden_size
            self.encoder = VAEEncoder(hidden_size=hidden_size, num_layers=num_layers, \
                           layout=layout, dropout=dropout, bidir=bidir)
            self.decoder = VAEDecoder(emb_size=emb_size, hidden_size=hidden_size, \
                           num_layers=num_layers, layout=layout, dropout=dropout, bidir=bidir)

    def forward(self, original_input, paraphrase_input):
        mu, lv = self.encoder(original_input)
        eps = nd.normal(loc=0, scale=1, shape=(original_input.shape[self.layout.find('N')], \
                                               self.hidden_size), ctx=model_ctx)
        latent_input = mu + nd.exp(0.5 * lv) * eps
        y = self.decoder(original_input, paraphrase_input, latent_input)
        self.output = y
        KL = 0.5 * nd.sum(1 + lv - mu * mu - nd.exp(lv), axis=1)
        logloss = nd.sum(paraphrase_input * nd.log(y + self.soft_zero) + (1 - paraphrase_input) * \
                  nd.log(1 - y + self.soft_zero), axis=1)
        loss = - logloss - KL
        return loss

                    