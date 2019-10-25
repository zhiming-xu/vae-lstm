import mxnet as mx
from mxnet.gluon import nn, rnn

class VAEEncoder(nn.HybridBlock):
    '''
    encoder part of the VAE model
    '''
    def __init__(self, hidden_size, num_layers=3, layout='TNC', dropout=0.3, bidir=True, **kwargs):
        '''
        init this class, create relevant rnns
        '''
        super(VAEEncoder, self).__init__(**kwargs)
        with self.name_scope():
            self.layout = layout
            self.original_encoder = rnn.LSTM(hidden_size=hidden_size, num_layers=num_layers, \
                                    layout=layout, dropout=dropout, bidirectional=bidir)
            self.paraphrase_encoder = rnn.LSTM(hidden_size=hidden_size, num_layers=num_layers, \
                                      layout=layout, dropout=dropout, bidirectional=bidir)
            # we will only use the last output of this lstm
            self.output_layer = rnn.LSTM(hidden_size=hidden_size, num_layers=1, layout=layout, \
                                dropout=dropout, bidirectional=bidir)

    def hybrid_forward(self, F, original_input, paraphrase_input):
        '''
        forward pass, inputs are embeddings of original sentences and paraphrase sentences
        '''        
        original_encoded = self.original_encoder(original_input)
        org_para_concated = F.concat(original_encoded, paraphrase_input, dim=self.layout.find('C'))
        paraphrase_encoded = self.paraphrase_encoder(org_para_concated)
        # only need the output of last time step
        output = self.output_layer(paraphrase_encoded)[-1]
        return output
        
class VAEDecoder(nn.HybridBlock):
    '''
    decoder part of the VAE model
    '''
    pass