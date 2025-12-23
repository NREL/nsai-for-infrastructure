
# The base TransformerEncoder including embedding and positional encoder
# output of forward is the raw encoding of each element of the input sequence.
# Caller can do whatever they want after that (see the example of forward_classifier() )

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
 
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        """
        Args:
            x: shape is [seq_len, batch_size, d_model]
        """
        # pe is shape [seq_len, 1, d_model]
        pe = self.pe[:x.size(0), :]
        # expand pe to [seq_len, batch_size, d_model]
        pe = pe.repeat(1, x.size(1), 1)
        x = x + pe
        return x

class EncoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(EncoderOnlyTransformer, self).__init__()
 
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        d_final = d_model
        self.lin0 = nn.Linear(d_model, d_final)
        self.classifier = nn.Linear(d_final, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.embedding(src)
        # PATRICK: need to permute dims of src to be [seq_len, batch_size, d_model]
        src = src.permute(1,0,2) 
#        print ("EMBEDDED", src)
        src = self.pos_encoder(src)
#        print ("POS", src)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask=src_key_padding_mask)
        return output
        
    def forward_classifier(self, src, src_mask=None, src_key_padding_mask=None):
        output = self.forward(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    # # works for classifier, but not generally desirable to use mean()
    #     output = output.permute(1,0,2) 
    #     output = output.mean(dim=1)  ### these are for using the output as a clasifier.
    # # Instead, Patrick says use the encoding for just the "special token" corresponding to the purpose of the net (classify, policy, value)
        output = output[0,:,:]  # 
        output = self.lin0(output) #.flatten()
        output = self.classifier(output)
        return output

    def forward_special_token(self, src, src_mask=None, src_key_padding_mask=None):
        # run the encoder, then assume the first token in the sequence is the special token
        # whose encoding we will use as the encoding for the whole sequence, take just that row
        # and pass it through a linear layer.  Caller can turn that into anything they want: policy, value, etc
        output = self.forward(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = output[0,:,:]  
        output = self.lin0(output)
        return output
