import model
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# PART 1.2
# Define the simple decoder model here with correct inputs as defined in models.py
def define_simple_decoder(hidden_size, input_vocab_len, output_vocab_len, max_length):
    """ Provides a simple decoder instance
        NOTE: Not all the function arguments are needed - you need to figure out which arguments to use

    :return: a simple decoder instance
    """
    decoder = None

    # Write your implementation here
    decoder = model.DecoderRNN(hidden_size, output_vocab_len)
    # End of implementation

    return decoder


# PART 1.2
# Run the decoder model with correct inputs as defined in models.py
def run_simple_decoder(simple_decoder, decoder_input, encoder_hidden, decoder_hidden, encoder_outputs):
    """ Runs the simple_decoder
        NOTE: Not all the function arguments are needed - you need to figure out which arguments to use

    :return: The appropriate values
            HINT: Look at what the caller of this function in seq2seq.py expects as well as the simple decoder
                    definition in model.py
    """
    results = None

    # Write your implementation here
    results = simple_decoder.forward(decoder_input, decoder_hidden)
    # End of implementation

    return results


# PART 2.2
class BidirectionalEncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(BidirectionalEncoderRNN, self).__init__()

        # Write your implementation here
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        # End of implementation

    def forward(self, input, hidden):
        # Write your implementation here
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden
        # End of implementation

    def initHidden(self):
        return torch.zeros(1*2, 1, self.hidden_size, device=device)

# PART 2.2
# Define the encoder model here
def define_bi_encoder(input_vocab_len, hidden_size):
    # Write your implementation here
    encoder = BidirectionalEncoderRNN(input_vocab_len, hidden_size)
    # End of implementation

    return encoder


# PART 2.2
# Correct the dimension of encoder output by adding the forward and backward representation
def fix_bi_encoder_output_dim(encoder_output, hidden_size):
    # Write your implementation here
    output = (encoder_output[:, :, :hidden_size] +
              encoder_output[:, :, hidden_size:])
    # End of implementation

    return output


# PART 2.2
# Correct the dimension of encoder hidden by considering only one sided layer
def fix_bi_encoder_hidden_dim(encoder_hidden):
    # Write your implementation here
    output = encoder_hidden[:-1]
    # End of implementation

    return output


# PART 2.2
class AttnDecoderRNNDot(nn.Module):

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNNDot, self).__init__()

        # Write your implementation here
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        # End of implementation

    def forward(self, input, hidden, encoder_outputs):

        # Write your implementation here
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # changed from AttnDecoderRNN: uses dot attention instead
        attn_weights = F.softmax(torch.matmul(hidden[0], torch.t(encoder_outputs)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights
        # End of implementation

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# PART 3.1 goes below this comment

class MultiLayerBiDirectionalEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, num_layers=num_layers)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



class MultiLayerAttnDotDecoder(nn.Module):
    """ Write class definition for AttnDecoderRNNDot
        Hint: Modify AttnDecoderRNN to use dot attention
    """

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=10, num_layers=2):
        super(AttnDecoderRNNDot, self).__init__()

        # Write your implementation here
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        # End of implementation

    def forward(self, input, hidden, encoder_outputs):

        # Write your implementation here
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # changed from AttnDecoderRNN: uses dot attention instead
        attn_weights = F.softmax(torch.matmul(hidden[0], torch.t(encoder_outputs)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0) # why is it a Linear layer?

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights
        # End of implementation

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
