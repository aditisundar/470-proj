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

    :param hidden_size:
    :param input_vocab_len
    :param output_vocab_len
    :param max_length

    :return: a simple decoder instance
    """
    decoder = None

    # Write your implementation here

    # End of implementation

    return decoder


# PART 1.2
# Run the decoder model with correct inputs as defined in models.py
def run_simple_decoder(simple_decoder, decoder_input, encoder_hidden, decoder_hidden, encoder_outputs):
    """ Runs the simple_decoder
        NOTE: Not all the function arguments are needed - you need to figure out which arguments to use

    :param simple_decoder: the simple decoder object
    :param decoder_input:
    :param decoder_hidden:
    :param encoder_hidden:
    :param encoder_outputs:

    :return: The appropriate values
            HINT: Look at what the caller of this function in seq2seq.py expects as well as the simple decoder
                    definition in model.py
    """
    results = None

    # Write your implementation here

    # End of implementation

    return results


# PART 2.2
class BidirectionalEncoderRNN(nn.Module):
    """Write class definition for BidirectionalEncoderRNN
    """

    def __init__(self, input_size, hidden_size):
        """

        :param input_size:
        :param hidden_size:
        """

        super(BidirectionalEncoderRNN, self).__init__()

        # Write your implementation here

        # End of implementation

    def forward(self, input, hidden):
        """

        :param input:
        :param hidden:

        :return: output, hidden

            Hint: Don't correct the dimensions of the return values at this stage.
                    Function skeletons for doing this are provided later.
        """

        # Write your implementation here

        # End of implementation

    def initHidden(self):
        return torch.zeros(1*2, 1, self.hidden_size, device=device)


# PART 2.2
# Define the encoder model here
def define_bi_encoder(input_vocab_len, hidden_size):
    """ Defines bidirectional encoder RNN

    :param input_vocab_len:
    :param hidden_size:
    :return:
    """

    encoder = None

    # Write your implementation here

    # End of implementation

    return encoder


# PART 2.2
# Correct the dimension of encoder output by adding the forward and backward representation
def fix_bi_encoder_output_dim(encoder_output, hidden_size):
    """

    :param encoder_output:
    :param hidden_size:
    :return:
    """
    output = None

    # Write your implementation here

    # End of implementation

    return output


# PART 2.2
# Correct the dimension of encoder hidden by considering only one sided layer
def fix_bi_encoder_hidden_dim(encoder_hidden):
    """

    :param encoder_hidden:
    :return:
    """

    output = None

    # Write your implementation here

    # End of implementation

    return output


# PART 2.2
class AttnDecoderRNNDot(nn.Module):
    """ Write class definition for AttnDecoderRNNDot
        Hint: Modify AttnDecoderRNN to use dot attention
    """

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNNDot, self).__init__()

        # Write your implementation here

        # End of implementation

    def forward(self, input, hidden, encoder_outputs):

        # Write your implementation here

        pass
        
        # End of implementation

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# PART 3.1 goes below this comment
