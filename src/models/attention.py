import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    Examples::

         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)

    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out_2 = nn.Linear(dim*2, dim)
        self.linear_out_3 = nn.Linear(dim*3, dim)
        self.mask = None


    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask


    def get_mix(self, output, context):
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size)).view(self.batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)
        return attn, mix


    def forward(self, output, context):
        self.batch_size = output.size(0)
        self.hidden_size = output.size(2)

        (question_out, review_outs) = context
        attn, question_mix = self.get_mix(output, question_out)

        if review_outs is not None:
            review_mixs = [self.get_mix(output, review_out)[1] for review_out in review_outs]
            review_mix = _mean(review_mixs)
            # concat -> (batch, out_len, 2*dim)
            combined = torch.cat((question_mix, review_mix, output), dim=2)
            # output -> (batch, out_len, dim)
            output = F.tanh(self.linear_out_3(combined.view(-1, 3 * self.hidden_size))).view(self.batch_size, -1, self.hidden_size)
        else:
            combined = torch.cat((question_mix, output), dim=2)
            # output -> (batch, out_len, dim)
            output = F.tanh(self.linear_out_2(combined.view(-1, 2 * self.hidden_size))).view(self.batch_size, -1, self.hidden_size)

        return output, attn


def _mean(vars):
    return torch.mean(torch.cat([i.unsqueeze(0) for i in vars], 0), 0)

