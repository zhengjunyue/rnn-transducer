import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout=0.2, bidirectional=True):
        super(BaseEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.output_proj = nn.Linear(2 * hidden_size if bidirectional else hidden_size,
                                     output_size,
                                     bias=True)

    def forward(self, inputs, input_lengths):
        assert inputs.dim() == 3

        if input_lengths is not None:
            sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True) # (按行降序排列)
            inputs = inputs[indices]
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, sorted_seq_lengths, batch_first=True) # 返回一个PackedSequence 对象

        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(inputs)

        if input_lengths is not None:
            _, desorted_indices = torch.sort(indices, descending=False)  # (按行升序排列)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True) # 把压紧的序列再填充回来， 返回的Varaible的值的size是 B×T×*
            outputs = outputs[desorted_indices]

        logits = self.output_proj(outputs)

        return logits, hidden

# （160,320,320,4,0.3）
def build_encoder(config):
    if config.enc.type == 'lstm':
        return BaseEncoder(
            input_size=config.feature_dim,
            hidden_size=config.enc.hidden_size,
            output_size=config.enc.output_size,
            n_layers=config.enc.n_layers,
            dropout=config.dropout,
            bidirectional=config.enc.bidirectional
        )
    else:
        raise NotImplementedError
