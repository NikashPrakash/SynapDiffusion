import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import typing as tp
from functools import partial
from math import ceil
from loadBatches import *

class ScaledEmbedding(nn.Module):
    """Scale up learning rate for the embedding, otherwise, it can move too slowly.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, scale: float = 10.):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data /= scale
        self.scale = scale

    @property
    def weight(self):
        return self.embedding.weight * self.scale

    def forward(self, x):
        return self.embedding(x) * self.scale


class SubjectLayers(nn.Module):
    """Per subject linear layer."""
    def __init__(self, in_channels: int, out_channels: int, n_subjects: int, init_id: bool = False):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_subjects, in_channels, out_channels))
        if init_id:
            assert in_channels == out_channels
            self.weights.data[:] = torch.eye(in_channels)[None]
        self.weights.data *= 1 / in_channels**0.5

    def forward(self, x, subjects):
        _, C, D = self.weights.shape
        weights = self.weights.gather(0, subjects.view(-1, 1, 1).expand(-1, C, D))
        return torch.einsum("bct,bcd->bdt", x, weights)

    def __repr__(self):
        S, C, D = self.weights.shape
        return f"SubjectLayers({C}, {D}, {S})"


class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    """
    def __init__(self, channels: int, init: float = 0.1, boost: float = 5.):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init / boost
        self.boost = boost

    def forward(self, x):
        return (self.boost * self.scale[:, None]) * x


class ConvSequence(nn.Module):

    def __init__(self, channels: tp.Sequence[int], kernel: int = 4, dilation_growth: int = 1,
                 dilation_period: tp.Optional[int] = None, stride: int = 2,
                 dropout: float = 0.0, leakiness: float = 0.0, groups: int = 1,
                 decode: bool = False, batch_norm: bool = False, dropout_input: float = 0,
                 skip: bool = False, scale: tp.Optional[float] = None, rewrite: bool = False,
                 activation_on_last: bool = True, post_skip: bool = False, glu: int = 0,
                 glu_context: int = 0, glu_glu: bool = True, activation: tp.Any = None) -> None:
        super().__init__()
        dilation = 1
        channels = tuple(channels)
        self.skip = skip
        self.sequence = nn.ModuleList()
        self.glus = nn.ModuleList()
        if activation is None:
            activation = partial(nn.LeakyReLU, leakiness)
        Conv = nn.Conv1d if not decode else nn.ConvTranspose1d
        # build layers
        for k, (chin, chout) in enumerate(zip(channels[:-1], channels[1:])):
            layers: tp.List[nn.Module] = []
            is_last = k == len(channels) - 2

            # Set dropout for the input of the conv sequence if defined
            if k == 0 and dropout_input:
                assert 0 < dropout_input < 1
                layers.append(nn.Dropout(dropout_input))

            # conv layer
            if dilation_growth > 1:
                assert kernel % 2 != 0, "Supports only odd kernel with dilation for now"
            if dilation_period and (k % dilation_period) == 0:
                dilation = 1
            pad = kernel // 2 * dilation
            layers.append(Conv(chin, chout, kernel, stride, pad,
                               dilation=dilation, groups=groups if k > 0 else 1))
            dilation *= dilation_growth
            # non-linearity
            if activation_on_last or not is_last:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(num_features=chout))
                layers.append(activation())
                if dropout:
                    layers.append(nn.Dropout(dropout))
                if rewrite:
                    layers += [nn.Conv1d(chout, chout, 1), nn.LeakyReLU(leakiness)]
                    # layers += [nn.Conv1d(chout, 2 * chout, 1), nn.GLU(dim=1)]
            if chin == chout and skip:
                if scale is not None:
                    layers.append(LayerScale(chout, scale))
                if post_skip:
                    layers.append(Conv(chout, chout, 1, groups=chout, bias=False))

            self.sequence.append(nn.Sequential(*layers))
            if glu and (k + 1) % glu == 0:
                ch = 2 * chout if glu_glu else chout
                act = nn.GLU(dim=1) if glu_glu else activation()
                self.glus.append(
                    nn.Sequential(
                        nn.Conv1d(chout, ch, 1 + 2 * glu_context, padding=glu_context), act))
            else:
                self.glus.append(None)

    def forward(self, x: tp.Any) -> tp.Any:
        for module_idx, module in enumerate(self.sequence):
            old_x = x
            x = module(x)
            if self.skip and x.shape == old_x.shape:
                x = x + old_x
            glu = self.glus[module_idx]
            if glu is not None:
                x = glu(x)
        return x

class LSTM(nn.Module):
    """A wrapper for nn.LSTM that outputs the same amount
    of features if bidirectional or not bidirectional.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=bidirectional)
        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, x):
        x, h = self.lstm(x)
        if self.linear:
            x = self.linear(x)
        return x, h


class Attention(nn.Module):
    def __init__(self, channels: int, radius: int = 50, heads: int = 4):
        super().__init__()
        assert channels % heads == 0
        self.content = nn.Conv1d(channels, channels, 1)
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        self.embedding = nn.Embedding(radius * 2 + 1, channels // heads)
        # Let's make this embedding a bit smoother
        weight = self.embedding.weight.data
        weight[:] = weight.cumsum(0) / torch.arange(1, len(weight) + 1).float().view(-1, 1).sqrt()
        self.heads = heads
        self.radius = radius

        self.bn = nn.BatchNorm1d(channels)
        self.fc = nn.Conv1d(channels, channels, 1)
        self.scale = nn.Parameter(torch.full([channels], 0.1))

    def forward(self, x):
        def _split(y):
            return y.view(y.shape[0], self.heads, -1, y.shape[2])

        content = _split(self.content(x))
        query = _split(self.query(x))
        key = _split(self.key(x))

        batch_size, _, dim, length = content.shape

        # first index `t` is query, second index `s` is key.
        dots = torch.einsum("bhct,bhcs->bhts", query, key)

        steps = torch.arange(length, device=x.device)
        relative = (steps[:, None] - steps[None, :])
        embs = self.embedding.weight.gather(
            0, self.radius + relative.clamp_(-self.radius, self.radius).view(-1, 1).expand(-1, dim))
        embs = embs.view(length, length, -1)
        dots += 0.3 * torch.einsum("bhct,tsc->bhts", query, embs)

        # we kill any reference outside of the radius
        dots = torch.where(
            relative.abs() <= self.radius, dots, torch.tensor(-float('inf')).to(embs))

        weights = torch.softmax(dots, dim=-1)
        out = torch.einsum("bhts,bhcs->bhct", weights, content)
        out += 0.3 * torch.einsum("bhts,tsc->bhct", weights, embs)
        out = out.reshape(batch_size, -1, length)
        out = F.relu(self.bn(self.fc(out))) * self.scale.view(1, -1, 1)
        return out

class MEGDecoder(nn.Module):
    def __init__(self,
                 # Channels
                 in_channels: tp.Dict[str, int],
                 out_channels: int,
                 hidden: tp.Dict[str, int],
                 # Overall structure
                 depth: int = 2,
                 linear_out: bool = False,
                 complex_out: bool = False,
                 concatenate: bool = False,  # concatenate the inputs
                 # Conv structure
                 kernel_size: int = 4,
                 stride: int = 2,
                 growth: float = 1.,
                 # LSTM
                 lstm: int = 2,
                 flip_lstm: bool = False,
                 bidirectional_lstm: bool = False,
                 # Attention
                 attention: int = 0,
                 heads: int = 4,
                 # Dropout, BN, activations
                 conv_dropout: float = 0.0,
                 lstm_dropout: float = 0.0,
                 dropout_input: float = 0.0,
                 batch_norm: bool = False,
                 relu_leakiness: float = 0.0,
                 # Subject embeddings,
                 n_subjects: int = 200,
                 subject_dim: int = 64,
                 embedding_location: tp.List[str] = ["lstm"],  # can be lstm or input
                 embedding_scale: float = 1.0,
                 subject_layers: bool = False,
                 subject_layers_dim: str = "input",  # or hidden
                 ):
        super().__init__()
        if set(in_channels.keys()) != set(hidden.keys()):
            raise ValueError("Channels and hidden keys must match "
                             f"({set(in_channels.keys())} and {set(hidden.keys())})")

        self._concatenate = concatenate
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.embedding_location = embedding_location

        self.subject_layers = None
        if subject_layers:
            assert "meg" in in_channels
            meg_dim = in_channels["meg"]
            dim = {"hidden": hidden["meg"], "input": meg_dim}[subject_layers_dim]
            self.subject_layers = SubjectLayers(meg_dim, dim, n_subjects)
            in_channels["meg"] = dim
        self.subject_embedding = None
        if subject_dim:
            self.subject_embedding = ScaledEmbedding(n_subjects, subject_dim, embedding_scale)
            if "input" in embedding_location:
                in_channels["meg"] += subject_dim

        # concatenate inputs if need be
        if concatenate:
            in_channels = {"concat": sum(in_channels.values())}
            hidden = {"concat": sum(hidden.values())}

        # compute the sequences of channel sizes
        sizes = {}
        for name in in_channels:
            sizes[name] = [in_channels[name]]
            sizes[name] += [int(round(hidden[name] * growth ** k)) for k in range(depth)]

        lstm_hidden = sum(sizes[n][-1] for n in in_channels)
        lstm_input = lstm_hidden
        if "lstm" in embedding_location:
            lstm_input += subject_dim

        # encoders and decoder
        params: tp.Dict[str, tp.Any]
        params = dict(kernel=kernel_size, stride=stride,
                      leakiness=relu_leakiness, dropout=conv_dropout, dropout_input=dropout_input,
                      batch_norm=batch_norm)
        self.encoders = nn.ModuleDict({name: ConvSequence(channels, **params)
                                       for name, channels in sizes.items()})

        # lstm
        self.lstm = None
        self.linear = None
        if lstm:
            self.lstm = LSTM(
                input_size=lstm_input,
                hidden_size=lstm_hidden,
                dropout=lstm_dropout,
                num_layers=lstm,
                bidirectional=bidirectional_lstm)
            self._flip_lstm = flip_lstm

        self.attentions = nn.ModuleList()
        for _ in range(attention):
            self.attentions.append(Attention(lstm_hidden, heads=heads))

        # decoder
        decoder_sizes = [int(round(lstm_hidden / growth ** k)) for k in range(depth + 1)]
        self.final = None
        if linear_out:
            assert not complex_out
            self.final = nn.Conv1d(decoder_sizes[-1], out_channels, 1)
        elif complex_out:
            self.final = nn.Sequential(
                nn.Conv1d(decoder_sizes[-1], 2 * decoder_sizes[-1], 1),
                nn.ReLU(),
                nn.Conv1d(2 * decoder_sizes[-1], out_channels, 1))
        else:
            params['activation_on_last'] = False
            decoder_sizes[-1] = out_channels
            assert depth > 0, "if no linear out, depth must be > 0"
        self.decoder = ConvSequence(decoder_sizes, decode=True, **params)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, (size of the input - kernel_size) % stride = 0.

        If the input has a valid length, the output
        will have exactly the same length.
        """
        for idx in range(self.depth):
            length = ceil(length / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride
        return int(length)

    def pad(self, x):
        length = x.size(-1)
        valid_length = self.valid_length(length)
        delta = valid_length - length
        return F.pad(x, (0, delta))

    def forward(self, inputs, batch):
        subjects = batch.subject_index
        length = next(iter(inputs.values())).shape[-1]  # length of any of the inputs

        if self.subject_layers is not None:
            inputs["meg"] = self.subject_layers(inputs["meg"], subjects)
        if self.subject_embedding is not None:
            emb = self.subject_embedding(subjects)[:, :, None]
            if "input" in self.embedding_location:
                inputs["meg"] = torch.cat([inputs["meg"], emb.expand(-1, -1, length)], dim=1)

        if self._concatenate:
            input_list = [input_ for _, input_ in sorted(inputs.items())]
            inputs = {"concat": torch.cat(input_list, dim=1)}

        inputs = {name: self.pad(input_) for name, input_ in inputs.items()}
        encoded = {}
        for name, x in inputs.items():
            encoded[name] = self.encoders[name](self.pad(x))

        inputs = [x[1] for x in sorted(encoded.items())]
        if "lstm" in self.embedding_location and self.subject_embedding is not None:
            inputs.append(emb.expand(-1, -1, inputs[0].shape[-1]))

        x = torch.cat(inputs, dim=1)
        if self.lstm is not None:
            x = x.permute(2, 0, 1)
            if self._flip_lstm:
                x = x.flip([0])
            x, _ = self.lstm(x)
            if self._flip_lstm:
                x = x.flip([0])
            x = x.permute(1, 2, 0)

        for attention in self.attentions:
            x = x + attention(x)

        x = self.decoder(x)

        if self.final is not None:
            x = self.final(x)

        out = x[:, :, :length]
        return out
        
    def forward(self, input_data):
        return input_data
    
    
class EEGDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): #
        super(EEGDecoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True) # USE TEMPORAL ATTENTION INSTEAD OF LSTM, transformer or custom impl.
        self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=output_size, kernel_size=3, padding=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        lstmOutput, _ = self.lstm(input_data)
        lstmOutput = lstmOutput.permute(0, 2, 1) # Dont forget this change  
        convOutput = self.conv1(lstmOutput)
        convOutput = self.relu(convOutput)
        convOutput = torch.max(convOutput, dim=2)[0]  
        output = self.fc(convOutput)
        output = self.softmax(output)
        return output
    
class NeuralGenerator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        return input_data
    
class EEGDataset(Dataset):
    def __init__(self, labels_file, eeg_file, transform=None, target_transform=None):
        self.img_labels = torch.load(labels_file,map_location=torch.device("cuda"))
        self.eeg_file = eeg_file
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        #img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) TODO READ FROM BIG FILE
        eegSnippet = read_image(eeg_file) #TODO IMPLEMENT THIS
        label = self.img_labels.iloc[idx, 1]
        return eegSnippet, label