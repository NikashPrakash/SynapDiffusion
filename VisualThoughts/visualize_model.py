from models.MEGDecoder import MEGDecoder
# from torchviz import make_dot
import torch
import dataclasses
import typing as tp

@dataclasses.dataclass
class SegmentBatch():
    """Collatable training data."""
    meg: torch.Tensor
    features: torch.Tensor
    features_mask: torch.Tensor
    subject_index: torch.Tensor

    def to(self, device: tp.Any) -> "SegmentBatch":
        """Creates a new instance on the appropriate device."""
        out: tp.Dict[str, torch.Tensor] = {}
        for field in dataclasses.fields(self):
            data = getattr(self, field.name)
            if isinstance(data, torch.Tensor):
                out[field.name] = data.to(device)
            else:
                out[field.name] = data
        return SegmentBatch(**out)

    def replace(self, **kwargs) -> "SegmentBatch":
        cls = self.__class__
        kw = {}
        for field in dataclasses.fields(cls):
            if field.name in kwargs:
                kw[field.name] = kwargs[field.name]
            else:
                kw[field.name] = getattr(self, field.name)
        return cls(**kw)

    def __getitem__(self, index) -> "SegmentBatch":
        cls = self.__class__
        kw = {}
        indexes = torch.arange(
            len(self), device=self.meg.device)[index].tolist()  # explicit indexes for lists
        for field in dataclasses.fields(cls):
            data = getattr(self, field.name)
            if isinstance(data, list):
                if data:
                    value = [data[idx] for idx in indexes]
                else:
                    value = []
            else:
                value = data[index]
            kw[field.name] = value
        return cls(**kw)

    def __len__(self) -> int:
        return len(self.meg)

    @classmethod
    def collate_fn(cls, meg_features_list: tp.List["SegmentBatch"]) -> "SegmentBatch":
        out: tp.Dict[str, torch.Tensor] = {}
        for field in dataclasses.fields(cls):
            data = [getattr(mf, field.name) for mf in meg_features_list]
            if isinstance(data[0], torch.Tensor):
                out[field.name] = torch.stack(data)
            else:
                out[field.name] = [x for y in data for x in y]
        meg_features = SegmentBatch(**out)
        # check that list sizes are either 0 or batch size
        batch_size = meg_features.meg.shape[0]
        for field in dataclasses.fields(meg_features):
            val = out[field.name]
            if isinstance(val, list):
                assert len(val) in (0, batch_size), f"Incorrect size for {field.name}"
        return meg_features


#sample data
meg = torch.randn(2, 12, 128)
features = torch.randn(2, 8, 128)
subject_index = torch.tensor([1, 0], dtype=torch.int64)
features_mask=torch.ones(2, 1, 128)
batch = SegmentBatch(meg=meg,
        features=features,
        features_mask=torch.ones(2, 1, 128),
        subject_index=subject_index)


model = MEGDecoder(in_channels=dict(meg=12, features=8),
                            out_channels=12,
                            hidden=dict(meg=200, features=20),
                            depth=1,
                            linear_out=True, attention=2,
                            conv_dropout=0.1, growth=1.2,
                            concatenate=True,subject_layers=True)
inp = {"meg":batch.meg, "features":batch.features}
y = model(inp, batch)
# generate a model architecture visualization - gives dot file -> graphviz
# make_dot(y.mean(),
#          params=dict(model.named_parameters()),
#          show_attrs=True,
#          show_saved=True).render("MegDecoder_viz", format="png")

'''
MEGDecoder(
  (subject_layers): SubjectLayers(12, 12, 200)
  (subject_embedding): ScaledEmbedding(
    (embedding): Embedding(200, 64)
  )
  (encoders): ModuleDict(
    (concat): ConvSequence(
      (sequence): ModuleList(
        (0): Sequential(
          (0): Conv1d(20, 220, kernel_size=(4,), stride=(2,), padding=(2,))
          (1): LeakyReLU(negative_slope=0.0)
          (2): Dropout(p=0.1, inplace=False)
        )
      )
      (glus): ModuleList(
        (0): None
      )
    )
  )
  (lstm): LSTM(
    (lstm): LSTM(284, 220, num_layers=2)
  )
  (attentions): ModuleList(
    (0-1): 2 x Attention(
      (content): Conv1d(220, 220, kernel_size=(1,), stride=(1,))
      (query): Conv1d(220, 220, kernel_size=(1,), stride=(1,))
      (key): Conv1d(220, 220, kernel_size=(1,), stride=(1,))
      (embedding): Embedding(101, 55)
      (bn): BatchNorm1d(220, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc): Conv1d(220, 220, kernel_size=(1,), stride=(1,))
    )
  )
  (final): Conv1d(183, 12, kernel_size=(1,), stride=(1,))
  (decoder): ConvSequence(
    (sequence): ModuleList(
      (0): Sequential(
        (0): ConvTranspose1d(220, 183, kernel_size=(4,), stride=(2,), padding=(2,))
        (1): LeakyReLU(negative_slope=0.0)
        (2): Dropout(p=0.1, inplace=False)
      )
    )
    (glus): ModuleList(
      (0): None
    )
  )
)
'''
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/meg_decoder_1')
writer.add_graph(model, (inp,batch))
writer.close()
