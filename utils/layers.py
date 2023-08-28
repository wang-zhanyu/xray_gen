import torch
from torch import nn


class TextFcLayer(nn.Module):
  """Layers used in mapping text embeddings to visual outputs."""

  def __init__(self, in_dim, out_dim, num_input_tokens, num_output_tokens):
    super().__init__()

    self.num_input_tokens = num_input_tokens
    self.num_output_tokens = num_output_tokens

    hidden_dim = 512
    self.fc = nn.Linear(in_dim, hidden_dim)
    self.tfm = nn.Transformer(batch_first=True, norm_first=True,
                              d_model=hidden_dim, num_encoder_layers=4, num_decoder_layers=4,
                              dim_feedforward=hidden_dim * 4, dropout=0.0, nhead=4)
    self.model = nn.Linear(hidden_dim, out_dim)
    self.query_embs = nn.Parameter(torch.randn(1, num_output_tokens, hidden_dim))


  def forward(self, x, input_embs):
    outputs = None
    x = x + input_embs
    x = self.fc(x)
    # self.query_embs = self.query_embs.to(x.dtype)
    query = self.query_embs.repeat(x.shape[0], 1, 1)
    x = self.tfm(x, query.half())
    # x = self.tfm(x, query)
    outputs = self.model(x)
    return outputs  # (N, T, D)


if __name__ == "__main__":
    model = TextFcLayer(4096, 768, 8, 77).cuda()
    x = torch.rand(4, 8, 4096).cuda().half()
    input_embs = torch.rand(4, 8, 4096).cuda().half()

    out = model(x, input_embs)
    print(out.shape)
    print(out.dtype)

