import torch
from torch import nn
import torch.nn.functional as F


class VisionTransformer(nn.Module):
  def __init__(self, in_features, num_patches, num_classes, proj_dim=64, num_heads=8,
              num_layers=8, mlp_size=2048, dropout=0.5):
    super().__init__()
    self.register_buffer(name='patches_pos', tensor=torch.arange(0, num_patches+1))
    self.projector = nn.Linear(in_features=in_features, out_features=proj_dim)
    self.pos_embedding = nn.Embedding(num_embeddings=num_patches+1, embedding_dim=proj_dim)
    self.cls_param = nn.Parameter(data=torch.empty(1, proj_dim))

    self.encoder_layer = nn.TransformerEncoderLayer(d_model=proj_dim, nhead=num_heads,
                                                    activation=F.gelu, batch_first=True,
                                                    norm_first=True)
    self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    self.norm = nn.LayerNorm(proj_dim, eps=1e-5)
    self.linear1 = nn.Linear(proj_dim, mlp_size)
    self.dropout1 = nn.Dropout(dropout)
    self.linear2 = nn.Linear(mlp_size, mlp_size//2)
    self.dropout2 = nn.Dropout(dropout)
    self.linear_out = nn.Linear(mlp_size//2, num_classes)
    
    self.cls_param.data.uniform_(-1, 1) #init weights
    

  def forward(self, x):
    x = self.projector(x)
    x = torch.cat((self.cls_param.expand(x.shape[0], -1, -1), x), dim=1)
    x = x + self.pos_embedding(self.patches_pos)

    x = self.encoder(x)
    pred_features = x[:, 0, :]

    out = self.norm(pred_features)
    out = F.gelu(self.linear1(out))
    out = self.dropout1(out)
    out = F.gelu(self.linear2(out))
    out = self.dropout2(out)
    return self.linear_out(out)