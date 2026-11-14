"""Scene + social Transformer for GC trajectory prediction (Transformer member's model).

Token sets fed to one cross-attention fusion block:
  ego      : the target's 5 observed steps after a temporal Transformer encoder
  neighbor : one token per neighbour slot (16), incl. a "stationary" feature
             (Yi et al., CVPR 2015: stationary groups dominate path choice)
  image    : 8x8 = 64 patch tokens from a small CNN over the anchor-frame crop
A single query token (ego summary + location) attends over all of them.

The output head (learned waypoint/end-goal + curve) mirrors the shared team
decoder in improved_forecast.SceneGoalForecast with one mode, so the loss,
units and evaluation stay identical to the team baseline. Shared files are only
imported, never modified.
"""
import math
import numpy as np
import torch
from torch import nn

STATIONARY_SPEED = .1  # 100-pixel units per 0.8 s step, i.e. < 10 px per step


def sinusoid(steps, width):
    position = torch.arange(steps, dtype=torch.float32)[:, None]
    frequency = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.) / width))
    table = torch.zeros(steps, width)
    table[:, 0::2] = torch.sin(position * frequency)
    table[:, 1::2] = torch.cos(position * frequency)
    return table


def position_features(data):
    """Same 17-dim location descriptor as the team model's spatial branch."""
    origin = data['origin']
    position = origin / torch.tensor([1920., 1080.], device=origin.device) * 2 - 1
    angles = position[:, :, None] * torch.tensor([1., 2., 4.], device=origin.device) * np.pi
    return torch.cat((position, angles.sin().flatten(1), angles.cos().flatten(1),
                      data['heading'], data['speed']), dim=-1)


def neighbor_features(neighbors, mask):
    """[B,N,T,4] heading-aligned history + [B,N,T] mask -> [B,N,F] and slot mask [B,N]."""
    m = mask.float()
    history = (neighbors * m[..., None]).flatten(2)                        # [B,N,T*4]
    delta_valid = mask[..., 1:] & mask[..., :-1]
    step_speed = neighbors[..., 1:, 2:].norm(dim=-1) * delta_valid       # [B,N,T-1]
    count = delta_valid.sum(dim=-1)
    speed = step_speed.sum(dim=-1) / count.clamp_min(1)
    stationary = ((speed < STATIONARY_SPEED) & (count > 0)).float()
    last = neighbors[..., -1, :2] * m[..., -1:]                          # position at anchor
    distance = last.norm(dim=-1)
    features = torch.cat((history, m, last, distance[..., None], speed[..., None],
                          stationary[..., None], (count > 0).float()[..., None]), dim=-1)
    return features, mask.any(dim=-1)


class ImageTokens(nn.Module):
    def __init__(self, width, in_channels=4):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1), nn.GroupNorm(8, o), nn.GELU())
        self.cnn = nn.Sequential(block(in_channels, 32), block(32, 64), block(64, width))
        self.width = width
        self.register_buffer('mean', torch.tensor([.45, .45, .45, 0.])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.25, .25, .25, 1.])[None, :, None, None])

    def forward(self, image):
        features = self.cnn((image - self.mean) / self.std)                  # [B,D,h,w]
        b, d, h, w = features.shape
        return features.flatten(2).transpose(1, 2), (h, w)                   # [B,h*w,D]


class SceneSocialTransformer(nn.Module):
    def __init__(self, width=64, heads=4, ffn=128, temporal_layers=2, fusion_layers=2,
                 dropout=.1, use_image=True, use_social=True, obs_len=5, image_size=64):
        super().__init__()
        self.use_image, self.use_social = use_image, use_social
        self.k = 1
        self.register_buffer('anchors', torch.zeros(1, 6, 2))  # kept for team checkpoint/eval helpers

        def encoder_layer():
            return nn.TransformerEncoderLayer(width, heads, ffn, dropout, activation='gelu',
                                              batch_first=True, norm_first=True)
        self.ego_in = nn.Linear(4, width)
        self.register_buffer('time_code', sinusoid(obs_len, width)[None])
        self.temporal = nn.TransformerEncoder(encoder_layer(), temporal_layers,
                                              norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.location = nn.Sequential(nn.Linear(17, 64), nn.GELU(), nn.Linear(64, width))
        self.type_code = nn.Parameter(torch.randn(3, width) * .02)  # ego / neighbour / image
        if use_social:
            self.neighbor_in = nn.Sequential(nn.Linear(obs_len * 5 + 6, 64), nn.GELU(), nn.Linear(64, width))
        if use_image:
            self.image = ImageTokens(width)
            grid = image_size // 8
            self.patch_code = nn.Parameter(torch.randn(grid * grid, width) * .02)
        fusion = nn.TransformerDecoderLayer(width, heads, ffn, dropout, activation='gelu',
                                            batch_first=True, norm_first=True)
        self.fusion = nn.TransformerDecoder(fusion, fusion_layers, norm=nn.LayerNorm(width))
        self.fuse = nn.Sequential(nn.Linear(width * 2 + 64, 128), nn.ReLU(), nn.Dropout(.05),
                                  nn.Linear(128, 128), nn.ReLU())
        self.position_mlp = nn.Sequential(nn.Linear(17, 64), nn.ReLU())
        # Output head: same structure as the shared team decoder (K = 1).
        self.goals = nn.Linear(128, 4)
        self.mode_embedding = nn.Parameter(torch.randn(1, 16) * .1)
        self.path = nn.Sequential(nn.Linear(128 + 16 + 4, 128), nn.ReLU(), nn.Linear(128, 12))
        self.scores = nn.Linear(128, 1)
        nn.init.zeros_(self.goals.weight); nn.init.zeros_(self.goals.bias)
        nn.init.zeros_(self.path[-1].weight); nn.init.zeros_(self.path[-1].bias)

    def forward(self, data):
        obs = data['obs']; b = len(obs)
        ego = self.temporal(self.ego_in(obs) + self.time_code[:, :obs.shape[1]])        # [B,5,D]
        location = position_features(data)
        query = (ego[:, -1] + self.location(location))[:, None]                          # [B,1,D]

        memory = [ego + self.type_code[0]]
        ignore = [torch.zeros(b, ego.shape[1], dtype=torch.bool, device=obs.device)]
        if self.use_social:
            features, slot = neighbor_features(data['neighbors'], data['neighbor_mask'])
            memory.append(self.neighbor_in(features) + self.type_code[1])
            ignore.append(~slot)  # empty neighbour slots are never attended
        if self.use_image:
            tokens, _ = self.image(data['image'])
            memory.append(tokens + self.patch_code + self.type_code[2])
            ignore.append(torch.zeros(b, tokens.shape[1], dtype=torch.bool, device=obs.device))
        memory, ignore = torch.cat(memory, dim=1), torch.cat(ignore, dim=1)
        # Ego tokens are always present, so no row is fully masked (no NaN softmax).
        fused = self.fusion(query, memory, memory_key_padding_mask=ignore)[:, 0]
        context = self.fuse(torch.cat((fused, ego[:, -1], self.position_mlp(location)), dim=-1))

        v = (obs[:, -1, :2] - obs[:, 0, :2]) / 4
        steps = torch.arange(1, 7, dtype=obs.dtype, device=obs.device)[None, :, None]
        reference = (v[:, None] * steps)[:, None] + self.anchors[None]                  # [B,1,6,2]
        adjustment = self.goals(context).reshape(b, 1, 2, 2)
        waypoint = reference[:, :, 2] + adjustment[:, :, 0]
        endpoint = reference[:, :, -1] + adjustment[:, :, 1]
        embedding = self.mode_embedding[None].expand(b, -1, -1)
        residual = self.path(torch.cat((context[:, None], embedding, waypoint, endpoint), dim=-1)).reshape(b, 1, 6, 2)
        t = steps[None] / 6
        control = 2 * waypoint - .5 * endpoint
        curve = 2 * t * (1 - t) * control[:, :, None] + t.square() * endpoint[:, :, None]
        curve = curve + residual * (16 * t * (1 - t) * (t - .5))
        return curve, self.scores(context)
