"""Scene-conditioned waypoint/end-goal trajectory models for the fixed GC camera.

Uses training-only trajectory anchors and a background estimated from training
frames. This is a custom baseline inspired by goal-conditioned and modality-
clustering work, not a reproduction of PECNet, Y-net or PCCSNet.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from motion_encoders import make_motion_encoder


def prepare_context(raw):
    """Interface coordinates -> heading-aligned coordinates, in 100-pixel units."""
    obs = raw['obs'].float() * 19.2
    neighbors = raw['neighbors'].float() * 19.2
    velocity = (obs[:, -1, :2] - obs[:, 0, :2]) / 4
    speed = velocity.norm(dim=-1, keepdim=True)
    unit = velocity / speed.clamp_min(1e-6)
    fallback = torch.tensor([1., 0.], dtype=unit.dtype, device=unit.device).expand_as(unit)
    unit = torch.where(speed > .02, unit, fallback)
    perpendicular = torch.stack((-unit[:, 1], unit[:, 0]), dim=-1)
    basis = torch.stack((unit, perpendicular), dim=-1)  # [B,world_xy,local_xy]
    local_obs = torch.cat((torch.einsum('bti,bij->btj', obs[:, :, :2], basis),
                           torch.einsum('bti,bij->btj', obs[:, :, 2:], basis)), dim=-1)
    local_neighbors = torch.cat((torch.einsum('bmti,bij->bmtj', neighbors[:, :, :, :2], basis),
                                 torch.einsum('bmti,bij->bmtj', neighbors[:, :, :, 2:], basis)), dim=-1)
    data = {'obs': local_obs, 'neighbors': local_neighbors,
            'neighbor_mask': raw['neighbor_mask'].bool(), 'origin': raw['origin'].float(),
            'basis': basis, 'heading': unit, 'speed': speed}
    for key in ('pedestrian_id', 'index'):
        if key in raw:
            data[key] = raw[key]
    if 'future' in raw:
        data['future'] = torch.einsum('bti,bij->btj', raw['future'].float()*19.2, basis)
    return data


def local_to_pixels(path, basis):
    if path.ndim == 4:
        return torch.einsum('bkti,bji->bktj', path, basis)*100
    return torch.einsum('bti,bji->btj', path, basis)*100


def turn_mask(data):
    """Training/evaluation label only, never a model input."""
    past = (data['obs'][:, :, :2]*torch.arange(-2,3,dtype=torch.float32,device=data['obs'].device)[None,:,None]).sum(dim=1)/10
    future_velocity = (data['future'][:, -1]-data['future'][:, -4])/3
    cos = (past*future_velocity).sum(dim=-1)/(past.norm(dim=-1)*future_velocity.norm(dim=-1)).clamp_min(1e-8)
    angle = cos.clamp(-1,1).acos()*180/np.pi
    moving = (past.norm(dim=-1)>=.05)&(future_velocity.norm(dim=-1)>=.05)
    return moving&(angle>=30)


def make_background(root, train_end_frame, output):
    paths = sorted((Path(root)/'Frame').glob('*.jpg'))
    paths = [p for p in paths if p.stem.isdigit() and int(p.stem)<=train_end_frame]
    chosen = np.linspace(0, len(paths)-1, 25, dtype=int)
    frames = []
    for i in chosen:
        with Image.open(paths[i]) as image:
            frames.append(np.asarray(image.convert('RGB').resize((256,144)), dtype=np.uint8))
    median = np.median(np.stack(frames),axis=0).astype(np.uint8)
    Image.fromarray(median).save(output)
    return torch.from_numpy(median.copy()).permute(2,0,1).float()[None]/255


def cluster_anchors(data, count=20, seed=29, sample_size=40000, trim_quantile=1.):
    """K-means on TRAIN future residuals. Neither validation nor test is used."""
    generator = torch.Generator().manual_seed(seed)
    subset = torch.randperm(len(data['obs']),generator=generator)[:sample_size]
    obs, future = data['obs'][subset], data['future'][subset]
    base_velocity = (obs[:, -1, :2]-obs[:, 0, :2])/4
    steps = torch.arange(1,7,dtype=torch.float32)[None,:,None]
    residual = (future-base_velocity[:,None]*steps).reshape(-1,12)
    weights = torch.tensor([1,1,1,1,1.5,1.5,1,1,1,1,2,2],dtype=torch.float32).sqrt()
    x = residual*weights
    if not 0 < trim_quantile <= 1:
        raise ValueError('trim_quantile must lie in (0,1]')
    if trim_quantile < 1:
        magnitude=x.norm(dim=-1)
        threshold=torch.quantile(magnitude,trim_quantile)
        x=x[magnitude<=threshold]
        if len(x)<count:raise ValueError('Too few training points for anchors')
    centers = [x[torch.randint(len(x),(1,),generator=generator).item()]]
    nearest = torch.full((len(x),),float('inf'))
    for _ in range(1,count):
        nearest = torch.minimum(nearest, (x-centers[-1]).square().sum(dim=-1))
        index = torch.multinomial(nearest.clamp_min(1e-12),1,generator=generator).item()
        centers.append(x[index])
    centers = torch.stack(centers)
    for _ in range(30):
        distances = torch.cdist(x,centers).square()
        labels = distances.argmin(dim=-1)
        sums = torch.zeros_like(centers).index_add_(0,labels,x)
        numbers = torch.bincount(labels,minlength=count).float()
        centers = torch.where(numbers[:,None]>0,sums/numbers[:,None].clamp_min(1),centers)
    return (centers/weights).reshape(count,6,2)


class SceneGoalForecast(nn.Module):
    def __init__(self, anchors, background=None, spatial=True, social=True, *,
                 encoder_type='gru',hidden_size=64,encoder_layers=None,
                 transformer_heads=4,transformer_ffn=128,encoder_dropout=.1):
        super().__init__()
        self.register_buffer('anchors',anchors.clone())
        self.k = len(anchors)
        self.spatial, self.social, self.has_scene = spatial, social, background is not None
        self.encoder_type=encoder_type.lower()
        self.motion = make_motion_encoder(self.encoder_type,hidden_size,encoder_layers,
                                          transformer_heads,transformer_ffn,encoder_dropout)
        width=hidden_size
        if spatial:
            self.position = nn.Sequential(nn.Linear(17,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
            width+=64
        if social:
            self.neighbor = nn.Sequential(nn.Linear(4,32),nn.ReLU(),nn.Linear(32,32))
            self.query = nn.Linear(hidden_size,32)
            width+=32
        if self.has_scene:
            self.register_buffer('background',background.clone())
            self.scene_cnn = nn.Sequential(nn.Conv2d(3,16,5,stride=2,padding=2),nn.ReLU(),
                                           nn.Conv2d(16,32,3,stride=2,padding=1),nn.ReLU(),
                                           nn.Conv2d(32,32,3,padding=1),nn.ReLU())
            self.scene_projection = nn.Sequential(nn.Linear(32*9,64),nn.ReLU())
            offsets = torch.tensor([[x,y] for y in [-256,0,256] for x in [-256,0,256]],dtype=torch.float32)
            self.register_buffer('scene_offsets',offsets)
            width+=64
        self.fuse=nn.Sequential(nn.Linear(width,128),nn.ReLU(),nn.Dropout(.05),nn.Linear(128,128),nn.ReLU())
        self.goals=nn.Linear(128,self.k*4)
        self.mode_embedding=nn.Parameter(torch.randn(self.k,16)*.1)
        self.path=nn.Sequential(nn.Linear(128+16+4,128),nn.ReLU(),nn.Linear(128,12))
        self.scores=nn.Linear(128,self.k)
        nn.init.zeros_(self.goals.weight);nn.init.zeros_(self.goals.bias)
        nn.init.zeros_(self.path[-1].weight);nn.init.zeros_(self.path[-1].bias)

    def forward(self, data):
        # Future and all diagnostic labels are intentionally not accessed here.
        obs=data['obs'];b=len(obs)
        _,hidden=self.motion(obs)
        h=hidden[-1];parts=[h]
        if self.spatial:
            position=data['origin']/torch.tensor([1920.,1080.],device=obs.device)*2-1
            frequencies=torch.tensor([1.,2.,4.],device=obs.device)*np.pi
            angles=position[:,:,None]*frequencies
            features=torch.cat((position,angles.sin().flatten(1),angles.cos().flatten(1),data['heading'],data['speed']),dim=-1)
            parts.append(self.position(features))
        if self.social:
            mask=data['neighbor_mask'];encoded=self.neighbor(data['neighbors'])
            n=(encoded*mask[...,None]).sum(dim=2)/mask.sum(dim=2,keepdim=True).clamp_min(1)
            valid=mask.any(dim=-1)
            logits=(n*self.query(h)[:,None]).sum(dim=-1)/(32**.5)
            weights=logits.masked_fill(~valid,-1e9).softmax(dim=-1)*valid
            weights=weights/weights.sum(dim=-1,keepdim=True).clamp_min(1e-12)
            parts.append((n*weights[...,None]).sum(dim=1))
        if self.has_scene:
            feature_map=self.scene_cnn(self.background)
            locations=data['origin'][:,None]+self.scene_offsets[None]
            grid=locations/torch.tensor([1920.,1080.],device=obs.device)*2-1
            sampled=F.grid_sample(feature_map,grid.reshape(1,b*9,1,2),align_corners=False,padding_mode='zeros')
            sampled=sampled[0,:,:,0].reshape(32,b,9).permute(1,0,2).flatten(1)
            parts.append(self.scene_projection(sampled))
        context=self.fuse(torch.cat(parts,dim=-1))
        v=(obs[:,-1,:2]-obs[:,0,:2])/4
        base=v[:,None]*torch.arange(1,7,dtype=obs.dtype,device=obs.device)[None,:,None]
        reference=base[:,None]+self.anchors[None]
        adjustment=self.goals(context).reshape(b,self.k,2,2)
        waypoint=reference[:,:,2]+adjustment[:,:,0]
        endpoint=reference[:,:,-1]+adjustment[:,:,1]
        context_k=context[:,None].expand(-1,self.k,-1)
        embedding=self.mode_embedding[None].expand(b,-1,-1)
        residual=self.path(torch.cat((context_k,embedding,waypoint,endpoint),dim=-1)).reshape(b,self.k,6,2)
        t=torch.arange(1,7,dtype=obs.dtype,device=obs.device)[None,None,:,None]/6
        control=2*waypoint-.5*endpoint
        curve=2*t*(1-t)*control[:,:,None]+t.square()*endpoint[:,:,None]
        # Corrections vanish at t=0,0.5,1: the learned waypoint and endpoint remain exact.
        curve=curve+residual*(16*t*(1-t)*(t-.5))
        return curve,self.scores(context)


def forecast_loss(paths, scores, future, turning=None, turn_weight=1.):
    if paths.shape[1]==1:
        winner=torch.zeros(len(paths),dtype=torch.int64,device=paths.device)
    else:
        with torch.no_grad():
            distance=(paths-future[:,None]).norm(dim=-1)
            winner=(distance.mean(dim=-1)+.5*distance[:,:,-1]).argmin(dim=-1)
    chosen=paths[torch.arange(len(paths),device=paths.device),winner]
    position=F.smooth_l1_loss(chosen,future,reduction='none',beta=.1).mean(dim=(1,2))
    endpoint=F.smooth_l1_loss(chosen[:,-1],future[:,-1],reduction='none',beta=.1).mean(dim=1)
    dp=torch.diff(torch.cat((torch.zeros_like(chosen[:,:1]),chosen),dim=1),dim=1)
    dy=torch.diff(torch.cat((torch.zeros_like(future[:,:1]),future),dim=1),dim=1)
    velocity=F.smooth_l1_loss(dp,dy,reduction='none',beta=.1).mean(dim=(1,2))
    weight=1+(turn_weight-1)*turning.float() if turning is not None else torch.ones_like(position)
    regression=((position+.5*endpoint+.3*velocity)*weight).mean()
    classification=F.cross_entropy(scores,winner) if paths.shape[1]>1 else scores.sum()*0
    return regression+.1*classification
