"""Shared model factory. Encoder is the only architectural choice in team runs."""
import copy
import json
from pathlib import Path
import torch
from improved_forecast import SceneGoalForecast


def load_team_config(path=None):
    path=Path(path) if path else Path(__file__).with_name('team_config.json')
    config=json.loads(path.read_text())
    for key in ('hidden_size','recurrent_layers','transformer_layers','transformer_heads',
                'transformer_ffn','batch_size','epochs','patience','min_epochs'):
        if not isinstance(config[key],int) or config[key]<1:raise ValueError(f'{key} must be a positive integer')
    for key in ('learning_rate','min_learning_rate','gradient_clip','turn_weight'):
        if config[key]<=0:raise ValueError(f'{key} must be positive')
    if config['weight_decay']<0 or not 0<=config['encoder_dropout']<1:
        raise ValueError('Invalid weight decay or dropout')
    if config['min_learning_rate']>config['learning_rate']:
        raise ValueError('min_learning_rate must not exceed learning_rate')
    if not all(isinstance(config[key],bool) for key in ('spatial','social')):
        raise ValueError('spatial and social must be booleans')
    if config['hidden_size']%config['transformer_heads']:
        raise ValueError('hidden_size must be divisible by transformer_heads')
    return config


def build_team_model(encoder,config=None):
    """Return a one-trajectory model; shared initial weights identical for a seed."""
    config=copy.deepcopy(config or load_team_config())
    encoder=encoder.lower()
    if encoder not in ('gru','lstm','transformer'):raise ValueError('Unknown encoder')
    kwargs=dict(spatial=config['spatial'],social=config['social'],hidden_size=config['hidden_size'],
                transformer_heads=config['transformer_heads'],transformer_ffn=config['transformer_ffn'],
                encoder_dropout=config['encoder_dropout'])
    # Scope initialization so construction does not consume the training sampler RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seed'])
        reference=SceneGoalForecast(torch.zeros(1,6,2),encoder_type='gru',
                                   encoder_layers=config['recurrent_layers'],**kwargs)
        if encoder=='gru':model=reference
        else:
            torch.manual_seed(config['seed'])
            model=SceneGoalForecast(torch.zeros(1,6,2),encoder_type=encoder,
                  encoder_layers=config['transformer_layers'] if encoder=='transformer' else config['recurrent_layers'],**kwargs)
            state=model.state_dict()
            for key,value in reference.state_dict().items():
                if not key.startswith('motion.'):state[key]=value.clone()
            model.load_state_dict(state)
    return model


def checkpoint_config(encoder,config):
    return {**config,'encoder_type':encoder,
            'encoder_layers':config['transformer_layers'] if encoder=='transformer' else config['recurrent_layers'],
            'modes':1,'scene':False}
