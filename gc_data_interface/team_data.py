"""Versioned shared trajectory cache, with source/config fingerprints."""
from dataclasses import asdict
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from gc_dataset import GCStore
from improved_forecast import prepare_context


CACHE_SCHEMA=2


def stable_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class TeamData:
    def __init__(self,root,cache_dir):
        self.store=GCStore(root)
        self.cache_dir=Path(cache_dir);self.cache_dir.mkdir(parents=True,exist_ok=True)
        # Semantic fingerprint: independent of local folder names, extraction
        # timestamps and line endings, so teammates can compare on different PCs.
        source=hashlib.sha256()
        for agent,(frames,xy) in sorted(self.store.tracks.items()):
            source.update(np.asarray([agent,len(frames)],dtype='<i8').tobytes())
            source.update(np.asarray(frames,dtype='<i8').tobytes())
            source.update(np.asarray(xy,dtype='<f4').tobytes())
        source.update(np.asarray(self.store.frame_ids,dtype='<i8').tobytes())
        source.update(np.asarray([self.store.image_width,self.store.image_height],dtype='<i8').tobytes())
        self.protocol={'schema':CACHE_SCHEMA,'config':asdict(self.store.config),
                       'split_ranges':self.store.split_ranges,'counts':self.store.report()['sample_counts'],
                       'source_fingerprint':source.hexdigest(),
                       'fingerprint_kind':'canonical_numeric_tracks_frames_and_image_dimensions',
                       'subset_seed':310}
        self.fingerprint=stable_hash(self.protocol)
        manifest=self.cache_dir/f'manifest_{self.fingerprint[:16]}.json'
        manifest.write_text(json.dumps(self.protocol,ensure_ascii=False,indent=2)+'\n')

    def raw(self,split,limit=None):
        from gc_dataset import GCDataset
        dataset=GCDataset(self.store,split,include_image=False)
        n=min(len(dataset),limit if limit is not None else len(dataset))
        if n<1:raise ValueError(f'{split} has no usable samples')
        indices=np.random.default_rng(310).permutation(len(dataset))[:n].astype(np.int64)
        path=self.cache_dir/f'{split}_{self.fingerprint[:16]}_{n}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as saved:
                if str(saved['fingerprint'].item())!=self.fingerprint:raise ValueError('Cache fingerprint mismatch')
                if not np.array_equal(saved['index'],indices):raise ValueError('Cache subset mismatch')
                return {k:torch.from_numpy(saved[k].copy()) for k in saved.files if k!='fingerprint'}
        sample=dataset[int(indices[0])]
        fields=['obs','neighbors','neighbor_mask','future','origin','pedestrian_id','anchor_frame',
                'obs_frame_ids','future_frame_ids']
        arrays={k:np.empty((n,)+np.asarray(sample[k]).shape,dtype=np.asarray(sample[k]).dtype) for k in fields}
        arrays['index']=indices
        offset=0
        for batch in DataLoader(Subset(dataset,indices.tolist()),batch_size=256,shuffle=False,num_workers=0):
            size=len(batch['obs'])
            for key in fields:arrays[key][offset:offset+size]=batch[key].numpy()
            offset+=size
            if offset%10240==0 or offset==n:print(f'CACHE {split}: {offset}/{n}',flush=True)
        # Atomic replacement allows interrupted preprocessing to be retried safely.
        with tempfile.NamedTemporaryFile(dir=self.cache_dir,suffix='.partial.npz',delete=False) as handle:
            temporary=Path(handle.name)
        try:
            np.savez(temporary,fingerprint=self.fingerprint,**arrays)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return {k:torch.from_numpy(v) for k,v in arrays.items()}

    def context(self,split,limit=None):
        return prepare_context(self.raw(split,limit))


def move_batch(data,indices,device):
    return {k:v[indices].to(device) for k,v in data.items()}
