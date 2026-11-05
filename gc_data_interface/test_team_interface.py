import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from team_model import build_team_model,load_team_config,checkpoint_config
from improved_forecast import prepare_context,forecast_loss,SceneGoalForecast
from train_improved import load_model
from motion_encoders import TransformerMotionEncoder
import test_improved_forecast as improved_tests


class TeamModelChecks(unittest.TestCase):
    def context(self):
        raw=improved_tests.ImprovedModelChecks().raw_batch()
        raw['neighbor_mask'][0]=False
        return prepare_context(raw)

    def test_non_motion_initial_weights_are_identical(self):
        models=[build_team_model(name) for name in ('gru','lstm','transformer')]
        reference=models[0].state_dict()
        for model in models[1:]:
            state=model.state_dict()
            for key,value in reference.items():
                if not key.startswith('motion.'):
                    self.assertTrue(torch.equal(value,state[key]),key)

    def test_all_encoders_train_and_reload_without_future_inputs(self):
        data=self.context();config=load_team_config()
        for encoder in ('gru','lstm','transformer'):
            with self.subTest(encoder=encoder), tempfile.TemporaryDirectory() as folder:
                model=build_team_model(encoder,config)
                optimizer=torch.optim.Adam(model.parameters(),lr=.001)
                # Two steps: decoder zero initialization initially blocks motion gradients.
                for _ in range(2):
                    model.train();optimizer.zero_grad()
                    paths,scores=model(data)
                    loss=forecast_loss(paths,scores,data['future'])
                    self.assertEqual(tuple(paths.shape),(3,1,6,2))
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward();optimizer.step()
                magnitude=sum(float(p.grad.abs().sum()) for p in model.motion.parameters() if p.grad is not None)
                self.assertGreater(magnitude,0)
                model.eval()
                with torch.no_grad():
                    expected,_=model(data)
                    actual,_=model({k:v for k,v in data.items() if k!='future'})
                    torch.testing.assert_close(expected,actual)
                path=Path(folder)/'model.pt'
                torch.save({'config':checkpoint_config(encoder,config),'anchors':model.anchors,
                            'background':None,'state_dict':model.state_dict()},path)
                restored=load_model(path)
                with torch.no_grad():torch.testing.assert_close(restored(data)[0],expected)

    def test_transformer_distinguishes_order_with_same_last_token(self):
        torch.manual_seed(83)
        model=TransformerMotionEncoder(dropout=0).eval()
        x=torch.randn(2,5,4)
        swapped=x.clone();swapped[:,[0,1]]=x[:,[1,0]]
        with torch.no_grad():
            before=model(x)[1];after=model(swapped)[1]
        self.assertGreater(float((before-after).abs().max()),1e-6)

    def test_old_gru_checkpoint_defaults_remain_supported(self):
        model=SceneGoalForecast(torch.zeros(1,6,2)).eval();data=self.context()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'old.pt'
            torch.save({'config':{'spatial':True},'anchors':model.anchors,
                        'background':None,'state_dict':model.state_dict()},path)
            restored=load_model(path)
            with torch.no_grad():torch.testing.assert_close(model(data)[0],restored(data)[0])

    def test_hidden_size_and_invalid_encoder_settings(self):
        config=load_team_config();config['hidden_size']=32
        for encoder in ('gru','lstm','transformer'):
            model=build_team_model(encoder,config)
            self.assertEqual(tuple(model(self.context())[0].shape),(3,1,6,2))
        with self.assertRaises(ValueError):build_team_model('wrong')
        with self.assertRaises(ValueError):TransformerMotionEncoder(hidden_size=63,heads=4)


class TeamCacheChecks(unittest.TestCase):
    def test_cache_reuse_and_portable_content_fingerprint(self):
        from test_gc_dataset import TestGCInterface
        from team_data import TeamData
        TestGCInterface.setUpClass()
        try:
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);copy_root=root/'copied'
                shutil.copytree(TestGCInterface.root,copy_root)
                a=TeamData(TestGCInterface.root,root/'cache')
                b=TeamData(copy_root,root/'cache')
                self.assertEqual(a.fingerprint,b.fingerprint)
                first=a.raw('train',8);second=b.raw('train',8)
                for key in first:torch.testing.assert_close(first[key],second[key])
                annotation=copy_root/'Annotation'/'000001.txt'
                values=np.loadtxt(annotation);values[0]+=1
                np.savetxt(annotation,values)
                changed=TeamData(copy_root,root/'cache')
                self.assertNotEqual(a.fingerprint,changed.fingerprint)
        finally:TestGCInterface.tearDownClass()


if __name__=='__main__':unittest.main(verbosity=2)
