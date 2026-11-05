import unittest
import torch
from improved_forecast import prepare_context, local_to_pixels, SceneGoalForecast, forecast_loss


class ImprovedModelChecks(unittest.TestCase):
    def raw_batch(self):
        torch.manual_seed(21)
        b=3
        xy=torch.randn(b,5,2)*.02
        xy=xy-xy[:,-1:]
        delta=torch.zeros_like(xy);delta[:,1:]=torch.diff(xy,dim=1)
        return {'obs':torch.cat((xy,delta),dim=-1),
                'neighbors':torch.randn(b,16,5,4)*.03,
                'neighbor_mask':torch.ones(b,16,5,dtype=torch.bool),
                'future':torch.randn(b,6,2)*.1,
                'origin':torch.tensor([[100.,100.],[900.,500.],[1800.,900.]]),
                'index':torch.arange(b),'pedestrian_id':torch.arange(b)}

    def test_heading_coordinate_round_trip(self):
        raw=self.raw_batch();data=prepare_context(raw)
        world=local_to_pixels(data['future'],data['basis'])
        torch.testing.assert_close(world,raw['future']*1920,atol=1e-3,rtol=1e-5)
        multiple=data['future'][:,None].expand(-1,20,-1,-1)
        restored=local_to_pixels(multiple,data['basis'])
        torch.testing.assert_close(restored[:,0],raw['future']*1920,atol=1e-3,rtol=1e-5)

    def test_future_label_not_required_and_does_not_affect_prediction(self):
        raw=self.raw_batch()
        del raw['index']
        data=prepare_context(raw)
        model=SceneGoalForecast(torch.randn(20,6,2)*.1,torch.rand(1,3,144,256)).eval()
        with torch.no_grad():
            expected,_=model(data)
            no_future={k:v for k,v in data.items() if k!='future'}
            actual,_=model(no_future)
        self.assertTrue(torch.equal(expected,actual))

    def test_midpoint_endpoint_initialization_and_finite_backward(self):
        data=prepare_context(self.raw_batch());data['neighbor_mask'][0]=False
        for modes in [1,20]:
            anchors=torch.randn(modes,6,2)*.1
            model=SceneGoalForecast(anchors)
            prediction,scores=model(data)
            v=(data['obs'][:,-1,:2]-data['obs'][:,0,:2])/4
            torch.testing.assert_close(prediction[:,:,2],v[:,None]*3+anchors[None,:,2])
            torch.testing.assert_close(prediction[:,:,-1],v[:,None]*6+anchors[None,:,-1])
            loss=forecast_loss(prediction,scores,data['future'],torch.tensor([True,False,True]),2.)
            loss.backward()
            self.assertTrue(torch.isfinite(prediction).all())
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__=='__main__':unittest.main(verbosity=2)
