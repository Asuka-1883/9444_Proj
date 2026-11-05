"""Drop-in inference adapter for batches returned by the original GCDataset."""
import torch
from improved_forecast import prepare_context,local_to_pixels
from train_improved import load_model


@torch.no_grad()
def predict_from_batch(model,batch):
    model.eval()
    device=next(model.parameters()).device
    # Explicit allow-list: no future labels, future frames or future identities.
    observed={key:batch[key].to(device) for key in ('obs','neighbors','neighbor_mask','origin')}
    data=prepare_context(observed)
    candidates,scores=model(data)
    relative_pixels=local_to_pixels(candidates,data['basis'])
    choice=scores.argmax(dim=-1)
    top=relative_pixels[torch.arange(len(choice),device=device),choice]
    return {'top1_relative':top/1920.,
            'top1_pixels':top+observed['origin'][:,None],
            'candidates_relative':relative_pixels/1920.,
            'candidates_pixels':relative_pixels+observed['origin'][:,None,None],
            'mode_logits':scores}
