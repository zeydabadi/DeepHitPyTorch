import numpy as np
import torch
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models.data import pair_rank_mat
from lifelines import KaplanMeierFitter

def set_seed(seed):
    import torch, numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class DiscreteTimeLabelTransform(LabTransDiscreteTime):
    def transform(self, durations, events):
        d_disc, is_event = super().transform(durations, events>0)
        events_disc = np.where(is_event, events, 0).astype('int64')
        return d_disc, events_disc

def compute_rank_matrix(times, events, device):
    t_np = times.cpu().numpy()
    e_np = events.cpu().numpy()
    rank = pair_rank_mat(t_np, e_np)
    return torch.tensor(rank, device=device, dtype=torch.int64)

def compute_brier_score(cif_vals, kmf, times, events, event_id, horizon):
    n = len(times)
    resid = np.zeros(n)
    for i, (t_i, e_i, p_i) in enumerate(zip(times, events, cif_vals)):
        if t_i > horizon:
            w = kmf.predict(horizon)
            resid[i] = p_i**2 / w
        else:
            w = kmf.predict(t_i)
            if e_i == event_id:
                resid[i] = (1-p_i)**2 / w
            elif e_i != 0:
                resid[i] = p_i**2 / w
    return resid.mean()
