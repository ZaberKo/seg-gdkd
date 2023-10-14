from .dist_kd import DIST
from .kd import KD, KDDIST
from .dkd import DKD, DKDDIST
from .gdkd import GDKD, GDKDDIST

def get_criterion_distiller(name, *args, **kwargs):
    return dict(
        kd=KD,
        kd_dist=KDDIST,
        dkd=DKD,
        dkd_dist=DKDDIST,
        gdkd=GDKD,
        gdkd_dist=GDKDDIST,
        dist=DIST
    )[name](*args, **kwargs)
