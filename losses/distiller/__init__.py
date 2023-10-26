from .dist import DIST
from .kd import KD, KDDIST
from .dkd import DKD, DKDDIST
from .gdkd import GDKD, GDKDDIST

def get_criterion_distiller(name, *args, **kwargs):
    return {
        'kd': KD,
        'kd-dist': KDDIST,
        'dkd': DKD,
        'dkd-dist': DKDDIST,
        'gdkd': GDKD,
        'gdkd-dist': GDKDDIST,
        'dist': DIST
    }[name](*args, **kwargs)
