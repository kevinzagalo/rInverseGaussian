"""Mode-parameterized inverse Gaussian models and response-time mixtures."""

from .rInvGauss import rInvGauss, RTInvGauss
from .rInvGaussMixture import rInvGaussMixture
from .RTInvGaussMixture import RTInvGaussMixture

__all__ = ["rInvGauss", "RTInvGauss", "rInvGaussMixture", "RTInvGaussMixture"]
__version__ = "0.2.0"
