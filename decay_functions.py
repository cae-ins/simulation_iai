import torch
import math
from utils import LabelledTensor
from typing import Literal, Dict, Callable, Any
import inspect


# ---------------------------------------------------------------------
# Utilitaire: normalisation linéaire pour garantir f(0)=1, f(1)=0
# Si g(d) est décroissante et finie en 0 et 1, alors
#   f(d) = (g(d) - g(1)) / (g(0) - g(1))
# ---------------------------------------------------------------------
def _normalize_minmax_inplace(t: torch.Tensor, g0: float, g1: float, eps: float = 1e-12):
    denom = (g0 - g1)
    if abs(denom) < eps:
        # évite une division par ~0 ; renvoie tout à 0
        t.zero_()
    else:
        t.add_(-g1).div_(denom)
    return t


# ---------------------------------------------------------------------
# 0) Exponentielle (variante proche de ton exemple)
#    g(d) = exp(-lambda * d^p)  -> normalisée vers f
#    p=1 : exponentielle standard ; p=2 : comme ton code "square()"
# ---------------------------------------------------------------------
def apply_exponential_decay(
    distance_tensor: LabelledTensor, lam: float = 0.5, p: float = 2.0
) -> LabelledTensor:
    """
    Applique f(d) = (exp(-lam * d^p) - exp(-lam * 1^p)) / (1 - exp(-lam))
    Assure f(0)=1, f(1)=0. Par défaut p=2 reproduit la structure de ton code.
    """
    x = distance_tensor.tensor.clone()
    x.clamp_(0.0, 1.0)

    if p != 1.0:
        x.pow_(p)  # d^p

    # g(d) = exp(-lam * d^p)
    x.mul_(-lam).exp_()

    g0 = math.exp(0.0)             # exp(0) = 1
    g1 = math.exp(-lam * 1.0)      # exp(-lam)

    _normalize_minmax_inplace(x, g0, g1)
    return LabelledTensor(x, distance_tensor.dim_labels, distance_tensor.index_to_label)


# ---------------------------------------------------------------------
# 1) Rationnelle / "power-like" sans singularité
#    g(d) = 1 / (1 + k d)^beta  -> normalisée
# ---------------------------------------------------------------------
def apply_rational_decay(
    distance_tensor: LabelledTensor, beta: float = 1.3, k: float = 3.0
) -> LabelledTensor:
    """
    f(d) = ( (1 + k d)^(-beta) - (1 + k)^(-beta) ) / ( 1 - (1 + k)^(-beta) )
    -> f(0)=1, f(1)=0 ; beta>0, k>=0
    """
    x = distance_tensor.tensor.clone()
    x.clamp_(0.0, 1.0)

    # g(d) = (1 + k d)^(-beta)
    x.mul_(k).add_(1.0).pow_(-beta)

    g0 = 1.0                       # (1 + 0)^(-beta)
    g1 = (1.0 + k) ** (-beta)

    _normalize_minmax_inplace(x, g0, g1)
    return LabelledTensor(x, distance_tensor.dim_labels, distance_tensor.index_to_label)


# ---------------------------------------------------------------------
# 2) Combinée exponentielle × rationnelle
#    g(d) = exp(-lam d) / (1 + k d)^beta  -> normalisée
# ---------------------------------------------------------------------
def apply_combined_decay(
    distance_tensor: LabelledTensor, lam: float = 1.0, beta: float = 1.0, k: float = 1.0
) -> LabelledTensor:
    """
    f(d) = ( g(d) - g(1) ) / ( g(0) - g(1) ),
    g(d) = exp(-lam d) / (1 + k d)^beta
    """
    x = distance_tensor.tensor.clone()
    x.clamp_(0.0, 1.0)

    # On calcule en log pour la stabilité, puis on exponentie
    # log g = -lam d - beta * log(1 + k d)
    logg = x.mul(-lam)  # -lam d
    tmp = distance_tensor.tensor.clone().clamp(0.0, 1.0).mul(k).add_(1.0).log_().mul_(beta)
    logg.add_(-tmp)

    x = logg.exp_()

    g0 = 1.0  # exp(0)/(1)^beta
    g1 = math.exp(-lam) / (1.0 + k) ** beta

    _normalize_minmax_inplace(x, g0, g1)
    return LabelledTensor(x, distance_tensor.dim_labels, distance_tensor.index_to_label)


# ---------------------------------------------------------------------
# 3) Logistique normalisée (coude en d0, raideur a)
#    g(d) = 1 / (1 + exp(a (d - d0))) ; puis f = normalisation min-max
# ---------------------------------------------------------------------
def apply_logistic_decay(
    distance_tensor: LabelledTensor, a: float = 10.0, d0: float = 0.5
) -> LabelledTensor:
    """
    f(d) = (L(d) - L(1)) / (L(0) - L(1)),
    L(d) = 1 / (1 + exp(a (d - d0)))
    -> f(0)=1, f(1)=0
    """
    x = distance_tensor.tensor.clone()
    x.clamp_(0.0, 1.0)

    # L(d)
    x.add_(-d0).mul_(a).exp_().add_(1.0).reciprocal_()

    L0 = 1.0 / (1.0 + math.exp(a * (0.0 - d0)))
    L1 = 1.0 / (1.0 + math.exp(a * (1.0 - d0)))

    _normalize_minmax_inplace(x, L0, L1)
    return LabelledTensor(x, distance_tensor.dim_labels, distance_tensor.index_to_label)


# ---------------------------------------------------------------------
# 4) Cutoff polynomial (déjà f(0)=1, f(1)=0)
#    f(d) = (1 - d)^gamma  (gamma>0)
# ---------------------------------------------------------------------
def apply_cutoff_polynomial_decay(
    distance_tensor: LabelledTensor, gamma: float = 2.0
) -> LabelledTensor:
    """
    f(d) = (1 - d)^gamma, gamma > 0
    -> satisfait déjà f(0)=1, f(1)=0 sur d∈[0,1]
    """
    x = distance_tensor.tensor.clone()
    x.clamp_(0.0, 1.0)
    x.neg_().add_(1.0).pow_(gamma)
    return LabelledTensor(x, distance_tensor.dim_labels, distance_tensor.index_to_label)


# ---------------------------------------------------------------------
# 5) “Mi-portée” (optionnel) : si tu veux fixer f(d50)=0.5
#    Exemple pour l’exponentielle standard (p=1) :
#       lam = ln(2) / d50
#    Exemple pour la rationnelle beta=1 :
#       k = (1/0.5) - 1  allongé par d50 -> k = (1/0.5 - 1)/d50
#   (à utiliser pour calibrer les paramètres passés aux fonctions ci-dessus)
# ---------------------------------------------------------------------

# Types of supported decays
DecayKind = Literal["exponential", "rational", "combined", "logistic", "cutoff"]

# Map a user-facing name to the actual implementation
_DECAY_IMPLS: Dict[DecayKind, Callable[..., "LabelledTensor"]] = {
    "exponential": apply_exponential_decay,          # params: lam, p
    "rational":    apply_rational_decay,             # params: beta, k
    "combined":    apply_combined_decay,             # params: lam, beta, k
    "logistic":    apply_logistic_decay,             # params: a, d0
    "cutoff":      apply_cutoff_polynomial_decay,    # params: gamma
}


def _filter_kwargs_for(func: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only kwargs accepted by `func` to avoid unexpected-kwargs errors.
    """
    sig = inspect.signature(func)
    accepted = {name for name, p in sig.parameters.items()
                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    return {k: v for k, v in kwargs.items() if k in accepted}


def apply_decay(distance_tensor: "LabelledTensor",
                kind: DecayKind = "exponential",
                **params: Any) -> "LabelledTensor":
    """
    Apply a chosen decay to `distance_tensor` (with distances normalized in [0,1]).

    Parameters
    ----------
    distance_tensor : LabelledTensor [i, j, t]
        Tensor of normalized distances (d ∈ [0,1]).
    kind : {'exponential','rational','combined','logistic','cutoff'}
        Which decay family to use.
    **params :
        Family-specific parameters:
          - exponential: lam=0.5, p=2.0
          - rational:    beta=1.3, k=3.0
          - combined:    lam=1.0, beta=1.0, k=1.0
          - logistic:    a=10.0, d0=0.5
          - cutoff:      gamma=2.0

    Returns
    -------
    LabelledTensor [i, j, t]
        Transformed tensor with f(0)=1 and f(1)=0 (cutoff already satisfies it).

    Raises
    ------
    ValueError
        If `kind` is unknown.
    """
    if kind not in _DECAY_IMPLS:
        raise ValueError(
            f"Unknown decay kind='{kind}'. "
            f"Choose one of {list(_DECAY_IMPLS.keys())}."
        )

    impl = _DECAY_IMPLS[kind]
    clean_params = _filter_kwargs_for(impl, params)
    return impl(distance_tensor, **clean_params)
