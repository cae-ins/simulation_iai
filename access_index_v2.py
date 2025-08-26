import torch
import math
import gc
import warnings
import logging
import polars as pl
from config import Config
from utils import LabelledTensor
from decay_functions import apply_decay
from reaching_proba import compute_reaching_proba

warnings.filterwarnings("ignore")

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_distance_tensor(distance_tensor: LabelledTensor, 
                            infra_tensor: LabelledTensor, 
                            device="cpu") -> LabelledTensor:
    """
    Construit un tenseur D[i, j, t] des distances inter localités [i, j], ne conservant 
    les distances que vers les localités j possédant une infrastructure de type t.

    Paramètres
    ----------
    distance_tensor : LabelledTensor [i, j]
        Distances entre localités i et j.
    infra_tensor : LabelledTensor [j, t]
        Disponibilité des infrastructures à chaque localité j.
    device : str
        Périphérique ('cpu' ou 'cuda').

    Retour
    ------
    LabelledTensor [i, j, t]
        Distances pondérées.
    """
    dist = distance_tensor.tensor.to(device)
    infra_mask = (infra_tensor.tensor.to(device) >= 0).float()   ################## JE VIENS DE MODIFIER >= 0

    dist_3d = dist.unsqueeze(-1)          # [i, j, 1]
    mask_3d = infra_mask.unsqueeze(0)     # [1, j, t]

    D_tensor = dist_3d * mask_3d          # [i, j, t]

    return LabelledTensor(D_tensor, ["i", "j", "t"], {
        "i": distance_tensor.index_to_label["i"],
        "j": distance_tensor.index_to_label["j"],
        "t": infra_tensor.index_to_label["t"]
    })


def compute_max_distance(distance_tensor: LabelledTensor) -> LabelledTensor:
    """
    Calcule Dmax[t] = max_ij D[i, j, t], soit la distance maximale observée par type.

    Paramètres
    ----------
    distance_tensor : LabelledTensor [i, j, t]

    Retour
    ------
    LabelledTensor [t]
    """
    distance_max = distance_tensor.tensor.amax(dim=(0, 1))
    return LabelledTensor(distance_max, ["t"], 
                          {"t": distance_tensor.index_to_label["t"]})



def normalize_distance_tensor(distance_tensor: LabelledTensor, 
                              distance_max: LabelledTensor) -> LabelledTensor:
    """
    Normalise chaque distance_tensor[i, j, t] par la distance maximale distance_max[t].

    Paramètres
    ----------
    distance_tensor : LabelledTensor [i, j, t]
    distance_max : LabelledTensor [t]

    Retour
    ------
    LabelledTensor [i, j, t]
    """
    D_norm = distance_tensor.tensor / distance_max.tensor.view(1, 1, -1)
    return LabelledTensor(D_norm, distance_tensor.dim_labels, 
                          distance_tensor.index_to_label)




def restrict_infra(distance_tensor: LabelledTensor, 
                   infra_tensor: LabelledTensor, 
                   restrict: bool = True) -> LabelledTensor:
    """
    Met à zéro les distances D[i, j, t] là où l'infrastructure [j, t] est absente.

    Paramètres
    ----------
    distance_tensor : LabelledTensor [i, j, t]
    infra_tensor : LabelledTensor [j, t]
    restrict : bool
        Si False, retourne D inchangé.

    Retour
    ------
    LabelledTensor [i, j, t]
    """
    if not restrict:
        return distance_tensor

    mask = (infra_tensor.tensor > 0).float().unsqueeze(0)  # [1, j, t]
    D_masked = distance_tensor.tensor * mask

    return LabelledTensor(D_masked, distance_tensor.dim_labels, 
                          distance_tensor.index_to_label)


def compute_weighted_sum(
    population_tensor: LabelledTensor,  # P[i]         dims ["i"]
    distance_tensor:  LabelledTensor,   # D[i,j,t]     dims ["i","j","t"]
    apply_inverse: bool = False,
) -> LabelledTensor:
    """
    Compute :

        SOMME[i, j, t] = P[i] + ∑_{k≠i} P[k] * D[k, j, t]

    Efficient formulation (only i, j, t dims):
        Let S_total[j, t] = ∑_k P[k] * D[k, j, t]
        Then:
            SOMME[i, j, t] = P[i] + S_total[j, t] - P[i] * D[i, j, t]

    This avoids any explicit [i,j,k,t] tensor. We only form:
      - S_total[j,t]  (via einsum), and
      - the final SOMME[i,j,t] using cheap broadcasting.

    Parameters
    ----------
    population_tensor : LabelledTensor, dims ["i"], shape [I]
        Vector of weights P (same index set as the first axis of D).
    distance_tensor : LabelledTensor, dims ["i","j","t"], shape [I,J,T]
        Distance/impedance tensor D.
    apply_inverse : bool, default False
        If True, return 1 / SOMME[i,j,t] (0 where SOMME==0).

    Returns
    -------
    LabelledTensor, dims ["i","j","t"], shape [I,J,T]
        The tensor SOMME (or its safe inverse).
    """
    P = population_tensor.tensor           # [I]
    D = distance_tensor.tensor             # [I,J,T]
    I, J, T = D.shape

    # 1) S_total[j, t] = ∑_k P[k] * D[k, j, t]
    S_total_jt = torch.einsum('i,ijt->jt', P, D)  # [J,T]

    # 2) SOMME[i, j, t] = P[i] + S_total[j, t] - P[i] * D[i, j, t]
    P_i   = P.view(I, 1, 1)                         # [I,1,1]
    SOMME = P_i + S_total_jt.view(1, J, T) - (P_i * D)   # [I,J,T]

    if apply_inverse:
        SOMME = torch.where(SOMME != 0, 1.0 / SOMME, torch.zeros_like(SOMME))

    return LabelledTensor(
        SOMME,
        ["i", "j", "t"],
        {
            "i": distance_tensor.index_to_label["i"],
            "j": distance_tensor.index_to_label["j"],
            "t": distance_tensor.index_to_label["t"],
        },
    )


def compute_weighted_product(
    infra_tensor: LabelledTensor,      # S[j, t]
) -> LabelledTensor:
    """
    Return PRODUIT[j, t] = S[j, t].

    This version reflects the new definition where the "product" no longer
    depends on i (nor on D). We keep the same function signature for compatibility.

    Parameters
    ----------
    infra_tensor : LabelledTensor with dims ["j", "t"]
        The destination weights S[j, t].
    distance_tensor : LabelledTensor with dims ["i", "j", "t"]
        Unused in the computation. Kept for API symmetry and (optionally)
        for propagating j/t labels if needed.

    Returns
    -------
    LabelledTensor with dims ["j", "t"]
        PRODUIT[j, t] = S[j, t].

    Notes
    -----
    • If a downstream function expects a [i, j, t] tensor, broadcast on the i dimension:
          I = distance_tensor.tensor.shape[0]
          produit_ijt = produit_jt.tensor.unsqueeze(0).expand(I, -1, -1)
      (No extra memory is allocated by expand().)
    """
    # Directly return S[j, t]
    S_jt = infra_tensor.tensor.clone()

    # Prefer labels from S; fall back to D’s labels if S doesn’t carry them
    labels_j = infra_tensor.index_to_label.get("j", infra_tensor.index_to_label.get("i"))
    labels_t = infra_tensor.index_to_label.get("t", infra_tensor.index_to_label.get("t"))

    return LabelledTensor(
        S_jt,
        ["j", "t"],
        {"j": labels_j, "t": labels_t},
    )


def compute_matrix_product(
    produit: LabelledTensor,   # PRODUIT[j, t]
    inv_sum: LabelledTensor,   # INV_SUM[i, j, t]
) -> LabelledTensor:
    """
    Compute ACCESS[i, t] = ∑_j PRODUIT[j, t] * INV_SUM[i, j, t]

    Shapes
    ------
    produit.tensor  : [J, T]    # weights per destination j and type t
    inv_sum.tensor  : [I, J, T] # inverse weighted sums per (i, j, t)
    result          : [I, T]

    Notes
    -----
    • Uses a single einsum ('ijt,jt->it') to avoid building a full [I, J, T]
      intermediate before summation over j. This is both fast and memory-light.
    """
    # Efficient contraction over j: (i,j,t) × (j,t) -> (i,t)
    access_it = torch.einsum('ijt,jt->it', inv_sum.tensor, produit.tensor)

    return LabelledTensor(
        access_it,
        ["i", "t"],
        {
            "i": inv_sum.index_to_label["i"],
            "t": inv_sum.index_to_label["t"],
        },
    )


def compute_access_formula(distance_tensor: LabelledTensor,
                           population_tensor: LabelledTensor,
                           infra_tensor: LabelledTensor,
                           device: str = "cpu") -> LabelledTensor:
    """
    Calcule l'indice final d'accessibilité pour chaque zone et chaque type d'infrastructure.

    Ce calcul suit les étapes suivantes :
    1. Génère D[i, j, t] à partir des distances et de la présence d'infrastructure.
    2. Normalise D par type t.
    3. Applique une fonction de décroissance exponentielle.
    4. Restreint les distances aux seules zones équipées.
    5. Calcule la probabilité de parcourir la distance D[i, j, t]
    6 Calcule un dénominateur : somme pondérée des proba de parcours selon l'offre.
    7. Calcule un numérateur : produit pondéré des proba de parcours selon l'offre.
    8. Calcule l'indice final ACCESS[i, t].

    Paramètres
    ----------
    distance_tensor : LabelledTensor [i, j]
        Matrice des distances entre localités.
    population_tensor : LabelledTensor [i]
        Population par localité.
    infra_tensor : LabelledTensor [j, t]
        Présence d'infrastructures par localité et type.

    Retour
    ------
    LabelledTensor [i, t]
        Indice d'ccessibilité final.
    """
    # Étape 0 — Récupération du max de D[i, j]
    logger.info("Récupération de la distance maximum...")
    distance_max = distance_tensor.tensor.max()
    decay_drop = Config.DROP_TIME * 1000 / distance_max
    # decay_speed = torch.log(9)/torch.min([decay_drop, 1 - decay_drop])

    # Étape 1 — Construction de D[i, j, t]
    logger.info("Construction du tenseur de distance pondérée D[i, j, t]...")
    distance_tensor = compute_distance_tensor(distance_tensor=distance_tensor,
                                              infra_tensor=infra_tensor,
                                              device=device)

    # Étape 2 — Distance maximale par type
    logger.info("Calcul des distances maximales par type d'infrastructure...")
    distance_max = compute_max_distance(distance_tensor)

    # Étape 3 — Normalisation par Dmax[t]
    logger.info("Normalisation des distances...")
    distance_tensor = normalize_distance_tensor(distance_tensor=distance_tensor,
                                                distance_max=distance_max)

    # Étape 4 — Application de la décroissance exponentielle
    logger.info("Application de la fonction de décroissance exponentielle...")
    distance_tensor = apply_decay(distance_tensor, kind="logistic", a=5, d0=decay_drop)


    # Étape 5 — Calcul de la probabilité de parcours des distances selon l'offre
    logger.info("Calcul de la probabilité de parcours des distances selon l'offre...")
    proba_tensor = compute_reaching_proba(distance_tensor=distance_tensor,
                                          infra_tensor=infra_tensor,
                                          population_tensor=population_tensor)


    # Étape 6 — Calcul du dénominateur : somme pondérée (avec inversion)
    logger.info("Calcul du dénominateur (somme pondérée inversée)...")
    denominator = compute_weighted_sum(
        population_tensor=population_tensor,
        distance_tensor=proba_tensor,
        apply_inverse=True
    )

    # Étape 7 — Calcul du numérateur : produit pondéré (avec carré de D)
    logger.info("Calcul du numérateur (produit pondéré)...")
    numerator = compute_weighted_product(
        infra_tensor=infra_tensor
    )

    # Libération explicite de la mémoire des tenseurs intermédiaires
    del distance_tensor
    del distance_max
    gc.collect()  # Nettoyage mémoire Python

    # Étape 8 — Division pour obtenir l'indice d'accessibilité
    logger.info("Calcul final de l'indice d'accessibilité...")
    index = compute_matrix_product(numerator, denominator)

    return index


def compute_access_index(distance_tensor: LabelledTensor,
                         population_tensor: LabelledTensor,
                         infra_tensor: LabelledTensor,
                         device: str = "cpu",
                         apply_minmax: bool = False) -> LabelledTensor:
    """
    Calcule un indice d'accessibilité relatif ACCESS / ACCESS_TH, avec normalisation min-max optionnelle.

    Ajoute une étape pour évaluer si chaque ACCESS_TH[i, t] ≥ ACCESS[i, t].

    Paramètres
    ----------
    distance_tensor : LabelledTensor [i, j, t]
        Tenseur des distances pondérées.
    population_tensor : LabelledTensor [i]
        Population par localisation.
    infra_tensor : LabelledTensor [j, t]
        Offre réelle d'infrastructure.
    infra_tensor_th : LabelledTensor [j, t]
        Offre théorique d'infrastructure.
    device : str
        Appareil de calcul ("cpu" ou "cuda").
    apply_minmax : bool
        Si True, applique une normalisation min-max par colonne t.

    Retour
    ------
    LabelledTensor [i, t]
        Indice d'accessibilité normalisé (ou brut).
    """

    # Étape 1 : accessibilité réelle
    access_real = compute_access_formula(
        distance_tensor=distance_tensor,
        population_tensor=population_tensor,
        infra_tensor=infra_tensor,
        device=device
    )

    # Étape 2 : Transformation en tenseur
    access_tensor = access_real.tensor

    # Étape 3 : normalisation min-max par colonne t
    if apply_minmax:
        min_vals, _ = access_tensor.min(dim=0, keepdim=True)  # [1, t]
        max_vals, _ = access_tensor.max(dim=0, keepdim=True)  # [1, t]
        range_vals = (max_vals - min_vals).clamp(min=1e-8)
        normalized = (access_tensor - min_vals) / range_vals
    else:
        normalized = access_tensor

    return LabelledTensor(
        normalized,
        dim_labels=access_real.dim_labels,
        index_to_label=access_real.index_to_label
    )