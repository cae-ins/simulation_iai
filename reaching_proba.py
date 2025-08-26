import torch
from utils import LabelledTensor



def compute_infra_per_population(
    infra_tensor: LabelledTensor,      # S_obs[i, t]
    population_tensor: LabelledTensor, # P[i]
) -> LabelledTensor:
    """
    Compute a per-capita infrastructure measure:

        out[i, t] = S_obs[i, t] / P[i]

    Notes
    -----
    - No scaling factor and no epsilon are added.
    - If P[i] == 0, PyTorch will produce inf/NaN (by design).
    """
    S = infra_tensor.tensor
    P = population_tensor.tensor.to(device=S.device, dtype=S.dtype)  # [I]

    I, T = S.shape
    out = S / P.view(I, 1).expand(I, T)  # broadcast P[i] across t

    return LabelledTensor(
        out,
        ["i", "t"],
        {
            "i": infra_tensor.index_to_label["i"],
            "t": infra_tensor.index_to_label["t"],
        },
    )


def compute_weight_sum(
    infra_tensor: LabelledTensor,      # S[k, t]
    distance_tensor: LabelledTensor,   # D[i, k, t]
    population_tensor: LabelledTensor,
    apply_inverse: bool = True) -> LabelledTensor:
    """
    Calcule la matrice

        SOMME[i, t] = ∑_k S[k, t] * D[i, k, t].

    Interprétation
    --------------
    Pour chaque origine i et type t, on agrège l'infrastructure S[·,t] sur
    toutes les destinations k en la pondérant par D[i,k,t] (impédance).

    Paramètres
    ----------
    infra_tensor : LabelledTensor [k, t]
        Offre/poids d'infrastructure par destination k et type t (S).
    distance_tensor : LabelledTensor [i, k, t]
        Tenseur (normalisé/pondéré) des distances/temps (D).
        (Si votre tenseur est noté D[i, j, t], considérez k ≡ j.)
    apply_inverse : bool, default=True
        Si True, retourne 1 / SOMME[i, t] (0 si SOMME==0) pour un usage direct
        comme dénominateur de normalisation.

    Retour
    ------
    LabelledTensor [i, t]
        SOMME[i, t] ou son inverse sécurisé.
    """
    infra_tensor = compute_infra_per_population(
                                infra_tensor=infra_tensor,
                                population_tensor=population_tensor)

    S = infra_tensor.tensor                  # [K, T]
    D = distance_tensor.tensor               # [I, K, T]

    # Pondération: diffuse S[k,t] sur i -> [1,K,T], multiplie D[i,k,t], puis somme sur k
    summed = (D * S.unsqueeze(0)).sum(dim=1)  # -> [I, T]

    if apply_inverse:
        summed = torch.where(summed != 0, 1.0 / summed, torch.zeros_like(summed))

    return LabelledTensor(
        summed,
        ["i", "t"],
        {
            "i": distance_tensor.index_to_label["i"],
            "t": distance_tensor.index_to_label["t"],
        },
    )


def compute_weight_product(infra_tensor: LabelledTensor,
                           distance_tensor: LabelledTensor,
                           population_tensor: LabelledTensor) -> LabelledTensor:
    """
    Calcule PRODUIT[i, j, t] = S[j, t] * D[i, j, t].

    Pondération appliquée colonne par colonne (j), pour chaque type t.

    Paramètres
    ----------
    S : LabelledTensor [j, t]
    D : LabelledTensor [i, j, t]

    Retour
    ------
    LabelledTensor [i, j, t]
    """
    infra_tensor = compute_infra_per_population(
                                infra_tensor=infra_tensor,
                                population_tensor=population_tensor)
    D = distance_tensor.tensor
    S_exp = infra_tensor.tensor.unsqueeze(0)  # [1, j, t]
    produit = D * S_exp

    return LabelledTensor(produit, distance_tensor.dim_labels,
                          distance_tensor.index_to_label)


def compute_proba_matrix(
    produit: LabelledTensor,          # PRODUIT[i, j, t]
    inv_sum: LabelledTensor,          # INV_SUM[i, t]
    infra_tensor: LabelledTensor,     # S[i, t]  (nb d'infra de type t dans la localité i)
    *,
    apply_local_capture: bool = True,   # active la règle locale
    zero_others_when_local: bool = True,# si True: met 0 sur j≠i (dur) ; sinon conserve les autres (souple)
    threshold: float = 1.0,             # déclenche la capture quand S[i,t] > threshold
    renormalize: bool = False,          # re-normalise ∑_j p[i,j,t]=1 (utile en mode "souple")
    eps: float = 1e-12,
) -> LabelledTensor:
    """
    Calcule PROBA[i, j, t] = PRODUIT[i, j, t] * INV_SUM[i, t]
    puis applique, si souhaité, une règle de "capture locale" basée sur S[i,t]:

      - Si apply_local_capture=True et S[i,t] > threshold :
          • zero_others_when_local=True  -> p[i,i,t]=1 et p[i,j≠i,t]=0  (capture dure)
          • zero_others_when_local=False -> p[i,i,t]=1 et on laisse p[i,j≠i,t] inchangés (capture souple)
            (optionnellement, renormalize=True pour forcer ∑_j p[i,j,t]=1)

    Paramètres
    ----------
    produit : LabelledTensor  # dims: ["i","j","t"], shape [I, J, T]
    inv_sum : LabelledTensor  # dims: ["i","t"],     shape [I, T]
    infra_tensor : LabelledTensor  # dims: ["i","t"], shape [I, T]
    apply_local_capture : bool, default=True
    zero_others_when_local : bool, default=True
    threshold : float, default=1.0
    renormalize : bool, default=False
    eps : float, default=1e-12

    Retour
    ------
    LabelledTensor  # dims: ["i","j","t"], shape [I, J, T]
    """
    # --- Probas de base : p = PRODUIT * INV_SUM (broadcast sur j)
    P = produit.tensor
    W = inv_sum.tensor.to(device=P.device, dtype=P.dtype)
    P = P * W.unsqueeze(1).expand_as(P)                  # [I,J,T]

    if apply_local_capture:
        S_it = infra_tensor.tensor.to(device=P.device, dtype=P.dtype)  # [I,T]
        I, J, T = P.shape

        # Masque des (i,t) à capturer : S[i,t] > threshold
        capture_it = (S_it > threshold).to(P.dtype)      # [I,T]

        # Masque diagonal (i==j), partagé sur T
        eye_ij = torch.eye(I, J, dtype=P.dtype, device=P.device).unsqueeze(2)  # [I,J,1]
        diag_mask = eye_ij * capture_it.unsqueeze(1)      # [I,J,T] = 1 sur (i,i,t) captés

        if zero_others_when_local:
            # Capture dure : on annule toute la ligne (i,·,t) pour les cas captés
            # puis on met 1 sur la diagonale correspondante
            P = P * (1.0 - capture_it.unsqueeze(1)) + diag_mask
        else:
            # Capture souple : on force seulement la diagonale à 1, on conserve les autres j
            # (la somme peut alors dépasser 1 ; utiliser renormalize=True si souhaité)
            P = P * (1.0 - diag_mask) + diag_mask

    if renormalize:
        # Normalise sur j pour chaque (i,t) afin que ∑_j p[i,j,t] = 1
        Z = P.sum(dim=1, keepdim=True)                  # [I,1,T]
        P = P / (Z + eps)

    return LabelledTensor(
        P,
        ["i", "j", "t"],
        {
            "i": produit.index_to_label["i"],
            "j": produit.index_to_label["j"],
            "t": produit.index_to_label["t"],
        },
    )


def compute_reaching_proba(distance_tensor: LabelledTensor,   # D[i, j, t]  (déjà transformé par la décroissance)
                           infra_tensor: LabelledTensor,      # S[·, t]     (utilisé pour la somme côté i ET le produit côté j)
                           population_tensor: LabelledTensor,
                           ) -> LabelledTensor:
    """
    Compose et retourne la matrice de probabilité P[i, j, t] telle que :

        P[i, j, t] = PRODUIT[i, j, t] * INV_SUM[j, t]

    où
      - PRODUIT[i, j, t] = S[j, t] * D[i, j, t]
      - INV_SUM[j, t]    = 1 / ∑_i S[i, t] * D[i, j, t]   (inverse sécurisé)

    Ce wrapper orchestre les trois étapes canoniques :
      (1) produit  = compute_weighted_product(S[·, t], D[i, j, t])        -> [i, j, t]
      (2) inv_sum  = compute_weighted_sum   (S[·, t], D[i, j, t])         -> [j, t]
      (3) proba    = compute_proba_matrix   (produit, inv_sum)            -> [i, j, t]

    Parameters
    ----------
    distance_tensor : LabelledTensor with dims ["i", "j", "t"]
        Tenseur des distances/impédances D (d ∈ [0,1]) déjà passé par une
        fonction de décroissance (plus petit = plus accessible).
    infra_tensor : LabelledTensor
        Tenseur S utilisé des deux côtés :
          • côté origine i pour la somme ∑_i S[i, t] * D[i, j, t]
          • côté destination j pour le produit  S[j, t] * D[i, j, t]

    Returns
    -------
    LabelledTensor with dims ["i", "j", "t"]
        Matrice des probabilités P[i, j, t].

    Notes
    -----
    - Par défaut, `compute_weighted_sum` renvoie l'**inverse sécurisé** de la
      somme (INV_SUM[j, t]) ce qui est exactement ce qu'attend `compute_proba_matrix`.
    """
    # 1) Produit pondéré par colonne j : PRODUIT[i, j, t] = S[j, t] * D[i, j, t]
    produit = compute_weight_product(
        infra_tensor=infra_tensor,
        distance_tensor=distance_tensor,
        population_tensor=population_tensor
    )

    # 2) Somme (puis inverse) le long de i : INV_SUM[j, t] = 1 / ∑_i S[i, t] * D[i, j, t]
    inv_sum = compute_weight_sum(
        infra_tensor=infra_tensor,
        distance_tensor=distance_tensor,   # apply_inverse=True par défaut
        population_tensor=population_tensor
    )

    # 3) Probabilités : P[i, j, t] = PRODUIT[i, j, t] * INV_SUM[j, t]
    proba_matrix = compute_proba_matrix(
        produit=produit,
        inv_sum=inv_sum,
        infra_tensor=infra_tensor
    )

    return proba_matrix
