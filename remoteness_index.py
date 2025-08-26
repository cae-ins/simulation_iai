import torch
import warnings
import polars as pl
from utils import LabelledTensor
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_min_distance(distance_matrix: LabelledTensor,
                         infrastructure_matrix: LabelledTensor) -> LabelledTensor:
    """
    Calcule la distance minimale depuis chaque localité i vers une infrastructure de type t.

    Y[i, t] = min_j { distance[i, j] | infrastructure[j, t] > 0 }

    Paramètres :
    ------------
    distance_matrix : LabelledTensor [i, j]
        Distances entre localités i et j.
    infrastructure_matrix : LabelledTensor [j, t]
        Présence d'infrastructure t en j.

    Retour :
    --------
    LabelledTensor [i, t]
        Distances minimales vers chaque type d'infrastructure.
    """
    D = distance_matrix.tensor
    I = infrastructure_matrix.tensor.bool()

    i_size, j_size = D.shape
    t_size = I.shape[1]

    result_per_type = []

    for t in range(t_size):
        has_infra = I[:, t]
        if has_infra.any():
            D_filtered = D[:, has_infra]
            min_dist = D_filtered.min(dim=1).values
        else:
            min_dist = torch.full((i_size,), float("inf"), device=D.device)

        result_per_type.append(min_dist)

    result_tensor = torch.stack(result_per_type, dim=1)  # [i, t]

    return LabelledTensor(result_tensor, ["i", "t"], {
        "i": distance_matrix.index_to_label["i"],
        "t": infrastructure_matrix.index_to_label["t"]
    })

def compute_mean_distance(distance_tensor: LabelledTensor,
                          population_tensor: LabelledTensor,
                          exclude_zero_values: bool = False) -> LabelledTensor:
    
    """
    Calcule la moyenne pondérée des distances minimales par type d'infrastructure.

    Paramètres
    ----------
    distance_tensor : LabelledTensor [i, t]
        Distances minimales vers infrastructures.
    population_tensor : LabelledTensor [i]
        Population de chaque localité.
    exclude_zero_values : bool
        Si True, ignore les distances nulles dans la moyenne.

    Retour
    ------
    LabelledTensor [t]
        Moyennes pondérées des distances.
    """
    D = distance_tensor.tensor
    P = population_tensor.tensor.view(-1, 1)

    if exclude_zero_values:
        mask = D != 0
        weighted_sum = D * P * mask
        weights = P * mask
    else:
        weighted_sum = D * P
        weights = P.expand_as(D)

    mean_dist = weighted_sum.sum(dim=0) / weights.sum(dim=0).clamp(min=1e-8)

    return LabelledTensor(mean_dist, ["t"], {
        "t": distance_tensor.index_to_label["t"]
    })


def normalize_distance(
    distance_tensor: LabelledTensor,
    mean_distance_tensor: LabelledTensor
) -> LabelledTensor:
    """
    Normalise chaque distance par la moyenne agrégée pour le type d'infrastructure correspondant.

    Paramètres :
    ------------
    distance_tensor : LabelledTensor [i, t]
        Distances minimales.
    mean_distance_tensor : LabelledTensor [t]
        Moyennes pondérées des distances.

    Retour :
    --------
    LabelledTensor [i, t]
        Distances normalisées.
    """
    normalized = distance_tensor.tensor / mean_distance_tensor.tensor.unsqueeze(0)
    return LabelledTensor(normalized, distance_tensor.dim_labels, distance_tensor.index_to_label)


def clip_max_distance(
    distance_tensor: LabelledTensor,
    max_value: float = 3.0
) -> LabelledTensor:
    """
    Tronque les valeurs de distance normalisée trop élevées.

    Paramètres :
    ------------
    distance_tensor : LabelledTensor [i, t]
        Distances normalisées.
    max_value : float
        Seuil maximal autorisé.

    Retour :
    --------
    LabelledTensor [i, t]
        Distances tronquées.
    """
    clamped = torch.clamp(distance_tensor.tensor, max=max_value)
    return LabelledTensor(clamped, distance_tensor.dim_labels, distance_tensor.index_to_label)


def compute_remoteness_index(
    distance_matrix: LabelledTensor,
    infrastructure_matrix: LabelledTensor,
    population_vector: LabelledTensor,
    clip_threshold: float 
    = 3.0
) -> LabelledTensor:
    """
    Calcule un indice d'éloignement relatif basé sur la distance, normalisé et borné.

    Étapes :
    -------
    1. Calcul de la distance minimale vers chaque type d'infrastructure.
    2. Moyenne pondérée par population.
    3. Normalisation des distances.
    4. Clipping des valeurs trop élevées.

    Paramètres :
    ------------
    distance_matrix : LabelledTensor [i, j]
        Distances entre localités.
    infrastructure_matrix : LabelledTensor [j, t]
        Présence d'infrastructures.
    population_vector : LabelledTensor [i]
        Population de chaque localité.
    clip_threshold : float
        Valeur maximale autorisée après normalisation.

    Retour :
    --------
    LabelledTensor [i, t]
        Tenseur d'éloignement relatif.
    """

    logger.info("Calcul de la distance minimale vers chaque type d'infrastructure...")
    min_distances = compute_min_distance(
        distance_matrix, infrastructure_matrix
    )

    logger.info("Calcul de la moyenne pondérée par la population (en excluant les valeurs nulles)...")
    mean_distances = compute_mean_distance(
        min_distances, population_vector, exclude_zero_values=True
    )

    logger.info("Normalisation des distances par les moyennes...")
    normalized = normalize_distance(
        min_distances, mean_distances
    )

    logger.info(f"Tronquage des distances normalisées au seuil {clip_threshold}...")
    final_index = clip_max_distance(normalized, max_value=clip_threshold)

    return final_index