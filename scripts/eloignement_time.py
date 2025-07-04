# %%
from io import BytesIO
import torch
import numpy as np
import polars as pl
import itertools
import warnings
import boto3
import json
import time
from typing import Optional, Tuple


# %%
warnings.filterwarnings("ignore")

def load_credentials(path: str, endpoint_url: str = None, bucket: str = None) -> dict:
    """
    Charge les credentials MinIO à partir d'un fichier JSON.

    Parameters
    ----------
    path : str
        Chemin vers credentials.json (contenant accessKey, secretKey, etc.)
    endpoint_url : str, optional
        Peut être passé manuellement si pas dans le fichier.
    bucket : str, optional
        Peut être passé manuellement si pas dans le fichier.

    Returns
    -------
    dict
        Dictionnaire prêt pour `boto3.client(...)`
    """
    with open(path, "r") as f:
        creds = json.load(f)

    return {
        "endpoint_url": endpoint_url or "http://localhost:9000",  # ou autre valeur par défaut
        "bucket": bucket or "remoteness",
        "aws_access_key_id": creds["accessKey"],
        "aws_secret_access_key": creds["secretKey"],
        "verify": False
    }


def load_parquet_from_minio(bucket: str, object_key: str, s3_client) -> pl.DataFrame:
    """
    Charge un fichier Parquet stocké dans MinIO et retourne un DataFrame Polars.
    """
    obj = s3_client.get_object(Bucket=bucket, Key=object_key)
    buffer = BytesIO(obj["Body"].read())
    return pl.read_parquet(buffer)

# %%
class LabelledTensor:
    """
    Classe enveloppe pour un tenseur PyTorch avec des étiquettes explicites pour chaque dimension.

    Attributs
    ---------
    tensor : torch.Tensor
        Le tenseur brut.
    dim_labels : list[str]
        Liste ordonnée des noms des dimensions.
    index_to_label : dict[str, list[str]]
        Dictionnaire associant à chaque nom de dimension la liste de ses étiquettes.
    """

    def __init__(self, tensor: torch.Tensor, dim_labels: list[str], index_to_label: dict[str, list[str]]):
        """
        Initialise un objet LabelledTensor.

        Paramètres
        ----------
        tensor : torch.Tensor
            Le tenseur PyTorch à encapsuler.
        dim_labels : list[str]
            Les noms des dimensions du tenseur.
        index_to_label : dict[str, list[str]]
            Les étiquettes associées à chaque dimension.
        """
        self.tensor = tensor
        self.dim_labels = dim_labels
        self.index_to_label = index_to_label

    def to(self, device):
        """
        Déplace le tenseur vers le périphérique spécifié (CPU ou GPU).

        Paramètre
        ---------
        device : str
            'cpu' ou 'cuda' (ou tout autre périphérique reconnu par PyTorch).
        
        Retourne
        --------
        self : LabelledTensor
            L'objet lui-même après déplacement.
        """
        self.tensor = self.tensor.to(device)
        return self

    def __repr__(self):
        """
        Représentation courte de l'objet pour un affichage rapide.
        """
        return f"LabelledTensor(shape={self.tensor.shape}, dims={self.dim_labels})"

    def display(self, max_elements=100) -> pl.DataFrame:
        """
        Affiche une vue lisible du tenseur sous forme d'un DataFrame Polars avec les étiquettes.

        Paramètres
        ----------
        max_elements : int
            Nombre maximal d'éléments à afficher (pour éviter une surcharge visuelle).

        Retourne
        --------
        pl.DataFrame
            Un DataFrame Polars contenant les valeurs du tenseur et leurs étiquettes.
        """
        dims = self.tensor.shape
        dim_labels = self.dim_labels
        idx_labels = self.index_to_label

        total = self.tensor.numel()
        if total > max_elements:
            print(f"[INFO] Tenseur trop grand ({total} éléments). Affichage des {max_elements} premiers éléments seulement.\n")

        # Création de toutes les combinaisons possibles d'indices
        indices = list(itertools.product(*[range(d) for d in dims]))
        records = []

        for idx in indices[:max_elements]:
            labels = [idx_labels[dim_labels[i]][idx[i]] for i in range(len(idx))]
            value = self.tensor[idx].item()
            records.append((*labels, value))

        columns = dim_labels + ["valeur"]
        df = pl.DataFrame(records, schema=columns)
        
        return df

# %%
def create_symmetric_matrix(df: pl.DataFrame, device="cpu") -> Tuple[LabelledTensor, float]:
    """
    Construit une matrice symétrique des temps de parcours à partir d'un DataFrame triangle supérieur.

    Paramètres :
    ------------
    df : pl.DataFrame
        Doit contenir les colonnes "Idloc_start", "Idloc_end", "temps_parcours".
    device : str
        'cpu' ou 'cuda' selon l'appareil souhaité.

    Retour :
    --------
    Tuple[LabelledTensor, float]
        Matrice [i, j] symétrique avec labels et temps de calcul en secondes.
    """
    start_time = time.time()
    
    # Étape 1 : identifiants uniques ordonnés
    unique_locs = pl.concat([df["Idloc_start"], df["Idloc_end"]]).unique().sort()
    idx_to_id = unique_locs.to_list()
    id_to_idx = {idloc: idx for idx, idloc in enumerate(idx_to_id)}
    n = len(idx_to_id)

    # Étape 2 : conversion en indices numpy
    i_idx = df["Idloc_start"].to_numpy()
    j_idx = df["Idloc_end"].to_numpy()
    values = df["temps_parcours"].to_numpy()

    # Étape 3 : mapping des ID vers index
    i_indices = np.vectorize(id_to_idx.get)(i_idx)
    j_indices = np.vectorize(id_to_idx.get)(j_idx)

    # Étape 4 : remplissage de la matrice via vecteurs
    T = torch.full((n, n), float("inf"), device=device)
    indices = torch.tensor(np.stack([i_indices, j_indices]), device=device)
    distances = torch.tensor(values, dtype=torch.float32, device=device)

    # Triangle supérieur
    T[indices[0], indices[1]] = distances
    # Symétrie
    T[indices[1], indices[0]] = distances
    # Diagonale
    T.fill_diagonal_(0.0)

    computation_time = time.time() - start_time
    
    return LabelledTensor(T, ["i", "j"], {"i": idx_to_id, "j": idx_to_id}), computation_time


# %%
def create_population_tensor(df_pop: pl.DataFrame, idloc_order: list[str], device="cpu") -> Tuple[LabelledTensor, float]:
    """
    Crée un vecteur de population ordonné selon idloc.

    Paramètres :
    ------------
    df_pop : pl.DataFrame 
        Contient les colonnes "idloc" et "taille_population".
    idloc_order : list[str]
        Ordre des localités à respecter.
    device : str
        Appareil.

    Retour :
    --------
    Tuple[LabelledTensor, float]
        Vecteur [i] avec les tailles de population et temps de calcul.
    """
    start_time = time.time()
    
    pop = torch.tensor(
        [df_pop.filter(pl.col("Idloc") == loc)["taille_population"][0] for loc in idloc_order],
        dtype=torch.float32,
        device=device
    )
    
    computation_time = time.time() - start_time
    
    return LabelledTensor(pop, ["i"], {"i": idloc_order}), computation_time


# %%
def create_infrastructure_tensor(df: pl.DataFrame, device="cpu") -> Tuple[LabelledTensor, float]:
    """
    Construit un tenseur D[i, t] représentant les infrastructures disponibles.

    Les noms des colonnes encodent le secteur (k) dans leurs deux premiers caractères.

    Paramètres :
    ------------
    df : pl.DataFrame
        Doit contenir la colonne "idloc" et des colonnes de sous-secteurs.
    device : str
        Appareil.

    Retour :
    --------
    Tuple[LabelledTensor, float]
        Tenseur [i, t] : localité × sous-secteur et temps de calcul.
    """
    start_time = time.time()
    
    idlocs = df["idloc"].to_list()
    df_data = df.drop("idloc")
    sous_secteurs = df_data.columns

    D = torch.tensor(
        df_data.to_numpy(),
        dtype=torch.float32,
        device=device
    )

    computation_time = time.time() - start_time

    return LabelledTensor(D, ["i", "t"], {
        "i": idlocs,
        "t": sous_secteurs
    }), computation_time


# %%
def compute_Y(T: LabelledTensor, D: LabelledTensor) -> Tuple[LabelledTensor, float]:
    """
    Calcule le temps minimal de chaque localité i vers une autre j disposant d'une infrastructure t.

    Y[i, t] = min_j { T[i, j] | D[j, t] > 0 }

    Version sans boucle explicite, entièrement vectorisée.

    Paramètres :
    ------------
    T : LabelledTensor
        Matrice des temps de parcours [i, j].
    D : LabelledTensor
        Tenseur d'infrastructure [j, t].

    Retour :
    --------
    Tuple[LabelledTensor, float]
        Tenseur [i, t] contenant les temps minimaux vers une infrastructure et temps de calcul.
    """
    start_time = time.time()
    
    T_tensor = T.tensor  # [i, j]
    D_tensor = D.tensor.bool()  # [j, t]

    i_size, j_size = T_tensor.shape
    t_size = D_tensor.shape[1]

    # Stocke les résultats pour chaque t
    results = []

    for t in range(t_size):
        # Indices des j où l'infrastructure t existe
        j_mask = D_tensor[:, t]  # [j]
        if j_mask.any():
            # Sous-matrice T[:, j sélectionnés]
            T_filtered = T_tensor[:, j_mask]
            min_t = T_filtered.min(dim=1).values  # [i]
        else:
            # Aucun j valide pour ce t
            min_t = torch.full((i_size,), float("inf"), device=T_tensor.device)

        results.append(min_t)

    # Concatène les résultats pour obtenir [i, t]
    Y_tensor = torch.stack(results, dim=1)  # [i, t]

    computation_time = time.time() - start_time

    return LabelledTensor(Y_tensor, ["i", "t"], {
        "i": T.index_to_label["i"],
        "t": D.index_to_label["t"]
    }), computation_time


# %%
def compute_Y_agg(Y: LabelledTensor, P: LabelledTensor) -> Tuple[LabelledTensor, float]:
    """
    Moyenne pondérée de Y[i, t] par la population de chaque localité i.

    Retourne Y_[t]

    Paramètres
    ----------
    Y : LabelledTensor
        Tenseur Y[i, t]
    P : LabelledTensor
        Population par localité [i]

    Retour
    ------
    Tuple[LabelledTensor, float]
        Moyenne pondérée [t] et temps de calcul.
    """
    start_time = time.time()
    
    P_vec = P.tensor.view(-1, 1)  # [i, 1]
    weighted = Y.tensor * P_vec   # [i, t]
    sum_pop = P.tensor.sum()

    Y_mean = (weighted.sum(dim=0) / sum_pop)  # [t]

    computation_time = time.time() - start_time

    return LabelledTensor(Y_mean, ["t"], {
        "t": Y.index_to_label["t"]
    }), computation_time


# %%
def normalize_Y(Y: LabelledTensor, Y_mean: LabelledTensor) -> Tuple[LabelledTensor, float]:
    """
    Normalise Y[i, t] en divisant chaque élément par la moyenne agrégée Ȳ[t].

    Paramètres :
    ------------
    Y : LabelledTensor
        Tenseur original [i, t].
    Y_mean : LabelledTensor
        Moyenne agrégée [t].

    Retour :
    --------
    Tuple[LabelledTensor, float]
        Tenseur normalisé [i, t] et temps de calcul.
    """
    start_time = time.time()
    
    norm_tensor = Y.tensor / Y_mean.tensor.unsqueeze(0) # [i, t] / [1, t]
    
    computation_time = time.time() - start_time
    
    return LabelledTensor(norm_tensor, Y.dim_labels, Y.index_to_label), computation_time


# %%
def clamp_Y(Y: LabelledTensor, max_value=3.0) -> Tuple[LabelledTensor, float]:
    """
    Tronque les valeurs de Y[i, t] à une valeur maximale.

    Paramètres :
    ------------
    Y : LabelledTensor
        Tenseur à borner.
    max_value : float
        Valeur max autorisée.

    Retour :
    --------
    Tuple[LabelledTensor, float]
        Tenseur borné et temps de calcul.
    """
    start_time = time.time()
    
    clamped = torch.clamp(Y.tensor, max=max_value)
    
    computation_time = time.time() - start_time
    
    return LabelledTensor(clamped, Y.dim_labels, Y.index_to_label), computation_time


# %%
def compute_remoteness_tensor(Mat_dist: LabelledTensor,
                                 Mat_infra: LabelledTensor,
                                 Mat_pop: LabelledTensor,
                                 clamp_max: float = 3.0,
                                 verbose: bool = True) -> Tuple[LabelledTensor, dict]:
    """
    Calcule le tenseur d'accessibilité final Y[i, t] normalisé et borné.

    Étapes :
    -------
    1. Calcul des temps minimaux Y[i, t] depuis chaque localité i vers une infrastructure de type t.
    2. Agrégation pondérée par la population pour obtenir Y_mean[t].
    3. Normalisation du tenseur Y[i, t] par Y_mean[t].
    4. Borne la valeur maximale de Y[i, t] à `clamp_max`.

    Paramètres :
    ------------
    Mat_dist : LabelledTensor
        Matrice des distances [i, j].

    Mat_infra : LabelledTensor
        Tenseur d'infrastructure [i, t], booléen ou réel.

    Mat_pop : LabelledTensor
        Vecteur des populations [i].

    clamp_max : float
        Valeur maximale autorisée pour la normalisation (clipping final).
        
    verbose : bool
        Si True, affiche les temps de calcul pour chaque étape.

    Retour :
    --------
    Tuple[LabelledTensor, dict]
        Tenseur d'accessibilité [i, t] normalisé et borné, et dictionnaire des temps de calcul.
    """
    timing_info = {}
    
    Mat_Y, time_Y = compute_Y(Mat_dist, Mat_infra)
    timing_info['compute_Y'] = time_Y
    if verbose:
        print(f"Temps de calcul Y[i,t]: {time_Y:.4f}s")
    
    Mat_Y_mean, time_Y_mean = compute_Y_agg(Mat_Y, Mat_pop)
    timing_info['compute_Y_agg'] = time_Y_mean
    if verbose:
        print(f"Temps de calcul Y_mean[t]: {time_Y_mean:.4f}s")
    
    Mat_Y_norm, time_Y_norm = normalize_Y(Mat_Y, Mat_Y_mean)
    timing_info['normalize_Y'] = time_Y_norm
    if verbose:
        print(f"Temps de normalisation Y: {time_Y_norm:.4f}s")
    
    Mat_Y_final, time_Y_clamp = clamp_Y(Mat_Y_norm, max_value=clamp_max)
    timing_info['clamp_Y'] = time_Y_clamp
    if verbose:
        print(f"Temps de clipping Y: {time_Y_clamp:.4f}s")
    
    timing_info['total_computation'] = sum(timing_info.values())
    if verbose:
        print(f"Temps total de calcul: {timing_info['total_computation']:.4f}s")

    return Mat_Y_final, timing_info


# %%
def load_all_matrices(
    path_dt: str,
    path_infra: str,
    path_pop: str,
    device: str = "cuda",
    mode: str = "local",
    minio_config: dict = None,
    verbose: bool = True
) -> Tuple[LabelledTensor, LabelledTensor, LabelledTensor, dict]:
    """
    Charge et construit les matrices LabelledTensor nécessaires au calcul d'accessibilité.

    Paramètres :
    ------------
    path_dt : str
        Chemin vers le fichier Parquet ou clé MinIO (selon le mode).
    path_infra : str
        Idem.
    path_pop : str
        Idem.
    device : str
        'cpu' ou 'cuda'.
    mode : str
        "local" ou "minio".
    minio_config : dict
        Dictionnaire contenant : {
            "bucket": str,
            "endpoint_url": str,
            "aws_access_key_id": str,
            "aws_secret_access_key": str,
            "verify": bool
        }
    verbose : bool
        Si True, affiche les temps de chargement et construction.

    Retour :
    --------
    Tuple[LabelledTensor, LabelledTensor, LabelledTensor, dict]
        Matrices et dictionnaire des temps de calcul.
    """
    timing_info = {}
    
    # Temps de chargement des données
    load_start_time = time.time()
    
    if mode == "local":
        df_dt = pl.read_parquet(path_dt)
        df_infra = pl.read_parquet(path_infra)
        df_pop = pl.read_parquet(path_pop)

    elif mode == "minio":
        if minio_config is None:
            raise ValueError("minio_config doit être fourni pour le mode 'minio'")

        s3_client = boto3.client(
            "s3",
            endpoint_url=minio_config["endpoint_url"],
            aws_access_key_id=minio_config["aws_access_key_id"],
            aws_secret_access_key=minio_config["aws_secret_access_key"],
            verify=minio_config.get("verify", False)
        )

        bucket = minio_config["bucket"]
        df_dt = load_parquet_from_minio(bucket, path_dt, s3_client)
        df_infra = load_parquet_from_minio(bucket, path_infra, s3_client)
        df_pop = load_parquet_from_minio(bucket, path_pop, s3_client)

    else:
        raise ValueError("mode doit être 'local' ou 'minio'")

    load_time = time.time() - load_start_time
    timing_info['data_loading'] = load_time
    if verbose:
        print(f"Temps de chargement des données: {load_time:.4f}s")

    # Construction des tenseurs avec mesure des temps
    Mat_dist, time_dist = create_symmetric_matrix(df_dt, device=device)
    timing_info['create_distance_matrix'] = time_dist
    if verbose:
        print(f"Temps de construction matrice distance: {time_dist:.4f}s")
    
    Mat_infra, time_infra = create_infrastructure_tensor(df_infra, device=device)
    timing_info['create_infrastructure_tensor'] = time_infra
    if verbose:
        print(f"Temps de construction tenseur infrastructure: {time_infra:.4f}s")
    
    Mat_pop, time_pop = create_population_tensor(df_pop, df_pop["Idloc"].to_list(), device=device)
    timing_info['create_population_tensor'] = time_pop
    if verbose:
        print(f"Temps de construction vecteur population: {time_pop:.4f}s")
    
    timing_info['total_loading'] = sum(timing_info.values())
    if verbose:
        print(f"Temps total de chargement et construction: {timing_info['total_loading']:.4f}s")

    return Mat_dist, Mat_infra, Mat_pop, timing_info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calcule le tenseur d'accessibilité normalisé")

    parser.add_argument("--mode", choices=["local", "minio"], default="local", help="Mode de chargement des données")
    parser.add_argument("--dt_path", type=str, required=True, help="Chemin vers la matrice de distance (Parquet)")
    parser.add_argument("--infra_path", type=str, required=True, help="Chemin vers la matrice d'infrastructure (Parquet)")
    parser.add_argument("--pop_path", type=str, required=True, help="Chemin vers la matrice de population (Parquet)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Appareil pour PyTorch")
    parser.add_argument("--clamp_max", type=float, default=3.0, help="Valeur max pour le clipping final")
    parser.add_argument("--verbose", action="store_true", help="Affiche les temps de calcul détaillés")

    # Arguments pour MinIO (si mode == "minio")
    parser.add_argument("--minio_bucket", type=str, help="Bucket MinIO")
    parser.add_argument("--minio_endpoint", type=str, help="Endpoint MinIO")
    parser.add_argument("--minio_key", type=str, help="Access key MinIO")
    parser.add_argument("--minio_secret", type=str, help="Secret key MinIO")
    parser.add_argument("--minio_verify", action="store_true", help="Vérifie SSL (par défaut: False)")

    args = parser.parse_args()

    # MinIO config si nécessaire
    minio_config = None
    if args.mode == "minio":
        minio_config = load_credentials(
            path="credentials.json",
            endpoint_url="http://192.168.1.230:30137",
            bucket="remoteness"
        )

    # Chargement avec mesure des temps
    Mat_dist, Mat_infra, Mat_pop, loading_times = load_all_matrices(
        path_dt=args.dt_path,
        path_infra=args.infra_path,
        path_pop=args.pop_path,
        device=args.device,
        mode=args.mode,
        minio_config=minio_config,
        verbose=args.verbose
    )

    # Calcul accessibilité avec mesure des temps
    Mat_Y, computation_times = compute_remoteness_tensor(
        Mat_dist, Mat_infra, Mat_pop, 
        clamp_max=args.clamp_max, 
        verbose=args.verbose
    )

    # Affichage des temps totaux
    print("\n" + "="*50)
    print("RÉSUMÉ DES TEMPS DE CALCUL")
    print("="*50)
    print(f"Temps total de chargement: {loading_times['total_loading']:.4f}s")
    print(f"Temps total de calcul: {computation_times['total_computation']:.4f}s")
    print(f"Temps total global: {loading_times['total_loading'] + computation_times['total_computation']:.4f}s")
    print("="*50)

    # Affichage (10 premiers résultats)
    df_Y = Mat_Y.display(max_elements=10)
    print("\nPremiers résultats:")
    print(df_Y)