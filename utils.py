import torch
import math
import gc
import numpy as np
import polars as pl
import itertools
import torch.nn.functional as F
from typing import Literal
from dataclasses import dataclass
import warnings


warnings.filterwarnings("ignore")


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

    
    def display(self, max_elements: int = 100, filters: dict[str, list[str]] = None) -> pl.DataFrame:
        """
        Affiche une vue lisible du tenseur sous forme d'un DataFrame Polars avec les étiquettes,
        en permettant un filtrage rapide par labels.

        Paramètres
        ----------
        max_elements : int, par défaut 100
            Nombre maximal d'éléments à afficher.
        filters : dict[str, list[str]], optional
            Dictionnaire {nom_dimension: [liste de labels à afficher]}.
            Permet de restreindre l'affichage à certaines étiquettes.

        Retour
        ------
        pl.DataFrame
            Un DataFrame Polars avec les étiquettes et les valeurs du tenseur.
        """
        flat_indices = torch.nonzero(self.tensor, as_tuple=False)

        # Si des filtres sont fournis
        if filters:
            mask = torch.ones(flat_indices.size(0), dtype=torch.bool)

            for dim_name, accepted_labels in filters.items():
                if dim_name not in self.dim_labels:
                    raise ValueError(f"Dimension '{dim_name}' inconnue.")

                dim_idx = self.dim_labels.index(dim_name)
                label_to_index = self.index_to_label[dim_name]

                try:
                    accepted_set = set(label_to_index.index(label) for label in accepted_labels)
                except ValueError as e:
                    raise ValueError(f"Un des labels fournis dans '{dim_name}' est invalide.") from e

                # Utilisation de torch.isin pour un filtrage rapide
                dim_values = flat_indices[:, dim_idx]
                accepted_tensor = torch.tensor(list(accepted_set), device=dim_values.device)
                mask &= torch.isin(dim_values, accepted_tensor)

            flat_indices = flat_indices[mask]

        # Limiter l'affichage
        if flat_indices.size(0) > max_elements:
            print(f"[INFO] Affichage des {max_elements} premiers éléments sur {flat_indices.size(0)} non nuls.")
            flat_indices = flat_indices[:max_elements]

        values = self.tensor[tuple(flat_indices.T)].cpu().tolist()

        # Récupération rapide des étiquettes
        records = []
        for i in range(flat_indices.size(0)):
            labels = [
                self.index_to_label[dim][flat_indices[i, j].item()]
                for j, dim in enumerate(self.dim_labels)
            ]
            records.append((*labels, values[i]))

        columns = self.dim_labels + ["valeur"]
        return pl.DataFrame(records, schema=columns)
    

    def to_dataframe(
        self,
        index_dim: str,
        column_dim: str | None = None,
        index_name: str | None = None,
        value_name: str = "valeur",
        fixed_dims: dict[str, str] = None,
    ) -> pl.DataFrame:
        """
        Convertit un LabelledTensor (1D, 2D ou 3D) en DataFrame Polars.

        Si 3D, nécessite de fixer les dimensions supplémentaires avec `fixed_dims`.

        Paramètres
        ----------
        index_dim : str
            Nom de la dimension pour les lignes.
        column_dim : str, optional
            Nom de la dimension pour les colonnes (requis pour 2D ou 3D).
        index_name : str, optional
            Nom personnalisé de la colonne d'index.
        value_name : str
            Nom de la colonne des valeurs (pour tenseurs 1D).
        fixed_dims : dict[str, str]
            Pour tenseurs 3D : dictionnaire {nom_dim: label_valeur} pour fixer la dimension restante.

        Retour
        ------
        pl.DataFrame
        """
        if index_dim not in self.dim_labels:
            raise ValueError(f"{index_dim} n'est pas une dimension valide.")

        if self.tensor.ndim == 1:
            if column_dim is not None:
                raise ValueError("column_dim doit être None pour les tenseurs 1D.")
            labels = self.index_to_label[index_dim]
            values = self.tensor.cpu().numpy().tolist()
            return pl.DataFrame({
                index_name or index_dim: labels,
                value_name: values
            })

        if self.tensor.ndim == 2:
            if column_dim is None:
                raise ValueError("column_dim est requis pour les tenseurs 2D.")
            if column_dim not in self.dim_labels:
                raise ValueError(f"{column_dim} n'est pas une dimension valide.")
            row_idx = self.dim_labels.index(index_dim)
            col_idx = self.dim_labels.index(column_dim)

            tensor = self.tensor.permute(row_idx, col_idx) if (row_idx, col_idx) != (0, 1) else self.tensor
            row_labels = self.index_to_label[index_dim]
            col_labels = self.index_to_label[column_dim]
            values_np = tensor.cpu().numpy()

            df_dict = {index_name or index_dim: row_labels}
            for j, col in enumerate(col_labels):
                df_dict[col] = values_np[:, j]

            return pl.DataFrame(df_dict)

        if self.tensor.ndim == 3:
            if fixed_dims is None or len(fixed_dims) != 1:
                raise ValueError("Pour les tenseurs 3D, il faut fixer une dimension avec `fixed_dims={dim: label}`.")

            fixed_dim_name, fixed_label = next(iter(fixed_dims.items()))

            if fixed_dim_name not in self.dim_labels:
                raise ValueError(f"Dimension '{fixed_dim_name}' non trouvée dans {self.dim_labels}.")

            dim_indices = {dim: self.dim_labels.index(dim) for dim in self.dim_labels}
            fixed_idx = self.index_to_label[fixed_dim_name].index(fixed_label)

            # Réorganiser pour amener la dimension fixée en dernière position, puis découper (ou extraire)
            perm = [dim_indices[index_dim], dim_indices[column_dim], dim_indices[fixed_dim_name]]
            tensor_perm = self.tensor.permute(perm)  # [i, j, t]
            tensor_2d = tensor_perm[:, :, fixed_idx]  # fix t

            # Préparer le DataFrame
            row_labels = self.index_to_label[index_dim]
            col_labels = self.index_to_label[column_dim]
            values_np = tensor_2d.cpu().numpy()

            df_dict = {index_name or index_dim: row_labels}
            for j, col in enumerate(col_labels):
                df_dict[col] = values_np[:, j]

            return pl.DataFrame(df_dict)

        raise ValueError("Seuls les tenseurs 1D, 2D ou 3D sont pris en charge.")
    

    def _align_and_apply(self, other, op):
        """
        Aligne deux LabelledTensors sur leurs dimensions et labels, 
        puis applique une opération élément par élément.

        Paramètres
        ----------
        other : LabelledTensor
            Le deuxième tenseur à utiliser dans l'opération.
        op : function
            Une fonction PyTorch à appliquer, comme torch.add, torch.mul, etc.

        Retour
        ------
        LabelledTensor
            Le résultat de l'opération entre les deux LabelledTensors, 
            avec conservation des métadonnées.

        Exceptions
        ----------
        TypeError : si `other` n'est pas un LabelledTensor.
        ValueError : si les dimensions ou les étiquettes ne correspondent pas parfaitement.
        """
        # Vérifie que l'autre objet est un LabelledTensor
        if not isinstance(other, LabelledTensor):
            raise TypeError("Opérations valides uniquement entre deux LabelledTensors.")

        # Vérifie que l'ordre et les noms des dimensions correspondent
        if self.dim_labels != other.dim_labels:
            raise ValueError("Les dimensions doivent être dans le même ordre.")

        # Vérifie que les étiquettes des dimensions correspondent une à une
        for dim in self.dim_labels:
            if self.index_to_label[dim] != other.index_to_label[dim]:
                raise ValueError(f"Les labels de la dimension '{dim}' ne correspondent pas.")

        # Applique l'opération élément par élément entre les tenseurs bruts
        result_tensor = op(self.tensor, other.tensor)

        # Retourne un nouveau LabelledTensor avec les mêmes métadonnées
        return LabelledTensor(result_tensor, self.dim_labels, self.index_to_label.copy())


    def __truediv__(self, other):
        """
        Division élément par élément entre deux LabelledTensors alignés.

        Paramètres
        ----------
        other : LabelledTensor

        Retour
        ------
        LabelledTensor
        """
        return self._align_and_apply(other, torch.div)


    def __mul__(self, other):
        """
        Multiplication élément par élément entre deux LabelledTensors alignés.

        Paramètres
        ----------
        other : LabelledTensor

        Retour
        ------
        LabelledTensor
        """
        return self._align_and_apply(other, torch.mul)


    def __add__(self, other):
        """
        Addition élément par élément entre deux LabelledTensors alignés.

        Paramètres
        ----------
        other : LabelledTensor

        Retour
        ------
        LabelledTensor
        """
        return self._align_and_apply(other, torch.add)


    def __sub__(self, other):
        """
        Soustraction élément par élément entre deux LabelledTensors alignés.

        Paramètres
        ----------
        other : LabelledTensor

        Retour
        ------
        LabelledTensor
        """
        return self._align_and_apply(other, torch.sub)
    


def create_symmetric_matrix(df: pl.DataFrame, 
                            device="cpu") -> LabelledTensor:
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
    LabelledTensor
        Matrice [i, j] symétrique, avec labels.
    """
    # Étape 1 : identifiants uniques ordonnés
    unique_locs = pl.concat([df["Idloc_start"], df["Idloc_end"]]).unique().sort()
    # print(unique_locs)
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
    T = torch.full((n, n), float("inf"), dtype=torch.float32, device=device)
    indices = torch.tensor(np.stack([i_indices, j_indices]), device=device)
    distances = torch.tensor(values, dtype=torch.float32, device=device)

    # Triangle supérieur
    T[indices[0], indices[1]] = distances
    # Symétrie
    T[indices[1], indices[0]] = distances
    # Diagonale
    T.fill_diagonal_(0.0)

    return LabelledTensor(T, ["i", "j"], {"i": idx_to_id, "j": idx_to_id})



def create_population_tensor(df_pop: pl.DataFrame, 
                             idloc_order: list[str], 
                             device: str = "cpu", 
                             normalization: str = "none") -> LabelledTensor:
    """
    Crée un vecteur de population ordonné selon idloc, avec option de normalisation.

    Paramètres
    ----------
    df_pop : pl.DataFrame
        Contient les colonnes "Idloc" et "taille_population".
    idloc_order : list[str]
        Ordre des localités à respecter.
    device : str, default="cpu"
        Appareil cible (ex. "cpu" ou "cuda").
    normalization : str, default="none"
        Méthode de normalisation à appliquer :
            - "none"   : pas de normalisation.
            - "minmax" : (x - min) / (max - min), borné entre 0 et 1.
            - "zscore" : (x - mean) / std, puis rendu positif (shifté par +|min| si nécessaire).

    Retour
    ------
    LabelledTensor
        Vecteur [i] des tailles de population (normalisées ou non).
    """
    # Extraction des valeurs de population selon l'ordre fourni
    raw_values = [
        df_pop.filter(pl.col("Idloc") == loc)["taille_population"][0]
        for loc in idloc_order
    ]
    pop = torch.tensor(raw_values, dtype=torch.float32, device=device)

    # Application d'une normalisation si demandée
    if normalization == "minmax":
        min_val = pop.min()
        max_val = pop.max()
        if max_val > min_val:
            pop = (pop - min_val) / (max_val - min_val)
        else:
            pop = torch.zeros_like(pop)  # évite division par 0

    elif normalization == "zscore":
        mean = pop.mean()
        std = pop.std()
        if std > 0:
            pop = (pop - mean) / std
        else:
            pop = pop - mean  # pas de std → centré uniquement
        # Décalage pour garantir positivité
        min_val = pop.min()
        if min_val < 0:
            pop = pop + (-min_val)

    elif normalization != "none":
        raise ValueError(f"Unknown normalization method: {normalization}")

    return LabelledTensor(pop, ["i"], {"i": idloc_order})



def create_infrastructure_tensor(df: pl.DataFrame, 
                                 device="cpu") -> LabelledTensor:
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
    LabelledTensor
        Tenseur [i, t] : localité × sous-secteur.
    """
    idlocs = df["idloc"].to_list()
    df_data = df.drop("idloc")
    sous_secteurs = df_data.columns

    D = torch.tensor(
        df_data.to_numpy(),
        dtype=torch.float32,
        device=device
    )

    return LabelledTensor(D, ["i", "t"], {
        "i": idlocs,
        "t": sous_secteurs
    })


def compute_infra_tensor(infra_tensor: LabelledTensor, 
                         prefix: str | list[str] = None) -> LabelledTensor:
    """
    Sélectionne les colonnes t du tenseur S[i, t] correspondant à un ou plusieurs préfixes.
    Le tenseur S[i, t] représente les infrastructures par type (t) et par localité (i).

    Paramètres
    ----------
    infra_tensor: LabelledTensor [i, t]
        Tenseur contenant les infrastructures par type.
    prefix : str ou list[str], optionnel
        Préfixe ou liste de préfixes à filtrer sur l'axe t.
        Si None, toutes les colonnes sont conservées.

    Retour
    ------
    LabelledTensor [i, t]
        Tenseur filtré.
    """
    tensor = infra_tensor.tensor
    labels_t = infra_tensor.index_to_label["t"]
    prefix = [prefix] if isinstance(prefix, str) else prefix

    if prefix is not None:
        keep = [i for i, label in enumerate(labels_t) if any(label.startswith(p) for p in prefix)]
        tensor = tensor[:, keep]
        labels_t = [labels_t[i] for i in keep]

    return LabelledTensor(tensor.clone(), infra_tensor.dim_labels, {
        "i": infra_tensor.index_to_label["i"],
        "t": labels_t
    })


def compute_theoretical_infra_tensor(infra_tensor: LabelledTensor,
                                     population_tensor: LabelledTensor,
                                     normalization_rules: dict[str, float] = None,
                                     default_ratio: float = 1.0) -> LabelledTensor:
    """
    Calcule le nombre théorique d'infrastructures nécessaires par localité et type.

    Pour chaque type t, on suppose qu'une infrastructure est nécessaire pour `normalization_rules[t]` habitants.

    Paramètres
    ----------
    infra_tensor : LabelledTensor [i, t]
        Nombre d'infrastructures observées par localité (i) et secteur (t).
    population : LabelledTensor [i]
        Population par localité.
    normalization_rules : dict[str, float], optionnel
        Règles de normalisation par préfixe de t (ex : {'S': 3000}).
    default_ratio : float, optionnel
        Ratio par défaut si aucune règle ne correspond.

    Retour
    ------
    LabelledTensor [i, t]
        Nombre d'infrastructures théoriques.
    """
    device = infra_tensor.tensor.device
    pop = population_tensor.tensor.view(-1, 1).to(dtype=torch.float32, device=device)

    labels_t = infra_tensor.index_to_label["t"]

    if normalization_rules is None:
        factors = torch.full((1, len(labels_t)), default_ratio, dtype=torch.float32, device=device)
    else:
        factors = []
        for t in labels_t:
            factor = next((v for k, v in normalization_rules.items() if t.startswith(k)), default_ratio)
            factors.append(factor)
        factors = torch.tensor(factors, dtype=torch.float32, device=device).view(1, -1)

    S_theoretical = pop / factors  # [i, t]

    return LabelledTensor(S_theoretical, ["i", "t"], {
        "i": infra_tensor.index_to_label["i"],
        "t": labels_t
    })


def load_matrices(path_dt: str,
                path_infra: str,
                path_pop: str,
                device: str = "cuda") -> tuple[LabelledTensor, LabelledTensor, LabelledTensor]:
    """
    Charge et construit les matrices LabelledTensor nécessaires au calcul d'accessibilité :
    - Matrice des distances symétrique [i, j]
    - Tenseur des infrastructures [i, t]
    - Vecteur de population [i]

    Paramètres :
    ------------
    path_dt : str
        Chemin vers le fichier Parquet contenant les temps de parcours (triangle supérieur).
    path_infra : str
        Chemin vers le fichier Parquet contenant les infrastructures.
    path_pop : str
        Chemin vers le fichier Parquet contenant les populations.
    device : str
        Périphérique ("cpu" ou "cuda").

    Retour :
    --------
    tuple[LabelledTensor, LabelledTensor, LabelledTensor]
        distance_tensor  : Matrice des temps de parcours [i, j]
        infra_tensor     : Tenseur des infrastructures [i, t]
        population_tensor: Vecteur des populations [i]
    """
    # Lecture des fichiers parquet
    distance = pl.read_parquet(path_dt)
    infrastructures = pl.read_parquet(path_infra)
    population = pl.read_parquet(path_pop)

    # Construction des matrices
    distance_tensor = create_symmetric_matrix(distance, device=device)
    infra_tensor = create_infrastructure_tensor(infrastructures, device=device)
    population_tensor = create_population_tensor(population, population["Idloc"].to_list(),
                                                 device=device)

    return distance_tensor, infra_tensor, population_tensor