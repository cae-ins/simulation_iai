import polars as pl
import logging
from config import Config
from utils import create_symmetric_matrix, \
    create_infrastructure_tensor, create_population_tensor


# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_voie_type_changes(geo: pl.DataFrame, voie_type_global_change, gid_modifications) -> pl.DataFrame:
    """
    Applique les modifications de type de voie à la table géographique des tronçons.
    """
    if voie_type_global_change:
        logger.info("Application des changements globaux de type de voie...")
        geo = geo.with_columns([
            pl.when(pl.col("type_voie").is_in(list(voie_type_global_change)))
              .then(pl.col("type_voie").map_elements(lambda t: voie_type_global_change[t]))
              .otherwise(pl.col("type_voie")).alias("type_voie")
        ])

    if gid_modifications:
        logger.info("Application des changements spécifiques de type de voie par gid...")
        gid_df = pl.DataFrame({
            "gid": list(gid_modifications),
            "type_voie_new": list(gid_modifications.values())
        }).with_columns(pl.col("gid").cast(pl.Int64))
        geo = geo.join(gid_df, on="gid", how="left").with_columns([
            pl.coalesce([pl.col("type_voie_new"), pl.col("type_voie")]).alias("type_voie")
        ]).select(["gid", "length", "type_voie"])

    return geo


def get_troncons_cibles(geo: pl.DataFrame, gid_modifications, voie_type_global_change) -> set[int]:
    """
    Identifie les tronçons (gids) dont le type de voie a été modifié.
    """
    troncons = set(gid_modifications or [])
    if voie_type_global_change:
        types_mod = list(voie_type_global_change.values())
        troncons |= set(
            geo.filter(pl.col("type_voie").is_in(types_mod))["gid"]
        )
    print(f"{len(troncons)} tronçon(s) affecté(s) identifié(s).")
    return troncons


def update_parcours_temps(path_mat_path: str,
                           troncons_cibles: set[int],
                           geo: pl.DataFrame,
                           mat_dt: pl.DataFrame) -> pl.DataFrame:
    """
    Met à jour les temps de parcours dans une matrice d'itinéraires unique
    contenant des listes de tronçons, en prenant en compte les tronçons modifiés.
    """
    if not troncons_cibles:
        logger.info("Aucun tronçon modifié : pas de recalcul nécessaire.")
        return mat_dt

    logger.info("Recalcul des temps de parcours depuis la matrice unique...")

    # Dictionnaire des tronçons {gid: (longueur, type_voie)}
    troncon_map = {
        int(gid): (length, type_voie)
        for gid, length, type_voie in geo.select(["gid", "length", "type_voie"]).rows()
    }

    df = pl.read_parquet(path_mat_path)

    if "gids" not in df.columns:
        logger.warning(f"Colonne 'gids' manquante dans le fichier : {path_mat_path}")
        return mat_dt

    df_filtered = df.filter(pl.col("gids").list.any(lambda g: g in troncons_cibles))


    if df_filtered.is_empty():
        logger.info("Aucun itinéraire impacté par les tronçons modifiés.")
        return mat_dt
    
    logger.info(f"{df_filtered.height} itinéraire(s) impacté(s) identifié(s).")

    mat_update = df_filtered.with_columns([
        pl.col("gids").map_elements(
            lambda gids: sum(
                troncon_map.get(gid, (0, "AUTOROUTE"))[0] *
                Config.COEFS.get(troncon_map.get(gid, (0, "AUTOROUTE"))[1], 1.0)
                for gid in gids
            )
        ).alias("delta_temps")
    ])

    mat_dt = mat_dt.join(
        mat_update.select(["Idloc_start", "Idloc_end", "delta_temps"]),
        on=["Idloc_start", "Idloc_end"],
        how="left"
    ).with_columns([
        (pl.col("temps_parcours") + pl.col("delta_temps").fill_null(0)).alias("temps_parcours")
    ]).drop("delta_temps")

    return mat_dt



def update_infrastructure(mat_infra: pl.DataFrame, 
                          infra_modifications) -> pl.DataFrame:
    """
    Met à jour la matrice des infrastructures avec les changements définis par localité.
    """
    if not infra_modifications:
        return mat_infra

    logger.info("Mise à jour des infrastructures locales...")

    delta_df = pl.DataFrame([
        {"idloc": idloc, **mods} for idloc, mods in infra_modifications.items()]).fill_null(0)

    mod_cols = [c for c in delta_df.columns if c != "idloc"]
    mat_infra = mat_infra.join(delta_df, on="idloc", how="left", suffix="_mod").with_columns([
        (pl.col(c) + pl.col(f"{c}_mod").fill_null(0)).alias(c) for c in mod_cols])
 
    return mat_infra.select([c for c in mat_infra.columns if not c.endswith("_mod")])


def update_population(mat_pop: pl.DataFrame, 
                      population_growth_rates, 
                      global_growth) -> pl.DataFrame:
    """
    Met à jour la population par localité avec un taux global ou spécifique.
    """
    if global_growth is not None:
        logger.info(f"Application d'un taux global de croissance de la population : {global_growth:.2%}")
        return mat_pop.with_columns([
            (pl.col("taille_population") * (1 + global_growth)).alias("taille_population")])

    if population_growth_rates:
        logger.info("Application de taux de croissance spécifiques à certaines localités...")
        return mat_pop.with_columns([
            (pl.col("taille_population") * (
                1 + pl.col("Idloc").map_elements(lambda loc: population_growth_rates.get(loc, 0.0))
            )).alias("taille_population")])

    return mat_pop

def simulate_scenario(
    gid_modifications: dict[int, str] | None = None,
    voie_type_global_change: dict[str, str] | None = None,
    infra_modifications: dict[str, dict[str, int]] | None = None,
    population_growth_rates: dict[str, float] | None = None,
    global_population_growth: float | None = None,
    path_mat_path: str = "",
    path_mat_infra: str = "",
    path_mat_distance_temps: str = "",
    path_mat_population: str = "",
    path_geometrie_troncons: str = ""
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Simule un scénario incluant :
    - des modifications des types de voies,
    - des changements d'infrastructures,
    - des évolutions de population,
    et met à jour les temps de parcours en conséquence.

    Retourne :
    - Matrice des temps de parcours mise à jour
    - Matrice d'infrastructures mise à jour
    - Matrice de population mise à jour
    """
    logger.info("Début de la simulation de scénario de transport...")

    # Chargement des matrices
    mat_infra = pl.read_parquet(path_mat_infra)
    mat_dt = pl.read_parquet(path_mat_distance_temps)
    mat_pop = pl.read_parquet(path_mat_population)
    geo = pl.read_parquet(path_geometrie_troncons)

    # 1. Appliquer les changements de types de voies
    geo = apply_voie_type_changes(geo, voie_type_global_change, gid_modifications)

    # 2. Identifier les tronçons affectés
    troncons_cibles = get_troncons_cibles(geo, gid_modifications, voie_type_global_change)

    # 3. Mettre à jour les temps de parcours
    mat_dt = update_parcours_temps(path_mat_path, troncons_cibles, geo, mat_dt)

    # 4. Appliquer les modifications d'infrastructure
    mat_infra = update_infrastructure(mat_infra, infra_modifications)

    # 5. Appliquer les taux de croissance de population
    mat_pop = update_population(mat_pop, population_growth_rates, global_population_growth)

    logger.info("Création des tenseurs labéllisés...")
    distance_tensor = create_symmetric_matrix(mat_dt, device=Config.DEVICE)
    infra_tensor = create_infrastructure_tensor(mat_infra, device=Config.DEVICE)
    population_tensor = create_population_tensor(mat_pop, mat_pop["Idloc"].to_list(),
                                                 device=Config.DEVICE)

    logger.info("Création des tenseurs de simulation...")
    return distance_tensor, infra_tensor, population_tensor