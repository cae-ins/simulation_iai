import os

class Config:
    """
    Configuration class to store global constants and file paths used across the simulation project.
    """
    # --- Device for tensor computations, can be 'cpu' or 'cuda' ---
    DEVICE = "cuda"  

    # --- Decay drop penaltie after a certain time of travel (in minutes) ---
    DROP_TIME = 30

    # --- Coefficients for route types ---
    COEFS = {
        "AUTOROUTE": 1.016781,
        "RB": 1.352656,
        "RNB": 2.059399,
        "PISTE": 3.034741,
        "VOIE_FLUVIALE": 3.676587,
    }

    # --- Base path for project data ---
    BASE_DIR = r"C:\Users\e_koffie\Documents\IAI_Project"
    SIM_DIR = os.path.join(BASE_DIR, "access_remote_index", "Data")
    PROJECT_DIR = os.path.join(BASE_DIR, "IAI_PROJECT", "Data")

    # --- File paths: Inputs & Outputs ---
    PATH_GEOMETRIE_TRONCONS = os.path.join(SIM_DIR, "path_info_matrix.parquet")

    PATH_ITINERAIRE_MATRIX = os.path.join(
        BASE_DIR, "access_remote_index", "Data", 
        "path_matrices", "mat_terrain_part_01.parquet"
    )

    PATH_DT_MATRIX = os.path.join(SIM_DIR, "dt_matrix_terrain_VF.parquet")
    PATH_INFRA_MATRIX = os.path.join(SIM_DIR, "infrastructure_matrix_VF.parquet")
    PATH_POP_MATRIX = os.path.join(SIM_DIR, "population_matrix.parquet")
