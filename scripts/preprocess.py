"""Preprocessing data kompetisi Structural Damage Classification.

Modul ini berisi definisi kolom dan path proyek, fungsi untuk memuat dan
membagi data, serta pembuatan ``ColumnTransformer`` yang dipakai di dalam
pipeline model.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder

# ---------------------------------------------------------------------------
# Path proyek (dihitung dari lokasi file ini, jadi aman dipanggil dari mana pun)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models" / "pipeline_model_rf.pkl"
SUBMISSION_PATH = PROJECT_ROOT / "submissions" / "submission.csv"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "feature_importance.png"

# ---------------------------------------------------------------------------
# Definisi kolom
# ---------------------------------------------------------------------------
ID_COL = "building_id"
TARGET_COL = "damage_grade"

NUMERICAL_COLUMNS = [
    "count_floors_pre_eq",
    "age_building",
    "plinth_area_sq_ft",
    "height_ft_pre_eq",
]
CATEGORICAL_COLUMNS = [
    "land_surface_condition",
    "foundation_type",
    "roof_type",
    "ground_floor_type",
    "other_floor_type",
    "position",
    "plan_configuration",
    "superstructure",
]
LOCATION_COLUMNS = ["district_id", "vdcmun_id", "ward_id"]

FEATURE_COLUMNS = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS + LOCATION_COLUMNS

RANDOM_STATE = 42


def load_data(filename):
    """Memuat file CSV dari folder ``data/``.

    Args:
        filename (str): Nama file, misalnya ``"train.csv"``.

    Returns:
        pandas.DataFrame: Isi file CSV.

    Raises:
        FileNotFoundError: Jika file tidak ada di folder ``data/``.
    """
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} tidak ditemukan. Unduh data dari Kaggle dan letakkan di folder data/."
        )
    return pd.read_csv(path)


def split_train_validation(X, y, test_size=0.2, random_state=RANDOM_STATE):
    """Membagi data menjadi data latih dan data validasi secara stratified.

    Stratifikasi dipakai karena distribusi ``damage_grade`` tidak seimbang.

    Args:
        X (pandas.DataFrame): Fitur.
        y (pandas.Series): Target.
        test_size (float): Proporsi data validasi.
        random_state (int): Seed agar pembagian dapat direproduksi.

    Returns:
        tuple: ``(X_train, X_val, y_train, y_val)``.
    """
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def build_preprocessor():
    """Membuat ``ColumnTransformer`` untuk seluruh fitur.

    - Numerik: imputasi median.
    - Kategorikal struktur bangunan: imputasi modus lalu ``OrdinalEncoder``.
    - Lokasi (kardinalitas tinggi): ``TargetEncoder`` (cross-fitting bawaan
      scikit-learn), bukan one-hot, agar dimensi fitur tidak meledak.

    Semua statistik (median, modus, rata-rata target per lokasi) hanya
    dipelajari dari data latih saat ``fit``, sehingga tidak ada kebocoran data.

    Returns:
        sklearn.compose.ColumnTransformer: Preprocessor yang belum di-fit.
    """
    numerical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    location_pipeline = Pipeline(
        [("target_enc", TargetEncoder(random_state=RANDOM_STATE))]
    )

    return ColumnTransformer(
        [
            ("numeric", numerical_pipeline, NUMERICAL_COLUMNS),
            ("categoric", categorical_pipeline, CATEGORICAL_COLUMNS),
            ("location", location_pipeline, LOCATION_COLUMNS),
        ]
    )
