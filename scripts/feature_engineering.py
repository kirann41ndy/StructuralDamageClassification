"""Pembersihan dan pembentukan fitur dari data mentah.
 
Fungsi di modul ini dipakai saat training dan saat prediksi
(``train.py`` dan ``predict.py``), sehingga definisi fitur selalu konsisten.
"""

import numpy as np

from scripts.preprocess import FEATURE_COLUMNS, LOCATION_COLUMNS

AGE_PLACEHOLDER = 999


def replace_age_placeholder(df, placeholder=AGE_PLACEHOLDER):
    """Mengubah nilai placeholder pada ``age_building`` menjadi ``NaN``.

    Nilai 999 pada ``age_building`` berarti "usia tidak diketahui" (sekitar
    0,41% data), bukan usia sebenarnya. Setelah menjadi ``NaN``, nilai ini
    diimputasi oleh pipeline memakai median dari data latih.

    Args:
        df (pandas.DataFrame): Data yang memuat kolom ``age_building``.
        placeholder (int): Nilai placeholder yang diganti.

    Returns:
        pandas.DataFrame: Salinan data dengan placeholder sudah diganti.
    """
    df = df.copy()
    df["age_building"] = df["age_building"].replace(placeholder, np.nan)
    return df


def cast_location_to_str(df, location_columns=LOCATION_COLUMNS):
    """Mengubah kolom lokasi menjadi string agar diperlakukan sebagai kategori.

    ID lokasi berupa angka, padahal tidak punya urutan atau jarak yang bermakna.

    Args:
        df (pandas.DataFrame): Data yang memuat kolom lokasi.
        location_columns (list[str]): Nama kolom lokasi.

    Returns:
        pandas.DataFrame: Salinan data dengan kolom lokasi bertipe string.
    """
    df = df.copy()
    for col in location_columns:
        df[col] = df[col].astype(str)
    return df


def build_features(df):
    """Mengubah data mentah menjadi matriks fitur siap masuk pipeline.

    Langkah: mengganti placeholder ``age_building``, mengubah kolom lokasi ke
    string, lalu memilih ``FEATURE_COLUMNS`` (``building_id`` dan target tidak
    ikut sebagai fitur).

    Args:
        df (pandas.DataFrame): Data mentah dari ``train.csv`` atau ``test.csv``.

    Returns:
        pandas.DataFrame: Fitur dengan urutan kolom ``FEATURE_COLUMNS``.

    Raises:
        ValueError: Jika ada kolom fitur yang tidak ditemukan pada data.
    """
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom fitur tidak ditemukan: {missing}")

    df = replace_age_placeholder(df)
    df = cast_location_to_str(df)
    return df[FEATURE_COLUMNS]
