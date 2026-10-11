"""Prediksi data uji dan pembuatan file submission.

Cara menjalankan dari root repository::

    python -m scripts.predict

Membaca ``models/pipeline_model_rf.pkl`` dan ``data/test.csv``, lalu menulis
``submissions/submission.csv`` dengan ``damage_grade`` berupa angka 1 sampai 5.
"""

import argparse

import joblib
import pandas as pd

from scripts.feature_engineering import build_features
from scripts.preprocess import (
    ID_COL,
    MODEL_PATH,
    SUBMISSION_PATH,
    TARGET_COL,
    load_data,
)

VALID_GRADES = {1, 2, 3, 4, 5}


def load_model(path=MODEL_PATH):
    """Memuat pipeline model yang sudah disimpan.

    Args:
        path (pathlib.Path): Lokasi file model.

    Returns:
        sklearn.pipeline.Pipeline: Pipeline hasil training.

    Raises:
        FileNotFoundError: Jika file model belum dibuat.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} tidak ditemukan. Jalankan 'python -m scripts.train' terlebih dahulu."
        )
    return joblib.load(path)


def labels_to_numeric(labels):
    """Mengubah label teks ``"Grade 3"`` menjadi angka ``3``.

    Args:
        labels (array-like): Label seperti ``"Grade 1"`` sampai ``"Grade 5"``.

    Returns:
        numpy.ndarray: Array integer.
    """
    return (
        pd.Series(labels).str.replace("Grade ", "", regex=False).astype(int).to_numpy()
    )


def build_submission(model, test_df):
    """Memprediksi data uji dan menyusunnya dalam format submission.

    Args:
        model (sklearn.pipeline.Pipeline): Pipeline hasil training.
        test_df (pandas.DataFrame): Data mentah dari ``test.csv``.

    Returns:
        pandas.DataFrame: Kolom ``building_id`` dan ``damage_grade`` (integer).
    """
    features = build_features(test_df)
    predictions = labels_to_numeric(model.predict(features))
    return pd.DataFrame(
        {ID_COL: test_df[ID_COL].to_numpy(), TARGET_COL: predictions}
    )


def validate_submission(submission, sample_submission):
    """Memeriksa format submission terhadap ``sample_submission.csv``.

    Pemeriksaan: jumlah baris, nama dan urutan kolom, urutan ``building_id``,
    tidak ada nilai kosong, dan label hanya bernilai 1 sampai 5.

    Args:
        submission (pandas.DataFrame): Submission yang akan diperiksa.
        sample_submission (pandas.DataFrame): Contoh format dari Kaggle.

    Raises:
        ValueError: Jika salah satu pemeriksaan gagal.
    """
    if len(submission) != len(sample_submission):
        raise ValueError("Jumlah baris berbeda dengan sample_submission.")
    if list(submission.columns) != list(sample_submission.columns):
        raise ValueError("Nama atau urutan kolom berbeda dengan sample_submission.")
    if not (submission[ID_COL].to_numpy() == sample_submission[ID_COL].to_numpy()).all():
        raise ValueError("building_id atau urutannya berbeda dengan sample_submission.")
    if submission.isna().any().any():
        raise ValueError("Ada nilai kosong pada submission.")
    if not set(submission[TARGET_COL].unique()) <= VALID_GRADES:
        raise ValueError("Ada label di luar 1-5.")


def save_submission(submission, path=SUBMISSION_PATH):
    """Menyimpan submission ke CSV tanpa kolom index.

    Args:
        submission (pandas.DataFrame): Submission yang sudah valid.
        path (pathlib.Path): Lokasi file. Folder dibuat jika belum ada.

    Returns:
        pathlib.Path: Lokasi file yang disimpan.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return path


def main(argv=None):
    """Menjalankan alur prediksi dari memuat model sampai file tersimpan.

    Args:
        argv (list[str] | None): Argumen command line (tidak ada opsi saat ini).

    Returns:
        pandas.DataFrame: Submission yang dihasilkan.
    """
    argparse.ArgumentParser(description="Buat submission.csv dari model tersimpan.").parse_args(argv)

    model = load_model()
    test_df = load_data("test.csv")
    sample_submission = load_data("sample_submission.csv")

    submission = build_submission(model, test_df)
    validate_submission(submission, sample_submission)

    path = save_submission(submission)
    print(f"Submission tersimpan di {path} ({len(submission)} baris)")
    print("Distribusi prediksi:")
    print(submission[TARGET_COL].value_counts(normalize=True).sort_index().round(4))
    return submission


if __name__ == "__main__":
    main()
