"""Training, tuning, evaluasi, dan penyimpanan model.

Cara menjalankan dari root repository::

    python -m scripts.train                 # training dengan parameter bawaan
    python -m scripts.train --tune          # tuning RandomizedSearchCV dulu

Hasil: ``models/pipeline_model_rf.pkl`` dan
``reports/figures/feature_importance.png``.
"""

import argparse
import joblib
import pandas as pd
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from scripts.feature_engineering import build_features
from scripts.preprocess import (
    FIGURE_PATH,
    LOCATION_COLUMNS,
    MODEL_PATH,
    RANDOM_STATE,
    TARGET_COL,
    build_preprocessor,
    load_data,
    split_train_validation,
)

DEFAULT_RF_PARAMS = {
    "n_estimators": 150,
    "min_samples_leaf": 5,
    "max_depth": 20
}

PARAM_DISTRIBUTIONS = {
    "algo__n_estimators": [50,100, 150],
    "algo__max_depth": [10, 20, 30, 40],
    "algo__min_samples_leaf": [3, 5, 10],
    "algo__class_weight": [None, "balanced", "balanced_subsample"],
}


def build_pipeline(**rf_params):
    """Membuat pipeline lengkap: preprocessing lalu Random Forest.

    Args:
        **rf_params: Parameter ``RandomForestClassifier`` yang menimpa
            ``DEFAULT_RF_PARAMS``.

    Returns:
        sklearn.pipeline.Pipeline: Pipeline dengan langkah ``prep`` dan ``algo``.
    """
    params = {**DEFAULT_RF_PARAMS, **rf_params}
    return Pipeline(
        [
            ("prep", build_preprocessor()),
            (
                "algo",
                RandomForestClassifier(
                    random_state=RANDOM_STATE, n_jobs=-1, **params
                ),
            ),
        ]
    )


def tune_pipeline(pipeline, X_train, y_train, n_iter=4, cv=2):
    """Mencari hyperparameter terbaik dengan ``RandomizedSearchCV``.

    Args:
        pipeline (sklearn.pipeline.Pipeline): Pipeline dari ``build_pipeline``.
        X_train (pandas.DataFrame): Fitur latih.
        y_train (pandas.Series): Target latih.
        n_iter (int): Jumlah kombinasi parameter yang dicoba.
        cv (int): Jumlah fold cross-validation.

    Returns:
        sklearn.model_selection.RandomizedSearchCV: Objek pencarian yang sudah
        di-fit (model terbaik ada di ``best_estimator_``).
    """
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        cv=cv,
        scoring="f1_macro",
        random_state=RANDOM_STATE,
        n_jobs=1,  # Random Forest sudah paralel sendiri (n_jobs=-1)
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def evaluate(model, X, y):
    """Menghitung accuracy dan macro F1.

    Args:
        model: Model/pipeline yang sudah di-fit.
        X (pandas.DataFrame): Fitur.
        y (pandas.Series): Label sebenarnya.

    Returns:
        dict: ``{"accuracy": float, "macro_f1": float}``.
    """
    pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
    }


def _base_feature_name(transformed_name):
    """Mengembalikan nama fitur asli dari nama kolom hasil preprocessing.

    ``TargetEncoder`` pada target multi-kelas menghasilkan satu kolom per kelas
    untuk tiap fitur lokasi (misalnya ``ward_id_Grade 3``). Fungsi ini membuang
    awalan transformer dan akhiran kelas agar kolom-kolom itu bisa digabung.

    Args:
        transformed_name (str): Nama dari ``get_feature_names_out()``.

    Returns:
        str: Nama fitur asli.
    """
    name = transformed_name.split("__", 1)[-1]
    for col in LOCATION_COLUMNS:
        if name.startswith(col + "_"):
            return col
    return name


def get_feature_importance(model):
    """Menghitung feature importance per fitur asli.

    Importance kolom hasil target encoding dijumlahkan kembali ke fitur lokasi
    asalnya, jadi hasilnya berisi 15 fitur dengan total 1.

    Args:
        model (sklearn.pipeline.Pipeline): Pipeline hasil training.

    Returns:
        pandas.Series: Importance per fitur, terurut menurun.
    """
    names = model.named_steps["prep"].get_feature_names_out()
    importances = model.named_steps["algo"].feature_importances_
    index = [_base_feature_name(name) for name in names]
    return (
        pd.Series(importances, index=index)
        .groupby(level=0)
        .sum()
        .sort_values(ascending=False)
    )


def save_feature_importance_figure(importance, path=FIGURE_PATH):
    """Menyimpan grafik feature importance ke file gambar.

    Memakai ``matplotlib.figure.Figure`` langsung sehingga
    tidak mengubah backend matplotlib. Dengan begitu fungsi ini aman diimpor
    dari notebook tanpa mengganggu tampilan grafik inline.

    Args:
        importance (pandas.Series): Hasil ``get_feature_importance``.
        path (pathlib.Path): Lokasi file PNG. Folder dibuat jika belum ada.

    Returns:
        pathlib.Path: Lokasi file yang disimpan.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = importance.sort_values(ascending=True)

    fig = Figure(figsize=(8, 6))
    ax = fig.subplots()
    ax.barh(ordered.index, ordered.values, color="seagreen")
    ax.set_title("Feature Importance - Random Forest")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def save_model(model, path=MODEL_PATH):
    """Menyimpan pipeline ke file dengan kompresi ``joblib``.

    Args:
        model (sklearn.pipeline.Pipeline): Pipeline hasil training.
        path (pathlib.Path): Lokasi file model. Folder dibuat jika belum ada.

    Returns:
        pathlib.Path: Lokasi file yang disimpan.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    return path


def main(argv=None):
    """Menjalankan alur training dari awal sampai model tersimpan.

    Args:
        argv (list[str] | None): Argumen command line (``None`` memakai
            ``sys.argv``). Opsi: ``--tune`` dan ``--sample N``.

    Returns:
        sklearn.pipeline.Pipeline: Model yang sudah dilatih dan disimpan.
    """
    parser = argparse.ArgumentParser(description="Training model damage_grade.")
    parser.add_argument(
        "--tune", action="store_true", help="jalankan RandomizedSearchCV dulu"
    )
    parser.add_argument(
        "--sample", type=int, default=None, help="pakai N baris saja (uji cepat)"
    )
    args = parser.parse_args(argv)

    df = load_data("train.csv")
    if args.sample:
        df = df.sample(args.sample, random_state=RANDOM_STATE)

    X = build_features(df)
    y = df[TARGET_COL]
    X_train, X_val, y_train, y_val = split_train_validation(X, y)
    print(f"Data latih: {len(X_train)} baris | validasi: {len(X_val)} baris")

    pipeline = build_pipeline()
    if args.tune:
        search = tune_pipeline(pipeline, X_train, y_train)
        print("Parameter terbaik:", search.best_params_)
        model = search.best_estimator_
    else:
        model = pipeline.fit(X_train, y_train)

    train_metrics = evaluate(model, X_train, y_train)
    val_metrics = evaluate(model, X_val, y_val)
    print(
        f"Train      -> accuracy {train_metrics['accuracy']:.4f} | "
        f"macro F1 {train_metrics['macro_f1']:.4f}"
    )
    print(
        f"Validation -> accuracy {val_metrics['accuracy']:.4f} | "
        f"macro F1 {val_metrics['macro_f1']:.4f}"
    )

    model_path = save_model(model)
    size_mb = model_path.stat().st_size / 1e6
    print(f"Model tersimpan di {model_path} ({size_mb:.1f} MB)")

    figure_path = save_feature_importance_figure(get_feature_importance(model))
    print(f"Grafik feature importance tersimpan di {figure_path}")
    return model


if __name__ == "__main__":
    main()