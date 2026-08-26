"""
============================================================
DOWNSTREAM TASK CLASSIFICATION - DiffPuter vs MRMD (Baseline)
============================================================
Script ini menjalankan klasifikasi (Logistic Regression, SVM, XGBoost)
sebagai downstream task evaluation terhadap hasil imputasi data.

Skenario yang dibandingkan:
  1. diffputer  -> proposed method (train_impute_{i}.csv / test_impute_{i}.csv)
  2. mrmd       -> baseline (train_impute_mrmd_{i}.csv / test_impute_mrmd_{i}.csv)

Dataset : ADULT, SHOPPERS
Mask    : mask 0 - 9 (10 repetisi tiap skenario)

Untuk tiap kombinasi (dataset x skenario x mask):
  - Encoding fitur kategorikal & target (fit di train, transform ke test)
  - Cross-validation (StratifiedKFold) dilakukan HANYA di data training
  - Model di-fit ke seluruh training set, lalu dievaluasi ke test set terpisah
  - Metric: Accuracy, Precision, Recall, F1, ROC-AUC (average="binary")

Semua hasil (CV & test, per repetisi + ringkasan rata-rata) disimpan ke:
  - hasil_downstream_classification.xlsx  (beberapa sheet)
  - hasil_downstream_classification.txt   (log lengkap, human-readable)
"""

import os
import json
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)

# ============================================================
# 1. KONFIGURASI GLOBAL
# ============================================================

# ------------------------------------------------------------
# PATH PORTABLE — otomatis relatif terhadap lokasi script ini
# (PROJECT_ROOT), supaya script bisa dijalankan di komputer manapun
# tanpa edit path absolut (D:\..., C:\..., dst).
#
# SYARAT: taruh file script ini LANGSUNG DI DALAM folder "RESULT"
# (folder yang isinya ADULT/, SHOPPERS/, baselines/, datasets/, dst).
# Kalau lokasi script berbeda dari folder RESULT, ubah PROJECT_ROOT
# di bawah ini secara manual.
# ------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

# Folder root project (folder "RESULT"). Default: sama dengan lokasi script.
PROJECT_ROOT = SCRIPT_DIR

# Folder utama tempat file CSV hasil imputasi berada, per dataset.
# Struktur: {RESULT_BASE_DIR}/{DATASET_FOLDER}/MCAR/60/CSV/...
RESULT_BASE_DIR = PROJECT_ROOT

# Folder utama tempat baseline mean/modus (hyperimpute) berada.
# Struktur: {MEAN_MODE_BASE_DIR}/{dataset_name}/mask_{i}/mean_mode_train.csv
MEAN_MODE_BASE_DIR = PROJECT_ROOT / "baselines" / "imputed_csv"

# Folder tempat file info.json berada (kolom target per dataset)
INFO_DIR = PROJECT_ROOT / "datasets" / "info"

# Folder output tempat hasil (excel & txt) akan disimpan
OUTPUT_DIR = PROJECT_ROOT / "downstream_results"

# Daftar dataset yang dievaluasi: nama tampilan -> nama folder & nama file info.json
# "folder"      -> nama folder dataset untuk skenario diffputer & mrmd (di RESULT_BASE_DIR)
# "mean_mode_folder" -> nama folder dataset untuk skenario mean_mode (di MEAN_MODE_BASE_DIR)
DATASETS = {
    "adult": {
        "folder": "ADULT",
        "mean_mode_folder": "adult",
        "info_json": os.path.join(INFO_DIR, "adult.json"),
    },
    "shoppers": {
        "folder": "SHOPPERS",
        "mean_mode_folder": "shoppers",
        "info_json": os.path.join(INFO_DIR, "shoppers.json"),
    },
}


def path_diffputer(dataset_cfg, mask_idx):
    """train_impute_{i}.csv / test_impute_{i}.csv -> proposed method (DiffPuter)"""
    csv_dir = os.path.join(RESULT_BASE_DIR, dataset_cfg["folder"], "MCAR", "60", "CSV")
    train_path = os.path.join(csv_dir, f"train_impute_{mask_idx}.csv")
    test_path = os.path.join(csv_dir, f"test_impute_{mask_idx}.csv")
    return train_path, test_path


def path_mrmd(dataset_cfg, mask_idx):
    """train_impute_mrmd_{i}.csv / test_impute_mrmd_{i}.csv -> baseline MRMD"""
    csv_dir = os.path.join(RESULT_BASE_DIR, dataset_cfg["folder"], "MCAR", "60", "CSV")
    train_path = os.path.join(csv_dir, f"train_impute_mrmd_{mask_idx}.csv")
    test_path = os.path.join(csv_dir, f"test_impute_mrmd_{mask_idx}.csv")
    return train_path, test_path


def path_mean_mode(dataset_cfg, mask_idx):
    """{MEAN_MODE_BASE_DIR}\\{dataset}\\mask_{i}\\mean_mode_train.csv -> baseline Mean/Modus"""
    mask_dir = os.path.join(MEAN_MODE_BASE_DIR, dataset_cfg["mean_mode_folder"], f"mask_{mask_idx}")
    train_path = os.path.join(mask_dir, "mean_mode_train.csv")
    test_path = os.path.join(mask_dir, "mean_mode_test.csv")
    return train_path, test_path


# Skenario yang dibandingkan: tiap skenario punya fungsi path builder sendiri,
# karena struktur foldernya berbeda-beda (bukan cuma beda nama file).
SCENARIOS = {
    "diffputer": path_diffputer,   # proposed method
    "mrmd": path_mrmd,             # baseline 1
    "mean_mode": path_mean_mode,   # baseline 2
}

# Jumlah mask/repetisi (mask_0 ... mask_9 -> index 0-9)
N_MASKS = 10

RANDOM_STATE = 42
N_SPLITS_CV = 5

CV_SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",   # average="binary" (default)
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def make_classifiers():
    """Definisikan model dengan hyperparameter default. Dibuat baru tiap
    kombinasi supaya tidak ada state model yang 'bocor' antar file."""
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=RANDOM_STATE)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=RANDOM_STATE)),
        ]),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        ),
    }


# ============================================================
# 2. FUNGSI BANTU
# ============================================================

def get_target_column(info_json_path):
    """Ambil nama/idx kolom target dari file info.json."""
    with open(info_json_path, "r") as f:
        info = json.load(f)

    if "target_col" in info:
        return info["target_col"]
    if "target_col_idx" in info:
        idx_list = info["target_col_idx"]
        idx = idx_list[0] if isinstance(idx_list, list) else idx_list
        if "column_names" in info:
            return info["column_names"][idx]
        return idx

    raise ValueError(
        f"Tidak ada 'target_col'/'target_col_idx' di {info_json_path}. "
        f"Key tersedia: {list(info.keys())}"
    )


def load_train_test(train_path, test_path, target_col):
    """Load CSV train & test, pisahkan X dan y."""
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    col = target_col
    if isinstance(col, int):
        col = df_train.columns[col]

    if col not in df_train.columns:
        raise ValueError(
            f"Kolom target '{col}' tidak ditemukan di {train_path}. "
            f"Kolom tersedia: {list(df_train.columns)}"
        )

    y_train = df_train[col]
    X_train = df_train.drop(columns=[col])

    y_test = df_test[col]
    X_test = df_test.drop(columns=[col])

    return X_train, y_train, X_test, y_test


def encode_train_test(X_train, y_train, X_test, y_test):
    """Encode fitur kategorikal & target. Fit HANYA di train, transform ke train & test.
    Kategori baru di test yang tidak ada di train -> di-map ke -1 (fallback aman)."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in X_train.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        le.fit(X_train[col].astype(str))

        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = X_test[col].astype(str).map(
            lambda v, classes=le.classes_, le=le: le.transform([v])[0] if v in classes else -1
        )

    y_train = y_train.astype(str).str.strip()
    y_test = y_test.astype(str).str.strip()

    le_target = LabelEncoder()
    le_target.fit(y_train)
    y_train_enc = le_target.transform(y_train)
    y_test_enc = le_target.transform(y_test)

    return X_train, y_train_enc, X_test, y_test_enc, le_target


def run_cv_and_test(X_train, y_train, X_test, y_test, log_lines):
    """Jalankan CV di training set + evaluasi akhir di test set untuk 3 classifier.
    Mengembalikan list of dict (satu dict per classifier) berisi metric CV & test."""
    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    classifiers = make_classifiers()

    rows = []

    for name, clf in classifiers.items():
        log_lines.append(f"\n{'-'*40}\nModel: {name}\n{'-'*40}")

        # ---- Cross-validation di training set ----
        scores = cross_validate(clf, X_train, y_train, cv=cv, scoring=CV_SCORING, n_jobs=-1)
        cv_metrics = {
            metric: (np.mean(scores[f"test_{metric}"]), np.std(scores[f"test_{metric}"]))
            for metric in CV_SCORING
        }

        log_lines.append("CV (Training Set):")
        for metric, (mean_val, std_val) in cv_metrics.items():
            log_lines.append(f"  {metric:10s}: {mean_val:.4f} (+/- {std_val:.4f})")

        # ---- Fit ke seluruh training set, evaluasi ke test set ----
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        test_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="binary", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="binary", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="binary", zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        log_lines.append("Test Set:")
        for metric, val in test_metrics.items():
            log_lines.append(f"  {metric:10s}: {val:.4f}")

        log_lines.append("\nClassification Report:")
        log_lines.append(classification_report(y_test, y_pred, zero_division=0))
        log_lines.append("Confusion Matrix:")
        log_lines.append(str(confusion_matrix(y_test, y_pred)))

        row = {"classifier": name}
        for metric, (mean_val, std_val) in cv_metrics.items():
            row[f"cv_{metric}_mean"] = round(mean_val, 4)
            row[f"cv_{metric}_std"] = round(std_val, 4)
        for metric, val in test_metrics.items():
            row[f"test_{metric}"] = round(val, 4)

        rows.append(row)

    return rows


# ============================================================
# 3. LOOP UTAMA: dataset -> skenario -> mask -> classifier
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []      # detail per dataset-skenario-mask-classifier
    full_log = []         # log lengkap (classification report, confusion matrix, dst)
    skipped = []          # file yang tidak ditemukan

    full_log.append("=" * 70)
    full_log.append("LOG DOWNSTREAM TASK CLASSIFICATION")
    full_log.append(f"Dijalankan pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    full_log.append("=" * 70)

    for dataset_name, dataset_cfg in DATASETS.items():
        info_json_path = dataset_cfg["info_json"]

        try:
            target_col = get_target_column(info_json_path)
        except Exception as e:
            msg = f"[ERROR] Gagal ambil target_col untuk dataset '{dataset_name}': {e}"
            print(msg)
            full_log.append(msg)
            continue

        for scenario_name, path_builder in SCENARIOS.items():

            for mask_idx in range(N_MASKS):
                train_path, test_path = path_builder(dataset_cfg, mask_idx)

                header = (
                    f"\n{'='*70}\n"
                    f"DATASET: {dataset_name} | SKENARIO: {scenario_name} | MASK: {mask_idx}\n"
                    f"{'='*70}"
                )
                print(header)
                full_log.append(header)

                if not os.path.exists(train_path) or not os.path.exists(test_path):
                    msg = f"[SKIP] File tidak ditemukan:\n  train: {train_path}\n  test : {test_path}"
                    print(msg)
                    full_log.append(msg)
                    skipped.append({
                        "dataset": dataset_name, "scenario": scenario_name,
                        "mask": mask_idx, "train_path": train_path, "test_path": test_path,
                    })
                    continue

                try:
                    X_train, y_train, X_test, y_test = load_train_test(train_path, test_path, target_col)

                    full_log.append(f"Shape X_train: {X_train.shape}, X_test: {X_test.shape}")

                    X_train, y_train, X_test, y_test, le_target = encode_train_test(
                        X_train, y_train, X_test, y_test
                    )
                    full_log.append(f"Kelas target (encoded): {list(le_target.classes_)} -> {list(range(len(le_target.classes_)))}")

                    rows = run_cv_and_test(X_train, y_train, X_test, y_test, full_log)

                    for row in rows:
                        row.update({
                            "dataset": dataset_name,
                            "scenario": scenario_name,
                            "mask": mask_idx,
                        })
                        all_results.append(row)

                except Exception as e:
                    msg = f"[ERROR] Gagal proses dataset={dataset_name}, scenario={scenario_name}, mask={mask_idx}: {e}"
                    print(msg)
                    full_log.append(msg)
                    full_log.append(traceback.format_exc())
                    continue

    # ============================================================
    # 4. SUSUN HASIL: DETAIL & RINGKASAN
    # ============================================================

    results_df = pd.DataFrame(all_results)

    if results_df.empty:
        print("\n[PERINGATAN] Tidak ada hasil sama sekali — cek kembali path konfigurasi.")
        full_log.append("\n[PERINGATAN] Tidak ada hasil sama sekali — cek kembali path konfigurasi.")
    else:
        # urutkan kolom biar rapi
        front_cols = ["dataset", "scenario", "mask", "classifier"]
        metric_cols = [c for c in results_df.columns if c not in front_cols]
        results_df = results_df[front_cols + metric_cols]
        results_df = results_df.sort_values(["dataset", "scenario", "classifier", "mask"]).reset_index(drop=True)

        # ringkasan: rata-rata & std ANTAR MASK, per dataset-scenario-classifier
        agg_cols = [c for c in metric_cols]
        summary_df = (
            results_df
            .groupby(["dataset", "scenario", "classifier"])[agg_cols]
            .agg(["mean", "std"])
        )
        summary_df.columns = ["_".join(c) for c in summary_df.columns]
        summary_df = summary_df.reset_index().round(4)
        summary_df = summary_df.sort_values(["dataset", "classifier", "scenario"]).reset_index(drop=True)

    skipped_df = pd.DataFrame(skipped)

    # ============================================================
    # 5. SIMPAN KE EXCEL (multi-sheet) & TXT
    # ============================================================

    excel_path = os.path.join(OUTPUT_DIR, "hasil_downstream_classification.xlsx")
    txt_path = os.path.join(OUTPUT_DIR, "hasil_downstream_classification.txt")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        if not results_df.empty:
            results_df.to_excel(writer, sheet_name="detail_per_mask", index=False)
            summary_df.to_excel(writer, sheet_name="ringkasan_rata2_mask", index=False)
        else:
            pd.DataFrame({"info": ["Tidak ada hasil"]}).to_excel(writer, sheet_name="detail_per_mask", index=False)

        if not skipped_df.empty:
            skipped_df.to_excel(writer, sheet_name="file_terlewat", index=False)

    print(f"\n[INFO] Hasil Excel disimpan ke: {excel_path}")
    full_log.append(f"\n[INFO] Hasil Excel disimpan ke: {excel_path}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(full_log))
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("RINGKASAN AKHIR (rata-rata & std antar mask)\n")
        f.write("=" * 70 + "\n")
        if not results_df.empty:
            f.write(summary_df.to_string(index=False))
        else:
            f.write("Tidak ada hasil.")
        if not skipped_df.empty:
            f.write("\n\n" + "=" * 70 + "\n")
            f.write("FILE YANG TIDAK DITEMUKAN (di-skip)\n")
            f.write("=" * 70 + "\n")
            f.write(skipped_df.to_string(index=False))

    print(f"[INFO] Log lengkap (txt) disimpan ke: {txt_path}")

    # tampilkan ringkasan singkat di console
    if not results_df.empty:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        print("\n" + "=" * 70)
        print("RINGKASAN AKHIR (rata-rata & std antar mask)")
        print("=" * 70)
        print(summary_df.to_string(index=False))

    if skipped:
        print(f"\n[PERINGATAN] {len(skipped)} kombinasi file dilewati karena tidak ditemukan. Lihat sheet 'file_terlewat' / txt log.")


if __name__ == "__main__":
    main()