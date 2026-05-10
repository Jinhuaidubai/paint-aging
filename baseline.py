"""
终极完全体 —— 五模型两层 Stacking + 多输出融合
===============================================
基模型：LGB, XGB, CatBoost, RF, ET
内层：5 折 GroupKFold 生成 OOF
外层：RidgeCV 自动学习融合权重（对 log1p 目标）
多输出：用 LGB 单独预训练 ΔL,Δa,Δb，计算物理 ΔE_calc
最终：Stacking 集成 ΔE 与物理 ΔE 按 0.7:0.3 融合
要求：所有模型可正常训练（无崩溃）
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

RANDOM_STATE = 42
TRAIN_CSV_PATH = "paint_aging_trainset.csv"
TEST_CSV_PATH  = "paint_aging_testset.csv"
PRED_OUTPUT_CSV = "paint_pred.csv"
TARGET_COL = "dietaE"


def load_raw():
    df_train = pd.read_csv(TRAIN_CSV_PATH, encoding="utf-8")
    df_test  = pd.read_csv(TEST_CSV_PATH, encoding="utf-8")
    df_train = df_train.dropna(axis=1, how="all")
    df_test  = df_test.dropna(axis=1, how="all")
    return df_train, df_test


def fit_aging_rate(df_train):
    rates = {}
    for (samp, cond), grp in df_train.groupby(["sample", "aging_condition"]):
        grp = grp.sort_values("aging_time_day")
        t = grp["aging_time_day"].values.astype(float)
        y = grp[TARGET_COL].values.astype(float)
        if len(t) < 2 or t.max() == 0:
            continue
        try:
            (k,), _ = curve_fit(lambda t_, k_: k_ * np.sqrt(t_), t, y, p0=[1.0])
            rates[f"{samp}|{cond}"] = float(k) if np.isfinite(k) else np.nan
        except:
            pass
    return rates


def build_features(df, aging_rate_map, global_mean_rate, train_samples_list=None):
    feat = pd.DataFrame(index=df.index)
    feat["uv_stress"]    = np.where(df["aging_condition"] == "UV", 1.0, 0.1)
    feat["humid_stress"] = np.where(df["aging_condition"] == "humid-_heat", 1.0, 0.2)
    feat["stress_ratio"] = feat["uv_stress"] / (feat["humid_stress"] + 0.01)

    t = df["aging_time_day"].astype(float)
    feat["aging_time_day"] = t
    feat["log1p_time"]     = np.log1p(t)
    feat["sqrt_time"]      = np.sqrt(t)
    feat["time_sq"]        = t ** 2

    L0, a0, b0 = df["L0"].astype(float), df["a0"].astype(float), df["b0"].astype(float)
    feat["L0"], feat["a0"], feat["b0"] = L0, a0, b0
    feat["chroma0"]        = np.sqrt(a0**2 + b0**2)
    feat["hue0"]           = np.arctan2(b0, a0)
    feat["sin_hue"]        = np.sin(feat["hue0"])
    feat["cos_hue"]        = np.cos(feat["hue0"])
    feat["lightness_norm"] = L0 / 100.0

    feat["UV_x_sqrt_t"]      = feat["uv_stress"] * feat["sqrt_time"]
    feat["chroma_x_sqrt_t"]  = feat["chroma0"] * feat["sqrt_time"]
    feat["L0_x_sqrt_t"]      = L0 * feat["sqrt_time"]

    keys = df["sample"].astype(str) + "|" + df["aging_condition"].astype(str)
    feat["aging_rate_prior"] = keys.map(aging_rate_map).fillna(global_mean_rate)
    feat["rate_x_sqrt_t"]    = feat["aging_rate_prior"] * feat["sqrt_time"]

    if train_samples_list is not None:
        for s_name in train_samples_list:
            feat[f"s_{s_name}"] = (df["sample"] == s_name).astype(int)

    return feat


def get_monotone_constraints(feature_names):
    core_time = {"aging_time_day", "log1p_time", "sqrt_time"}
    return [1 if name in core_time else 0 for name in feature_names]


if __name__ == "__main__":
    df_train, df_test = load_raw()
    rate_map = fit_aging_rate(df_train)
    global_mean = np.nanmean(list(rate_map.values())) if rate_map else 0.0
    print(f"[PRIOR] 老化速率: {len(rate_map)} 组, 全局均值={global_mean:.4f}")
    train_samples = list(df_train["sample"].unique())

    X_all = build_features(df_train, rate_map, global_mean, train_samples)
    y_all = df_train[TARGET_COL].values.astype(float)
    y_dL = df_train["L"].values - df_train["L0"].values
    y_da = df_train["a"].values - df_train["a0"].values
    y_db = df_train["b"].values - df_train["b0"].values
    groups = df_train["sample"].astype(str).values
    feature_names = list(X_all.columns)
    constraints = get_monotone_constraints(feature_names)
    print(f"[FEATURES] 数量: {len(feature_names)}")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_all_t = imputer.fit_transform(X_all)
    X_all_s = scaler.fit_transform(X_all_t)
    y_train_log = np.log1p(y_all)

    # ---------- 内层 GroupKFold 生成五模型 OOF ----------
    gkf = GroupKFold(n_splits=5)
    oof_lgb = np.zeros(len(y_all))
    oof_xgb = np.zeros(len(y_all))
    oof_cat = np.zeros(len(y_all))
    oof_rf  = np.zeros(len(y_all))
    oof_et  = np.zeros(len(y_all))

    for fold, (train_idx, valid_idx) in enumerate(gkf.split(X_all_s, y_train_log, groups=groups)):
        print(f"\n--- Fold {fold+1} ---")
        X_tr, X_val = X_all_s[train_idx], X_all_s[valid_idx]
        y_tr = y_train_log[train_idx]

        # LightGBM (带单调性约束)
        print("  [LGB] 训练...")
        lgb = LGBMRegressor(
            n_estimators=500, max_depth=7, learning_rate=0.03,
            num_leaves=80, subsample=0.85, colsample_bytree=0.85,
            objective='huber', alpha=0.9,
            monotone_constraints=constraints,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
        lgb.fit(X_tr, y_tr)
        oof_lgb[valid_idx] = lgb.predict(X_val)

        # XGBoost (不带单调性约束，兼容性好)
        print("  [XGB] 训练...")
        xgb = XGBRegressor(
            n_estimators=500, max_depth=7, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
        )
        xgb.fit(X_tr, y_tr)
        oof_xgb[valid_idx] = xgb.predict(X_val)

        # CatBoost
        print("  [Cat] 训练...")
        cat = CatBoostRegressor(
            iterations=500, depth=6, learning_rate=0.03,
            random_seed=RANDOM_STATE, thread_count=-1, silent=True
        )
        cat.fit(X_tr, y_tr)
        oof_cat[valid_idx] = cat.predict(X_val)

        # RandomForest
        print("  [RF] 训练...")
        rf = RandomForestRegressor(
            n_estimators=400, max_depth=12,
            random_state=RANDOM_STATE, n_jobs=-1
        )
        rf.fit(X_tr, y_tr)
        oof_rf[valid_idx] = rf.predict(X_val)

        # ExtraTrees
        print("  [ET] 训练...")
        et = ExtraTreesRegressor(
            n_estimators=400, max_depth=12,
            random_state=RANDOM_STATE, n_jobs=-1
        )
        et.fit(X_tr, y_tr)
        oof_et[valid_idx] = et.predict(X_val)

    # 元模型训练（RidgeCV）
    X_meta = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_rf, oof_et])
    meta = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    meta.fit(X_meta, y_train_log)
    oof_meta_log = meta.predict(X_meta)
    oof_meta = np.clip(np.expm1(oof_meta_log), 0, None)
    mae = mean_absolute_error(y_all, oof_meta)
    print(f"\n[Meta] 系数: LGB={meta.coef_[0]:.3f}, XGB={meta.coef_[1]:.3f}, Cat={meta.coef_[2]:.3f}, RF={meta.coef_[3]:.3f}, ET={meta.coef_[4]:.3f}")
    print(f"[OOF] Stacking 集成 MAE: {mae:.4f}")

    # ---------- 多输出分量模型（仅 LightGBM） ----------
    print("\n[分量] 训练 ΔL, Δa, Δb 模型...")
    lgb_L = LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        num_leaves=60, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    lgb_L.fit(X_all_s, y_dL)
    pred_L_test = lgb_L.predict(X_all_s[:1])  # 先占位，后面预测测试集会用全量

    lgb_a = LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        num_leaves=60, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    lgb_a.fit(X_all_s, y_da)

    lgb_b = LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        num_leaves=60, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    lgb_b.fit(X_all_s, y_db)

    # ---------- 全量训练基模型 ----------
    print("\n[全量训练] 五模型...")
    # 测试集特征
    X_test = build_features(df_test, rate_map, global_mean, train_samples)
    X_test_final = X_test[feature_names]
    X_te_t = imputer.transform(X_test_final)
    X_te_s = scaler.transform(X_te_t)

    lgb_full = LGBMRegressor(
        n_estimators=800, max_depth=8, learning_rate=0.02,
        num_leaves=100, subsample=0.9, colsample_bytree=0.9,
        objective='huber', alpha=0.9,
        monotone_constraints=constraints,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    lgb_full.fit(X_all_s, y_train_log)
    pred_lgb_test = lgb_full.predict(X_te_s)

    xgb_full = XGBRegressor(
        n_estimators=800, max_depth=8, learning_rate=0.02,
        subsample=0.9, colsample_bytree=0.9,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
    )
    xgb_full.fit(X_all_s, y_train_log)
    pred_xgb_test = xgb_full.predict(X_te_s)

    cat_full = CatBoostRegressor(
        iterations=800, depth=8, learning_rate=0.02,
        random_seed=RANDOM_STATE, thread_count=-1, silent=True
    )
    cat_full.fit(X_all_s, y_train_log)
    pred_cat_test = cat_full.predict(X_te_s)

    rf_full = RandomForestRegressor(
        n_estimators=500, max_depth=12,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_full.fit(X_all_s, y_train_log)
    pred_rf_test = rf_full.predict(X_te_s)

    et_full = ExtraTreesRegressor(
        n_estimators=500, max_depth=12,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    et_full.fit(X_all_s, y_train_log)
    pred_et_test = et_full.predict(X_te_s)

    # Stacking 集成预测（测试集）
    X_meta_test = np.column_stack([pred_lgb_test, pred_xgb_test, pred_cat_test, pred_rf_test, pred_et_test])
    pred_stacking_log = meta.predict(X_meta_test)
    pred_stacking = np.expm1(pred_stacking_log)

    # 分量预测并计算物理 ΔE
    pred_L_test = lgb_L.predict(X_te_s)
    pred_a_test = lgb_a.predict(X_te_s)
    pred_b_test = lgb_b.predict(X_te_s)
    pred_E_calc = np.sqrt(pred_L_test**2 + pred_a_test**2 + pred_b_test**2)

    # 最终融合：0.7 Stacking + 0.3 物理
    final_pred = 0.7 * pred_stacking + 0.3 * pred_E_calc
    final_pred[df_test["aging_time_day"] == 0] = 0.0
    final_pred = np.clip(final_pred, 0, None)

    # 保存
    pred_df = pd.DataFrame({TARGET_COL: final_pred})
    pred_df.to_csv(PRED_OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n[OK] 终极预测已保存: {PRED_OUTPUT_CSV}")
    print(pred_df.describe())
