"""
分类融合版 —— 四分类 + 回归平滑
================================
- 将 dietaE 分为 4 个区间进行多分类预测
- 基模型：LGB, XGB, CatBoost, RF
- 类别概率 Stacking，取众数中位数作为基础预测
- 再用轻量 LGB 回归修正，最终加权平均
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

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
    train_samples = list(df_train["sample"].unique())

    X_all = build_features(df_train, rate_map, global_mean, train_samples)
    y_all = df_train[TARGET_COL].values.astype(float)
    groups = df_train["sample"].astype(str).values
    feature_names = list(X_all.columns)
    constraints = get_monotone_constraints(feature_names)
    print(f"[FEATURES] 数量: {len(feature_names)}")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_all_t = imputer.fit_transform(X_all)
    X_all_s = scaler.fit_transform(X_all_t)

    # ---- 构建分类标签 ----
    y_quantiles = np.percentile(y_all, [25, 50, 75])
    print(f"[分箱] 阈值: 25%={y_quantiles[0]:.2f}, 50%={y_quantiles[1]:.2f}, 75%={y_quantiles[2]:.2f}")
    y_class = np.digitize(y_all, bins=y_quantiles)  # 0,1,2,3
    class_centers = [np.median(y_all[y_class == c]) for c in range(4)]
    print(f"[分箱] 类别中位数: {class_centers}")

    # ---- GroupKFold OOF 分类概率 ----
    gkf = GroupKFold(n_splits=5)
    oof_prob_lgb = np.zeros((len(y_all), 4))
    oof_prob_xgb = np.zeros((len(y_all), 4))
    oof_prob_cat = np.zeros((len(y_all), 4))
    oof_prob_rf  = np.zeros((len(y_all), 4))

    for fold, (train_idx, valid_idx) in enumerate(gkf.split(X_all_s, y_class, groups=groups)):
        print(f"\n--- Fold {fold+1} ---")
        X_tr, X_val = X_all_s[train_idx], X_all_s[valid_idx]
        y_tr = y_class[train_idx]

        # LightGBM 分类
        print("  [LGB] 分类...")
        lgb = LGBMClassifier(
            n_estimators=500, max_depth=7, learning_rate=0.03,
            num_leaves=80, subsample=0.85, colsample_bytree=0.85,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
        lgb.fit(X_tr, y_tr)
        oof_prob_lgb[valid_idx] = lgb.predict_proba(X_val)

        # XGBoost 分类
        print("  [XGB] 分类...")
        xgb = XGBClassifier(
            n_estimators=500, max_depth=7, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
        )
        xgb.fit(X_tr, y_tr)
        oof_prob_xgb[valid_idx] = xgb.predict_proba(X_val)

        # CatBoost 分类
        print("  [Cat] 分类...")
        cat = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.03,
            random_seed=RANDOM_STATE, thread_count=-1, silent=True
        )
        cat.fit(X_tr, y_tr)
        oof_prob_cat[valid_idx] = cat.predict_proba(X_val)

        # RandomForest 分类
        print("  [RF] 分类...")
        rf = RandomForestClassifier(
            n_estimators=400, max_depth=12,
            random_state=RANDOM_STATE, n_jobs=-1
        )
        rf.fit(X_tr, y_tr)
        oof_prob_rf[valid_idx] = rf.predict_proba(X_val)

    # 元模型融合概率
    meta_models = []
    oof_class_pred = np.zeros(len(y_all))
    for c in range(4):
        X_meta_c = np.column_stack([
            oof_prob_lgb[:, c], oof_prob_xgb[:, c], oof_prob_cat[:, c], oof_prob_rf[:, c]
        ])
        meta = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
        meta.fit(X_meta_c, (y_class == c).astype(int))
        meta_models.append(meta)
        oof_class_pred += meta.predict(X_meta_c)  # 加权概率求和

    # 取最大概率类别
    oof_pred_class_probs = np.column_stack([m.predict(X_meta_c) for m, X_meta_c in
        [(meta_models[c], np.column_stack([oof_prob_lgb[:, c], oof_prob_xgb[:, c], oof_prob_cat[:, c], oof_prob_rf[:, c]])) for c in range(4)]])
    oof_class = np.argmax(oof_pred_class_probs, axis=1)
    oof_reg_pred = np.array([class_centers[cl] for cl in oof_class])
    mae_class = np.mean(np.abs(y_all - oof_reg_pred))
    print(f"\n[分类基] MAE: {mae_class:.4f}")

    # ---- 轻量回归修正 ----
    print("\n[修正] 训练轻量回归模型...")
    lgb_reg = LGBMRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.03,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    lgb_reg.fit(X_all_s, np.log1p(y_all))
    oof_reg_log = lgb_reg.predict(X_all_s)
    oof_reg = np.expm1(oof_reg_log)

    # 加权融合
    final_alpha = 0.5
    oof_final = final_alpha * oof_reg_pred + (1 - final_alpha) * oof_reg
    mae_final = np.mean(np.abs(y_all - oof_final))
    print(f"[融合] 分类+回归 加权 OOF MAE: {mae_final:.4f}")

    # ---- 全量训练基分类器与回归器，预测测试集 ----
    print("\n[全量] 训练分类器与回归器...")
    X_test = build_features(df_test, rate_map, global_mean, train_samples)
    X_test_final = X_test[feature_names]
    X_te_t = imputer.transform(X_test_final)
    X_te_s = scaler.transform(X_te_t)

    # 全量分类器
    lgb_full_clf = LGBMClassifier(n_estimators=800, max_depth=8, learning_rate=0.02,
                                  num_leaves=100, subsample=0.9, colsample_bytree=0.9,
                                  random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    lgb_full_clf.fit(X_all_s, y_class)
    prob_lgb_test = lgb_full_clf.predict_proba(X_te_s)

    xgb_full_clf = XGBClassifier(n_estimators=800, max_depth=8, learning_rate=0.02,
                                 subsample=0.9, colsample_bytree=0.9,
                                 random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    xgb_full_clf.fit(X_all_s, y_class)
    prob_xgb_test = xgb_full_clf.predict_proba(X_te_s)

    cat_full_clf = CatBoostClassifier(iterations=800, depth=8, learning_rate=0.02,
                                      random_seed=RANDOM_STATE, thread_count=-1, silent=True)
    cat_full_clf.fit(X_all_s, y_class)
    prob_cat_test = cat_full_clf.predict_proba(X_te_s)

    rf_full_clf = RandomForestClassifier(n_estimators=500, max_depth=12,
                                         random_state=RANDOM_STATE, n_jobs=-1)
    rf_full_clf.fit(X_all_s, y_class)
    prob_rf_test = rf_full_clf.predict_proba(X_te_s)

    # 元模型融合测试集概率
    final_probs_test = np.zeros((len(X_test), 4))
    for c in range(4):
        X_meta_test_c = np.column_stack([prob_lgb_test[:, c], prob_xgb_test[:, c],
                                         prob_cat_test[:, c], prob_rf_test[:, c]])
        final_probs_test[:, c] = meta_models[c].predict(X_meta_test_c)

    test_class = np.argmax(final_probs_test, axis=1)
    test_reg_pred = np.array([class_centers[cl] for cl in test_class])

    # 全量回归修正模型
    lgb_reg_full = LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.02,
        num_leaves=60, subsample=0.85, colsample_bytree=0.85,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    lgb_reg_full.fit(X_all_s, np.log1p(y_all))
    test_reg = np.expm1(lgb_reg_full.predict(X_te_s))

    final_pred = final_alpha * test_reg_pred + (1 - final_alpha) * test_reg
    final_pred[df_test["aging_time_day"] == 0] = 0.0
    final_pred = np.clip(final_pred, 0, None)

    pred_df = pd.DataFrame({TARGET_COL: final_pred})
    pred_df.to_csv(PRED_OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n[OK] 分类融合预测已保存: {PRED_OUTPUT_CSV}")
    print(pred_df.describe())
