import ast
import os
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
import rasterio
try:
    from qgis.PyQt.QtCore import QVariant
    from qgis.core import (
        QgsCoordinateReferenceSystem,
        QgsRasterLayer,
        QgsCoordinateTransform,
        QgsProject,
        QgsPointXY,
    )
except ImportError:
    QVariant = None
    QgsCoordinateReferenceSystem = None
    QgsRasterLayer = None
    QgsCoordinateTransform = None
    QgsProject = None
    QgsPointXY = None

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge, HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class CustomEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimators, method="Average"):
        self.estimators = estimators  # List of (name, model)
        self.method = method
        self.fitted_estimators_ = []
        self.uncert_models_ = []
        self.meta_learner_ = None

    def fit(self, X, y, sample_weight=None):
        self.fitted_estimators_ = []
        self.uncert_models_ = []
        
        for name, est in self.estimators:
            model_to_fit = clone(est)
            fit_params = {}
            if sample_weight is not None:
                import inspect
                sig = inspect.signature(model_to_fit.fit)
                if "sample_weight" in sig.parameters:
                    fit_params["sample_weight"] = sample_weight
            model_to_fit.fit(X, y, **fit_params)
            self.fitted_estimators_.append(model_to_fit)

            # Train a light residual uncertainty model for each estimator if Uncertainty-Weighted Fusion is active
            if self.method in ["Uncertainty-Weighted Fusion", "Uncertainty-Weighted"]:
                y_pred = model_to_fit.predict(X)
                abs_residuals = np.abs(y - y_pred) * 1.96
                from sklearn.ensemble import RandomForestRegressor
                uncert_model = RandomForestRegressor(
                    n_estimators=30, max_depth=5, random_state=42, n_jobs=-1
                )
                uncert_model.fit(X, abs_residuals)
                self.uncert_models_.append(uncert_model)

        if self.method == "Stacking":
            base_preds = np.column_stack([est.predict(X) for est in self.fitted_estimators_])
            self.meta_learner_ = Ridge(alpha=1.0)
            self.meta_learner_.fit(base_preds, y, sample_weight=sample_weight)
        
        return self

    def predict(self, X):
        if not self.fitted_estimators_:
            raise ValueError("Ensemble is not fitted yet.")
        
        preds = np.column_stack([est.predict(X) for est in self.fitted_estimators_])

        if self.method == "Average":
            return np.mean(preds, axis=1)
        elif self.method == "Median":
            return np.median(preds, axis=1)
        elif self.method == "Stacking":
            if self.meta_learner_ is None:
                raise ValueError("Meta learner is not fitted.")
            return self.meta_learner_.predict(preds)
        elif self.method in ["Uncertainty-Weighted Fusion", "Uncertainty-Weighted"]:
            if not self.uncert_models_:
                return np.mean(preds, axis=1)
            
            sigmas = np.column_stack([u_mod.predict(X) for u_mod in self.uncert_models_])
            weights = 1.0 / (np.square(sigmas) + 1e-4)
            weights_sum = np.sum(weights, axis=1, keepdims=True)
            weighted_z = np.sum(preds * weights, axis=1, keepdims=True) / weights_sum
            return weighted_z.flatten()

        return np.mean(preds, axis=1)

try:
    from qgis.core import (
        QgsCoordinateTransform,
        QgsFeature,
        QgsField,
        QgsFields,
        QgsGeometry,
        QgsPointXY,
        QgsProcessingException,
        QgsProject,
        QgsRasterLayer,
        QgsVectorFileWriter,
        QgsWkbTypes,
    )
except ImportError:
    QgsFields = None
    QgsGeometry = None
    QgsProcessingException = Exception
    QgsVectorFileWriter = None
    QgsWkbTypes = None


try:
    from ...infrastructure.logging import append_log
except (ImportError, ValueError):
    from infrastructure.logging import append_log


try:
    from skopt import BayesSearchCV
    from skopt.space import Categorical, Integer, Real

    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

try:
    from scipy.ndimage import median_filter

    scipy_is_available = True
except ImportError:
    scipy_is_available = False

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

OPTIMIZER_LIST = ["Random Search", "Grid Search", "Bayesian Search"]


def parse_param_string(param_str):
    if not param_str:
        return {}
    try:
        return ast.literal_eval("{" + param_str + "}")
    except Exception:
        return {}


def convert_to_bayes(params_dict):
    bayes_params = {}
    for k, v in params_dict.items():
        if isinstance(v, list) and len(v) > 0:
            if len(v) == 1:
                bayes_params[k] = Categorical(v)
            elif all(isinstance(x, int) for x in v):
                if min(v) == max(v):
                    bayes_params[k] = Categorical(v)
                else:
                    bayes_params[k] = Integer(min(v), max(v))
            elif all(isinstance(x, (int, float)) for x in v):
                if min(v) == max(v):
                    bayes_params[k] = Categorical(v)
                else:
                    bayes_params[k] = Real(min(v), max(v))
            else:
                bayes_params[k] = Categorical(v)
        else:
            bayes_params[k] = Categorical([v])
    return bayes_params


def save_training_points(out_path, coords, depths, weights, X_data, ref_raster, crs, feature_names=None):
    fields = QgsFields()
    fields.append(QgsField("Depth_Used", QVariant.Double))
    fields.append(QgsField("Weight_Used", QVariant.Double))
    fields.append(QgsField("Row_Idx", QVariant.Int))
    fields.append(QgsField("Col_Idx", QVariant.Int))

    num_bands = X_data.shape[1]
    for b in range(num_bands):
        name = feature_names[b] if feature_names and b < len(feature_names) else f"Band_{b + 1}"
        fields.append(QgsField(name[:10], QVariant.Double))

    writer = QgsVectorFileWriter(
        out_path, "UTF-8", fields, QgsWkbTypes.Point, crs, "ESRI Shapefile"
    )
    if writer.hasError() != QgsVectorFileWriter.NoError:
        return

    with rasterio.open(ref_raster) as src:
        transform = src.transform
        for i, (r, c) in enumerate(coords):
            x, y = rasterio.transform.xy(transform, r, c, offset="center")
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))

            attrs = [float(depths[i]), float(weights[i]), int(r), int(c)]
            band_values = [float(val) for val in X_data[i]]
            attrs.extend(band_values)

            feat.setAttributes(attrs)
            writer.addFeature(feat)
    del writer


def _get_pixel_coords(geom, src, v_crs, r_crs):
    raw_pt = geom.asMultiPoint()[0] if geom.isMultipart() else geom.asPoint()
    h, w = src.height, src.width
    try:
        r_raw, c_raw = src.index(raw_pt.x(), raw_pt.y())
        if 0 <= r_raw < h and 0 <= c_raw < w:
            return r_raw, c_raw, raw_pt
    except Exception:
        pass

    try:
        tr = QgsCoordinateTransform(v_crs, r_crs, QgsProject.instance())
        geom_trans = QgsGeometry(geom)
        geom_trans.transform(tr)
        trans_pt = geom_trans.asMultiPoint()[0] if geom_trans.isMultipart() else geom_trans.asPoint()
        r_trans, c_trans = src.index(trans_pt.x(), trans_pt.y())
        if 0 <= r_trans < h and 0 <= c_trans < w:
            return r_trans, c_trans, trans_pt
    except Exception:
        pass

    return None, None, None


def extract_samples(ras_path, vec_layer, d_fld, w_fld, mode):
    rlayer = QgsRasterLayer(ras_path)
    
    if isinstance(vec_layer, str):
        from qgis.core import QgsVectorLayer
        vec_layer = QgsVectorLayer(vec_layer, "pts", "ogr")
    
    v_crs = vec_layer.crs() if vec_layer and vec_layer.crs().isValid() else QgsCoordinateReferenceSystem("EPSG:4326")
    r_crs = rlayer.crs() if rlayer and rlayer.crs().isValid() else QgsCoordinateReferenceSystem("EPSG:4326")

    # Auto-detect if QGIS incorrectly assigned the raster's projected CRS to a Lat/Long file
    if v_crs == r_crs and not r_crs.isGeographic():
        bbox = vec_layer.extent()
        if bbox.xMinimum() >= -180 and bbox.xMaximum() <= 180 and bbox.yMinimum() >= -90 and bbox.yMaximum() <= 90:
            v_crs = QgsCoordinateReferenceSystem("EPSG:4326")
            vec_layer.setCrs(v_crs)
    
    fields = [f.name() for f in vec_layer.fields()]
    actual_d_fld = None
    if d_fld:
        d_clean = str(d_fld).strip()
        for f in fields:
            if f == d_clean or f.lower() == d_clean.lower() or f.lower() == d_clean.lower()[:10]:
                actual_d_fld = f
                break
    if not actual_d_fld:
        from qgis.core import QgsProcessingException
        raise QgsProcessingException(f"❌ ERROR: Specified depth field '{d_fld}' was not found in layer '{vec_layer.name()}'. Available fields in the layer are: {', '.join(fields)}. Please specify the exact depth field name.")
    
    fields_lower = [f.lower() for f in fields]
    actual_w_fld = None
    if w_fld and w_fld.lower() in fields_lower:
        actual_w_fld = [f.name() for f in vec_layer.fields() if f.name().lower() == w_fld.lower()][0]
    else:
        for fallback in ['confidence', 'sdb_uncert', 'weight', 'uncertainty']:
            if fallback in fields_lower:
                actual_w_fld = [f.name() for f in vec_layer.fields() if f.name().lower() == fallback][0]
                break
                
    X_out, y_out, w_out, c_out = [], [], [], []

    with rasterio.open(ras_path) as src:
        d = src.read()
        h, w = src.height, src.width
        rst_transform = src.transform
        pixel_size = abs(src.res[0])

        if mode == 0:
            for f in vec_layer.getFeatures():
                geom = QgsGeometry(f.geometry())
                r, c, pt = _get_pixel_coords(geom, src, v_crs, r_crs)
                if r is not None and c is not None:
                    val = d[:, r, c]
                    if np.all(np.isfinite(val)) and not np.any(val == -9999):
                        X_out.append(val)
                        y_out.append(f[actual_d_fld])
                        c_out.append([r, c])
                        w_out.append(f[actual_w_fld] if actual_w_fld else 1.0)
                    else:
                        from qgis.core import QgsMessageLog, Qgis
                        QgsMessageLog.logMessage(
                            f"Point dropped at [r={r}, c={c}]. val: {val.tolist()}", 
                            "Bathymetrix_AI_Debug", Qgis.Warning)
                else:
                    from qgis.core import QgsMessageLog, Qgis
                    QgsMessageLog.logMessage(
                        f"Point out of bounds or transformation failed. Geom: {geom.asWkt()}", 
                        "Bathymetrix_AI_Debug", Qgis.Warning)
            return np.array(X_out), np.array(y_out), np.array(w_out), c_out

        pixel_registry = {}
        for f in vec_layer.getFeatures():
            geom = QgsGeometry(f.geometry())
            r, c, pt = _get_pixel_coords(geom, src, v_crs, r_crs)
            if r is not None and c is not None:
                pixel_registry.setdefault((r, c), []).append(
                    {"d": f[actual_d_fld], "w": f[actual_w_fld] if actual_w_fld else 1.0, "pt": pt}
                )

        for (r, c), items in pixel_registry.items():
            val = d[:, r, c]
            if not np.all(np.isfinite(val)) or np.any(val == -9999):
                from qgis.core import QgsMessageLog, Qgis
                QgsMessageLog.logMessage(
                    f"Point dropped at [r={r}, c={c}]. val: {val.tolist()}", 
                    "Bathymetrix_AI_Debug", Qgis.Warning)
                continue

            final_depth, final_weight = 0.0, 1.0
            if mode == 1:
                best = sorted(items, key=lambda x: x["w"], reverse=True)[0]
                final_depth, final_weight = best["d"], best["w"]
            elif mode == 2:
                cx, cy = rasterio.transform.xy(rst_transform, r, c, offset="center")
                best = min(
                    items,
                    key=lambda x: (x["pt"].x() - cx) ** 2 + (x["pt"].y() - cy) ** 2,
                )
                final_depth, final_weight = best["d"], best["w"]
            elif mode == 3:
                cx, cy = rasterio.transform.xy(rst_transform, r, c, offset="center")
                close = [
                    i
                    for i in items
                    if ((i["pt"].x() - cx) ** 2 + (i["pt"].y() - cy) ** 2) ** 0.5 <= pixel_size / 2
                ]
                target = close if close else items
                vals = np.array([i["d"] for i in target])
                ws = np.array([i["w"] for i in target])
                final_depth = (
                    np.average(vals, weights=ws) if np.sum(ws) > 0 else np.mean(vals)
                )
                final_weight = np.mean(ws)
            elif mode == 4:
                cx, cy = rasterio.transform.xy(rst_transform, r, c, offset="center")
                close = [
                    i
                    for i in items
                    if ((i["pt"].x() - cx) ** 2 + (i["pt"].y() - cy) ** 2) ** 0.5 <= pixel_size / 2
                ]
                if not close:
                    continue
                vals = np.array([i["d"] for i in close])
                ws = np.array([i["w"] for i in close])
                final_depth = (
                    np.average(vals, weights=ws) if np.sum(ws) > 0 else np.mean(vals)
                )
                final_weight = np.mean(ws)

            X_out.append(val)
            y_out.append(final_depth)
            w_out.append(final_weight)
            c_out.append([r, c])

    return np.array(X_out), np.array(y_out), np.array(w_out), c_out


def save_algo_artifacts(y_t, y_p, pct, name, folder, r2, rmse, mape, params, scaler_name="None"):
    with open(os.path.join(folder, "Results.txt"), "w") as f:
        f.write(
            f"Algo: {name}\nFeature Scaling: {scaler_name}\nR2: {r2:.4f}\nRMSE: {rmse:.4f}\nwMAPE: {mape:.2f}%\nParams: {params}"
        )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(8, 6))
        # Plot predictions vs actuals
        plt.scatter(y_t, y_p, color='#1f77b4', alpha=0.6, edgecolors='none', s=20, label='Validation Points')
        
        # Perfect fit line
        min_val = min(float(np.min(y_t)), float(np.min(y_p)))
        max_val = max(float(np.max(y_t)), float(np.max(y_p)))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')
        
        plt.title(f'{name} - Validation Scatter Plot\nR² = {r2:.3f} | RMSE = {rmse:.2f}m | wMAPE = {mape:.2f}%', fontsize=11, fontweight='bold', pad=10)
        plt.xlabel('Observed Depth (m)', fontsize=10)
        plt.ylabel('Predicted Depth (m)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(folder, "Validation_Scatter_Plot.png"), dpi=150)
        plt.close()
    except Exception:  # nosec B110
        pass


def get_model_and_params(index, opt_idx=0, random_state=42, n_jobs=-1):
    is_bayes = opt_idx == 2 and SKOPT_AVAILABLE
    if index == 0:
        return "Linear Regression", LinearRegression(), {}
    if index == 1:
        return (
            "Random Forest",
            RandomForestRegressor(random_state=random_state, n_jobs=n_jobs),
            (
                {"n_estimators": Integer(100, 500)}
                if is_bayes
                else {"n_estimators": [100, 500]}
            ),
        )
    if index == 2:
        return (
            "Gradient Boosting",
            GradientBoostingRegressor(random_state=random_state),
            (
                {"learning_rate": Real(0.01, 0.2)}
                if is_bayes
                else {"learning_rate": [0.05, 0.1]}
            ),
        )
    if index == 3:
        return (
            "Extra Trees",
            ExtraTreesRegressor(random_state=random_state, n_jobs=n_jobs),
            (
                {"n_estimators": Integer(100, 500)}
                if is_bayes
                else {"n_estimators": [100, 500]}
            ),
        )
    if index == 4:
        return (
            "Ridge",
            Ridge(),
            ({"alpha": Real(0.1, 10.0)} if is_bayes else {"alpha": [0.1, 1.0]}),
        )
    if index == 5:
        return (
            "Lasso",
            Lasso(),
            ({"alpha": Real(0.01, 1.0)} if is_bayes else {"alpha": [0.01, 0.1]}),
        )
    if index == 6:
        return (
            "ElasticNet",
            ElasticNet(),
            ({"l1_ratio": Real(0.1, 0.9)} if is_bayes else {"l1_ratio": [0.5]}),
        )
    if index == 7:
        return (
            "KNN",
            KNeighborsRegressor(n_jobs=n_jobs),
            ({"n_neighbors": Integer(3, 15)} if is_bayes else {"n_neighbors": [5, 10]}),
        )
    if index == 8:
        return (
            "Decision Tree",
            DecisionTreeRegressor(random_state=random_state),
            ({"max_depth": Integer(5, 20)} if is_bayes else {"max_depth": [5, 10]}),
        )
    if index == 9:
        return (
            "MLP (Neural Net)",
            MLPRegressor(random_state=random_state),
            {
                "hidden_layer_sizes": [(100,), (100, 50)],
                "activation": ["relu", "tanh"],
                "learning_rate_init": [0.001, 0.01],
            },
        )
    if index == 10:
        return (
            "SVR",
            SVR(),
            (
                {"C": Real(1.0, 100.0)}
                if is_bayes
                else {"C": [10, 100], "kernel": ["rbf"]}
            ),
        )
    if index == 11:
        return (
            "Huber Regressor",
            HuberRegressor(),
            (
                {"epsilon": Real(1.1, 1.5)}
                if is_bayes
                else {"epsilon": [1.1, 1.35, 1.5]}
            ),
        )
    if index == 12:
        if not XGB_AVAILABLE:
            return "XGBoost", None, {}
        return (
            "XGBoost",
            xgb.XGBRegressor(random_state=random_state, n_jobs=n_jobs),
            (
                {
                    "n_estimators": Integer(50, 300),
                    "max_depth": Integer(3, 10),
                    "learning_rate": Real(0.01, 0.2),
                }
                if is_bayes
                else {"n_estimators": [100, 200], "max_depth": [4, 6], "learning_rate": [0.05, 0.1]}
            ),
        )
    if index == 13:
        if not LGBM_AVAILABLE:
            return "LightGBM", None, {}
        return (
            "LightGBM",
            lgb.LGBMRegressor(random_state=random_state, n_jobs=n_jobs, verbose=-1),
            (
                {
                    "n_estimators": Integer(50, 300),
                    "max_depth": Integer(3, 10),
                    "learning_rate": Real(0.01, 0.2),
                }
                if is_bayes
                else {"n_estimators": [100, 200], "max_depth": [4, 6], "learning_rate": [0.05, 0.1]}
            ),
        )
    if index == 14:
        if not CATBOOST_AVAILABLE:
            return "CatBoost", None, {}
        return (
            "CatBoost",
            cb.CatBoostRegressor(random_state=random_state, thread_count=n_jobs, verbose=0),
            (
                {
                    "iterations": Integer(50, 300),
                    "depth": Integer(3, 10),
                    "learning_rate": Real(0.01, 0.2),
                }
                if is_bayes
                else {"iterations": [100, 200], "depth": [4, 6], "learning_rate": [0.05, 0.1]}
            ),
        )
    return "Unknown", LinearRegression(), {}


def run_optuna_search(base_model, name, param_distributions, X, y, fit_params, n_iter=20, cv=3, random_state=42, n_jobs=-1, groups=None):
    import optuna
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {}
        for param_name, dist in param_distributions.items():
            cls_name = type(dist).__name__
            if "Integer" in cls_name or "Real" in cls_name:
                low = dist.low
                high = dist.high
                if "Integer" in cls_name:
                    params[param_name] = trial.suggest_int(param_name, int(low), int(high))
                else:
                    is_log = getattr(dist, "prior", None) == "log-uniform"
                    params[param_name] = trial.suggest_float(param_name, float(low), float(high), log=is_log)
            elif "Categorical" in cls_name:
                params[param_name] = trial.suggest_categorical(param_name, dist.categories)
            elif hasattr(dist, 'bounds') and hasattr(dist, 'low') and hasattr(dist, 'high'):
                low = dist.low
                high = dist.high
                params[param_name] = trial.suggest_float(param_name, float(low), float(high))
            elif isinstance(dist, list):
                if all(isinstance(x, int) for x in dist):
                    params[param_name] = trial.suggest_int(param_name, min(dist), max(dist))
                elif all(isinstance(x, (int, float)) for x in dist):
                    params[param_name] = trial.suggest_float(param_name, min(dist), max(dist))
                else:
                    params[param_name] = trial.suggest_categorical(param_name, dist)
            else:
                params[param_name] = trial.suggest_categorical(param_name, [dist])

        if groups is not None:
            from sklearn.model_selection import GroupKFold
            n_unique_groups = len(np.unique(groups))
            if n_unique_groups >= 2:
                actual_cv = min(cv, n_unique_groups)
                kf = GroupKFold(n_splits=actual_cv)
                split_iterator = kf.split(X, y, groups=groups)
            else:
                kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
                split_iterator = kf.split(X)
        else:
            kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
            split_iterator = kf.split(X)

        scores = []
        for train_idx, val_idx in split_iterator:
            X_tr_cv, X_val_cv = X[train_idx], X[val_idx]
            y_tr_cv, y_val_cv = y[train_idx], y[val_idx]
            
            split_fit_params = {}
            if "sample_weight" in fit_params:
                split_fit_params["sample_weight"] = fit_params["sample_weight"][train_idx]

            fold_model = clone(base_model)
            fold_model.set_params(**params)
            fold_model.fit(X_tr_cv, y_tr_cv, **split_fit_params)
            preds = fold_model.predict(X_val_cv)
            scores.append(mean_squared_error(y_val_cv, preds))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_iter, n_jobs=1)
    
    best_params = study.best_params
    best_model = clone(base_model)
    best_model.set_params(**best_params)
    return best_model, best_params


def export_feature_importance(model, win_name, X_val, y_val, out_dir, log_path, feedback, selected_indices=None, feature_names=None):
    """
    Extracts, plots, and saves feature importances (or coefficients/permutation importance)
    for the winning model or ensemble.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    num_features = X_val.shape[1]
    if feature_names is not None:
        if selected_indices is not None and len(feature_names) > len(selected_indices):
            final_names = [feature_names[idx] for idx in selected_indices]
        else:
            final_names = feature_names
    else:
        final_names = [f"Band_{i+1}" for i in range(num_features)]
        if selected_indices is not None:
            final_names = [f"Band_{idx+1}" for idx in selected_indices]
            
    importances = None
    method_used = "Feature Importance"
    
    # 1. Check for native feature importances (Tree-based models)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        method_used = "Native Feature Importance"
    # 2. Check for coefficients (Linear models)
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        total = np.sum(importances)
        if total > 0:
            importances = importances / total
        method_used = "Normalized Absolute Coefficients"
    # 3. Check if it's our CustomEnsembleRegressor
    elif win_name.startswith("Ensemble") and hasattr(model, "fitted_estimators_"):
        base_importances = []
        for est in model.fitted_estimators_:
            if hasattr(est, "feature_importances_"):
                base_importances.append(est.feature_importances_)
            elif hasattr(est, "coef_"):
                coefs = np.abs(est.coef_)
                total = np.sum(coefs)
                if total > 0:
                    base_importances.append(coefs / total)
        if len(base_importances) > 0:
            importances = np.mean(base_importances, axis=0)
            method_used = "Ensemble Average Importance"
            
    # 4. Fallback to Permutation Importance
    if importances is None:
        append_log("   Calculating Permutation Importance (Fallback)...", log_path, feedback)
        from sklearn.inspection import permutation_importance
        subset_size = min(1000, len(X_val))
        X_sub = X_val[:subset_size]
        y_sub = y_val[:subset_size]
        
        result = permutation_importance(model, X_sub, y_sub, n_repeats=3, random_state=42, n_jobs=1)
        importances = np.clip(result.importances_mean, 0, None)
        total = np.sum(importances)
        if total > 0:
            importances = importances / total
        method_used = "Permutation Importance (Normalized)"
        
    # Save report
    df_imp = pd.DataFrame({
        "Feature": final_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    csv_path = os.path.join(out_dir, "3_Model_Feature_Importance_Report.csv")
    df_imp.to_csv(csv_path, index=False)
    
    # Plot importances
    plt.figure(figsize=(10, 6))
    df_plot = df_imp.sort_values(by="Importance", ascending=True)
    plt.barh(df_plot["Feature"], df_plot["Importance"], color="dodgerblue", edgecolor="black", alpha=0.8)
    plt.xlabel("Normalized Importance / Contribution")
    plt.ylabel("Features (Spectral Bands)")
    plt.title(f"Model Feature Importance ({win_name})\nMethod: {method_used}")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    img_path = os.path.join(out_dir, "3_Model_Feature_Importance.png")
    plt.savefig(img_path, dpi=150)
    plt.close()
    
    append_log(f"   Saved feature importance plot and report to: {out_dir}", log_path, feedback)


def parse_score_config(algorithm, parameters, context=None) -> dict:
    """
    Parses customizable score settings and model selection strategy from QGIS algorithm parameters.
    Supports:
      - SCORE_SELECTION_STRATEGY: Strategy Enum (0 to 6)
      - SCORE_METRICS: Multi-select checkboxes for metrics [0: R2, 1: RMSE, 2: wMAPE, 3: Bias, 4: MAE]
      - SCORE_CUSTOM_CONFIG: Optional custom weights (e.g. 'R2: 70, RMSE: 30') and simulation settings
    """
    config = {
        "selection_strategy": 0,
        "strategy_name": "Winner Stability (Monte Carlo Sensitivity)",
        "weights": {
            "R2": 0.35,
            "RMSE": 0.30,
            "wMAPE": 0.20,
            "Bias": 0.15,
            "MAE": 0.0,
        },
        "n_rounds": 20,
        "variation_pct": 0.35,
        "weight_ranges": "",
    }

    if parameters is None:
        return config

    def _get_val(param_name, default_val):
        if hasattr(algorithm, "parameterDefinition") and algorithm.parameterDefinition(param_name):
            if isinstance(default_val, bool):
                return algorithm.parameterAsBool(parameters, param_name, context)
            elif isinstance(default_val, int):
                return algorithm.parameterAsInt(parameters, param_name, context)
            elif isinstance(default_val, float):
                return algorithm.parameterAsDouble(parameters, param_name, context)
            elif isinstance(default_val, list):
                return algorithm.parameterAsEnums(parameters, param_name, context)
            elif isinstance(default_val, str):
                return algorithm.parameterAsString(parameters, param_name, context)
        # Fallback to dict lookup
        if isinstance(parameters, dict) and param_name in parameters:
            val = parameters[param_name]
            if val is not None and str(val).strip() != "":
                try:
                    if isinstance(default_val, list):
                        if isinstance(val, (list, tuple)):
                            return [int(x) for x in val]
                        return [int(x.strip()) for x in str(val).split(",") if x.strip().isdigit()]
                    elif isinstance(default_val, int):
                        return int(val)
                    elif isinstance(default_val, float):
                        return float(val)
                    elif isinstance(default_val, bool):
                        return bool(val)
                    return str(val)
                except Exception:
                    pass
        return default_val

    # 1. Strategy Enum
    strategy_options = [
        "Winner Stability (Monte Carlo Sensitivity)",
        "Highest SDB Composite Score (Max Baseline Score)",
        "Highest R² Accuracy",
        "Lowest RMSE (Minimum Vertical Error)",
        "Lowest wMAPE (%)",
        "Lowest |Bias| (Zero-Mean Residual Offset)",
        "Lowest MAE (Mean Absolute Error)",
    ]
    strat_idx = _get_val("SCORE_SELECTION_STRATEGY", 0)
    if isinstance(strat_idx, int) and 0 <= strat_idx < len(strategy_options):
        config["selection_strategy"] = strat_idx
        config["strategy_name"] = strategy_options[strat_idx]
    elif isinstance(strat_idx, str):
        config["strategy_name"] = strat_idx
        for idx, opt in enumerate(strategy_options):
            if opt.lower().startswith(strat_idx.lower()) or strat_idx.lower() in opt.lower():
                config["selection_strategy"] = idx
                config["strategy_name"] = opt
                break

    # 2. Metric Checkboxes (0: R2, 1: RMSE, 2: wMAPE, 3: Bias, 4: MAE)
    metric_keys = ["R2", "RMSE", "wMAPE", "Bias", "MAE"]
    default_base_weights = {"R2": 35.0, "RMSE": 30.0, "wMAPE": 20.0, "Bias": 15.0, "MAE": 15.0}
    
    selected_metric_indices = _get_val("SCORE_METRICS", [0, 1, 2, 3, 4])
    if not isinstance(selected_metric_indices, list) or len(selected_metric_indices) == 0:
        selected_metric_indices = [0, 1, 2, 3, 4]

    active_metrics = set()
    for m_idx in selected_metric_indices:
        try:
            m_int = int(m_idx)
            if 0 <= m_int < len(metric_keys):
                active_metrics.add(metric_keys[m_int])
        except Exception:
            pass

    if not active_metrics:
        active_metrics = {"R2", "RMSE"}

    # Compute initial default weights for active metrics
    raw_weights = {}
    for m in metric_keys:
        if m in active_metrics:
            raw_weights[m] = default_base_weights.get(m, 20.0)
        else:
            raw_weights[m] = 0.0

    # 3. Custom Config String (Custom Weights & Monte Carlo settings)
    custom_cfg_str = _get_val("SCORE_CUSTOM_CONFIG", _get_val("SCORE_WEIGHT_RANGES", ""))
    
    # Check for legacy parameter overrides if present
    for legacy_k, metric_name in [("SCORE_WEIGHT_R2", "R2"), ("SCORE_WEIGHT_RMSE", "RMSE"), 
                                  ("SCORE_WEIGHT_WMAPE", "wMAPE"), ("SCORE_WEIGHT_BIAS", "Bias"), 
                                  ("SCORE_WEIGHT_MAE", "MAE")]:
        if isinstance(parameters, dict) and legacy_k in parameters:
            try:
                val = float(parameters[legacy_k])
                raw_weights[metric_name] = max(0.0, val)
                if val > 0:
                    active_metrics.add(metric_name)
            except Exception:
                pass

    if custom_cfg_str and isinstance(custom_cfg_str, str) and custom_cfg_str.strip():
        import re
        # Check for Rounds / n_rounds: e.g. "Rounds: 50"
        rounds_m = re.search(r"(?:rounds|iterations|n_rounds|sim_rounds)\s*[:=]\s*(\d+)", custom_cfg_str, re.IGNORECASE)
        if rounds_m:
            config["n_rounds"] = max(1, int(rounds_m.group(1)))
            
        # Check for Variation / Variation_pct: e.g. "Variation: +/-25%" or "Variation: 30"
        var_m = re.search(r"(?:variation|var|pct|tolerance)\s*[:=]\s*[\+\-\/\s]*([0-9.]+)\s*%?", custom_cfg_str, re.IGNORECASE)
        if var_m:
            v_val = float(var_m.group(1))
            if v_val > 1.0:
                v_val = v_val / 100.0
            config["variation_pct"] = max(0.05, min(0.95, v_val))

        # Check for custom weight assignments: e.g. "R2: 70, RMSE: 30" or "R2=50"
        custom_weights_dict = {}
        for m in metric_keys:
            m_regex = re.compile(rf"\b{re.escape(m)}\b\s*[:=]\s*([0-9.]+)(?![\s,]*[0-9.]+\])", re.IGNORECASE)
            match = m_regex.search(custom_cfg_str)
            if match:
                val = float(match.group(1))
                custom_weights_dict[m] = max(0.0, val)

        if custom_weights_dict:
            max_c_val = max(custom_weights_dict.values())
            scale_factor = 1.0 if (max_c_val <= 1.0 and sum(custom_weights_dict.values()) <= 1.5) else 100.0

            if len(custom_weights_dict) >= len(metric_keys) - 1:
                # Full matrix template provided
                for m in metric_keys:
                    if m in active_metrics:
                        c_val = custom_weights_dict.get(m, 0.0)
                        if c_val > 0:
                            raw_weights[m] = c_val
                        else:
                            # Metric is active in checkboxes; assign proportional base weight
                            raw_weights[m] = (default_base_weights.get(m, 20.0) / 100.0 * scale_factor) if scale_factor > 1.0 else (default_base_weights.get(m, 20.0) / 100.0)
                    else:
                        raw_weights[m] = 0.0
            else:
                # Short custom string (e.g. "R2: 70, RMSE: 30")
                active_metrics = set(k for k, v in custom_weights_dict.items() if v > 0)
                for m in metric_keys:
                    raw_weights[m] = custom_weights_dict.get(m, 0.0)

        config["weight_ranges"] = custom_cfg_str

    # Simulation rounds direct fallback
    if isinstance(parameters, dict) and "SCORE_SIM_ROUNDS" in parameters:
        try:
            config["n_rounds"] = max(1, int(parameters["SCORE_SIM_ROUNDS"]))
        except Exception:
            pass

    # Normalize weights to sum exactly to 1.0 across active metrics
    active_sum = sum(raw_weights[m] for m in active_metrics)
    if active_sum <= 0:
        # Fallback to equal weighting among active metrics
        eq_w = 1.0 / len(active_metrics) if len(active_metrics) > 0 else 0.5
        config["weights"] = {m: (eq_w if m in active_metrics else 0.0) for m in metric_keys}
    else:
        config["weights"] = {m: ((raw_weights[m] / active_sum) if m in active_metrics else 0.0) for m in metric_keys}

    return config


def parse_weight_ranges(ranges_str: str, base_weights: dict, variation_pct: float = 0.35) -> dict:
    """
    Parses custom weight bounds for Monte Carlo simulation.
    Automatically scales bounds around active base weights using variation_pct (default +/-35%)
    while clamping zero-weighted / unselected metrics to (0.0, 0.0).
    """
    ranges = {}
    for k, w in base_weights.items():
        if w <= 0.0:
            ranges[k] = (0.0, 0.0)
        else:
            low_b = max(0.01, w * (1.0 - variation_pct))
            high_b = min(1.0, w * (1.0 + variation_pct))
            ranges[k] = (low_b, high_b)

    if ranges_str and isinstance(ranges_str, str) and ranges_str.strip():
        import re
        pattern = re.compile(r"([A-Za-z0-9_]+)\s*[:=]\s*\[\s*([0-9.]+)\s*[,;\-\s]+\s*([0-9.]+)\s*\]")
        for match in pattern.finditer(ranges_str):
            metric_name, min_v, max_v = match.groups()
            for k in base_weights.keys():
                if metric_name.lower() == k.lower():
                    try:
                        low = float(min_v)
                        high = float(max_v)
                        if low > high:
                            low, high = high, low
                        if base_weights[k] <= 0.0:
                            ranges[k] = (0.0, 0.0)
                        else:
                            ranges[k] = (max(0.0, low), max(low, high))
                    except Exception:
                        pass
    return ranges


def calculate_sdb_composite_score(
    df: pd.DataFrame,
    n_rounds: int = 20,
    random_state: int = 42,
    score_config: dict = None,
    out_dir: str = None,
    prefix: str = "3_",
    **kwargs
) -> pd.DataFrame:
    """
    Computes the SDB Composite Score (0-100) and executes a Multi-Weight Sensitivity
    & Winner Stability Analysis across randomized weight configurations.
    
    Supports customizable weights, Monte Carlo bounds, and multiple model selection strategies:
      - 0: "Winner Stability (Monte Carlo Sensitivity)" [Default]
      - 1: "Highest SDB Composite Score (Max Baseline Score)"
      - 2: "Highest R² Accuracy"
      - 3: "Lowest RMSE (Minimum Vertical Error)"
      - 4: "Lowest wMAPE (%)"
      - 5: "Lowest |Bias| (Zero-Mean Residual Offset)"
      - 6: "Lowest MAE (Mean Absolute Error)"
    """
    df = df.copy()
    if len(df) == 0:
        return df

    if score_config is None:
        score_config = {}

    strat = score_config.get("selection_strategy", 0)
    user_weights = score_config.get("weights", {
        "R2": 0.35, "RMSE": 0.30, "wMAPE": 0.20, "Bias": 0.15, "MAE": 0.0
    })
    sim_rounds = score_config.get("n_rounds", n_rounds)
    ranges_str = score_config.get("weight_ranges", "")

    # Ensure required columns exist
    for col in ["R2", "RMSE", "wMAPE"]:
        if col not in df.columns:
            df[col] = 0.0
    if "Bias" not in df.columns:
        df["Bias"] = 0.0
    if "MAE" not in df.columns:
        df["MAE"] = df["Bias"].abs()

    # Normalize user weights to sum to 1.0
    tot_w = sum(user_weights.values())
    if tot_w <= 0:
        norm_weights = {"R2": 0.35, "RMSE": 0.30, "wMAPE": 0.20, "Bias": 0.15, "MAE": 0.0}
    else:
        norm_weights = {k: v / tot_w for k, v in user_weights.items()}

    if len(df) == 1:
        df["SDB_Score"] = 100.0
        df["Stability"] = 100.0
        df["Wins"] = f"{sim_rounds}/{sim_rounds}"
        df["Mean_Score"] = 100.0
        return df

    # Normalization helper (0 to 1)
    def _norm_higher(series):
        s_min = float(series.min())
        s_max = float(series.max())
        if np.isclose(s_max, s_min) or s_max == s_min:
            return np.ones(len(series))
        return ((series - s_min) / (s_max - s_min)).to_numpy()

    def _norm_lower(series):
        s_min = float(series.min())
        s_max = float(series.max())
        if np.isclose(s_max, s_min) or s_max == s_min:
            return np.ones(len(series))
        return ((s_max - series) / (s_max - s_min)).to_numpy()

    r2_n = _norm_higher(df["R2"])
    rmse_n = _norm_lower(df["RMSE"])
    wmape_n = _norm_lower(df["wMAPE"])
    bias_n = _norm_lower(df["Bias"].abs())
    mae_n = _norm_lower(df["MAE"])

    norm_series = {
        "R2": r2_n,
        "RMSE": rmse_n,
        "wMAPE": wmape_n,
        "Bias": bias_n,
        "MAE": mae_n,
    }

    # Standard baseline composite score (0-100)
    baseline_score = np.zeros(len(df))
    for k, w in norm_weights.items():
        if w > 0.0 and k in norm_series:
            baseline_score += w * norm_series[k]
    baseline_score *= 100.0

    # Multi-Weight Sensitivity Simulation (sim_rounds)
    rng = np.random.RandomState(random_state)
    sim_bounds = parse_weight_ranges(ranges_str, norm_weights, variation_pct=score_config.get("variation_pct", 0.35))

    round_scores = np.zeros((len(df), sim_rounds))
    wins = np.zeros(len(df), dtype=int)
    rounds_log = []

    for k in range(sim_rounds):
        sampled_u = {}
        tot_u = 0.0
        for metric, (low_b, high_b) in sim_bounds.items():
            if norm_weights.get(metric, 0.0) <= 0.0 or high_b <= 0.0:
                sampled_u[metric] = 0.0
            else:
                val = rng.uniform(low_b, high_b)
                sampled_u[metric] = val
                tot_u += val

        if tot_u <= 0.0:
            tot_u = 1.0
            sampled_u = norm_weights.copy()

        score_k = np.zeros(len(df))
        normalized_weights_k = {}
        for metric, val in sampled_u.items():
            w_k = val / tot_u
            normalized_weights_k[metric] = round(float(w_k), 4)
            if w_k > 0.0 and metric in norm_series:
                score_k += w_k * norm_series[metric]
        score_k *= 100.0

        round_scores[:, k] = score_k
        winner_k = int(np.argmax(score_k))
        wins[winner_k] += 1

        if out_dir:
            round_entry = {
                "Round": k + 1,
                "Weight_R2": normalized_weights_k.get("R2", 0.0),
                "Weight_RMSE": normalized_weights_k.get("RMSE", 0.0),
                "Weight_wMAPE": normalized_weights_k.get("wMAPE", 0.0),
                "Weight_Bias": normalized_weights_k.get("Bias", 0.0),
                "Weight_MAE": normalized_weights_k.get("MAE", 0.0),
            }
            if "Algorithm" in df.columns:
                for algo_idx, algo_name in enumerate(df["Algorithm"]):
                    round_entry[f"Score_{algo_name}"] = round(float(score_k[algo_idx]), 2)
                round_entry["Round_Winner"] = str(df.iloc[winner_k]["Algorithm"])
            else:
                round_entry["Round_Winner"] = f"Model_{winner_k + 1}"
            round_entry["Winning_Score"] = round(float(score_k[winner_k]), 2)
            rounds_log.append(round_entry)

    if out_dir and len(rounds_log) > 0:
        try:
            os.makedirs(out_dir, exist_ok=True)
            csv_name = f"{prefix}Score_Sensitivity_Rounds.csv"
            rounds_csv_path = os.path.join(out_dir, csv_name)
            df_rounds = pd.DataFrame(rounds_log)
            df_rounds.to_csv(rounds_csv_path, index=False, encoding="utf-8")
        except Exception:
            pass

    df["Stability"] = np.round((wins / sim_rounds) * 100.0, 1)
    df["Wins"] = [f"{w}/{sim_rounds}" for w in wins]
    df["Mean_Score"] = np.round(np.mean(round_scores, axis=1), 2)
    df["SDB_Score"] = np.round(baseline_score, 2)

    # Ranking and Winner Selection Strategy
    df["Abs_Bias"] = df["Bias"].abs()
    if strat == 1 or strat == "Highest Composite Score (SDB Score)" or strat == "Highest SDB Composite Score (Max Baseline Score)":
        # Sort by SDB Composite Score descending
        df = df.sort_values(by=["SDB_Score", "R2", "Stability"], ascending=[False, False, False]).reset_index(drop=True)
    elif strat == 2 or strat == "Highest R² Accuracy":
        # Sort by R2 descending
        df = df.sort_values(by=["R2", "SDB_Score", "Stability"], ascending=[False, False, False]).reset_index(drop=True)
    elif strat == 3 or strat == "Lowest RMSE (Minimum Vertical Error)":
        # Sort by RMSE ascending
        df = df.sort_values(by=["RMSE", "SDB_Score"], ascending=[True, False]).reset_index(drop=True)
    elif strat == 4 or strat == "Lowest wMAPE (%)":
        # Sort by wMAPE ascending
        df = df.sort_values(by=["wMAPE", "SDB_Score"], ascending=[True, False]).reset_index(drop=True)
    elif strat == 5 or strat == "Lowest |Bias| (Zero-Mean Residual Offset)":
        # Sort by Abs_Bias ascending
        df = df.sort_values(by=["Abs_Bias", "SDB_Score"], ascending=[True, False]).reset_index(drop=True)
    elif strat == 6 or strat == "Lowest MAE (Mean Absolute Error)":
        # Sort by MAE ascending
        df = df.sort_values(by=["MAE", "SDB_Score"], ascending=[True, False]).reset_index(drop=True)
    else:
        # Default: Winner Stability % (Wins), tie-break with Mean_Score, then SDB_Score, then R2
        df = df.sort_values(by=["Stability", "Mean_Score", "SDB_Score", "R2"], ascending=[False, False, False, False]).reset_index(drop=True)

    if "Abs_Bias" in df.columns:
        df = df.drop(columns=["Abs_Bias"])

    return df


def run_benchmarking(
    X, y, weights, indices, n_iter, out_dir, feedback, opt_idx, log_path, custom_params,
    test_size=0.2, random_state=42, n_jobs=-1, enable_ensemble=False, ensemble_method="All Ensemble Methods (Auto-Select)",
    spatial_cv=False, coords=None, selected_indices=None, ensemble_size=3, feature_names=None,
    cv_folds=3, enable_depth_variance_corr=False, score_config=None
):
    X = np.nan_to_num(X, nan=0.0)
    
    groups_tr = None
    if spatial_cv and coords is not None:
        from sklearn.cluster import KMeans
        from sklearn.model_selection import GroupShuffleSplit
        kmeans = KMeans(n_clusters=5, random_state=random_state, n_init=10)
        spatial_groups = kmeans.fit_predict(coords)
        
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, val_idx = next(gss.split(X, y, groups=spatial_groups))
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr = weights[train_idx]
        groups_tr = spatial_groups[train_idx]
        append_log("   [Spatial CV] Split data into 5 geographic clusters using KMeans.", log_path, feedback)
    else:
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val, w_tr, _ = train_test_split(
            X, y, weights, test_size=test_size, random_state=random_state
        )

    all_indices = [int(i) for i in indices]
    base_indices = [i for i in all_indices if i < 15]
    ensemble_selected_indices = [i for i in all_indices if i >= 15]

    if not base_indices:
        base_indices = [3, 12, 13, 14]

    results = []

    for idx in base_indices:
        name, raw_base_model, default_params = get_model_and_params(idx, opt_idx, random_state, n_jobs)
        if raw_base_model is None:
            append_log(f"      ! Skipping {name}: Library is not installed.", log_path, feedback)
            continue

        if name in custom_params and custom_params[name]:
            parsed_dict = custom_params[name]
            base_params = {k: (v[0] if isinstance(v, list) and len(v)>0 else v) for k, v in parsed_dict.items()}
            raw_base_model.set_params(**base_params)

            if opt_idx == 2 and SKOPT_AVAILABLE:
                params = convert_to_bayes(parsed_dict)
            else:
                params = parsed_dict
        else:
            params = default_params

        algo_dir = os.path.join(out_dir, name.replace(" ", "_"))
        os.makedirs(algo_dir, exist_ok=True)

        try:
            fit_params = {}
            if name in [
                "Random Forest",
                "Gradient Boosting",
                "Extra Trees",
                "Ridge",
                "Lasso",
                "Decision Tree",
                "SVR",
                "Linear Regression",
                "Huber Regressor",
                "XGBoost",
                "LightGBM",
                "CatBoost",
            ]:
                fit_params["sample_weight"] = w_tr

            search_n_jobs = 1 if name in ["Random Forest", "Extra Trees", "KNN", "XGBoost", "LightGBM", "CatBoost"] else n_jobs

            with joblib.parallel_backend("threading", n_jobs=search_n_jobs):
                if params and n_iter > 0:
                    search = None
                    current_opt_idx = opt_idx
                    if name == "MLP (Neural Net)":
                        current_opt_idx = 0
                    
                    cv_splitter = cv_folds
                    if groups_tr is not None:
                        n_unique_groups = len(np.unique(groups_tr))
                        if n_unique_groups >= 2:
                            from sklearn.model_selection import GroupKFold
                            cv_splitter = GroupKFold(n_splits=min(cv_folds, n_unique_groups))
                        else:
                            groups_tr = None

                    if current_opt_idx == 0:
                        import scipy.stats as stats
                        opt_params = {}
                        for k, v in params.items():
                            if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
                                if all(isinstance(x, int) for x in v) and v[0] < v[1]:
                                    opt_params[k] = stats.randint(v[0], v[1] + 1)
                                elif v[0] < v[1]:
                                    opt_params[k] = stats.uniform(v[0], v[1] - v[0])
                                else:
                                    opt_params[k] = v
                            else:
                                opt_params[k] = v
                        search = RandomizedSearchCV(
                            clone(raw_base_model), opt_params, n_iter=n_iter, cv=cv_splitter, n_jobs=search_n_jobs, random_state=random_state
                        )
                    elif current_opt_idx == 1:
                        opt_params = {}
                        for k, v in params.items():
                            if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
                                if all(isinstance(x, int) for x in v) and v[0] < v[1]:
                                    opt_params[k] = list(np.linspace(v[0], v[1], min(5, v[1] - v[0] + 1), dtype=int))
                                elif v[0] < v[1]:
                                    opt_params[k] = list(np.linspace(v[0], v[1], 5))
                                else:
                                    opt_params[k] = v
                            else:
                                opt_params[k] = v
                        search = GridSearchCV(clone(raw_base_model), opt_params, cv=cv_splitter, n_jobs=search_n_jobs)
                    elif current_opt_idx == 2:
                        if OPTUNA_AVAILABLE:
                            best_m, best_params = run_optuna_search(
                                clone(raw_base_model), name, params, X_tr, y_tr, fit_params,
                                n_iter=n_iter, cv=cv_folds, random_state=random_state, n_jobs=search_n_jobs,
                                groups=groups_tr
                            )
                            model = best_m
                            model.fit(X_tr, y_tr, **fit_params)
                            search = None
                            params_str = str(best_params)
                        else:
                            import scipy.stats as stats
                            opt_params = {}
                            for k, v in params.items():
                                if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
                                    if all(isinstance(x, int) for x in v) and v[0] < v[1]:
                                        opt_params[k] = stats.randint(v[0], v[1] + 1)
                                    elif v[0] < v[1]:
                                        opt_params[k] = stats.uniform(v[0], v[1] - v[0])
                                    else:
                                        opt_params[k] = v
                                else:
                                    opt_params[k] = v
                            search = (
                                BayesSearchCV(clone(raw_base_model), params, n_iter=n_iter, cv=cv_splitter, n_jobs=search_n_jobs)
                                if SKOPT_AVAILABLE
                                else RandomizedSearchCV(
                                    clone(raw_base_model), opt_params, n_iter=n_iter, cv=cv_splitter, n_jobs=search_n_jobs, random_state=random_state
                                )
                            )

                    if search:
                        if groups_tr is not None:
                            search.fit(X_tr, y_tr, groups=groups_tr, **fit_params)
                        else:
                            search.fit(X_tr, y_tr, **fit_params)
                        model = search.best_estimator_
                        params_str = str(search.best_params_)
                    elif not OPTUNA_AVAILABLE or current_opt_idx != 2:
                        model = clone(raw_base_model)
                        model.fit(X_tr, y_tr, **fit_params)
                        params_str = "Default / Custom"
                else:
                    model = clone(raw_base_model)
                    model.fit(X_tr, y_tr, **fit_params)
                    params_str = "Default / Custom"

            y_p = model.predict(X_val)

            r2 = float(r2_score(y_val, y_p))

            rmse = float(np.sqrt(mean_squared_error(y_val, y_p)))
            sum_abs_diff = np.sum(np.abs(y_val - y_p))
            sum_abs_true = np.sum(np.abs(y_val))
            wmape = float((sum_abs_diff / sum_abs_true) * 100 if sum_abs_true != 0 else 0.0)
            bias = float(np.mean(y_p - y_val))
            mae = float(mean_absolute_error(y_val, y_p))

            save_algo_artifacts(
                y_val,
                y_p,
                np.abs(y_val - y_p),
                name,
                algo_dir,
                r2,
                rmse,
                wmape,
                params_str,
                scaler_name="None",
            )

            results.append(
                {
                    "Algorithm": name,
                    "Feature Scaling": "None",
                    "Model": model,
                    "R2": r2,
                    "RMSE": rmse,
                    "wMAPE": wmape,
                    "Bias": bias,
                    "MAE": mae,
                }
            )
            append_log(
                f"      > {name}: R2={r2:.3f}, RMSE={rmse:.2f}m, wMAPE={wmape:.1f}%, Bias={bias:+.3f}m, MAE={mae:.2f}m", log_path, feedback
            )

        except Exception as e:
            append_log(f"      ! Failed {name}: {e}", log_path, feedback)

    if not results:
        raise QgsProcessingException("All selected algorithms failed.")

    has_ensemble_selected = len(ensemble_selected_indices) > 0 or enable_ensemble
    if has_ensemble_selected and len(results) >= 2:
        df_base = calculate_sdb_composite_score(pd.DataFrame(results), random_state=random_state, score_config=score_config)
        top_results = df_base.sort_values(by="SDB_Score", ascending=False).head(ensemble_size).to_dict("records")
        estimators = [(r["Algorithm"], r["Model"]) for r in top_results]
        append_log(f"   [Ensemble] Blending top {len(estimators)} base models ({[r['Algorithm'] for r in top_results]})", log_path, feedback)

        methods_to_test = []
        if 19 in ensemble_selected_indices or (enable_ensemble and ensemble_method in ["All Ensemble Methods (Auto-Select)", "All"]):
            methods_to_test = ["Average", "Median", "Stacking", "Uncertainty-Weighted Fusion"]
        else:
            if 15 in ensemble_selected_indices:
                methods_to_test.append("Average")
            if 16 in ensemble_selected_indices:
                methods_to_test.append("Median")
            if 17 in ensemble_selected_indices:
                methods_to_test.append("Stacking")
            if 18 in ensemble_selected_indices:
                methods_to_test.append("Uncertainty-Weighted Fusion")
            if not methods_to_test and enable_ensemble:
                methods_to_test = [ensemble_method]

        for m_name in methods_to_test:
            try:
                ensemble_model = CustomEnsembleRegressor(estimators=estimators, method=m_name)
                ensemble_model.fit(X_tr, y_tr, sample_weight=w_tr)

                y_p = ensemble_model.predict(X_val)
                r2 = float(r2_score(y_val, y_p))
                rmse = float(np.sqrt(mean_squared_error(y_val, y_p)))
                sum_abs_diff = np.sum(np.abs(y_val - y_p))
                sum_abs_true = np.sum(np.abs(y_val))
                wmape = float((sum_abs_diff / sum_abs_true) * 100 if sum_abs_true != 0 else 0.0)
                bias = float(np.mean(y_p - y_val))
                mae = float(mean_absolute_error(y_val, y_p))

                ens_dir = os.path.join(out_dir, f"Ensemble_{m_name.replace(' ', '_')}")
                os.makedirs(ens_dir, exist_ok=True)
                save_algo_artifacts(
                    y_val,
                    y_p,
                    np.abs(y_val - y_p),
                    f"Ensemble ({m_name})",
                    ens_dir,
                    r2,
                    rmse,
                    wmape,
                    f"Method: {m_name}, Models: {[r['Algorithm'] for r in top_results]}",
                    scaler_name="Composite",
                )

                results.append(
                    {
                        "Algorithm": f"Ensemble ({m_name})",
                        "Feature Scaling": "Composite",
                        "Model": ensemble_model,
                        "R2": r2,
                        "RMSE": rmse,
                        "wMAPE": wmape,
                        "Bias": bias,
                        "MAE": mae,
                    }
                )
                append_log(
                    f"      > Ensemble ({m_name}): R2={r2:.3f}, RMSE={rmse:.2f}m, wMAPE={wmape:.1f}%, Bias={bias:+.3f}m, MAE={mae:.2f}m", log_path, feedback
                )
            except Exception as e:
                append_log(f"      ! Failed Ensemble ({m_name}): {e}", log_path, feedback)

    df = pd.DataFrame(results)
    df = calculate_sdb_composite_score(df, random_state=random_state, score_config=score_config, out_dir=out_dir, prefix="3_")
    winner = df.iloc[0]

    strat_title = score_config.get("strategy_name", "Winner Stability & SDB Composite Ranking") if score_config else "Winner Stability & SDB Composite Ranking"
    append_log(f"\n   📊 [Phase 03 Auto-ML Leaderboard - Selection Strategy: {strat_title}]:", log_path, feedback)
    for rank, row in df.iterrows():
        prefix = "🥇" if rank == 0 else ("🥈" if rank == 1 else ("🥉" if rank == 2 else "  "))
        append_log(
            f"      {prefix} {row['Algorithm']:<32} | Stability: {row['Stability']:>5.1f}% ({row['Wins']:>5}) | Mean Score: {row['Mean_Score']:>5.2f} | Baseline Score: {row['SDB_Score']:>5.2f} | R²={row['R2']:>6.4f} | RMSE={row['RMSE']:>5.2f}m | wMAPE={row['wMAPE']:>5.2f}% | Bias={row['Bias']:>+6.3f}m",
            log_path, feedback
        )

    append_log(f"\n   ⭐ Winner Selected: {winner['Algorithm']} (Stability: {winner['Stability']:.1f}% [{winner['Wins']}], SDB Score: {winner['SDB_Score']:.2f}, Mean Score: {winner['Mean_Score']:.2f}/100)", log_path, feedback)

    final_model = winner["Model"]
    fit_params_final = {}
    if winner["Algorithm"].startswith("Ensemble"):
        fit_params_final["sample_weight"] = weights
    elif winner["Algorithm"] in [
        "Random Forest",
        "Gradient Boosting",
        "Extra Trees",
        "Ridge",
        "Lasso",
        "Decision Tree",
        "SVR",
        "Linear Regression",
        "Huber Regressor",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    ]:
        fit_params_final["sample_weight"] = weights

    final_search_n_jobs = 1 if winner["Algorithm"] in ["Random Forest", "Extra Trees", "KNN", "XGBoost", "LightGBM", "CatBoost"] else n_jobs
    with joblib.parallel_backend("threading", n_jobs=final_search_n_jobs):
        final_model.fit(X, y, **fit_params_final)

    try:
        export_feature_importance(
            final_model,
            winner["Algorithm"],
            X_val,
            y_val,
            out_dir,
            log_path,
            feedback,
            selected_indices,
            feature_names
        )
    except Exception as e:
        append_log(f"   [Warning] Failed to generate feature importance plot: {e}", log_path, feedback)

    return df, {
        "name": winner["Algorithm"],
        "scaler": winner.get("Feature Scaling", "None"),
        "model": final_model,
        "score": winner["SDB_Score"],
        "r2": winner["R2"],
        "rmse": winner["RMSE"],
        "wmape": winner["wMAPE"],
        "bias": winner.get("Bias", 0.0),
        "mae": winner.get("MAE", 0.0),
    }


def predict_map(model, stack_path, mask_path, out_path, med_size, output_format="float32", selected_indices=None, extra_features=None, feedback=None):
    with rasterio.open(stack_path) as s:
        d = s.read()
        h, w = s.height, s.width
        d_flat = d.reshape(d.shape[0], -1).T
        prof = s.profile.copy()
    
    if mask_path and str(mask_path).strip() and mask_path != "None":
        with rasterio.open(mask_path) as m:
            if m.shape == (h, w) and m.crs == prof['crs'] and m.transform == prof['transform']:
                mask_arr = m.read(1).flatten()
            else:
                from rasterio.warp import reproject, Resampling
                mask_resampled = np.zeros((h, w), dtype=np.uint8)
                reproject(
                    source=m.read(1),
                    destination=mask_resampled,
                    src_transform=m.transform,
                    src_crs=m.crs,
                    dst_transform=prof['transform'],
                    dst_crs=prof['crs'],
                    resampling=Resampling.nearest,
                )
                mask_arr = mask_resampled.flatten()
    else:
        mask_arr = np.ones(h * w, dtype=np.uint8)
        
    water_idx = np.where(mask_arr > 0)[0]
    if len(water_idx) == 0:
        return
        
    out_img = np.full(h * w, -9999.0, dtype="float32")
    model_str = str(model)
    if "KNeighbors" in model_str or "GaussianProcess" in model_str or "RobustSpatialKNN" in model_str or "KNN" in model_str:
        chunk_size = 5000
    else:
        chunk_size = 500000
    n_pixels = len(water_idx)
    
    for start in range(0, n_pixels, chunk_size):
        if feedback and feedback.isCanceled():
            break
        if feedback:
            feedback.setProgress(int((start / n_pixels) * 100))
            
        end = min(start + chunk_size, n_pixels)
        batch_idx = water_idx[start:end]
        
        X_chunk = d_flat[batch_idx]
        
        # Identify pixels where any feature is missing (-9999 or NaN)
        # We do this BEFORE adding extra_features (which are global) 
        # and BEFORE filling NaNs with 0.0
        invalid_mask = np.any(np.isnan(X_chunk) | (X_chunk == -9999.0) | (X_chunk < -9000), axis=1)

        if extra_features is not None:
            extra_cols = np.tile(extra_features, (X_chunk.shape[0], 1))
            X_chunk = np.hstack((X_chunk, extra_cols))
            
        if selected_indices is not None and len(selected_indices) > 0:
            X_chunk = X_chunk[:, selected_indices]
            
        np.nan_to_num(X_chunk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        chunk_preds = model.predict(X_chunk)
        
        # Overwrite predictions for invalid feature pixels to NoData
        chunk_preds[invalid_mask] = -9999.0
        
        out_img[batch_idx] = chunk_preds

    out_img = out_img.reshape(h, w)
    if med_size > 0 and scipy_is_available:
        valid = out_img != -9999.0
        temp = out_img.copy()
        temp[~valid] = 0
        filt = median_filter(temp, size=med_size)
        filt[~valid] = -9999.0
        out_img = filt
        
    nodata_val = -9999.0
    if output_format == "uint16":
        valid = out_img != -9999.0
        out_img[valid] = np.clip(out_img[valid], 0, None)
        out_img[~valid] = 65535
        out_img = out_img.astype("uint16")
        nodata_val = 65535.0
    elif output_format == "float64":
        out_img = out_img.astype("float64")
    else:
        out_img = out_img.astype("float32")

    prof.update(count=1, dtype=output_format, nodata=nodata_val)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(out_img, 1)


def run_phase03_initial_modeling(algorithm, parameters, context, feedback, pre_extracted_data=None):
    if "OUTPUT_FOLDER" in parameters and isinstance(parameters["OUTPUT_FOLDER"], str) and parameters["OUTPUT_FOLDER"]:
        out_dir = parameters["OUTPUT_FOLDER"]
        log_path = parameters.get("LOG_FILE", os.path.join(out_dir, "Phase_03_Log.txt"))
    elif hasattr(algorithm, "OUTPUT_FOLDER"):
        out_dir = algorithm.parameterAsString(parameters, getattr(algorithm, "OUTPUT_FOLDER", "OUTPUT_FOLDER"), context)
        log_path = algorithm.parameterAsString(parameters, getattr(algorithm, "LOG_FILE", "LOG_FILE"), context)
    else:
        out_dir = algorithm.parameterAsString(parameters, getattr(algorithm, "OUTPUT_MASTER_FOLDER", "OUTPUT_MASTER_FOLDER"), context)
        out_dir = os.path.join(out_dir, "Global_Model")
        log_path = os.path.join(out_dir, "Global_Model_Log.txt")
        
    os.makedirs(out_dir, exist_ok=True)

    custom_params = {
        "Random Forest": parse_param_string(
            algorithm.parameterAsString(parameters, getattr(algorithm, "PARAM_RF", "PARAM_RF"), context)
        ),
        "Gradient Boosting": parse_param_string(
            algorithm.parameterAsString(parameters, getattr(algorithm, "PARAM_GB", "PARAM_GB"), context)
        ),
        "Extra Trees": parse_param_string(
            algorithm.parameterAsString(parameters, getattr(algorithm, "PARAM_ET", "PARAM_ET"), context)
        ),
        "SVR": parse_param_string(
            algorithm.parameterAsString(parameters, getattr(algorithm, "PARAM_SVR", "PARAM_SVR"), context)
        ),
        "MLP (Neural Net)": parse_param_string(
            algorithm.parameterAsString(parameters, getattr(algorithm, "PARAM_MLP", "PARAM_MLP"), context)
        ),
        "Ridge": parse_param_string(
            algorithm.parameterAsString(parameters, getattr(algorithm, "PARAM_RIDGE", "PARAM_RIDGE"), context)
        ),
        "Lasso": parse_param_string(
            algorithm.parameterAsString(parameters, getattr(algorithm, "PARAM_LASSO", "PARAM_LASSO"), context)
        ),
        "ElasticNet": parse_param_string(
            algorithm.parameterAsString(parameters, getattr(algorithm, "PARAM_ELASTICNET", "PARAM_ELASTICNET"), context)
        ),
        "KNN": parse_param_string(
            algorithm.parameterAsString(parameters, getattr(algorithm, "PARAM_KNN", "PARAM_KNN"), context)
        ),
        "Decision Tree": parse_param_string(
            algorithm.parameterAsString(parameters, getattr(algorithm, "PARAM_DT", "PARAM_DT"), context)
        ),
        "Huber Regressor": parse_param_string(
            algorithm.parameterAsString(parameters, "PARAM_HUBER", context)
        ),
        "XGBoost": parse_param_string(
            algorithm.parameterAsString(parameters, "PARAM_XGB", context)
        ),
        "LightGBM": parse_param_string(
            algorithm.parameterAsString(parameters, "PARAM_LGBM", context)
        ),
        "CatBoost": parse_param_string(
            algorithm.parameterAsString(parameters, "PARAM_CATBOOST", context)
        ),
    }

    train_ratio = algorithm.parameterAsDouble(parameters, getattr(algorithm, "TRAIN_TEST_SPLIT", "TRAIN_TEST_SPLIT"), context)
    test_size = 1.0 - train_ratio
    if test_size <= 0.0 or test_size >= 1.0:
        test_size = 0.2
    
    random_state = algorithm.parameterAsInt(parameters, getattr(algorithm, "RANDOM_STATE", "RANDOM_STATE"), context)
    n_jobs = algorithm.parameterAsInt(parameters, getattr(algorithm, "NUM_THREADS", "NUM_THREADS"), context)
    
    cv_folds = 3
    if algorithm.parameterDefinition("CV_FOLDS"):
        try:
            val = algorithm.parameterAsInt(parameters, "CV_FOLDS", context)
            if val > 1: cv_folds = val
        except: pass

    uncert_trees = 50
    if algorithm.parameterDefinition("UNCERT_TREES"):
        try:
            val = algorithm.parameterAsInt(parameters, "UNCERT_TREES", context)
            if val > 0: uncert_trees = val
        except: pass

    fmt_idx = algorithm.parameterAsEnum(parameters, getattr(algorithm, "OUTPUT_FORMAT", "OUTPUT_FORMAT"), context)
    fmt_map = {0: "float32", 1: "float64", 2: "uint16"}
    output_format = fmt_map.get(fmt_idx, "float32")

    stack_layer = algorithm.parameterAsRasterLayer(
        parameters, getattr(algorithm, "INPUT_STACK", "INPUT_STACK"), context
    )
    stack_path = stack_layer.source() if stack_layer else parameters.get(getattr(algorithm, "INPUT_STACK", "INPUT_STACK"))

    mask_layer = algorithm.parameterAsRasterLayer(
        parameters, getattr(algorithm, "INPUT_MASK", "INPUT_MASK"), context
    )
    mask_path = mask_layer.source() if mask_layer else parameters.get(getattr(algorithm, "INPUT_MASK", "INPUT_MASK"))

    points_layer = algorithm.parameterAsVectorLayer(
        parameters, getattr(algorithm, "INPUT_POINTS", "INPUT_POINTS"), context
    )
    if points_layer is None:
        points_layer = parameters.get(getattr(algorithm, "INPUT_POINTS", "INPUT_POINTS"))
    depth_fld = algorithm.parameterAsString(parameters, getattr(algorithm, "FIELD_DEPTH", "FIELD_DEPTH"), context)
    weight_fld = algorithm.parameterAsString(
        parameters, getattr(algorithm, "FIELD_WEIGHT", "FIELD_WEIGHT"), context
    )
    if not weight_fld:
        weight_fld = None

    n_iter = algorithm.parameterAsInt(parameters, getattr(algorithm, "N_ITERATIONS", "N_ITERATIONS"), context)
    med_size = algorithm.parameterAsInt(parameters, getattr(algorithm, "MEDIAN_SIZE", "MEDIAN_SIZE"), context)
    sel_idx = algorithm.parameterAsEnums(parameters, getattr(algorithm, "SELECTED_ALGOS", "SELECTED_ALGOS"), context)
    opt_idx = algorithm.parameterAsInt(parameters, getattr(algorithm, "OPTIMIZER_METHOD", "OPTIMIZER_METHOD"), context)
    col_mode = algorithm.parameterAsInt(
        parameters, getattr(algorithm, "COLLISION_HANDLING", "COLLISION_HANDLING"), context
    )
    
    try:
        val_idx = algorithm.parameterAsEnum(parameters, getattr(algorithm, "FEATURE_CORR_THRESHOLD", "FEATURE_CORR_THRESHOLD"), context)
        corr_threshold = val_idx * 0.1
    except:
        try:
            corr_threshold = algorithm.parameterAsDouble(parameters, getattr(algorithm, "FEATURE_CORR_THRESHOLD", "FEATURE_CORR_THRESHOLD"), context)
        except:
            corr_threshold = 0.2

    try:
        corr_method_idx = algorithm.parameterAsEnum(parameters, getattr(algorithm, "FEATURE_CORR_METHOD", "FEATURE_CORR_METHOD"), context)
    except:
        corr_method_idx = 3

    append_log(
        f"MODULE 03 START: Optimizer = {OPTIMIZER_LIST[opt_idx]}", log_path, feedback
    )

    if pre_extracted_data is not None:
        X = pre_extracted_data["X"]
        y = pre_extracted_data["y"]
        final_weights = pre_extracted_data["weights"]
        coords = pre_extracted_data["coords"]
        append_log(">>> Using PRE-EXTRACTED Global Matrix data.", log_path, feedback)
    else:
        X, y, final_weights, coords = extract_samples(
            stack_path, points_layer, depth_fld, weight_fld, col_mode
        )
    if len(y) < 10:
        raise QgsProcessingException("Critically low training points (<10).")
    append_log(f"   Extracted {len(y)} training pixels.", log_path, feedback)

    actual_pts_path = os.path.join(out_dir, "3_Actual_Model_Input_Points.shp")
    
    # Try to extract feature names
    if pre_extracted_data is not None and "feature_names" in pre_extracted_data:
        feature_names = pre_extracted_data["feature_names"]
    else:
        feature_names = []
        try:
            import rasterio
            with rasterio.open(stack_path) as src:
                for i, desc in enumerate(src.descriptions):
                    if desc and str(desc).strip():
                        feature_names.append(str(desc).strip())
                    else:
                        feature_names.append(f"Band_{i+1}")
        except Exception:
            feature_names = [f"Band_{i+1}" for i in range(X.shape[1])]
        
        if not feature_names or len(feature_names) != X.shape[1]:
            feature_names = [f"Band_{i+1}" for i in range(X.shape[1])]

    save_training_points(
        actual_pts_path,
        coords,
        y,
        final_weights,
        X,
        stack_path,
        QgsRasterLayer(stack_path).crs(),
        feature_names
    )
    
    selected_indices = None
    if corr_method_idx == 3:
        append_log(f"   [Feature Analysis] Running Automatic-RANSAC Selection...", log_path, feedback)
        num_bands = X.shape[1]
        correlations = []
        method_name = "Automatic-RANSAC (Robust Pearson)"
        
        try:
            from sklearn.linear_model import RANSACRegressor, LinearRegression
        except ImportError:
            append_log(f"   [Warning] sklearn not found. Falling back to Pearson.", log_path, feedback)
            corr_method_idx = 1
            
    if corr_method_idx == 4:
        append_log(f"   [Feature Analysis] Running Automatic-Random Forest Selection...", log_path, feedback)
        num_bands = X.shape[1]
        method_name = "Automatic-Random Forest (Importance)"
        
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            append_log(f"   [Warning] sklearn not found. Falling back to Pearson.", log_path, feedback)
            corr_method_idx = 1

    if corr_method_idx == 3:
        for b in range(num_bands):
            X_b = X[:, b].reshape(-1, 1)
            try:
                ransac = RANSACRegressor(estimator=LinearRegression(), random_state=42)
                ransac.fit(X_b, y)
                inlier_mask = ransac.inlier_mask_
                
                if np.sum(inlier_mask) > 1:
                    r = np.corrcoef(X[inlier_mask, b], y[inlier_mask])[0, 1]
                else:
                    r = 0.0
            except Exception:
                r = 0.0
                
            if np.isnan(r):
                r = 0.0
            correlations.append(r)
            
        correlations = np.array(correlations)
        abs_correlations = np.abs(correlations)
        plot_scores = abs_correlations
        
        valid_scores = abs_correlations[abs_correlations > 0]
        if len(valid_scores) > 0:
            mean_score = float(np.mean(valid_scores))
            std_score = float(np.std(valid_scores))
            corr_threshold = max(0.3, mean_score - std_score)
        else:
            corr_threshold = 0.0
            
        append_log(f"   [Feature Analysis] Auto-Calculated Threshold = {corr_threshold:.3f}", log_path, feedback)
        selected_indices = np.where(abs_correlations >= corr_threshold)[0]
        
    elif corr_method_idx == 4:
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        plot_scores = rf.feature_importances_
        correlations = plot_scores
        
        corr_threshold = max(0.02, 1.0 / (num_bands * 2))
        append_log(f"   [Feature Analysis] Auto-Calculated RF Threshold = {corr_threshold:.3f}", log_path, feedback)
        selected_indices = np.where(plot_scores >= corr_threshold)[0]

    elif corr_method_idx in [1, 2] and corr_threshold > 0.0:
        append_log(f"   [Feature Analysis] Running with threshold >= {corr_threshold}", log_path, feedback)
        num_bands = X.shape[1]
        correlations = []
        method_name = "Spearman" if corr_method_idx == 2 else "Pearson"
        
        if corr_method_idx == 2:
            try:
                from scipy.stats import spearmanr
            except ImportError:
                method_name = "Pearson (Fallback)"
                corr_method_idx = 1
                
        for b in range(num_bands):
            std_X = np.std(X[:, b])
            std_y = np.std(y)
            if std_X == 0 or std_y == 0:
                r = 0.0
            else:
                if corr_method_idx == 2:
                    r = spearmanr(X[:, b], y)[0]
                else:
                    r = np.corrcoef(X[:, b], y)[0, 1]
            if np.isnan(r):
                r = 0.0
            correlations.append(r)
        
        correlations = np.array(correlations)
        abs_correlations = np.abs(correlations)
        plot_scores = abs_correlations
        selected_indices = np.where(abs_correlations >= corr_threshold)[0]
        
    if corr_method_idx > 0 and (corr_threshold > 0.0 or corr_method_idx in [3, 4]):
        if len(selected_indices) == 0:
            append_log(f"   [Warning] No bands met threshold {corr_threshold:.3f}. Using all bands.", log_path, feedback)
            selected_indices = np.arange(num_bands)
        else:
            selected_indices_list = [int(x) for x in selected_indices]
            append_log(f"   [Feature Analysis] Selected {len(selected_indices)} bands: {selected_indices_list}", log_path, feedback)
            X = X[:, selected_indices]
            
        report_path = os.path.join(out_dir, "3_Feature_Analysis_Report.txt")
        with open(report_path, "w") as f:
            f.write(f"Feature Analysis - {method_name}\n")
            if corr_method_idx in [3, 4]:
                f.write(f"Automatically Calculated Threshold: {corr_threshold:.3f}\n")
            f.write("-" * 50 + "\n")
            for b in range(num_bands):
                status = "Selected" if b in selected_indices else "Discarded"
                fname = feature_names[b]
                score_label = "Importance" if corr_method_idx == 4 else "abs(r)"
                f.write(f"{fname}: {score_label} = {plot_scores[b]:.4f} [{status}]\n")

        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(1, num_bands + 1), plot_scores, color='skyblue')
            plt.axhline(y=corr_threshold, color='r', linestyle='--', label=f'Threshold ({corr_threshold:.3f})')
            for i, b_bar in enumerate(bars):
                if i not in selected_indices:
                    b_bar.set_color('lightgray')
            plt.xlabel('Band Number')
            y_label = "Feature Importance" if corr_method_idx == 4 else f"Absolute {method_name} Correlation (|r|)"
            plt.ylabel(y_label)
            plt.title(f'Feature Analysis: {method_name}')
            plt.xticks(range(1, num_bands + 1), feature_names, rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "3_Feature_Correlation_Plot.png"), dpi=150)
            plt.close()
        except Exception as e:
            append_log(f"   [Warning] Failed to generate correlation plot: {e}", log_path, feedback)

    # Direct layer addition removed to prevent auto-loading in panel.
    # The output is returned to the processing framework instead.
    # QgsProject.instance().addMapLayer(QgsVectorLayer(actual_pts_path, "3_Actual_Model_Input_Points", "ogr"))

    try:
        enable_ensemble = algorithm.parameterAsBool(parameters, "ENABLE_ENSEMBLE", context)
    except Exception:
        enable_ensemble = False

    try:
        ens_idx = algorithm.parameterAsEnum(parameters, "ENSEMBLE_METHOD", context)
        ens_map = {0: "All Ensemble Methods (Auto-Select)", 1: "Average", 2: "Median", 3: "Stacking", 4: "Uncertainty-Weighted Fusion"}
        ensemble_method = ens_map.get(ens_idx, "All Ensemble Methods (Auto-Select)")
    except Exception:
        ensemble_method = "All Ensemble Methods (Auto-Select)"

    try:
        enable_depth_variance_corr = algorithm.parameterAsBool(parameters, "ENABLE_DEPTH_VARIANCE_CORR", context)
    except Exception:
        enable_depth_variance_corr = False

    spatial_cv = parameters.get("SPATIAL_CV")
    if not isinstance(spatial_cv, bool):
        try:
            spatial_cv = algorithm.parameterAsBool(parameters, "SPATIAL_CV", context)
        except Exception:
            spatial_cv = False

    try:
        ensemble_size = algorithm.parameterAsInt(parameters, "ENSEMBLE_SIZE", context)
    except Exception:
        ensemble_size = 3

    score_config = parse_score_config(algorithm, parameters, context)

    results_df, best_algo_data = run_benchmarking(
        X,
        y,
        final_weights,
        sel_idx,
        n_iter,
        out_dir,
        feedback,
        opt_idx,
        log_path,
        custom_params,
        test_size,
        random_state,
        n_jobs,
        enable_ensemble,
        ensemble_method,
        spatial_cv,
        coords,
        selected_indices,
        ensemble_size,
        feature_names,
        cv_folds,
        enable_depth_variance_corr=enable_depth_variance_corr,
        score_config=score_config
    )
    try:
        csv_cols = [c for c in ["Algorithm", "Stability", "Wins", "Mean_Score", "SDB_Score", "R2", "RMSE", "wMAPE", "Bias", "MAE", "Feature Scaling"] if c in results_df.columns]
        results_df[csv_cols].to_csv(
            os.path.join(out_dir, "3_All_Algorithms_Benchmark.csv"), index=False
        )
    except Exception:
        results_df.to_csv(
            os.path.join(out_dir, "3_All_Algorithms_Benchmark.csv"), index=False
        )

    # ---------------------------------------------------------
    # Generate strictly Linear Regression SDB ONLY if manually selected by user
    # ---------------------------------------------------------
    if pre_extracted_data is None:
        lr_algo_name = "Linear Regression"
        if lr_algo_name in results_df["Algorithm"].values:
            lr_model = results_df[results_df["Algorithm"] == lr_algo_name].iloc[0]["Model"]
            append_log("   [Analytics] Linear Regression was manually selected. Generating its depth map...", log_path, feedback)
            if lr_model is not None:
                lr_dir = os.path.join(out_dir, "Linear_Regression")
                os.makedirs(lr_dir, exist_ok=True)
                lr_map_path = os.path.join(lr_dir, "Linear_Regression_Depth.tif")
                lr_uncert_path = os.path.join(lr_dir, "Linear_Regression_Uncertainty.tif")
                try:
                    from sklearn.base import clone
                    final_lr_model = clone(lr_model)
                    fit_kwargs = {"sample_weight": final_weights}
                    final_lr_model.fit(X, y, **fit_kwargs)
                    
                    predict_map(final_lr_model, stack_path, mask_path, lr_map_path, med_size, output_format, selected_indices, feedback=feedback)
                    joblib.dump(final_lr_model, os.path.join(lr_dir, "Linear_Regression_Model.pkl"))

                    # Generate Uncertainty map for Linear Regression
                    y_lr_pred = final_lr_model.predict(X)
                    lr_abs_residuals = np.abs(y - y_lr_pred) * 1.96
                    from sklearn.ensemble import RandomForestRegressor
                    lr_uncert_model = RandomForestRegressor(n_estimators=uncert_trees, random_state=random_state, n_jobs=n_jobs)
                    lr_uncert_model.fit(X, lr_abs_residuals)
                    # Post-process LR depth map if cleanup enabled
                    try:
                        enable_slope_lr = algorithm.parameterAsBool(parameters, "ENABLE_SLOPE_FILTER", context) if (algorithm and hasattr(algorithm, "parameterDefinition") and algorithm.parameterDefinition("ENABLE_SLOPE_FILTER")) else False
                        remove_pos_lr = algorithm.parameterAsBool(parameters, "REMOVE_POSITIVES", context) if (algorithm and hasattr(algorithm, "parameterDefinition") and algorithm.parameterDefinition("REMOVE_POSITIVES")) else False
                        en_max_d_lr = algorithm.parameterAsBool(parameters, "ENABLE_MAX_DEPTH_FILTER", context) if (algorithm and hasattr(algorithm, "parameterDefinition") and algorithm.parameterDefinition("ENABLE_MAX_DEPTH_FILTER")) else False
                        max_depth_lr = algorithm.parameterAsDouble(parameters, "MAX_DEPTH_THRESHOLD", context) if (en_max_d_lr and algorithm and hasattr(algorithm, "parameterDefinition") and algorithm.parameterDefinition("MAX_DEPTH_THRESHOLD")) else -999999.0
                        slope_thresh_lr = algorithm.parameterAsDouble(parameters, "SLOPE_THRESHOLD", context) if (algorithm and hasattr(algorithm, "parameterDefinition") and algorithm.parameterDefinition("SLOPE_THRESHOLD")) else 35.0

                        if (en_max_d_lr or enable_slope_lr or remove_pos_lr) and os.path.exists(lr_map_path):
                            try:
                                from Bathymetrix_AI.infrastructure.raster_io import clean_depth_map, slope_filter_depth, remove_positive_pixels, safe_replace_or_copy
                            except (ImportError, ValueError):
                                from infrastructure.raster_io import clean_depth_map, slope_filter_depth, remove_positive_pixels, safe_replace_or_copy
                            cur_lr = lr_map_path
                            if en_max_d_lr:
                                lr_cleaned = os.path.join(lr_dir, "Linear_Regression_Cleaned.tif")
                                ref_mask = mask_path if mask_path and os.path.exists(mask_path) else stack_path
                                clean_depth_map(cur_lr, ref_mask, max_depth_lr, lr_cleaned, context, feedback)
                                cur_lr = lr_cleaned
                            if enable_slope_lr:
                                lr_slope = os.path.join(lr_dir, "Linear_Regression_SlopeFiltered.tif")
                                cur_lr = slope_filter_depth(cur_lr, slope_thresh_lr, lr_slope, context, feedback)
                            if remove_pos_lr:
                                lr_nopos = os.path.join(lr_dir, "Linear_Regression_NoPositives.tif")
                                remove_positive_pixels(cur_lr, lr_nopos, feedback)
                                cur_lr = lr_nopos
                            if cur_lr != lr_map_path:
                                safe_replace_or_copy(cur_lr, lr_map_path)
                    except Exception as e:
                        append_log(f"   [Warning] Linear Regression cleanup failed: {e}", log_path, feedback)

                    append_log(f"   [Analytics] Linear Regression depth & uncertainty analytics saved to: {lr_dir}", log_path, feedback)
                except Exception as e:
                    append_log(f"   [Warning] Failed to generate Linear Regression map: {e}", log_path, feedback)
    # ---------------------------------------------------------

    win_name = best_algo_data["name"]
    append_log(
        f"\n   >>> WINNER: {win_name} (Score={best_algo_data['score']:.4f})",
        log_path,
        feedback,
    )

    joblib.dump(
        best_algo_data["model"], os.path.join(out_dir, "3_Best_Global_Model.pkl")
    )
    p_map = os.path.join(out_dir, "3_Initial_Global_Depth.tif")

    if enable_depth_variance_corr:
        append_log("   [Phase 03] Fitting Depth Variance Correction (Huber) model...", log_path, feedback)
        try:
            from sklearn.linear_model import HuberRegressor
            y_train_pred = best_algo_data["model"].predict(X)
            raw_residuals = y - y_train_pred
            mean_bias = float(np.mean(raw_residuals))
            append_log(f"   [Phase 03] Raw mean residual offset for variance correction: {mean_bias:.4f}m", log_path, feedback)
            residuals = raw_residuals - mean_bias

            huber_model = HuberRegressor(epsilon=1.35)
            huber_model.fit(y_train_pred.reshape(-1, 1), residuals)
            
            joblib.dump({"model": huber_model, "bias": mean_bias}, os.path.join(out_dir, "3_Huber_Variance_Model.pkl"))
        except Exception as e:
            append_log(f"   [Warning] Failed to train Phase 03 Depth Variance Correction: {e}", log_path, feedback)
            huber_model = None
            mean_bias = 0.0

    append_log("   Generating prediction map...", log_path, feedback)
    if pre_extracted_data is None:
        predict_map(best_algo_data["model"], stack_path, mask_path, p_map, med_size, output_format, selected_indices, feedback=feedback)
        
        if enable_depth_variance_corr and 'huber_model' in locals() and huber_model is not None:
            append_log("   Applying Depth Variance Correction to prediction map...", log_path, feedback)
            try:
                with rasterio.open(p_map, "r+") as dst:
                    depth_arr = dst.read(1)
                    valid_mask = (depth_arr != -9999.0)
                    if np.any(valid_mask):
                        valid_depths = depth_arr[valid_mask]
                        residual_grid = huber_model.predict(valid_depths.reshape(-1, 1))
                        corrected_depths = valid_depths + residual_grid + mean_bias
                        depth_arr[valid_mask] = corrected_depths
                        dst.write(depth_arr, 1)
                append_log("   [Phase 03] Depth Variance Correction applied successfully.", log_path, feedback)
            except Exception as e:
                append_log(f"   [Warning] Failed to apply Depth Variance Correction: {e}", log_path, feedback)

        # Apply Post-Processing / Cleanup Filters if requested
        try:
            enable_slope = algorithm.parameterAsBool(parameters, "ENABLE_SLOPE_FILTER", context) if (algorithm and hasattr(algorithm, "parameterDefinition") and algorithm.parameterDefinition("ENABLE_SLOPE_FILTER")) else False
            remove_pos = algorithm.parameterAsBool(parameters, "REMOVE_POSITIVES", context) if (algorithm and hasattr(algorithm, "parameterDefinition") and algorithm.parameterDefinition("REMOVE_POSITIVES")) else False
            en_max_d = algorithm.parameterAsBool(parameters, "ENABLE_MAX_DEPTH_FILTER", context) if (algorithm and hasattr(algorithm, "parameterDefinition") and algorithm.parameterDefinition("ENABLE_MAX_DEPTH_FILTER")) else False
            max_depth = algorithm.parameterAsDouble(parameters, "MAX_DEPTH_THRESHOLD", context) if (en_max_d and algorithm and hasattr(algorithm, "parameterDefinition") and algorithm.parameterDefinition("MAX_DEPTH_THRESHOLD")) else -999999.0
            slope_thresh = algorithm.parameterAsDouble(parameters, "SLOPE_THRESHOLD", context) if (algorithm and hasattr(algorithm, "parameterDefinition") and algorithm.parameterDefinition("SLOPE_THRESHOLD")) else 35.0

            if (en_max_d or enable_slope or remove_pos) and os.path.exists(p_map):
                try:
                    from Bathymetrix_AI.infrastructure.raster_io import clean_depth_map, slope_filter_depth, remove_positive_pixels, safe_replace_or_copy
                except (ImportError, ValueError):
                    from infrastructure.raster_io import clean_depth_map, slope_filter_depth, remove_positive_pixels, safe_replace_or_copy
                append_log("   [Cleanup] Applying post-prediction cleanup filters to Phase 03 depth map...", log_path, feedback)
                cur_map = p_map
                if en_max_d:
                    p_cleaned = os.path.join(out_dir, "3_Initial_Global_Depth_Cleaned.tif")
                    ref_mask = mask_path if mask_path and os.path.exists(mask_path) else stack_path
                    clean_depth_map(cur_map, ref_mask, max_depth, p_cleaned, context, feedback)
                    cur_map = p_cleaned
                if enable_slope:
                    p_slope = os.path.join(out_dir, "3_Initial_Global_Depth_SlopeFiltered.tif")
                    cur_map = slope_filter_depth(cur_map, slope_thresh, p_slope, context, feedback)
                if remove_pos:
                    p_nopos = os.path.join(out_dir, "3_Initial_Global_Depth_NoPositives.tif")
                    remove_positive_pixels(cur_map, p_nopos, feedback)
                    cur_map = p_nopos
                
                if cur_map != p_map:
                    safe_replace_or_copy(cur_map, p_map)
                append_log("   [Cleanup] Phase 03 depth map cleanup completed.", log_path, feedback)
        except Exception as e:
            append_log(f"   [Warning] Post-prediction cleanup failed: {e}", log_path, feedback)

    else:
        append_log("   [Global Model] Skipping map generation since pre_extracted_data was used (to be done per-year).", log_path, feedback)
    # Direct layer addition removed to prevent auto-loading in panel.
    # The output is returned to the processing framework instead.
    # QgsProject.instance().addMapLayer(QgsRasterLayer(p_map, f"3_Initial_Global_Depth ({win_name})"))

    p_uncert_map = os.path.join(out_dir, "3_Initial_Global_Uncertainty.tif")
    p_uncert_model = os.path.join(out_dir, "3_Uncertainty_Global_Model.pkl")
    try:
        append_log("   Fitting Phase 03 uncertainty model (Empirical Residual Regressor)...", log_path, feedback)
        y_train_pred = best_algo_data["model"].predict(X)
        abs_residuals = np.abs(y - y_train_pred)
        uncert_y = abs_residuals * 1.96
        
        from sklearn.ensemble import RandomForestRegressor
        uncertainty_model = RandomForestRegressor(n_estimators=uncert_trees, random_state=random_state, n_jobs=n_jobs)
        uncertainty_model.fit(X, uncert_y)
        
        joblib.dump(uncertainty_model, p_uncert_model)
        
        append_log("   Generating Phase 03 uncertainty prediction map...", log_path, feedback)
        if pre_extracted_data is None:
            predict_map(uncertainty_model, stack_path, mask_path, p_uncert_map, med_size, "float32", selected_indices, feedback=feedback)
        else:
            append_log("   [Global Model] Skipping map generation since pre_extracted_data was used (to be done per-year).", log_path, feedback)
            p_uncert_map = None
    except Exception as e:
        append_log(f"   [Warning] Failed to generate Phase 03 uncertainty model: {e}", log_path, feedback)
        p_uncert_map = None
        p_uncert_model = None

    # Generate standardized ocean bathymetry .qml style alongside depth map
    if p_map and os.path.exists(p_map):
        try:
            from Bathymetrix_AI.infrastructure.raster_io import write_qml_style
            write_qml_style(p_map)
        except Exception:
            pass

    try:
        from Bathymetrix_AI.infrastructure.logging import log_module_completion
        primary_files = {
            "Phase 03 Depth Map": p_map,
            "Uncertainty Map": p_uncert_map,
            "Best ML Model": os.path.join(out_dir, "3_Best_Global_Model.pkl"),
            "Algorithms Benchmark": os.path.join(out_dir, "3_All_Algorithms_Benchmark.csv"),
            "Validation Plot": os.path.join(out_dir, "3_Validation_Scatter_Plot.png")
        }
        log_module_completion(
            module_title=f"Phase 03: Global Auto-ML Modeling (Winner: {win_name})",
            out_dir=out_dir,
            primary_files=primary_files,
            log_path=log_path,
            feedback=feedback
        )
    except Exception:
        pass

    return {
        "OUTPUT_DEPTH_MAP": p_map,
        "OUTPUT_UNCERT_MAP": p_uncert_map,
        "OUTPUT_MODEL_PKL": os.path.join(out_dir, "3_Best_Global_Model.pkl"),
        "OUTPUT_UNCERT_MODEL_PKL": p_uncert_model,
        "BEST_R2": best_algo_data["r2"],
        "BEST_RMSE": best_algo_data["rmse"],
        "BEST_MAE": best_algo_data.get("mae", 0.0),
        "BEST_BIAS": best_algo_data.get("bias", 0.0),
        "BEST_WMAPE": best_algo_data.get("wmape", 0.0),
        "SELECTED_INDICES": selected_indices
    }

