import ast
import os
import joblib
import warnings

import matplotlib
import numpy as np
import pandas as pd
import rasterio
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
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from .trainers import (
    CustomEnsembleRegressor,
    run_optuna_search,
    calculate_sdb_composite_score,
    parse_score_config,
    OPTUNA_AVAILABLE,
    XGB_AVAILABLE,
    LGBM_AVAILABLE,
    CATBOOST_AVAILABLE,
)

if XGB_AVAILABLE:
    import xgboost as xgb
if LGBM_AVAILABLE:
    import lightgbm as lgb
if CATBOOST_AVAILABLE:
    import catboost as cb

try:
    from qgis.core import (
        QgsCoordinateTransform,
        QgsProcessingException,
        QgsProject,
        QgsRasterLayer,
    )
except ImportError:
    QgsCoordinateTransform = None
    QgsProcessingException = Exception
    QgsProject = None
    QgsRasterLayer = None

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


def smooth_idw_weights(distances):
    return 1.0 / (distances + 1.0)

def parse_param_string(param_str):
    if not param_str or param_str.strip() == "":
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


def extract_values(ras, vec, fld, mode, logger_file, fb):
    from qgis.core import QgsProcessingException
    from Bathymetrix_AI.infrastructure.logging import append_log
    
    fields = vec.fields()
    field_names = [f.name() for f in fields]
    
    matched_field = None
    if fld:
        fld_clean = str(fld).strip()
        for fn in field_names:
            if fn == fld_clean or fn.lower() == fld_clean.lower() or fn.lower() == fld_clean.lower()[:10]:
                matched_field = fn
                break
                
    if not matched_field:
        err = f"❌ ERROR: Specified depth field '{fld}' was not found in layer '{vec.name()}'. Available fields in the layer are: {', '.join(field_names)}. Please specify the exact depth field name."
        append_log(err, logger_file, fb)
        raise QgsProcessingException(err)
        
    fld = matched_field

    with rasterio.open(ras) as ds:
        d = ds.read()
        h, w = ds.height, ds.width
        tr = QgsCoordinateTransform(
            vec.sourceCrs(), QgsRasterLayer(ras).crs(), QgsProject.instance()
        )
        X_out, y_out, c_out = [], [], []

        for f in vec.getFeatures():
            g = f.geometry()
            if g.isNull():
                continue
            g.transform(tr)
            try:
                pt = g.asMultiPoint()[0] if g.isMultipart() else g.asPoint()
                r, c = ds.index(pt.x(), pt.y())
                if 0 <= r < h and 0 <= c < w:
                    val = d[:, r, c]
                    if np.all(np.isfinite(val)) and not np.any(val == -9999):
                        X_out.append(val)
                        y_out.append(f[fld])
                        c_out.append([r, c])
            except Exception:
                continue

    return np.array(X_out), np.array(y_out), np.array(c_out)


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


class RobustSpatialKNN:
    def __init__(self, n_neighbors=15):
        self.n_neighbors = n_neighbors
        self.coords_tr = None
        self.residuals = None
        self.huber_w = None

    def fit(self, coords_tr, residuals):
        self.coords_tr = coords_tr
        self.residuals = residuals
        from sklearn.neighbors import NearestNeighbors
        
        # Use Leave-One-Out (LOO) to calculate true spatial errors
        k = min(self.n_neighbors + 1, len(coords_tr))
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(coords_tr)
        dists, indices = nn.kneighbors(coords_tr)
        
        # Exclude the point itself (first column is dist=0)
        loo_dists = dists[:, 1:]
        loo_indices = indices[:, 1:]
        
        # Use a +1.0 smoothing to prevent divide-by-zero & extreme spikes
        inv_dists = 1.0 / (loo_dists + 1.0)
        w_sum = np.sum(inv_dists, axis=1, keepdims=True)
        w_sum[w_sum == 0] = 1.0
        
        loo_preds = np.sum(inv_dists * residuals[loo_indices], axis=1) / w_sum.flatten()
        
        # Calculate robust Huber weights based on LOO errors
        errors = residuals - loo_preds
        mad = np.median(np.abs(errors))
        scale = 1.4826 * mad if mad > 0 else 1e-4
        
        self.huber_w = np.clip(1.35 * scale / (np.abs(errors) + 1e-6), 0.0, 1.0)
        return self

    def predict(self, X_query):
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        nn.fit(self.coords_tr)
        dists, indices = nn.kneighbors(X_query)
        
        # Use +1.0 smoothing here too so Huber weights can effectively downweight outliers
        inv_dists = 1.0 / (dists + 1.0)
        w = inv_dists * self.huber_w[indices]
        w_sum = np.sum(w, axis=1, keepdims=True)
        w_sum[w_sum == 0] = 1.0
        
        predictions = np.sum(w * self.residuals[indices], axis=1) / w_sum.flatten()
        return predictions


def run_phase04_spatial_retraining(algorithm, parameters, context, feedback):
    out_dir = algorithm.parameterAsString(parameters, algorithm.OUTPUT_FOLDER, context)
    os.makedirs(out_dir, exist_ok=True)
    log_path = algorithm.parameterAsString(parameters, algorithm.LOG_FILE, context)

    custom_params = {
        "Random Forest": parse_param_string(
            algorithm.parameterAsString(parameters, algorithm.PARAM_RF, context)
        ),
        "Gradient Boosting": parse_param_string(
            algorithm.parameterAsString(parameters, algorithm.PARAM_GB, context)
        ),
        "Extra Trees": parse_param_string(
            algorithm.parameterAsString(parameters, algorithm.PARAM_ET, context)
        ),
        "SVR": parse_param_string(
            algorithm.parameterAsString(parameters, algorithm.PARAM_SVR, context)
        ),
        "MLP (Neural Net)": parse_param_string(
            algorithm.parameterAsString(parameters, algorithm.PARAM_MLP, context)
        ),
        "Ridge": parse_param_string(
            algorithm.parameterAsString(parameters, algorithm.PARAM_RIDGE, context)
        ),
        "Lasso": parse_param_string(
            algorithm.parameterAsString(parameters, algorithm.PARAM_LASSO, context)
        ),
        "ElasticNet": parse_param_string(
            algorithm.parameterAsString(parameters, algorithm.PARAM_ELASTICNET, context)
        ),
        "KNN": parse_param_string(
            algorithm.parameterAsString(parameters, algorithm.PARAM_KNN, context)
        ),
        "Decision Tree": parse_param_string(
            algorithm.parameterAsString(parameters, algorithm.PARAM_DT, context)
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

    train_ratio = algorithm.parameterAsDouble(parameters, algorithm.TRAIN_TEST_SPLIT, context)
    test_size = 1.0 - train_ratio
    if test_size <= 0.0 or test_size >= 1.0:
        test_size = 0.2
        
    random_state = algorithm.parameterAsInt(parameters, algorithm.RANDOM_STATE, context)
    n_jobs = algorithm.parameterAsInt(parameters, algorithm.NUM_THREADS, context)
    if n_jobs == 0:
        n_jobs = -1
    
    score_config = parse_score_config(algorithm, parameters, context)
    
    max_gpr_samples = 1500
    if algorithm.parameterDefinition("MAX_GPR_SAMPLES"):
        try:
            val = algorithm.parameterAsInt(parameters, "MAX_GPR_SAMPLES", context)
            if val > 1: max_gpr_samples = val
        except: pass
        
    fmt_idx = algorithm.parameterAsEnum(parameters, algorithm.OUTPUT_FORMAT, context)
    fmt_map = {0: "float32", 1: "float64", 2: "uint16"}
    output_format = fmt_map.get(fmt_idx, "float32")

    global_path = algorithm.parameterAsRasterLayer(
        parameters, algorithm.INPUT_GLOBAL_RASTER, context
    ).source()
    feat_path = algorithm.parameterAsRasterLayer(
        parameters, algorithm.INPUT_ORIGINAL_FEAT, context
    ).source()
    mask_layer = algorithm.parameterAsRasterLayer(
        parameters, algorithm.INPUT_MASK, context
    )
    mask_path = mask_layer.source() if mask_layer else None
    train_lyr = algorithm.parameterAsVectorLayer(
        parameters, algorithm.INPUT_TRAIN, context
    )
    train_fld = algorithm.parameterAsString(parameters, algorithm.FIELD_TRAIN, context)

    sel_idx = algorithm.parameterAsEnums(parameters, algorithm.SELECTED_ALGOS, context)
    n_iter = algorithm.parameterAsInt(parameters, algorithm.N_ITERATIONS, context)
    med_size = algorithm.parameterAsInt(parameters, algorithm.MEDIAN_SIZE, context)
    opt_idx = algorithm.parameterAsInt(parameters, algorithm.OPTIMIZER_METHOD, context)

    col_mode = 0
    
    try:
        val_idx = algorithm.parameterAsEnum(parameters, algorithm.FEATURE_CORR_THRESHOLD, context)
        corr_threshold = val_idx * 0.1
    except:
        try:
            corr_threshold = algorithm.parameterAsDouble(parameters, algorithm.FEATURE_CORR_THRESHOLD, context)
        except:
            corr_threshold = 0.2

    try:
        corr_method_idx = algorithm.parameterAsEnum(parameters, algorithm.FEATURE_CORR_METHOD, context)
    except:
        corr_method_idx = 3

    append_log(
        f"MODULE 04 START: Refinement Phase | Opt={OPTIMIZER_LIST[opt_idx]}",
        log_path,
        feedback,
    )

    z_pred3, z_true, coords_tr = extract_values(
        global_path, train_lyr, train_fld, col_mode, log_path, feedback
    )

    if len(z_pred3) < 5:
        raise QgsProcessingException(
            "Not enough points for Module 4 (Spatial Refinement)."
        )

    try:
        enable_var_corr = algorithm.parameterAsBool(parameters, "ENABLE_DEPTH_VARIANCE_CORR", context)
    except Exception:
        enable_var_corr = False
    if not enable_var_corr:
        try:
            enable_var_corr = algorithm.parameterAsBool(parameters, "ENABLE_DEPTH_VARIANCE_CORR_P4", context)
        except Exception:
            enable_var_corr = parameters.get("ENABLE_DEPTH_VARIANCE_CORR", parameters.get("ENABLE_DEPTH_VARIANCE_CORR_P4", False))

    try:
        enable_spatial_residual = algorithm.parameterAsBool(parameters, "ENABLE_SPATIAL_RESIDUAL_CORR", context)
    except Exception:
        enable_spatial_residual = True
    if not enable_spatial_residual and isinstance(enable_spatial_residual, bool) is False:
        try:
            enable_spatial_residual = algorithm.parameterAsBool(parameters, "ENABLE_SPATIAL_RESIDUAL_CORR_P4", context)
        except Exception:
            enable_spatial_residual = parameters.get("ENABLE_SPATIAL_RESIDUAL_CORR", parameters.get("ENABLE_SPATIAL_RESIDUAL_CORR_P4", True))
    if enable_spatial_residual is None:
        enable_spatial_residual = True

    raw_residuals = z_true.flatten() - z_pred3.flatten()
    mean_bias = float(np.mean(raw_residuals)) if len(raw_residuals) > 0 else 0.0
    if np.isnan(mean_bias) or np.isinf(mean_bias):
        mean_bias = 0.0

    if enable_var_corr:
        append_log(f"   [Phase 04] Depth Variance Correction ENABLED (Control Points Mean Shift: {mean_bias:+.4f}m)", log_path, feedback)
        z_pred3 = z_pred3 + mean_bias
        residuals = z_true.flatten() - z_pred3.flatten()
        applied_shift = mean_bias
        residual_mean_bias = 0.0  # Zero-mean is baked into the shifted depth map baseline
    else:
        append_log(f"   [Phase 04] Raw mean residual offset: {mean_bias:.4f}m (Zero-mean centered for temporal consistency)", log_path, feedback)
        residuals = raw_residuals - mean_bias
        applied_shift = 0.0
        residual_mean_bias = mean_bias

    try:
        interp_idx = algorithm.parameterAsEnum(parameters, "RESIDUAL_INTERP_METHOD", context)
    except Exception:
        interp_idx = 0

    try:
        spatial_cv = algorithm.parameterAsBool(parameters, "SPATIAL_CV", context)
    except Exception:
        spatial_cv = False

    try:
        knn_k = algorithm.parameterAsInt(parameters, "KNN_NEIGHBORS", context)
    except Exception:
        knn_k = 15

    if mask_path and str(mask_path).strip() and mask_path != "None":
        with rasterio.open(mask_path) as m:
            mask_arr = m.read(1)
            meta = m.profile
            h, w = m.height, m.width
    else:
        with rasterio.open(global_path) as m:
            meta = m.profile
            h, w = m.height, m.width
        mask_arr = np.ones((h, w), dtype=np.uint8)

    with rasterio.open(global_path) as g:
        p3_map = g.read(1)

    if enable_var_corr and applied_shift != 0.0:
        valid_water_p3 = (p3_map != -9999.0)
        if mask_path and str(mask_path).strip() and mask_path != "None":
            valid_water_p3 = valid_water_p3 & (mask_arr > 0)
        p3_map[valid_water_p3] += applied_shift
        append_log(f"   [Phase 04] Applied Depth Variance Correction ({applied_shift:+.4f}m shift) across all valid depth pixels.", log_path, feedback)
        
    water_indices = np.where((mask_arr > 0) & (p3_map != -9999.0))
    water_coords = np.column_stack((water_indices[0], water_indices[1]))

    residual_grid = np.zeros((h, w), dtype="float32")
    spatial_model = None
    chunk_size = 5000

    if enable_spatial_residual:
        if interp_idx == 0:
            interp_name = "KNN Standard"
            append_log(f"   Fitting Standard Spatial KNN (K={knn_k})...", log_path, feedback)
            spatial_model = KNeighborsRegressor(n_neighbors=knn_k, weights=smooth_idw_weights, n_jobs=n_jobs)
            spatial_model.fit(coords_tr, residuals)
        elif interp_idx == 1:
            interp_name = "KNN Robust"
            append_log(f"   Fitting Robust Spatial KNN (Huber weights, K={knn_k})...", log_path, feedback)
            spatial_model = RobustSpatialKNN(n_neighbors=knn_k)
            spatial_model.fit(coords_tr, residuals)
        elif interp_idx == 2:
            interp_name = "Kriging/GP"
            append_log("   Fitting Gaussian Process Regression (Kriging)...", log_path, feedback)
            if len(coords_tr) > max_gpr_samples:
                np.random.seed(random_state)
                gpr_idx = np.random.choice(len(coords_tr), size=max_gpr_samples, replace=False)
                coords_gpr = coords_tr[gpr_idx]
                residuals_gpr = residuals[gpr_idx]
            else:
                coords_gpr = coords_tr
                residuals_gpr = residuals
            
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=100.0, length_scale_bounds=(10, 10000))
            spatial_model = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=2, random_state=random_state)
            spatial_model.fit(coords_gpr, residuals_gpr)

        append_log(f"   Interpolating Residual Grid ({interp_name})...", log_path, feedback)
        
        model_str = str(spatial_model)
        if "KNeighbors" in model_str or "GaussianProcess" in model_str or "RobustSpatialKNN" in model_str or "KNN" in model_str:
            chunk_size = 5000
        else:
            chunk_size = 500000
        for i in range(0, len(water_coords), chunk_size):
            chunk = water_coords[i:i + chunk_size]
            if len(chunk) > 0:
                residual_grid[chunk[:, 0], chunk[:, 1]] = spatial_model.predict(chunk)

        append_log("   Saving Residual Error Map...", log_path, feedback)
        p_residual = os.path.join(out_dir, "4-Residual_Error_Map.tif")
        meta_res = meta.copy()
        meta_res.update(count=1, dtype="float32", nodata=-9999.0)

        res_map_to_save = np.full((h, w), -9999.0, dtype="float32")
        res_map_to_save[water_indices] = residual_grid[water_indices] + residual_mean_bias

        with rasterio.open(p_residual, "w", **meta_res) as dst:
            dst.write(res_map_to_save, 1)
    else:
        append_log("   [Phase 04] Spatial Residual Correction is DISABLED. Bypassing spatial error surface fitting.", log_path, feedback)

    with rasterio.open(feat_path) as f:
        orig_bands = f.read()
        try:
            feat_names = []
            for i, d in enumerate(f.descriptions):
                if d and str(d).strip():
                    feat_names.append(str(d).strip())
                else:
                    feat_names.append(f"Band_{i+1}")
        except Exception:
            feat_names = [f"Band_{i+1}" for i in range(orig_bands.shape[0])]

    try:
        stack_comps = algorithm.parameterAsEnums(parameters, "STACK_COMPONENTS", context)
    except Exception:
        stack_comps = [0, 1, 2]

    stack_layers = []
    stack_names = []
    if 0 in stack_comps:
        stack_layers.append(orig_bands)
        stack_names.extend(feat_names)
    if 1 in stack_comps:
        stack_layers.append(p3_map[np.newaxis, :, :])
        stack_names.append("Phase03_Global_Depth")
    if 2 in stack_comps and enable_spatial_residual:
        stack_layers.append((residual_grid + residual_mean_bias)[np.newaxis, :, :])
        stack_names.append("Residual_Error_Grid")

    if not stack_layers:
        stack_layers.append(p3_map[np.newaxis, :, :])
        stack_names.append("Phase03_Global_Depth")

    if 0 not in stack_comps:
        append_log("\n   [Notice] Feature Stack (Phase 01) is not selected. Bypassing ML Refinement...", log_path, feedback)
        append_log("   Computing final depth purely using Direct Alignment / Spatial Addition.", log_path, feedback)
        
        final_map = np.full((h, w), -9999.0, dtype="float32")
        valid_mask = (p3_map != -9999.0)
        if mask_path and str(mask_path).strip() and mask_path != "None":
            valid_mask = valid_mask & (mask_arr > 0)
        
        if enable_spatial_residual:
            if 1 in stack_comps and 2 in stack_comps:
                final_map[valid_mask] = p3_map[valid_mask] + residual_grid[valid_mask] + residual_mean_bias
            elif 1 in stack_comps:
                final_map[valid_mask] = p3_map[valid_mask]
            elif 2 in stack_comps:
                final_map[valid_mask] = residual_grid[valid_mask] + residual_mean_bias
            else:
                final_map[valid_mask] = p3_map[valid_mask]
        else:
            final_map[valid_mask] = p3_map[valid_mask]
            
        if med_size > 0 and scipy_is_available:
            from scipy.ndimage import distance_transform_edt
            temp = final_map.copy()
            invalid = ~valid_mask
            if np.any(invalid):
                dist, inds = distance_transform_edt(invalid, return_indices=True)
                temp[invalid] = temp[inds[0][invalid], inds[1][invalid]]
            filtered = median_filter(temp, size=med_size)
            final_map[valid_mask] = filtered[valid_mask]

        nodata_val = -9999.0
        if output_format == "uint16":
            final_map[valid_mask] = np.clip(final_map[valid_mask], 0, None)
            final_map[~valid_mask] = 65535
            final_map = final_map.astype("uint16")
            nodata_val = 65535.0

        p_depth = os.path.join(out_dir, "Phase4_Adaptive_Depth.tif")
        meta.update(count=1, dtype=final_map.dtype, nodata=nodata_val)
        with rasterio.open(p_depth, "w", **meta) as dst:
            dst.write(final_map, 1)

        p_uncert = os.path.join(out_dir, "4-Refined_Uncertainty.tif")
        try:
            append_log("   Fitting Phase 04 Math Uncertainty spatial model...", log_path, feedback)
            if enable_spatial_residual and spatial_model is not None:
                math_residuals = residuals - spatial_model.predict(coords_tr)
            else:
                math_residuals = residuals
            abs_math_residuals = np.abs(math_residuals) * 1.96
            
            if interp_idx == 0:
                spatial_uncert_model = KNeighborsRegressor(n_neighbors=knn_k, weights=smooth_idw_weights, n_jobs=n_jobs)
            elif interp_idx == 1:
                spatial_uncert_model = RobustSpatialKNN(n_neighbors=knn_k)
            else:
                kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=100.0, length_scale_bounds=(10, 10000))
                spatial_uncert_model = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=2, random_state=random_state)
                
            spatial_uncert_model.fit(coords_tr, abs_math_residuals)
            
            append_log("   Interpolating Phase 04 Math Uncertainty Grid...", log_path, feedback)
            uncert_map = np.full((h, w), -9999.0, dtype="float32")
            for idx_c in range(0, len(water_coords), chunk_size):
                chunk_c = water_coords[idx_c:idx_c + chunk_size]
                if len(chunk_c) > 0:
                    uncert_map[chunk_c[:, 0], chunk_c[:, 1]] = spatial_uncert_model.predict(chunk_c)
        except Exception as e:
            append_log(f"   [Warning] Failed to interpolate math uncertainty grid spatially: {e}", log_path, feedback)
            uncert_map = np.full((h, w), -9999.0, dtype="float32")
            uncert_map[valid_mask] = 0.0
            
        meta.update(count=1, dtype="float32", nodata=-9999.0)
        with rasterio.open(p_uncert, "w", **meta) as dst:
            dst.write(uncert_map, 1)

        # Extract values for metrics and scatter plot
        X_val_math, y_val_math, _ = extract_values(
            p_depth, train_lyr, train_fld, col_mode, log_path, feedback
        )
        y_pred = X_val_math[:, 0].flatten() if X_val_math.ndim > 1 else X_val_math.flatten()
        y_true = y_val_math.flatten()
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        sum_abs_diff = np.sum(np.abs(y_true - y_pred))
        sum_abs_true = np.sum(np.abs(y_true))
        wmape = (sum_abs_diff / sum_abs_true) * 100 if sum_abs_true != 0 else 0
        
        append_log(f"   [Math Mode] RMSE: {rmse:.2f}m | R2: {r2:.3f} | wMAPE: {wmape:.1f}%", log_path, feedback)
        
        scatter_path = os.path.join(out_dir, "4_Phase04_Refined_Scatter.png")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from scipy.stats import gaussian_kde
            
            fig, ax = plt.subplots(figsize=(8, 6))
            use_kde = len(y_true) <= 5000
            if use_kde:
                try:
                    xy = np.vstack([y_true, y_pred])
                    z = gaussian_kde(xy)(xy)
                    sc = ax.scatter(y_true, y_pred, c=z, s=20, cmap="viridis", edgecolors="none")
                    plt.colorbar(sc, ax=ax, label="Point Density")
                except Exception:
                    ax.scatter(y_true, y_pred, c="navy", alpha=0.4, s=15)
            else:
                ax.scatter(y_true, y_pred, c="navy", alpha=0.3, s=10)
            
            min_val = min(np.min(y_true), np.min(y_pred))
            max_val = max(np.max(y_true), np.max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="1:1 Line")
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            ax.set_aspect("equal", adjustable="box")
            
            ax.set_title("Phase 04: Refined / Math Addition", fontsize=11, fontweight="bold")
            ax.set_xlabel("Observed Depth (m)")
            ax.set_ylabel("Predicted Depth (m)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
            plt.tight_layout()
            plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            append_log(f"   [Warning] Could not generate scatter plot: {e}", log_path, feedback)

        # Apply Post-Processing / Cleanup Filters if requested
        try:
            enable_slope = algorithm.parameterAsBool(parameters, "ENABLE_SLOPE_FILTER", context) if (algorithm and hasattr(algorithm, "parameterDefinition") and algorithm.parameterDefinition("ENABLE_SLOPE_FILTER")) else False
            remove_pos = algorithm.parameterAsBool(parameters, "REMOVE_POSITIVES", context) if (algorithm and hasattr(algorithm, "parameterDefinition") and algorithm.parameterDefinition("REMOVE_POSITIVES")) else False
            en_max_d = algorithm.parameterAsBool(parameters, "ENABLE_MAX_DEPTH_FILTER", context) if (algorithm and hasattr(algorithm, "parameterDefinition") and algorithm.parameterDefinition("ENABLE_MAX_DEPTH_FILTER")) else False
            max_depth = algorithm.parameterAsDouble(parameters, "MAX_DEPTH_THRESHOLD", context) if (en_max_d and algorithm and hasattr(algorithm, "parameterDefinition") and algorithm.parameterDefinition("MAX_DEPTH_THRESHOLD")) else -999999.0
            slope_thresh = algorithm.parameterAsDouble(parameters, "SLOPE_THRESHOLD", context) if (algorithm and hasattr(algorithm, "parameterDefinition") and algorithm.parameterDefinition("SLOPE_THRESHOLD")) else 35.0

            if (en_max_d or enable_slope or remove_pos) and os.path.exists(p_depth):
                from Bathymetrix_AI.infrastructure.raster_io import clean_depth_map, slope_filter_depth, remove_positive_pixels
                append_log("   [Cleanup] Applying post-prediction cleanup filters to Phase 04 depth map...", log_path, feedback)
                cur_map = p_depth
                if en_max_d:
                    p_cleaned = os.path.join(out_dir, "4_Phase04_Depth_Cleaned.tif")
                    ref_mask = mask_path if mask_path and os.path.exists(mask_path) else global_path
                    clean_depth_map(cur_map, ref_mask, max_depth, p_cleaned, context, feedback)
                    cur_map = p_cleaned
                if enable_slope:
                    p_slope = os.path.join(out_dir, "4_Phase04_Depth_SlopeFiltered.tif")
                    cur_map = slope_filter_depth(cur_map, slope_thresh, p_slope, context, feedback)
                if remove_pos:
                    p_nopos = os.path.join(out_dir, "4_Phase04_Depth_NoPositives.tif")
                    remove_positive_pixels(cur_map, p_nopos, feedback)
                    cur_map = p_nopos
                
                if cur_map != p_depth:
                    import shutil
                    shutil.copy2(cur_map, p_depth)
                append_log("   [Cleanup] Phase 04 depth map cleanup completed.", log_path, feedback)
        except Exception as e:
            append_log(f"   [Warning] Phase 04 cleanup failed: {e}", log_path, feedback)

        return {
            "OUTPUT_FINAL": p_depth,
            "OUTPUT_UNCERT": p_uncert,
            "BEST_RMSE": rmse,
            "BEST_R2": r2,
            "BEST_WMAPE": wmape,
        }

    stack = np.concatenate(stack_layers, axis=0)
    p_stack = os.path.join(out_dir, "Phase4_Input_Stack.tif")
    meta.update(count=stack.shape[0], dtype="float32")
    with rasterio.open(p_stack, "w", **meta) as dst:
        dst.write(stack.astype("float32"))

    X_final, y_final, _ = extract_values(
        p_stack, train_lyr, train_fld, col_mode, log_path, feedback
    )
    y_final = y_final.flatten()

    selected_indices = None
    if corr_method_idx == 3:
        append_log(f"   [Feature Analysis] Running Automatic-RANSAC Selection...", log_path, feedback)
        num_bands = X_final.shape[1]
        correlations = []
        method_name = "Automatic-RANSAC (Robust Pearson)"
        
        try:
            from sklearn.linear_model import RANSACRegressor, LinearRegression
        except ImportError:
            append_log(f"   [Warning] sklearn not found. Falling back to Pearson.", log_path, feedback)
            corr_method_idx = 1
            
    if corr_method_idx == 4:
        append_log(f"   [Feature Analysis] Running Automatic-Random Forest Selection...", log_path, feedback)
        num_bands = X_final.shape[1]
        method_name = "Automatic-Random Forest (Importance)"
        
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            append_log(f"   [Warning] sklearn not found. Falling back to Pearson.", log_path, feedback)
            corr_method_idx = 1
            
    if corr_method_idx == 3:
        for b in range(num_bands):
            X_b = X_final[:, b].reshape(-1, 1)
            try:
                ransac = RANSACRegressor(estimator=LinearRegression(), random_state=42)
                ransac.fit(X_b, y_final)
                inlier_mask = ransac.inlier_mask_
                
                if np.sum(inlier_mask) > 1:
                    r = np.corrcoef(X_final[inlier_mask, b], y_final[inlier_mask])[0, 1]
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
        rf.fit(X_final, y_final)
        plot_scores = rf.feature_importances_
        correlations = plot_scores
        
        corr_threshold = max(0.02, 1.0 / (num_bands * 2))
        append_log(f"   [Feature Analysis] Auto-Calculated RF Threshold = {corr_threshold:.3f}", log_path, feedback)
        selected_indices = np.where(plot_scores >= corr_threshold)[0]
            
    elif corr_method_idx in [1, 2] and corr_threshold > 0.0:
        append_log(f"   [Feature Analysis] Running with threshold >= {corr_threshold}", log_path, feedback)
        num_bands = X_final.shape[1]
        correlations = []
        method_name = "Spearman" if corr_method_idx == 2 else "Pearson"
        
        if corr_method_idx == 2:
            try:
                from scipy.stats import spearmanr
            except ImportError:
                method_name = "Pearson (Fallback)"
                corr_method_idx = 1
                
        for b in range(num_bands):
            std_X = np.std(X_final[:, b])
            std_y = np.std(y_final)
            if std_X == 0 or std_y == 0:
                r = 0.0
            else:
                if corr_method_idx == 2:
                    r = spearmanr(X_final[:, b], y_final)[0]
                else:
                    r = np.corrcoef(X_final[:, b], y_final)[0, 1]
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
            X_final = X_final[:, selected_indices]
            
        report_path = os.path.join(out_dir, "4_Feature_Analysis_Report.txt")
        with open(report_path, "w") as f:
            f.write(f"Phase 04 Feature Analysis - {method_name}\n")
            if corr_method_idx in [3, 4]:
                f.write(f"Automatically Calculated Threshold: {corr_threshold:.3f}\n")
            f.write("-" * 50 + "\n")
            for b in range(num_bands):
                status = "Selected" if b in selected_indices else "Discarded"
                fname = stack_names[b] if b < len(stack_names) else f"Feature_{b+1}"
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
            plt.xlabel('Feature Number')
            y_label = "Feature Importance" if corr_method_idx == 4 else f"Absolute {method_name} Correlation (|r|)"
            plt.ylabel(y_label)
            plt.title(f'Phase 04 Feature Analysis: {method_name}')
            x_labels = [stack_names[i] if i < len(stack_names) else f"F{i+1}" for i in range(num_bands)]
            plt.xticks(range(1, num_bands + 1), x_labels, rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "4_Feature_Correlation_Plot.png"), dpi=150)
            plt.close()
        except Exception as e:
            append_log(f"   [Warning] Failed to generate correlation plot: {e}", log_path, feedback)

    groups_tr = None
    if spatial_cv and coords_tr is not None:
        from sklearn.cluster import KMeans
        from sklearn.model_selection import GroupShuffleSplit
        kmeans = KMeans(n_clusters=5, random_state=random_state, n_init=10)
        spatial_groups = kmeans.fit_predict(coords_tr)
        
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, val_idx = next(gss.split(X_final, y_final, groups=spatial_groups))
        X_train, X_val = X_final[train_idx], X_final[val_idx]
        y_train, y_val = y_final[train_idx], y_final[val_idx]
        groups_tr = spatial_groups[train_idx]
        append_log(f"   [Spatial CV] Split Phase 04 retraining data into 5 geographic clusters.", log_path, feedback)
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_final, y_final, test_size=test_size, random_state=random_state
        )

    best_rmse = float("inf")
    best_model = None
    best_algo_name = ""
    best_r2 = 0.0
    best_wmape = 0.0

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
        ensemble_size = algorithm.parameterAsInt(parameters, "ENSEMBLE_SIZE", context)
    except Exception:
        ensemble_size = 3

    all_indices = [int(i) for i in sel_idx]
    base_indices = [i for i in all_indices if i < 15]
    ensemble_selected_indices = [i for i in all_indices if i >= 15]

    if not base_indices and ensemble_selected_indices:
        base_indices = [3, 12, 13, 14] # Extra Trees, XGBoost, LightGBM, CatBoost

    all_models_p4 = []

    for idx in base_indices:
        name, raw_model_inst, default_params = get_model_and_params(idx, opt_idx, random_state, n_jobs)
        if raw_model_inst is None:
            append_log(f"       ! Skipping {name}: Library is not installed.", log_path, feedback)
            continue

        if name in custom_params and custom_params[name]:
            parsed_dict = custom_params[name]
            base_params = {k: (v[0] if isinstance(v, list) and len(v)>0 else v) for k, v in parsed_dict.items()}
            raw_model_inst.set_params(**base_params)

            if opt_idx == 2 and SKOPT_AVAILABLE:
                params = convert_to_bayes(parsed_dict)
            else:
                params = parsed_dict
        else:
            params = default_params

        try:
            with joblib.parallel_backend("threading", n_jobs=n_jobs):
                if params and n_iter > 0:
                    search = None
                    current_opt_idx = opt_idx
                    if name == "MLP (Neural Net)":
                        current_opt_idx = 0

                    cv_splitter = 3
                    if groups_tr is not None:
                        n_unique_groups = len(np.unique(groups_tr))
                        if n_unique_groups >= 2:
                            from sklearn.model_selection import GroupKFold
                            cv_splitter = GroupKFold(n_splits=min(3, n_unique_groups))
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
                            clone(raw_model_inst), opt_params, n_iter=n_iter, cv=cv_splitter, n_jobs=n_jobs, random_state=random_state
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
                        search = GridSearchCV(clone(raw_model_inst), opt_params, cv=cv_splitter, n_jobs=n_jobs)
                    elif current_opt_idx == 2:
                        if OPTUNA_AVAILABLE:
                            best_m, best_params = run_optuna_search(
                                clone(raw_model_inst), name, params, X_train, y_train, {},
                                n_iter=n_iter, cv=3, random_state=random_state, n_jobs=n_jobs,
                                groups=groups_tr
                            )
                            curr_model = best_m
                            curr_model.fit(X_train, y_train)
                            search = None
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
                                BayesSearchCV(clone(raw_model_inst), params, n_iter=n_iter, cv=cv_splitter, n_jobs=n_jobs)
                                if SKOPT_AVAILABLE
                                else RandomizedSearchCV(
                                    clone(raw_model_inst), opt_params, n_iter=n_iter, cv=cv_splitter, n_jobs=n_jobs, random_state=random_state
                                )
                            )

                    if search:
                        if groups_tr is not None:
                            search.fit(X_train, y_train, groups=groups_tr)
                        else:
                            search.fit(X_train, y_train)
                        curr_model = search.best_estimator_
                    elif not OPTUNA_AVAILABLE or current_opt_idx != 2:
                        curr_model = clone(raw_model_inst)
                        curr_model.fit(X_train, y_train)
                else:
                    curr_model = clone(raw_model_inst)
                    curr_model.fit(X_train, y_train)

            y_pred = curr_model.predict(X_val)
            rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
            r2 = float(r2_score(y_val, y_pred))
            sum_abs_diff = np.sum(np.abs(y_val - y_pred))
            sum_abs_true = np.sum(np.abs(y_val))
            wmape = float((sum_abs_diff / sum_abs_true) * 100 if sum_abs_true != 0 else 0.0)
            bias = float(np.mean(y_pred - y_val))
            mae = float(mean_absolute_error(y_val, y_pred))

            append_log(f"       > {name}: R2={r2:.3f}, RMSE={rmse:.2f}m, wMAPE={wmape:.1f}%, Bias={bias:+.3f}m, MAE={mae:.2f}m", log_path, feedback)
            all_models_p4.append({
                "Algorithm": name,
                "Feature Scaling": "None",
                "Model": curr_model,
                "RMSE": rmse,
                "R2": r2,
                "wMAPE": wmape,
                "Bias": bias,
                "MAE": mae,
            })

        except Exception as e:
            append_log(f"Error in {name}: {str(e)}", log_path, feedback)

    if not all_models_p4:
        raise QgsProcessingException("All refinement models failed.")

    has_ensemble_selected = len(ensemble_selected_indices) > 0 or enable_ensemble
    if has_ensemble_selected and len(all_models_p4) >= 2:
        df_base_p4 = calculate_sdb_composite_score(pd.DataFrame(all_models_p4), random_state=random_state, score_config=score_config)
        top_p4 = df_base_p4.sort_values(by="SDB_Score", ascending=False).head(ensemble_size).to_dict("records")
        estimators_p4 = [(m["Algorithm"], m["Model"]) for m in top_p4]
        append_log(f"   [Ensemble] Blending top {len(estimators_p4)} base models ({[m['Algorithm'] for m in top_p4]})", log_path, feedback)

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
                ensemble_model = CustomEnsembleRegressor(estimators=estimators_p4, method=m_name)
                ensemble_model.fit(X_train, y_train)
                
                y_pred = ensemble_model.predict(X_val)
                ens_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
                ens_r2 = float(r2_score(y_val, y_pred))
                sum_abs_diff = np.sum(np.abs(y_val - y_pred))
                sum_abs_true = np.sum(np.abs(y_val))
                ens_wmape = float((sum_abs_diff / sum_abs_true) * 100 if sum_abs_true != 0 else 0.0)
                ens_bias = float(np.mean(y_pred - y_val))
                ens_mae = float(mean_absolute_error(y_val, y_pred))

                all_models_p4.append({
                    "Algorithm": f"Ensemble ({m_name})",
                    "Feature Scaling": "Composite",
                    "Model": ensemble_model,
                    "RMSE": ens_rmse,
                    "R2": ens_r2,
                    "wMAPE": ens_wmape,
                    "Bias": ens_bias,
                    "MAE": ens_mae,
                })
                append_log(f"       > Ensemble ({m_name}): R2={ens_r2:.3f}, RMSE={ens_rmse:.2f}m, wMAPE={ens_wmape:.1f}%, Bias={ens_bias:+.3f}m, MAE={ens_mae:.2f}m", log_path, feedback)
            except Exception as e:
                append_log(f"       ! Failed Ensemble ({m_name}): {e}", log_path, feedback)

    df_p4 = pd.DataFrame(all_models_p4)
    df_p4 = calculate_sdb_composite_score(df_p4, random_state=random_state, score_config=score_config, out_dir=out_dir, prefix="4_")
    winner_p4 = df_p4.iloc[0]

    try:
        csv_cols = [c for c in ["Algorithm", "Stability", "Wins", "Mean_Score", "SDB_Score", "R2", "RMSE", "wMAPE", "Bias", "MAE", "Feature Scaling"] if c in df_p4.columns]
        df_p4[csv_cols].to_csv(os.path.join(out_dir, "4_All_Algorithms_Benchmark.csv"), index=False)
    except Exception:
        df_p4.to_csv(os.path.join(out_dir, "4_All_Algorithms_Benchmark.csv"), index=False)

    best_model = winner_p4["Model"]
    best_algo_name = winner_p4["Algorithm"]
    best_rmse = winner_p4["RMSE"]
    best_r2 = winner_p4["R2"]
    best_wmape = winner_p4["wMAPE"]

    strat_title_p4 = score_config.get("strategy_name", "Winner Stability & SDB Composite Ranking") if score_config else "Winner Stability & SDB Composite Ranking"
    append_log(f"\n   📊 [Phase 04 Auto-ML Leaderboard - Selection Strategy: {strat_title_p4}]:", log_path, feedback)
    for rank, row in df_p4.iterrows():
        prefix = "🥇" if rank == 0 else ("🥈" if rank == 1 else ("🥉" if rank == 2 else "  "))
        append_log(
            f"      {prefix} {row['Algorithm']:<32} | Stability: {row['Stability']:>5.1f}% ({row['Wins']:>5}) | Mean Score: {row['Mean_Score']:>5.2f} | Baseline Score: {row['SDB_Score']:>5.2f} | R²={row['R2']:>6.4f} | RMSE={row['RMSE']:>5.2f}m | wMAPE={row['wMAPE']:>5.2f}% | Bias={row['Bias']:>+6.3f}m",
            log_path, feedback
        )

    append_log(f"\n   ⭐ Winner Selected: {best_algo_name} (Stability: {winner_p4['Stability']:.1f}% [{winner_p4['Wins']}], SDB Score: {winner_p4['SDB_Score']:.2f}, Mean Score: {winner_p4['Mean_Score']:.2f}/100)", log_path, feedback)

    try:
        from .trainers import export_feature_importance
        export_feature_importance(
            best_model,
            best_algo_name,
            X_val,
            y_val,
            out_dir,
            log_path,
            feedback,
            selected_indices,
            feature_names=stack_names
        )
    except Exception as e:
        append_log(f"   [Warning] Failed to generate feature importance: {e}", log_path, feedback)

    append_log(f"   Predicting Final Map using {best_algo_name}...", log_path, feedback)
    n_water_pts = len(water_indices[0])
    z_out = np.empty(n_water_pts, dtype="float32")
    model_str = str(best_model)
    if "KNeighbors" in model_str or "GaussianProcess" in model_str or "RobustSpatialKNN" in model_str or "KNN" in model_str:
        chunk_size = 5000
    else:
        chunk_size = 500000

    for start in range(0, n_water_pts, chunk_size):
        end = min(start + chunk_size, n_water_pts)
        idx_r = (water_indices[0][start:end], water_indices[1][start:end])
        X_chunk = stack[:, idx_r[0], idx_r[1]].T
        if selected_indices is not None and len(selected_indices) > 0:
            X_chunk = X_chunk[:, selected_indices]
        np.nan_to_num(X_chunk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        z_out[start:end] = best_model.predict(X_chunk)

    final_map = np.full((h, w), -9999.0, dtype="float32")
    final_map[water_indices] = z_out

    if med_size > 0 and scipy_is_available:
        valid = final_map != -9999
        temp = final_map.copy()
        temp[~valid] = np.nan
        filtered = median_filter(temp, size=med_size)
        final_map[valid] = filtered[valid]

    nodata_val = -9999.0
    if output_format == "uint16":
        valid = final_map != -9999.0
        final_map[valid] = np.clip(final_map[valid], 0, None)
        final_map[~valid] = 65535
        final_map = final_map.astype("uint16")
        nodata_val = 65535.0
    elif output_format == "float64":
        final_map = final_map.astype("float64")
    else:
        final_map = final_map.astype("float32")

    p_final = os.path.join(out_dir, "4-Refined_Model.tif")
    meta.update(count=1, dtype=output_format, nodata=nodata_val)
    with rasterio.open(p_final, "w", **meta) as dst:
        dst.write(final_map, 1)

    p_uncert = os.path.join(out_dir, "4-Refined_Uncertainty.tif")
    try:
        append_log("   Fitting Phase 04 uncertainty model (Empirical Residual Regressor)...", log_path, feedback)
        y_train_pred = best_model.predict(X_train)
        abs_residuals = np.abs(y_train - y_train_pred)
        uncert_y = abs_residuals * 1.96
        
        from sklearn.ensemble import RandomForestRegressor
        uncertainty_model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=random_state, n_jobs=n_jobs)
        uncertainty_model.fit(X_train, uncert_y)
        
        append_log("   Generating Phase 04 uncertainty prediction map...", log_path, feedback)
        uncert_out = np.empty(n_water_pts, dtype="float32")
        for start in range(0, n_water_pts, chunk_size):
            end = min(start + chunk_size, n_water_pts)
            idx_r = (water_indices[0][start:end], water_indices[1][start:end])
            X_chunk = stack[:, idx_r[0], idx_r[1]].T
            if selected_indices is not None and len(selected_indices) > 0:
                X_chunk = X_chunk[:, selected_indices]
            np.nan_to_num(X_chunk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            uncert_out[start:end] = uncertainty_model.predict(X_chunk)
        
        uncert_map = np.full((h, w), -9999.0, dtype="float32")
        uncert_map[water_indices] = uncert_out
        
        meta_uncert = meta.copy()
        meta_uncert.update(count=1, dtype="float32", nodata=-9999.0)
        with rasterio.open(p_uncert, "w", **meta_uncert) as dst:
            dst.write(uncert_map, 1)
    except Exception as e:
        append_log(f"   [Warning] Failed to generate Phase 04 uncertainty map: {e}", log_path, feedback)
        try:
            uncert_map = np.full((h, w), -9999.0, dtype="float32")
            uncert_map[water_indices] = 0.0
            meta_uncert = meta.copy()
            meta_uncert.update(count=1, dtype="float32", nodata=-9999.0)
            with rasterio.open(p_uncert, "w", **meta_uncert) as dst:
                dst.write(uncert_map, 1)
        except Exception:
            p_uncert = None

    try:
        csv_cols = [c for c in ["Algorithm", "Stability", "Wins", "Mean_Score", "SDB_Score", "R2", "RMSE", "wMAPE", "Bias", "MAE", "Feature Scaling"] if c in df_p4.columns]
        df_p4[csv_cols].to_csv(
            os.path.join(out_dir, "4_All_Algorithms_Benchmark.csv"), index=False
        )
    except Exception as e:
        append_log(f"   [Warning] Failed to write Phase 04 benchmark CSV: {e}", log_path, feedback)

    # Apply Post-Processing / Cleanup Filters if requested
    try:
        enable_slope = algorithm.parameterAsBool(parameters, "ENABLE_SLOPE_FILTER", context) if (algorithm and hasattr(algorithm, "parameterDefinition") and algorithm.parameterDefinition("ENABLE_SLOPE_FILTER")) else False
        remove_pos = algorithm.parameterAsBool(parameters, "REMOVE_POSITIVES", context) if (algorithm and hasattr(algorithm, "parameterDefinition") and algorithm.parameterDefinition("REMOVE_POSITIVES")) else False
        en_max_d = algorithm.parameterAsBool(parameters, "ENABLE_MAX_DEPTH_FILTER", context) if (algorithm and hasattr(algorithm, "parameterDefinition") and algorithm.parameterDefinition("ENABLE_MAX_DEPTH_FILTER")) else False
        max_depth = algorithm.parameterAsDouble(parameters, "MAX_DEPTH_THRESHOLD", context) if (en_max_d and algorithm and hasattr(algorithm, "parameterDefinition") and algorithm.parameterDefinition("MAX_DEPTH_THRESHOLD")) else -999999.0
        slope_thresh = algorithm.parameterAsDouble(parameters, "SLOPE_THRESHOLD", context) if (algorithm and hasattr(algorithm, "parameterDefinition") and algorithm.parameterDefinition("SLOPE_THRESHOLD")) else 35.0

        if (en_max_d or enable_slope or remove_pos) and p_final and os.path.exists(p_final):
            from Bathymetrix_AI.infrastructure.raster_io import clean_depth_map, slope_filter_depth, remove_positive_pixels
            append_log("   [Cleanup] Applying post-prediction cleanup filters to Phase 04 depth map...", log_path, feedback)
            cur_map = p_final
            if en_max_d:
                p_cleaned = os.path.join(out_dir, "4_Phase04_Depth_Cleaned.tif")
                ref_mask = mask_path if mask_path and os.path.exists(mask_path) else global_path
                clean_depth_map(cur_map, ref_mask, max_depth, p_cleaned, context, feedback)
                cur_map = p_cleaned
            if enable_slope:
                p_slope = os.path.join(out_dir, "4_Phase04_Depth_SlopeFiltered.tif")
                cur_map = slope_filter_depth(cur_map, slope_thresh, p_slope, context, feedback)
            if remove_pos:
                p_nopos = os.path.join(out_dir, "4_Phase04_Depth_NoPositives.tif")
                remove_positive_pixels(cur_map, p_nopos, feedback)
                cur_map = p_nopos
            
            if cur_map != p_final:
                import shutil
                shutil.copy2(cur_map, p_final)
            append_log("   [Cleanup] Phase 04 depth map cleanup completed.", log_path, feedback)
    except Exception as e:
        append_log(f"   [Warning] Phase 04 cleanup failed: {e}", log_path, feedback)

    # Generate standardized ocean bathymetry .qml style alongside Phase 04 depth map
    if p_final and os.path.exists(p_final):
        try:
            from Bathymetrix_AI.infrastructure.raster_io import write_qml_style
            write_qml_style(p_final)
        except Exception:
            pass

    try:
        from Bathymetrix_AI.infrastructure.logging import log_module_completion
        primary_files = {
            "Phase 04 Refined Map": p_final,
            "Residual Map": p_residual,
            "Uncertainty Map": p_uncert,
            "Algorithms Benchmark": os.path.join(out_dir, "4_All_Algorithms_Benchmark.csv"),
            "Phase 04 Model": os.path.join(out_dir, "4_Phase04_Adaptive_Model.pkl")
        }
        log_module_completion(
            module_title=f"Phase 04: Spatial Error Calibration (Winner: {best_algo_name})",
            out_dir=out_dir,
            primary_files=primary_files,
            log_path=log_path,
            feedback=feedback
        )
    except Exception:
        pass

    return {
        "OUTPUT_FINAL": p_final,
        "OUTPUT_RESIDUAL": p_residual,
        "OUTPUT_UNCERT": p_uncert,
        "BEST_R2": best_r2,
        "BEST_RMSE": best_rmse,
        "BEST_WMAPE": best_wmape,
    }
