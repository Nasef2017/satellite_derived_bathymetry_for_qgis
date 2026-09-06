import warnings
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFile,
    QgsProcessingParameterString,
    QgsProcessingParameterBoolean,
    QgsProcessingOutputRasterLayer,
    QgsProcessingOutputNumber,
    QgsProcessingParameterDefinition,
)

warnings.filterwarnings("ignore")


class SDBModule03(QgsProcessingAlgorithm):
    INPUT_STACK = "INPUT_STACK"
    INPUT_MASK = "INPUT_MASK"
    INPUT_POINTS = "INPUT_POINTS"
    FIELD_DEPTH = "FIELD_DEPTH"
    FIELD_WEIGHT = "FIELD_WEIGHT"
    SELECTED_ALGOS = "SELECTED_ALGOS"
    OPTIMIZER_METHOD = "OPTIMIZER_METHOD"
    COLLISION_HANDLING = "COLLISION_HANDLING"
    N_ITERATIONS = "N_ITERATIONS"
    MEDIAN_SIZE = "MEDIAN_SIZE"
    FEATURE_CORR_THRESHOLD = "FEATURE_CORR_THRESHOLD"
    FEATURE_CORR_METHOD = "FEATURE_CORR_METHOD"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"
    LOG_FILE = "LOG_FILE"
    OUTPUT_DEPTH_MAP = "OUTPUT_DEPTH_MAP"
    BEST_R2 = "BEST_R2"
    BEST_RMSE = "BEST_RMSE"

    TRAIN_TEST_SPLIT = "TRAIN_TEST_SPLIT"
    RANDOM_STATE = "RANDOM_STATE"
    NUM_THREADS = "NUM_THREADS"
    CV_FOLDS = "CV_FOLDS"
    UNCERT_TREES = "UNCERT_TREES"
    OUTPUT_FORMAT = "OUTPUT_FORMAT"

    PARAM_RF = "PARAM_RF"
    PARAM_GB = "PARAM_GB"
    PARAM_ET = "PARAM_ET"
    PARAM_SVR = "PARAM_SVR"
    PARAM_MLP = "PARAM_MLP"
    PARAM_RIDGE = "PARAM_RIDGE"
    PARAM_LASSO = "PARAM_LASSO"
    PARAM_ELASTICNET = "PARAM_ELASTICNET"
    PARAM_KNN = "PARAM_KNN"
    PARAM_DT = "PARAM_DT"
    PARAM_HUBER = "PARAM_HUBER"
    PARAM_XGB = "PARAM_XGB"
    PARAM_LGBM = "PARAM_LGBM"
    PARAM_CATBOOST = "PARAM_CATBOOST"

    ENABLE_ENSEMBLE = "ENABLE_ENSEMBLE"
    ENSEMBLE_METHOD = "ENSEMBLE_METHOD"
    ENSEMBLE_SIZE = "ENSEMBLE_SIZE"
    SPATIAL_CV = "SPATIAL_CV"

    SCORE_SELECTION_STRATEGY = "SCORE_SELECTION_STRATEGY"
    SCORE_METRICS = "SCORE_METRICS"
    SCORE_CUSTOM_CONFIG = "SCORE_CUSTOM_CONFIG"

    SCORE_STRATEGY_OPTIONS = [
        "Winner Stability (Monte Carlo Sensitivity Analysis) [Default]",
        "Highest SDB Composite Score (Max Baseline Score 0-100)",
        "Highest R² Accuracy",
        "Lowest RMSE (Minimum Vertical Error)",
        "Lowest wMAPE (%)",
        "Lowest |Bias| (Zero-Mean Residual Offset)",
        "Lowest MAE (Mean Absolute Error)",
    ]

    SCORE_METRIC_OPTIONS = [
        "R² Accuracy (Correlation & Explained Variance)",
        "RMSE (Root Mean Squared Vertical Error)",
        "wMAPE (Weighted Mean Absolute Percentage Error)",
        "|Bias| (Zero-Mean Residual Shift Offset)",
        "MAE (Mean Absolute Error)",
    ]

    REMOVE_POSITIVES = "REMOVE_POSITIVES"
    ENABLE_SLOPE_FILTER = "ENABLE_SLOPE_FILTER"
    SLOPE_THRESHOLD = "SLOPE_THRESHOLD"
    MAX_DEPTH_THRESHOLD = "MAX_DEPTH_THRESHOLD"

    FEATURE_CORR_THRESHOLDS = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]

    MODEL_LIST = [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting",
        "Extra Trees",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "KNN",
        "Decision Tree",
        "MLP (Neural Net)",
        "SVR",
        "Huber Regressor",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "Ensemble (Average)",
        "Ensemble (Median)",
        "Ensemble (Stacking)",
        "Ensemble (Uncertainty-Weighted Fusion)",
    ]
    OPTIMIZER_LIST = ["Random Search", "Grid Search", "Bayesian Search"]
    COLLISION_LIST = [
        "Keep All Points",
        "Highest Confidence",
        "Closest to Pixel Center",
        "Hybrid",
        "Strict Center",
    ]

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(self.INPUT_STACK, "Input Feature Stack")
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_MASK, "Input Water Mask", optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_POINTS, "Cleaned Training Points"
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_DEPTH,
                "Depth Field",
                defaultValue="ortho_h",
                parentLayerParameterName=self.INPUT_POINTS,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_WEIGHT,
                "Weight Field",
                defaultValue="confidence",
                parentLayerParameterName=self.INPUT_POINTS,
                type=QgsProcessingParameterField.Numeric,
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.SELECTED_ALGOS,
                "Select Algorithms",
                options=self.MODEL_LIST,
                allowMultiple=True,
                defaultValue=[3, 12, 13, 14, 15, 17], # Extra Trees, XGBoost, LightGBM, CatBoost, Ensemble Average, Ensemble Stacking
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.OPTIMIZER_METHOD,
                "Optimizer",
                options=self.OPTIMIZER_LIST,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.COLLISION_HANDLING,
                "Collision Handling",
                options=self.COLLISION_LIST,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_ITERATIONS,
                "Optimization Iterations",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=10,
            )
        )
        p_ens_size = QgsProcessingParameterNumber(
            self.ENSEMBLE_SIZE,
            "📊 Ensemble Size (Top N Models to blend)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=3,
            minValue=2,
            maxValue=5,
        )
        p_ens_size.setFlags(p_ens_size.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_ens_size)
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.SPATIAL_CV,
                "🌍 Enable Spatial Block Cross-Validation",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.FEATURE_CORR_METHOD,
                "Feature Correlation Method",
                options=["Disabled", "Pearson (Linear)", "Spearman (Rank)", "Automatic-RANSAC", "Automatic-Random Forest"],
                defaultValue=3,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.FEATURE_CORR_THRESHOLD,
                "Feature Correlation Threshold",
                options=self.FEATURE_CORR_THRESHOLDS,
                defaultValue=2,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.MEDIAN_SIZE, "Output Median Filter Size", defaultValue=3
            )
        )
        self.addParameter(
            QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, "Output Folder")
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.LOG_FILE, "Log File (Optional)", optional=True
            )
        )

        p_split = QgsProcessingParameterNumber(
            self.TRAIN_TEST_SPLIT, "Training Data Ratio (e.g., 0.8 for 80%)", type=QgsProcessingParameterNumber.Double, defaultValue=0.8
        )
        p_split.setFlags(p_split.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_split)

        p_rs = QgsProcessingParameterNumber(
            self.RANDOM_STATE, "Random State", type=QgsProcessingParameterNumber.Integer, defaultValue=42
        )
        p_rs.setFlags(p_rs.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_rs)

        p_thr = QgsProcessingParameterNumber(
            self.NUM_THREADS, "Num Threads (n_jobs)", type=QgsProcessingParameterNumber.Integer, defaultValue=-1
        )
        p_thr.setFlags(p_thr.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_thr)

        p_fmt = QgsProcessingParameterEnum(
            self.OUTPUT_FORMAT, "Output Format", options=["float32", "float64", "uint16"], defaultValue=0
        )
        p_fmt.setFlags(p_fmt.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_fmt)

        p_cv = QgsProcessingParameterNumber(
            self.CV_FOLDS, "ML Cross-Validation Folds", type=QgsProcessingParameterNumber.Integer, defaultValue=5
        )
        p_cv.setFlags(p_cv.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_cv)

        p_uncert = QgsProcessingParameterNumber(
            self.UNCERT_TREES, "Uncertainty Model Estimators (Trees)", type=QgsProcessingParameterNumber.Integer, defaultValue=200
        )
        p_uncert.setFlags(p_uncert.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_uncert)

        p_rf = QgsProcessingParameterString(self.PARAM_RF, "RF Params", defaultValue="'n_estimators':[100, 500], 'max_depth':[10, 30]", optional=True)
        p_rf.setFlags(p_rf.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_rf)

        p_gb = QgsProcessingParameterString(self.PARAM_GB, "GB Params", defaultValue="'n_estimators':[100, 300], 'learning_rate':[0.05, 0.1]", optional=True)
        p_gb.setFlags(p_gb.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_gb)

        p_et = QgsProcessingParameterString(self.PARAM_ET, "ET Params", defaultValue="'n_estimators':[100, 500], 'max_depth':[10, 30]", optional=True)
        p_et.setFlags(p_et.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_et)

        p_svr = QgsProcessingParameterString(self.PARAM_SVR, "SVR Params", defaultValue="'C':[1, 10, 100], 'kernel':['rbf'], 'cache_size':[1000], 'max_iter':[20000]", optional=True)
        p_svr.setFlags(p_svr.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_svr)

        p_mlp = QgsProcessingParameterString(self.PARAM_MLP, "MLP Params", defaultValue="'hidden_layer_sizes':[(100,), (50, 50)], 'max_iter':[500]", optional=True)
        p_mlp.setFlags(p_mlp.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_mlp)

        p_ridge = QgsProcessingParameterString(self.PARAM_RIDGE, "Ridge Params", defaultValue="'alpha':[0.1, 1.0]", optional=True)
        p_ridge.setFlags(p_ridge.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_ridge)

        p_lasso = QgsProcessingParameterString(self.PARAM_LASSO, "Lasso Params", defaultValue="'alpha':[0.01, 0.1]", optional=True)
        p_lasso.setFlags(p_lasso.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_lasso)

        p_en = QgsProcessingParameterString(self.PARAM_ELASTICNET, "ElasticNet Params", defaultValue="'l1_ratio':[0.5]", optional=True)
        p_en.setFlags(p_en.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_en)

        p_knn = QgsProcessingParameterString(self.PARAM_KNN, "KNN Params", defaultValue="'n_neighbors':[5, 10]", optional=True)
        p_knn.setFlags(p_knn.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_knn)

        p_dt = QgsProcessingParameterString(self.PARAM_DT, "Decision Tree Params", defaultValue="'max_depth':[5, 10]", optional=True)
        p_dt.setFlags(p_dt.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_dt)

        p_huber = QgsProcessingParameterString(self.PARAM_HUBER, "Huber Params", defaultValue="'epsilon':[1.1, 1.35, 1.5]", optional=True)
        p_huber.setFlags(p_huber.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_huber)

        p_xgb = QgsProcessingParameterString(self.PARAM_XGB, "XGBoost Params", defaultValue="'n_estimators':[100, 200], 'max_depth':[4, 6], 'learning_rate':[0.05, 0.1]", optional=True)
        p_xgb.setFlags(p_xgb.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_xgb)

        p_lgbm = QgsProcessingParameterString(self.PARAM_LGBM, "LightGBM Params", defaultValue="'n_estimators':[100, 200], 'max_depth':[4, 6], 'learning_rate':[0.05, 0.1]", optional=True)
        p_lgbm.setFlags(p_lgbm.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_lgbm)

        p_cat = QgsProcessingParameterString(self.PARAM_CATBOOST, "CatBoost Params", defaultValue="'iterations':[100, 200], 'depth':[4, 6], 'learning_rate':[0.05, 0.1]", optional=True)
        p_cat.setFlags(p_cat.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_cat)

        p_rem_pos = QgsProcessingParameterBoolean(
            self.REMOVE_POSITIVES,
            "🧽 [Cleanup] Remove Positive Depths (>= 0)",
            defaultValue=True,
            optional=True,
        )
        p_rem_pos.setFlags(p_rem_pos.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_rem_pos)

        p_slope_flt = QgsProcessingParameterBoolean(
            self.ENABLE_SLOPE_FILTER,
            "🧽 [Cleanup] Apply Slope Filter (Remove sharp jumps)",
            defaultValue=True,
            optional=True,
        )
        p_slope_flt.setFlags(p_slope_flt.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_slope_flt)

        p_slope_thr = QgsProcessingParameterNumber(
            self.SLOPE_THRESHOLD,
            "🧽 [Cleanup] Slope Filter Threshold (Degrees)",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=35.0,
            optional=True,
        )
        p_slope_thr.setFlags(p_slope_thr.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_slope_thr)


        p_max_d = QgsProcessingParameterNumber(
            self.MAX_DEPTH_THRESHOLD,
            "🧽 [Cleanup] Max Depth Threshold (e.g. -30.0)",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=-30.0,
            optional=True,
        )
        p_max_d.setFlags(p_max_d.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_max_d)

        # --- SDB Composite Score & Model Selection Strategy ---
        p_strat = QgsProcessingParameterEnum(
            self.SCORE_SELECTION_STRATEGY,
            "🎯 [Auto-ML Ranking] Model Selection Strategy / Criterion",
            options=self.SCORE_STRATEGY_OPTIONS,
            defaultValue=0,
        )
        p_strat.setFlags(p_strat.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_strat)

        p_metrics = QgsProcessingParameterEnum(
            self.SCORE_METRICS,
            "⚖️ [Score Equation] Included Evaluation Metrics (Auto-Balanced)",
            options=self.SCORE_METRIC_OPTIONS,
            allowMultiple=True,
            defaultValue=[0, 1, 2, 3, 4],
        )
        p_metrics.setFlags(p_metrics.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_metrics)

        p_custom_cfg = QgsProcessingParameterString(
            self.SCORE_CUSTOM_CONFIG,
            "🎛️ [Custom Score Matrix] Optional Weights (e.g. 'R2: 50, MAE: 50') & Simulation Settings",
            defaultValue="R2: 35, RMSE: 30, wMAPE: 20, Bias: 15, Rounds: 20, Variation: +/-35%",
            optional=True,
        )
        p_custom_cfg.setFlags(p_custom_cfg.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_custom_cfg)

        self.addOutput(
            QgsProcessingOutputRasterLayer(self.OUTPUT_DEPTH_MAP, "Output Depth Map")
        )
        self.addOutput(
            QgsProcessingOutputNumber(self.BEST_R2, "Best R2 Score")
        )
        self.addOutput(
            QgsProcessingOutputNumber(self.BEST_RMSE, "Best RMSE (m)")
        )


    def name(self):
        return "sdb_03_initial_modeling"

    def displayName(self):
        return "3.3 Phase 03: Global Auto-ML SDB Modeling"

    def group(self):
        return "3. Step-by-Step Modular Pipeline"

    def groupId(self):
        return "modular_pipeline"

    def createInstance(self):
        return SDBModule03()

    def shortHelpString(self):
        return """
        <div style="font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.5; color: #2C3E50;">
            <h2 style="margin-bottom: 5px; color: #2E86C1;">🤖 SDB Module 03: Global Auto-ML & Feature Analysis</h2>
            <p style="margin-top: 0; margin-bottom: 15px; font-size: 13px;">
                Executes an automated Machine Learning pipeline for optical bathymetry: handles feature selection, multicollinearity reduction, algorithm benchmarking, hyperparameter optimization, spatial cross-validation, and chunked raster prediction.
            </p>
            
            <h3 style="color: #D35400; margin-bottom: 5px; border-bottom: 2px solid #D35400; padding-bottom: 3px;">🤖 Core Machine Learning Features</h3>
            <ul style="font-size: 12px; margin-top: 5px; padding-left: 20px;">
                <li><b>Feature Selection & Collinearity:</b> Evaluates Pearson / Spearman correlation, Auto-RANSAC, and Auto-Random Forest to drop weak or redundant bands and spectral indices.</li>
                <li><b>Model Benchmarking:</b> Automatically evaluates <b>15+ ML Regressors</b> (Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, SVR, MLP, Extra Trees, Ridge, Lasso, etc.).</li>
                <li><b>Hyperparameter Tuning:</b> Searches optimal model hyperparameter space using <b>Bayesian Optimization</b> (Optuna / scikit-optimize), Random Search, or Grid Search.</li>
                <li><b>Spatial Block Cross-Validation:</b> Validates model generalization across spatial coordinates to prevent spatial autocorrelation leakage.</li>
                <li><b>Ensemble Blending:</b> Supports Standard Average, Median, Stacking, and <b>Uncertainty-Weighted Pixel Fusion</b>.</li>
                <li><b>Memory-Efficient Prediction:</b> Employs block-chunked processing to safely predict high-resolution rasters without out-of-memory errors.</li>
            </ul>
            <br><b>Developer:</b> Mohamed Aly Nasef
        </div>
        """

    def helpString(self):
        return self.shortHelpString()

    def processAlgorithm(self, parameters, context, feedback):
        import os
        from ...core.ml.trainers import run_phase03_initial_modeling
        from ...infrastructure.raster_io import StylePostProcessor

        results = run_phase03_initial_modeling(self, parameters, context, feedback)

        try:
            depth_map = results.get("OUTPUT_DEPTH_MAP")
            if (
                depth_map
                and os.path.exists(depth_map)
                and context
                and hasattr(context, "layerToLoadOnCompletionDetails")
                and parameters.get(self.OUTPUT_DEPTH_MAP) is not None
            ):
                qml_path = os.path.splitext(depth_map)[0] + ".qml"
                if os.path.exists(qml_path) and StylePostProcessor:
                    details = context.layerToLoadOnCompletionDetails(self.OUTPUT_DEPTH_MAP)
                    if details:
                        details.setPostProcessor(StylePostProcessor(qml_path))
        except Exception:
            pass


        return results
