import os
import glob
import re
import warnings

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterFile,
    QgsProcessingParameterEnum,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterDefinition,
    QgsProcessingException,
    QgsProcessingContext,
    QgsProject,
)
from qgis import processing

try:
    from Bathymetrix_AI.core.spectral.aggregation import (
        spatiospectral_aggregate,
        spatiospectral_mask_intersection,
        AGGREGATION_METHODS,
        compute_r2_rmse_weight,
    )
    from Bathymetrix_AI.infrastructure.raster_io import (
        clean_depth_map,
        remove_positive_pixels,
        slope_filter_depth,
        get_raster_min_max,
        write_qml_style,
        StylePostProcessor,
    )
    from Bathymetrix_AI.infrastructure.canvas import add_raster_to_canvas
    from Bathymetrix_AI.infrastructure.logging import log_module_completion, append_log
    from Bathymetrix_AI.core.pipeline import generate_html_dashboard, generate_3d_seabed_png
except (ImportError, ValueError):
    from core.spectral.aggregation import (
        spatiospectral_aggregate,
        spatiospectral_mask_intersection,
        AGGREGATION_METHODS,
        compute_r2_rmse_weight,
    )
    from infrastructure.raster_io import (
        clean_depth_map,
        remove_positive_pixels,
        slope_filter_depth,
        get_raster_min_max,
        write_qml_style,
        StylePostProcessor,
    )
    from infrastructure.canvas import add_raster_to_canvas
    from infrastructure.logging import log_module_completion, append_log
    from core.pipeline import generate_html_dashboard, generate_3d_seabed_png

warnings.filterwarnings("ignore")


class PostSpatioSpectralAggregator(QgsProcessingAlgorithm):
    """
    Post-SpatioSpectral Aggregator:
    Re-aggregates depth maps from a completed SDB SpatioSpectral Masterflow workspace,
    and runs downstream Phase 04 Adaptive Refinement, Phase 05 Validation, and Cleanup
    using the exact same shared parameters and options as SDB SpatioSpectral Masterflow.
    """

    # 1. Parameter Constants (Matching SDB SpatioSpectral Flow exactly)
    INPUT_WORKSPACE = "INPUT_WORKSPACE"
    SPATIOSPECTRAL_AGGREGATION = "SPATIOSPECTRAL_AGGREGATION"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"

    # [4] Phase 04: Adaptive Refinement
    ENABLE_ADAPTIVE = "ENABLE_ADAPTIVE"
    STACK_COMPONENTS_P4 = "STACK_COMPONENTS_P4"
    FEATURE_CORR_METHOD_P4 = "FEATURE_CORR_METHOD_P4"
    FEATURE_CORR_THRESHOLD_P4 = "FEATURE_CORR_THRESHOLD_P4"
    RESIDUAL_INTERP_METHOD = "RESIDUAL_INTERP_METHOD"
    KNN_NEIGHBORS = "KNN_NEIGHBORS"
    MAX_GPR_SAMPLES = "MAX_GPR_SAMPLES"
    SPATIAL_CV_P4 = "SPATIAL_CV_P4"
    ENABLE_DEPTH_VARIANCE_CORR_P4 = "ENABLE_DEPTH_VARIANCE_CORR_P4"
    ENABLE_SPATIAL_RESIDUAL_CORR_P4 = "ENABLE_SPATIAL_RESIDUAL_CORR_P4"
    INPUT_ADAPTIVE_TRAIN = "INPUT_ADAPTIVE_TRAIN"
    FIELD_ADAPTIVE_DEPTH = "FIELD_ADAPTIVE_DEPTH"

    # [5] Phase 05: Validation & Reporting
    ENABLE_VALIDATION = "ENABLE_VALIDATION"
    INPUT_TEST = "INPUT_TEST"
    FIELD_TEST_DEPTH = "FIELD_TEST_DEPTH"

    # SDB Composite Score & Model Selection Strategy
    SCORE_SELECTION_STRATEGY = "SCORE_SELECTION_STRATEGY"
    SCORE_METRICS = "SCORE_METRICS"
    SCORE_CUSTOM_CONFIG = "SCORE_CUSTOM_CONFIG"

    SCORE_STRATEGY_OPTIONS = [
        "0 - Winner Stability (Monte Carlo Sensitivity Analysis) [Recommended]",
        "1 - Max SDB Composite Score (0-100 Benchmark Matrix)",
        "2 - Highest R2 (Variance Explained)",
        "3 - Lowest RMSE (Overall Vertical Error)",
        "4 - Lowest wMAPE (Shallow-Depth Relative Error)",
        "5 - Lowest |Bias| (Zero-Mean Residual Offset)",
        "6 - Lowest MAE (Mean Absolute Error)",
    ]

    SCORE_METRIC_OPTIONS = [
        "R2 Accuracy (Correlation & Fit)",
        "RMSE (Root Mean Squared Error in meters)",
        "wMAPE (Weighted Mean Absolute Percentage Error %)",
        "|Bias| (Zero-Mean Residual Offset in meters)",
        "MAE (Mean Absolute Error in meters)",
    ]

    # Cleanup & Thresholds
    ENABLE_MAX_DEPTH_FILTER = "ENABLE_MAX_DEPTH_FILTER"
    MAX_DEPTH_THRESHOLD = "MAX_DEPTH_THRESHOLD"
    REMOVE_POSITIVES = "REMOVE_POSITIVES"
    ENABLE_SLOPE_FILTER = "ENABLE_SLOPE_FILTER"
    SLOPE_THRESHOLD = "SLOPE_THRESHOLD"

    # Dropdown Options (Matching SpatioSpectral Flow)
    AGGREGATION_METHODS = AGGREGATION_METHODS

    FEATURE_CORR_THRESHOLDS_P4 = [
        "Use Phase 03 (-1.0)",
        "0.0",
        "0.1",
        "0.2",
        "0.3",
        "0.4",
        "0.5",
        "0.6",
        "0.7",
        "0.8",
        "0.9",
        "1.0",
    ]


    def initAlgorithm(self, config=None):
        # ===================================================================
        # [0] General Settings (Input Workspace & Aggregation Method)
        # ===================================================================
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_WORKSPACE,
                "📁 [0.1] SpatioSpectral Master Output Folder (Contains Scene_* folders & Logs)",
                behavior=QgsProcessingParameterFile.Folder,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.SPATIOSPECTRAL_AGGREGATION,
                "🧩 [0.2] SpatioSpectral Aggregation Method",
                options=self.AGGREGATION_METHODS,
                defaultValue=4,  # Weighted Median default
            )
        )

        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_FOLDER, "📁 [0.3] Output Folder"
            )
        )

        # ===================================================================
        # [4] Phase 04: Adaptive Refinement (Matching SpatioSpectral Flow)
        # ===================================================================
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_ADAPTIVE,
                "━━━━━━━━━ 🎯 [4] Phase 04: Adaptive Refinement ━━━━━━━━━",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.STACK_COMPONENTS_P4,
                "🎯 [4] Features for Retraining",
                options=["Phase 03 Depth Map", "Residual Error Grid"],
                allowMultiple=True,
                defaultValue=[0, 1],
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.RESIDUAL_INTERP_METHOD,
                "📍 [4] Spatial Residual Interpolation Method",
                options=[
                    "Standard KNN",
                    "Robust KNN (Huber Weights)",
                    "Gaussian Process / Kriging",
                ],
                defaultValue=0,
            )
        )

        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_ADAPTIVE_TRAIN, "🎯 [4] Adaptive Points", optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_ADAPTIVE_DEPTH,
                "🎯 [4] Adaptive Depth Field",
                defaultValue="ortho_h",
                parentLayerParameterName=self.INPUT_ADAPTIVE_TRAIN,
                type=QgsProcessingParameterField.Numeric,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_DEPTH_VARIANCE_CORR_P4,
                "🎛️ [4] Enable Depth Variance Correction (Datum Mean Shift)",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_SPATIAL_RESIDUAL_CORR_P4,
                "📍 [4] Enable Spatial Residual Correction (KNN / Kriging Grid)",
                defaultValue=True,
            )
        )

        # ===================================================================
        # [5] Phase 05: Validation & Reporting (Matching SpatioSpectral Flow)
        # ===================================================================
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_VALIDATION,
                "━━━━━━━━━ 📉 [5] Phase 05: Validation & Reporting ━━━━━━━━━",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_TEST, "📉 [5] Validation Points", optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_TEST_DEPTH,
                "📉 [5] Validation Depth Field",
                defaultValue="ortho_h",
                parentLayerParameterName=self.INPUT_TEST,
                type=QgsProcessingParameterField.Numeric,
                optional=True,
            )
        )

        # ===================================================================
        # ➕ ADVANCED PARAMETERS (Organized cleanly by phase)
        # ===================================================================

        # --- [Phase 02 & Filtering Thresholds] ---
        p_en_max_d = QgsProcessingParameterBoolean(
            self.ENABLE_MAX_DEPTH_FILTER,
            "🛑 [Phase 02] Enable Maximum Depth Cutoff Filter",
            defaultValue=False,
        )
        p_en_max_d.setFlags(p_en_max_d.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_en_max_d)

        p_max_d = QgsProcessingParameterNumber(
            self.MAX_DEPTH_THRESHOLD,
            "🛑 [Phase 02] Maximum Depth Cutoff Threshold (e.g. -30.0)",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=-30.0,
        )
        p_max_d.setFlags(p_max_d.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_max_d)

        # --- [Phase 04] ---
        p_knn = QgsProcessingParameterNumber(
            self.KNN_NEIGHBORS,
            "📍 [Phase 04] KNN Nearest Neighbors (K) for Residuals",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=15,
            minValue=1,
            maxValue=100,
        )
        p_knn.setFlags(p_knn.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_knn)

        p_gpr = QgsProcessingParameterNumber(
            self.MAX_GPR_SAMPLES,
            "📍 [Phase 04] Max GPR Training Samples",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=1500,
        )
        p_gpr.setFlags(p_gpr.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_gpr)

        p_sp_p4 = QgsProcessingParameterBoolean(
            self.SPATIAL_CV_P4,
            "🌍 [Phase 04] Enable Spatial Block Cross-Validation",
            defaultValue=False,
        )
        p_sp_p4.setFlags(p_sp_p4.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_sp_p4)

        p_corr_m_p4 = QgsProcessingParameterEnum(
            self.FEATURE_CORR_METHOD_P4,
            "🤖 [Phase 04] Feature Correlation Method",
            options=[
                "Disabled",
                "Pearson (Linear)",
                "Spearman (Rank)",
                "Automatic-RANSAC",
                "Automatic-Random Forest",
            ],
            defaultValue=3,
        )
        p_corr_m_p4.setFlags(p_corr_m_p4.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_corr_m_p4)

        p_corr_th_p4 = QgsProcessingParameterEnum(
            self.FEATURE_CORR_THRESHOLD_P4,
            "🤖 [Phase 04] Feature Correlation Threshold",
            options=self.FEATURE_CORR_THRESHOLDS_P4,
            defaultValue=0,
        )
        p_corr_th_p4.setFlags(p_corr_th_p4.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_corr_th_p4)

        # --- [Post-Processing Cleanup & Filtering] ---
        p_rem_pos = QgsProcessingParameterBoolean(
            self.REMOVE_POSITIVES,
            "🧽 [Post-Processing] Remove Positive Depths (>= 0)",
            defaultValue=True,
        )
        p_rem_pos.setFlags(p_rem_pos.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_rem_pos)

        p_slope_f = QgsProcessingParameterBoolean(
            self.ENABLE_SLOPE_FILTER,
            "🧽 [Post-Processing] Apply Physical Slope Filter",
            defaultValue=True,
        )
        p_slope_f.setFlags(p_slope_f.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_slope_f)

        p_slope_th = QgsProcessingParameterNumber(
            self.SLOPE_THRESHOLD,
            "🧽 [Post-Processing] Slope Filter Threshold (Degrees)",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=35.0,
        )
        p_slope_th.setFlags(p_slope_th.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_slope_th)

        # --- [Auto-ML Ranking & Validation Matrix] ---
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


    def name(self):
        return "sdb_post_spatiospectral_aggregator"

    def displayName(self):
        return "4.1 Post-SpatioSpectral Aggregator & Fusion"

    def group(self):
        return "4. Post-Processing & Dynamics"

    def groupId(self):
        return "post_dynamics"

    def createInstance(self):
        return PostSpatioSpectralAggregator()

    def shortHelpString(self):
        return """
        <div style="font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.5; color: #2C3E50;">
            <h2 style="margin-bottom: 5px; color: #D35400;">🧩 Bathymetrix-AI: Post-SpatioSpectral Aggregator</h2>
            <p style="margin-top: 0; margin-bottom: 15px; font-size: 13px;">
                Aggregates multiple Satellite-Derived Bathymetry (SDB) depth maps generated across different scenes into a single, highly accurate consensus model.
                Allows re-running or fine-tuning the aggregation step using a completed <b>SDB SpatioSpectral Masterflow</b> workspace without re-executing Phase 01–03.
            </p>
            
            <h3 style="color: #117A65; margin-bottom: 5px; border-bottom: 2px solid #117A65; padding-bottom: 3px;">📊 Aggregation Methods</h3>
            <ul style="font-size: 12px; margin-top: 5px; padding-left: 20px;">
                <li><b>Weighted Median (R2/RMSE) [Recommended]:</b> Combines all depth maps, giving higher weight to scenes with stronger training accuracy while remaining robust against local outliers.</li>
                <li><b>Weighted Mean (R2/RMSE):</b> Accuracy-weighted linear average with weights automatically extracted from the Masterflow log file.</li>
                <li><b>Select Best Scene (High R2 / Low RMSE):</b> Automatically selects the single best-performing scene based on training metrics.</li>
                <li><b>Median / Mean:</b> Combines all scenes equally using standard statistical central tendencies.</li>
                <li><b>Max (Deepest) / Min (Shallowest):</b> Extreme depth envelope extractions.</li>
            </ul>

            <h3 style="color: #8E44AD; margin-top: 15px; margin-bottom: 5px; border-bottom: 2px solid #8E44AD; padding-bottom: 3px;">📌 Downstream Phases</h3>
            <ul style="font-size: 12px; margin-top: 5px; padding-left: 20px;">
                <li><b>Phase 04 Adaptive Refinement:</b> Spatial residual calibration applied directly to the synthesized consensus map.</li>
                <li><b>Phase 05 Validation:</b> Independent accuracy assessment with automated <b>IHO S-44 (Order 1a/1b/2 TVU)</b> compliance reporting and HTML Dashboard.</li>
            </ul>
            <br><b>Developer:</b> Mohamed Aly Nasef
        </div>
        """

    def helpString(self):
        return self.shortHelpString()

    def processAlgorithm(self, parameters, context, feedback):
        import time
        start_time = time.time()

        workspace = self.parameterAsString(parameters, self.INPUT_WORKSPACE, context)
        if workspace and os.path.isfile(workspace):
            workspace = os.path.dirname(workspace)
        try:
            agg_method_idx = self.parameterAsInt(
                parameters, self.SPATIOSPECTRAL_AGGREGATION, context
            )
            agg_method = self.AGGREGATION_METHODS[agg_method_idx] if 0 <= agg_method_idx < len(self.AGGREGATION_METHODS) else self.AGGREGATION_METHODS[4]
        except Exception:
            raw_val = parameters.get(self.SPATIOSPECTRAL_AGGREGATION, 4)
            if isinstance(raw_val, str) and raw_val in self.AGGREGATION_METHODS:
                agg_method = raw_val
            else:
                try:
                    idx = int(raw_val)
                    agg_method = self.AGGREGATION_METHODS[idx] if 0 <= idx < len(self.AGGREGATION_METHODS) else self.AGGREGATION_METHODS[4]
                except Exception:
                    agg_method = self.AGGREGATION_METHODS[4]

        if not os.path.isdir(workspace):
            raise QgsProcessingException(f"Workspace directory not found: {workspace}")

        # 1. Discover depth maps, masks, and scene metrics using natural numerical sorting
        import re
        def scene_sort_key(folder_name):
            match = re.search(r"Scene_?(\d+)", folder_name, re.IGNORECASE)
            return int(match.group(1)) if match else 999999

        scene_folders = sorted(
            [f for f in os.listdir(workspace) if f.startswith("Scene_")],
            key=scene_sort_key
        )
        if not scene_folders:
            raise QgsProcessingException(f"No Scene_* folders found in workspace: {workspace}")

        feedback.pushInfo(f"Found {len(scene_folders)} scenes in workspace.")

        p3_depth_maps = []
        valid_scenes = []
        p1_masks = []
        p1_osw_polys = []

        for scene_folder in scene_folders:
            scene_path = os.path.join(workspace, scene_folder)

            # Find Depth Map with prioritized cleaned candidates
            p3_dir = os.path.join(scene_path, "Phase_03_Initial_Modeling")
            depth_candidates = [
                os.path.join(p3_dir, "3_Initial_Global_Depth_OSW_Clipped.tif"),
                os.path.join(p3_dir, "3_Initial_Global_Depth_NoPositives.tif"),
                os.path.join(p3_dir, "3_Initial_Global_Depth_SlopeFiltered.tif"),
                os.path.join(p3_dir, "3_Initial_Global_Depth_Cleaned.tif"),
                os.path.join(p3_dir, "3_Initial_Global_Depth.tif"),
            ]
            depth_map = None
            for cand in depth_candidates:
                if os.path.exists(cand):
                    depth_map = cand
                    break

            if depth_map:
                p3_depth_maps.append(depth_map)
                valid_scenes.append(scene_folder)
            else:
                feedback.pushInfo(f"Missing depth map for {scene_folder}.")

            # Find Mask & OSW Polygon
            p1_dir = os.path.join(scene_path, "Phase_01_Preprocessing")
            mask_files = glob.glob(os.path.join(p1_dir, "*Mask*.tif"))
            if mask_files:
                p1_masks.append(mask_files[0])
            else:
                feedback.pushInfo(f"Missing mask for {scene_folder}.")

            osw_poly_candidates = (
                glob.glob(os.path.join(p1_dir, "*OSW*.gpkg"))
                + glob.glob(os.path.join(p1_dir, "*Boundary*.gpkg"))
                + glob.glob(os.path.join(p1_dir, "*Mask*.gpkg"))
                + [os.path.join(p1_dir, "07_OSW_Boundary_Polygon.gpkg")]
            )
            for cand in osw_poly_candidates:
                if os.path.exists(cand) and cand not in p1_osw_polys:
                    p1_osw_polys.append(cand)
                    break

        if not p3_depth_maps:
            raise QgsProcessingException("No valid Phase 03 depth maps found in the workspace.")

        # Extract weights from log aligned strictly with valid scenes
        p3_weights = []
        if "Weighted" in agg_method or "Best Scene" in agg_method:
            log_files = glob.glob(os.path.join(workspace, "*Log*.txt"))
            log_content = ""
            if log_files:
                try:
                    with open(log_files[0], "r", encoding="utf-8") as f:
                        log_content = f.read()
                except Exception:
                    pass

            weights_in_log = []
            if log_content:
                weights_in_log = re.findall(r"Weight:\s*([0-9.]+)", log_content)

            for scene_folder in valid_scenes:
                scene_idx = scene_folders.index(scene_folder)
                w = 1.0
                if len(weights_in_log) > scene_idx:
                    try:
                        w = float(weights_in_log[scene_idx])
                    except ValueError:
                        w = 1.0
                p3_weights.append(w)
            feedback.pushInfo(f"Extracted Weights for {len(p3_weights)} valid scenes: {p3_weights}")

        # 2. Output directory
        out_folder = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        if not out_folder:
            out_folder = os.path.join(workspace, "Aggregated_Results")

        os.makedirs(out_folder, exist_ok=True)
        agg_log_path = os.path.join(out_folder, "Post_Aggregator_Log.txt")

        safe_agg_method_name = (
            agg_method.replace("/", "_").replace("\\", "_").replace(" ", "_")
        )
        aggregated_depth_path = os.path.join(
            out_folder, f"Aggregated_Depth_{safe_agg_method_name}.tif"
        )
        aggregated_mask_path = os.path.join(out_folder, "Aggregated_Intersection_Mask.tif")

        # 3. Run Aggregation
        if agg_method == "Select Best Scene (High R2 / Low RMSE)":
            best_idx = p3_weights.index(max(p3_weights)) if p3_weights else 0
            best_depth_map = p3_depth_maps[best_idx]
            feedback.pushInfo(
                f"Selecting Scene {best_idx+1} as the best scene based on R2/RMSE weight ({p3_weights[best_idx] if p3_weights else 1.0})."
            )

            import shutil
            shutil.copy2(best_depth_map, aggregated_depth_path)

            feedback.pushInfo("Copying mask for the best scene...")
            if p1_masks and best_idx < len(p1_masks):
                shutil.copy2(p1_masks[best_idx], aggregated_mask_path)
        else:
            feedback.pushInfo(f"Aggregating using method: {agg_method}...")
            agg_kwargs = {"method": agg_method, "feedback": feedback}
            if "Weighted" in agg_method and p3_weights:
                agg_kwargs["weights"] = p3_weights

            spatiospectral_aggregate(p3_depth_maps, aggregated_depth_path, **agg_kwargs)

            feedback.pushInfo("Generating aggregated intersection mask...")
            if p1_masks:
                spatiospectral_mask_intersection(p1_masks, aggregated_mask_path, feedback=feedback)

        feedback.pushInfo(f"Aggregation complete. Output saved to: {aggregated_depth_path}")

        # Extract CRS for GDAL clipping
        crs_id = None
        try:
            from qgis.core import QgsRasterLayer
            rl = QgsRasterLayer(aggregated_depth_path, "agg_depth")
            if rl.isValid() and rl.crs().isValid():
                crs_id = rl.crs().authid()
        except Exception:
            pass

        # Post-Aggregation Cleanup: Clamp max depth, slope filter, remove positives & OSW Clip
        aggregated_osw_poly = os.path.join(out_folder, "Aggregated_OSW_Boundary_Polygon.gpkg")
        if p1_osw_polys:
            feedback.pushInfo("Generating aggregated OSW boundary polygon from scene vectors...")
            try:
                if len(p1_osw_polys) == 1:
                    import shutil
                    shutil.copy2(p1_osw_polys[0], aggregated_osw_poly)
                else:
                    merge_res = processing.run(
                        "native:mergevectorlayers",
                        {"LAYERS": p1_osw_polys, "OUTPUT": aggregated_osw_poly},
                        context=context,
                        feedback=feedback,
                        is_child_algorithm=True,
                    )
                    if os.path.exists(merge_res["OUTPUT"]):
                        aggregated_osw_poly = merge_res["OUTPUT"]
            except Exception as e:
                feedback.pushInfo(f"Warning: Failed to aggregate OSW polygons: {e}")

        # Fallback: If no vector polygon exists in scene folders, create it on-the-fly from the aggregated mask!
        if (not os.path.exists(aggregated_osw_poly) or os.path.getsize(aggregated_osw_poly) == 0) and os.path.exists(aggregated_mask_path):
            feedback.pushInfo("Generating Aggregated OSW Boundary Polygon from Aggregated Intersection Mask...")
            try:
                poly_res = processing.run(
                    "gdal:polygonize",
                    {
                        "INPUT": aggregated_mask_path,
                        "BAND": 1,
                        "FIELD": "DN",
                        "EIGHT_CONNECTEDNESS": False,
                        "OUTPUT": "TEMPORARY_OUTPUT",
                    },
                    context=context,
                    feedback=feedback,
                    is_child_algorithm=True,
                )
                processing.run(
                    "native:extractbyexpression",
                    {
                        "INPUT": poly_res["OUTPUT"],
                        "EXPRESSION": '"DN" = 1',
                        "OUTPUT": aggregated_osw_poly,
                    },
                    context=context,
                    feedback=feedback,
                    is_child_algorithm=True,
                )
            except Exception as e:
                feedback.pushInfo(f"Warning: Failed to polygonize aggregated mask: {e}")

        if os.path.exists(aggregated_depth_path):
            feedback.pushInfo("Applying Post-Aggregation Cleanup (Clamping, Slope Filter, Positive Removal)...")
            max_depth = self.parameterAsDouble(parameters, self.MAX_DEPTH_THRESHOLD, context)
            agg_clamped = os.path.join(out_folder, f"Aggregated_Depth_{safe_agg_method_name}_Cleaned.tif")
            ref_feat = aggregated_mask_path if os.path.exists(aggregated_mask_path) else aggregated_depth_path
            clean_depth_map(aggregated_depth_path, ref_feat, max_depth, agg_clamped, context, feedback)
            current_agg = agg_clamped

            if self.parameterAsBool(parameters, self.ENABLE_SLOPE_FILTER, context):
                slope_threshold = self.parameterAsDouble(parameters, self.SLOPE_THRESHOLD, context)

                agg_slope = os.path.join(out_folder, f"Aggregated_Depth_{safe_agg_method_name}_SlopeFiltered.tif")
                current_agg = slope_filter_depth(
                    current_agg,
                    slope_threshold=slope_threshold,
                    out_path=agg_slope,
                    context=context,
                    feedback=feedback,
                )

            if self.parameterAsBool(parameters, self.REMOVE_POSITIVES, context):
                agg_no_pos = os.path.join(out_folder, f"Aggregated_Depth_{safe_agg_method_name}_NoPositives.tif")
                remove_positive_pixels(current_agg, agg_no_pos, feedback)
                current_agg = agg_no_pos

            if os.path.exists(aggregated_osw_poly) and os.path.getsize(aggregated_osw_poly) > 0:
                feedback.pushInfo("Clipping Aggregated Depth Map with OSW Polygon...")
                agg_osw_clipped = os.path.join(out_folder, f"Aggregated_Depth_{safe_agg_method_name}_OSW_Clipped.tif")
                try:
                    clip_params = {
                        "INPUT": current_agg,
                        "MASK": aggregated_osw_poly,
                        "NODATA": -9999.0,
                        "ALPHA_BAND": False,
                        "CROP_TO_CUTLINE": False,
                        "KEEP_RESOLUTION": True,
                        "DATA_TYPE": 0,
                        "OUTPUT": agg_osw_clipped,
                    }
                    if crs_id:
                        clip_params["SOURCE_CRS"] = crs_id
                        clip_params["TARGET_CRS"] = crs_id
                    processing.run(
                        "gdal:cliprasterbymasklayer",
                        clip_params,
                        context=context,
                        feedback=feedback,
                        is_child_algorithm=True,
                    )
                    if os.path.exists(agg_osw_clipped):
                        current_agg = agg_osw_clipped
                        import shutil
                        shutil.copy2(agg_osw_clipped, aggregated_depth_path)
                except Exception as e:
                    feedback.pushInfo(f"Warning: Failed to clip Aggregated Depth with OSW Polygon: {e}")

            aggregated_depth_path = current_agg

        write_qml_style(aggregated_depth_path)

        # 5. Optional Phase 04: Adaptive Refinement (Matching SpatioSpectral Flow)
        enable_adaptive = self.parameterAsBool(parameters, self.ENABLE_ADAPTIVE, context)
        p4_final_depth = None
        p4_dir = None

        if enable_adaptive:
            adaptive_train_layer = self.parameterAsVectorLayer(
                parameters, self.INPUT_ADAPTIVE_TRAIN, context
            )
            if not adaptive_train_layer:
                clean_vecs = glob.glob(os.path.join(workspace, "Scene_*", "Phase_02_Filtering", "*Clean*.gpkg")) + \
                             glob.glob(os.path.join(workspace, "Scene_*", "Phase_02_Filtering", "*Points*.gpkg"))
                if clean_vecs:
                    adaptive_train_layer = clean_vecs[0]
                    feedback.pushInfo(f"Using discovered training points from workspace: {adaptive_train_layer}")
                else:
                    raise QgsProcessingException(
                        "Adaptive Points layer (INPUT_ADAPTIVE_TRAIN) is required to run Phase 04."
                    )

            feedback.pushInfo("Running Phase 04 (Adaptive Refinement)...")
            p4_dir = os.path.join(out_folder, "Phase_04_Adaptive_Refinement")
            os.makedirs(p4_dir, exist_ok=True)

            stack_indices = self.parameterAsEnums(
                parameters, self.STACK_COMPONENTS_P4, context
            )
            # Map [0, 1] to 1-based band indexes [1, 2]
            stack_components = [idx + 1 for idx in stack_indices] if stack_indices else [1, 2]

            p4_thresh_idx = self.parameterAsInt(parameters, self.FEATURE_CORR_THRESHOLD_P4, context)
            mapped_thresh = max(0, p4_thresh_idx - 1) if p4_thresh_idx > 0 else 0

            p4_params = {
                "INPUT_GLOBAL_RASTER": aggregated_depth_path,
                "INPUT_ORIGINAL_FEAT": aggregated_depth_path,
                "STACK_COMPONENTS": stack_components,
                "INPUT_MASK": aggregated_mask_path if os.path.exists(aggregated_mask_path) else None,
                "INPUT_TRAIN": adaptive_train_layer,
                "FIELD_TRAIN": self.parameterAsString(parameters, self.FIELD_ADAPTIVE_DEPTH, context),
                "FEATURE_CORR_METHOD": self.parameterAsInt(parameters, self.FEATURE_CORR_METHOD_P4, context),
                "FEATURE_CORR_THRESHOLD": mapped_thresh,
                "RESIDUAL_INTERP_METHOD": self.parameterAsInt(parameters, self.RESIDUAL_INTERP_METHOD, context),
                "KNN_NEIGHBORS": self.parameterAsInt(parameters, self.KNN_NEIGHBORS, context),
                "MAX_GPR_SAMPLES": self.parameterAsInt(parameters, self.MAX_GPR_SAMPLES, context),
                "SPATIAL_CV": self.parameterAsBool(parameters, self.SPATIAL_CV_P4, context),
                "ENABLE_DEPTH_VARIANCE_CORR": self.parameterAsBool(parameters, self.ENABLE_DEPTH_VARIANCE_CORR_P4, context),
                "ENABLE_SPATIAL_RESIDUAL_CORR": self.parameterAsBool(parameters, self.ENABLE_SPATIAL_RESIDUAL_CORR_P4, context),
                "OUTPUT_FOLDER": p4_dir,
            }

            p4 = processing.run(
                "sdb_tools:sdb_phase4_adaptive",
                p4_params,
                is_child_algorithm=True,
                context=context,
                feedback=feedback,
            )
            raw_p4_depth = p4["OUTPUT_FINAL"]

            if raw_p4_depth and os.path.exists(raw_p4_depth):
                max_depth = self.parameterAsDouble(parameters, self.MAX_DEPTH_THRESHOLD, context)
                p4_clamped = os.path.join(p4_dir, "4_Phase04_Depth_Cleaned.tif")
                clean_depth_map(
                    raw_p4_depth, aggregated_depth_path, max_depth, p4_clamped, context, feedback
                )
                current_p4 = p4_clamped

                if self.parameterAsBool(parameters, self.ENABLE_SLOPE_FILTER, context):
                    slope_threshold = self.parameterAsDouble(
                        parameters, self.SLOPE_THRESHOLD, context
                    )

                    p4_slope = os.path.join(p4_dir, "4_Phase04_Depth_SlopeFiltered.tif")
                    current_p4 = slope_filter_depth(
                        current_p4,
                        slope_threshold=slope_threshold,
                        out_path=p4_slope,
                        context=context,
                        feedback=feedback,
                    )

                if self.parameterAsBool(parameters, self.REMOVE_POSITIVES, context):
                    p4_no_pos = os.path.join(p4_dir, "4_Phase04_Depth_NoPositives.tif")
                    remove_positive_pixels(current_p4, p4_no_pos, feedback)
                    p4_final_depth = p4_no_pos
                else:
                    p4_final_depth = current_p4

                if os.path.exists(aggregated_osw_poly) and os.path.getsize(aggregated_osw_poly) > 0:
                    feedback.pushInfo("Clipping Phase 04 Map with Aggregated OSW Polygon...")
                    p4_osw_clipped = os.path.join(p4_dir, "Phase04_Final_Depth_OSW_Clipped.tif")
                    try:
                        clip_params = {
                            "INPUT": p4_final_depth,
                            "MASK": aggregated_osw_poly,
                            "NODATA": -9999.0,
                            "ALPHA_BAND": False,
                            "CROP_TO_CUTLINE": False,
                            "KEEP_RESOLUTION": True,
                            "DATA_TYPE": 0,
                            "OUTPUT": p4_osw_clipped,
                        }
                        if crs_id:
                            clip_params["SOURCE_CRS"] = crs_id
                            clip_params["TARGET_CRS"] = crs_id
                        processing.run(
                            "gdal:cliprasterbymasklayer",
                            clip_params,
                            context=context,
                            feedback=feedback,
                            is_child_algorithm=True,
                        )
                        if os.path.exists(p4_osw_clipped):
                            p4_final_depth = p4_osw_clipped
                            if raw_p4_depth and os.path.exists(raw_p4_depth) and raw_p4_depth != p4_osw_clipped:
                                import shutil
                                shutil.copy2(p4_osw_clipped, raw_p4_depth)
                    except Exception as e:
                        feedback.pushInfo(f"Warning: Failed to clip Phase 04 with OSW Polygon: {e}")

                write_qml_style(p4_final_depth)
            else:
                p4_final_depth = raw_p4_depth

            if p4_final_depth and os.path.exists(p4_final_depth):
                write_qml_style(p4_final_depth)

            feedback.pushInfo(f"Phase 04 complete. Output saved to: {p4_final_depth}")

        # 6. Optional Phase 05: Validation & Reporting (Matching SpatioSpectral Flow)
        enable_val = self.parameterAsBool(parameters, self.ENABLE_VALIDATION, context)
        p5_dir = None

        if enable_val:
            test_layer = self.parameterAsVectorLayer(parameters, self.INPUT_TEST, context)
            train_ref = self.parameterAsVectorLayer(
                parameters, self.INPUT_ADAPTIVE_TRAIN, context
            )
            if not train_ref:
                clean_vecs = glob.glob(os.path.join(workspace, "Scene_*", "Phase_02_Filtering", "*Clean*.gpkg")) + \
                             glob.glob(os.path.join(workspace, "Scene_*", "Phase_02_Filtering", "*Points*.gpkg"))
                if clean_vecs:
                    train_ref = clean_vecs[0]
                elif test_layer:
                    train_ref = test_layer

            feedback.pushInfo("Running Phase 05 (Validation & Reporting)...")
            p5_dir = os.path.join(out_folder, "Phase_05_Scientific_Validation")
            os.makedirs(p5_dir, exist_ok=True)

            p5_params = {
                "INPUT_MAP_P3": aggregated_depth_path,
                "INPUT_MAP_P4": p4_final_depth if p4_final_depth else aggregated_depth_path,
                "INPUT_TRAIN": train_ref,
                "FIELD_TRAIN": self.parameterAsString(parameters, self.FIELD_ADAPTIVE_DEPTH, context) if train_ref else "ortho_h",
                "INPUT_VALIDATION": test_layer,
                "FIELD_VAL_DEPTH": self.parameterAsString(parameters, self.FIELD_TEST_DEPTH, context) if test_layer else "ortho_h",
                "OUTPUT_FOLDER": p5_dir,
            }

            processing.run(
                "sdb_tools:sdb_05_reporting",
                p5_params,
                is_child_algorithm=True,
                context=context,
                feedback=feedback,
            )
            feedback.pushInfo(f"Phase 05 complete. Reports & IHO S-44 analysis saved to: {p5_dir}")

            # Generate static 3D Seabed PNG
            final_depth_for_3d = p4_final_depth if (enable_adaptive and p4_final_depth) else aggregated_depth_path
            if final_depth_for_3d and os.path.exists(final_depth_for_3d):
                try:
                    out_3d_png = os.path.join(p5_dir, "5_Plot_3D_Seabed.png")
                    generate_3d_seabed_png(final_depth_for_3d, out_3d_png, feedback)
                except Exception as e:
                    feedback.pushWarning(f"3D Seabed plot failed: {str(e)}")

            feedback.pushInfo("Generating Interactive HTML Dashboard & S-44 Report...")
            try:
                generate_html_dashboard(
                    out_dir=out_folder,
                    p3_dir=out_folder,
                    p4_dir=p4_dir if enable_adaptive else None,
                    spatial_cv_p3=False,
                    spatial_cv_p4=self.parameterAsBool(parameters, self.SPATIAL_CV_P4, context),
                    field_depth=self.parameterAsString(parameters, self.FIELD_ADAPTIVE_DEPTH, context) if train_ref else None,
                    feedback=feedback,
                    raster_name="SpatioSpectral Aggregated Scenes",
                    train_name=train_ref.name() if train_ref else "Adaptive Points",
                    test_name=test_layer.name() if test_layer else "Validation Points",
                    final_raster_path=p4_final_depth if (enable_adaptive and p4_final_depth) else aggregated_depth_path,
                    is_spatiospectral=True,
                    p5_dir=p5_dir,
                )
            except Exception as e:
                feedback.pushWarning(f"Dashboard generation failed: {str(e)}")

        # 7. Layer loading with unified ocean styling on map canvas
        if aggregated_depth_path and os.path.exists(aggregated_depth_path):
            qml_agg = write_qml_style(aggregated_depth_path)
            add_raster_to_canvas(
                aggregated_depth_path,
                f"Aggregated Depth ({safe_agg_method_name})",
                context=context,
                style_path=qml_agg,
            )

        if enable_adaptive and p4_final_depth and os.path.exists(p4_final_depth):
            qml_p4 = write_qml_style(p4_final_depth)
            add_raster_to_canvas(
                p4_final_depth,
                f"Phase 4 Final Depth ({safe_agg_method_name})",
                context=context,
                style_path=qml_p4,
            )

        if aggregated_mask_path and os.path.exists(aggregated_mask_path):
            add_raster_to_canvas(aggregated_mask_path, "Aggregated Intersection Mask", context=context)


        # 8. Log module completion banner with clickable URLs
        elapsed = time.time() - start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)

        dash_file = os.path.join(out_folder, "SDB_Validation_Dashboard.html")
        if not os.path.exists(dash_file) and p5_dir:
            dash_file = os.path.join(p5_dir, "SDB_Validation_Dashboard.html")

        strat_file = os.path.join(p5_dir, "5_Stratified_Error_Analysis.csv") if p5_dir else None

        primary_files = {
            "Aggregated Depth Map": aggregated_depth_path,
            "Refined Depth Map": p4_final_depth,
            "Intersection Mask": aggregated_mask_path,
            "HTML Dashboard": dash_file if os.path.exists(dash_file) else None,
            "IHO S-44 Assessment CSV": strat_file if strat_file and os.path.exists(strat_file) else None,
        }

        try:
            log_module_completion(
                module_title=f"Post-SpatioSpectral Aggregator ({agg_method} - Elapsed: {mins}m {secs}s)",
                out_dir=out_folder,
                primary_files=primary_files,
                log_path=agg_log_path,
                feedback=feedback,
            )
        except Exception:
            pass

        return {
            "OUTPUT_DEPTH": p4_final_depth if (enable_adaptive and p4_final_depth) else aggregated_depth_path,
            "OUTPUT_FOLDER": out_folder,
        }
