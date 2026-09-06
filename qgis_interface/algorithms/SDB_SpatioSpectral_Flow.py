import os
import warnings
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterFile,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterField,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterBand,
    QgsProcessingParameterString,
    QgsProcessingParameterDefinition,
    QgsProcessingException,
    QgsProcessingFeedback
)

try:
    from .SDB_MasterFlow import SDBMasterOrchestrator
    from ...core.spectral.spatiospectral_runner import SpatioSpectralSDBRunner
    from ...core.spectral.aggregation import AGGREGATION_METHODS
    from ...infrastructure.logging import append_log, log_module_completion, format_clickable_url
except (ImportError, ValueError):
    from qgis_interface.algorithms.SDB_MasterFlow import SDBMasterOrchestrator
    from core.spectral.spatiospectral_runner import SpatioSpectralSDBRunner
    from core.spectral.aggregation import AGGREGATION_METHODS
    from infrastructure.logging import append_log, log_module_completion, format_clickable_url

warnings.filterwarnings("ignore")

class SDBSpatioSpectralFlow(SDBMasterOrchestrator):
    """
    SpatioSpectral Masterflow: Aggregates multiple scenes of the same year
    (Phase 01, 02, 03 independent -> Pixel-wise Aggregation -> Phase 04, 05).
    Inherits all parameters from the Single Masterflow.
    """
    
    # We override the INPUT_RASTER to be a folder of images
    INPUT_IMAGE_ROOT = "INPUT_IMAGE_ROOT"
    OUTPUT_MASTER_FOLDER = "OUTPUT_MASTER_FOLDER"
    SPATIOSPECTRAL_AGGREGATION = "SPATIOSPECTRAL_AGGREGATION"

    def name(self):
        return "sdb_spatiospectral_flow"

    def displayName(self):
        return "1.2 SDB SpatioSpectral Masterflow (Multi-Scene)"

    def group(self):
        return "1. End-to-End Masterflows"

    def groupId(self):
        return "masterflows"

    def createInstance(self):
        return SDBSpatioSpectralFlow()

    def shortHelpString(self):
        return """
        <div style="font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.5; color: #2C3E50;">
            <h2 style="margin-bottom: 5px; color: #8E44AD;">🛰️ Bathymetrix-AI: SpatioSpectral Masterflow</h2>
            <p style="margin-top: 0; margin-bottom: 15px; font-size: 13px;">
                An advanced <b>SpatioSpectral Auto-ML Pipeline</b> designed to evaluate and fuse multiple overlapping satellite scenes for a <b>single time period</b>. 
                Instead of relying on a single snapshot that might suffer from sun glint, clouds, or waves, this module iterates through all provided candidate scenes, 
                generates independent AI depth models, and performs pixel-wise <b>SpatioSpectral Aggregation</b> (Weighted Median, Weighted Mean, Best Scene Selection, Median, or Mean) 
                to synthesize a highly accurate, noise-free consensus depth map.
            </p>

              <h3 style="color: #117A65; margin-bottom: 5px; border-bottom: 2px solid #117A65; padding-bottom: 3px;">📂 Required Workspace Structure</h3>
              <pre style="background: #F8F9F9; padding: 10px; border-left: 4px solid #8E44AD; font-family: Consolas, monospace; font-size: 12px; margin-top: 5px; border-radius: 4px;">
  📁 Input_Scenes_Folder/
     ├── Scene_1.tif                  (Overlapping Scene 1 - GeoTIFF)
     ├── Scene_2.tif                  (Overlapping Scene 2 - GeoTIFF)
     ├── Scene_3.tif                  (Overlapping Scene 3 - GeoTIFF)
     └── ...                          (Any .tif satellite scenes covering the area)
              </pre>
              <p style="margin-top: 10px; font-size: 12px;">
                  <b>Note:</b> Simply place your candidate satellite scenes (e.g. <code>Scene_1.tif</code>, <code>Scene_2.tif</code>, <code>Scene_3.tif</code>, etc.) in one input folder, provide <b>one</b> global ICESat-2 training layer, and configure your parameters. The module will automatically align, project, and extract data across all scenes in the folder.
              </p>
            
            <h3 style="color: #D35400; margin-top: 20px; margin-bottom: 5px; border-bottom: 2px solid #D35400; padding-bottom: 3px;">⚙️ Pipeline Phases</h3>
            <ul style="font-size: 12px; margin-top: 5px; padding-left: 20px;">
                <li><b>Phase 01:</b> Automated atmospheric/glint correction, water masking, and flexible Raw/Log-Ratio feature engineering for each scene individually.</li>
                <li><b>Phase 02:</b> Training point filtering and RANSAC / LS / Huber outlier rejection for each scene.</li>
                <li><b>Phase 03:</b> <i>Scene-Specific AI Modeling</i> — benchmarks 15+ ML algorithms, generates independent Initial Depth Maps, and calculates scene accuracy weights (R² / RMSE).</li>
                <li><b>Aggregation:</b> <i>Pixel-wise SpatioSpectral Synthesis</i> — merges all scene depth maps into one superior consensus map using <b>Weighted Median</b> (Recommended), <b>Weighted Mean</b>, <b>Select Best Scene</b>, or standard <b>Median / Mean</b>.</li>
                <li><b>Phase 04:</b> Adaptive spatial residual correction (Zero-Mean Centering & LOO Huber IDW error surface) applied <i>strictly on the aggregated consensus map</i>, accompanied by 95% Confidence Spatial Uncertainty modeling.</li>
                <li><b>Phase 05:</b> Independent scientific validation, <b>IHO S-44 (Order 1a/2 TVU)</b> compliance assessment, and automated interactive HTML dashboard reporting.</li>
            </ul>
            <br><b>Developer:</b> Mohamed Aly Nasef
        </div>
        """

    def helpString(self):
        return self.shortHelpString()

    def initAlgorithm(self, config=None):

        # ===================================================================
        # [0] General Settings
        # ===================================================================
        self.addParameter(
            QgsProcessingParameterFile(self.INPUT_IMAGE_ROOT, "📁 [0.1] Input Folder containing multiple .tif Scenes (Same Year)", behavior=QgsProcessingParameterFile.Folder)
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.SPATIOSPECTRAL_AGGREGATION,
                "🧩 [0.2] SpatioSpectral Aggregation Method",
                options=AGGREGATION_METHODS,
                defaultValue=4,
            )
        )

        self.addParameter(
            QgsProcessingParameterFolderDestination(self.OUTPUT_MASTER_FOLDER, "📁 [0.3] Master Output Workspace")
        )

        # ===================================================================
        # [1] Phase 01: Advanced Pre-processing
        # ===================================================================
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_PREPROCESSING,
                "━━━━━━━━━ ⚙️ [1] Phase 01: Pre-processing & Feature Extraction ━━━━━━━━━",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.COASTAL_BAND,
                "📡 [1.1] Coastal Band",
                defaultValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.BLUE_BAND,
                "📡 [1.1] Blue Band",
                defaultValue=2,
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.GREEN_BAND,
                "📡 [1.1] Green Band",
                defaultValue=3,
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.RED_BAND,
                "📡 [1.1] Red Band",
                defaultValue=4,
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.NIR_BAND,
                "🌍 [1] NIR Band",
                defaultValue=8,
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.SWIR_BAND,
                "🌍 [1] SWIR Band (For 3 Indices Mask)",
                defaultValue=11,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.APPLY_SUNGLINT,
                "☀️ [1.2] Apply Sunglint Correction",
                defaultValue=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_MASKING,
                "🏖️ [1.3] Enable Automated Water Masking",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.MASKING_METHOD,
                "🏖️ [1.3] Water Masking Method",
                options=self.MASK_METHODS_NAMES,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.WATER_MASK_POLY,
                "🗺️ [1.3] Ready-made Water Mask Polygon",
                types=[QgsProcessing.TypeVectorPolygon],
                optional=True,
            )
        )

        default_feats = list(range(1, len(self.FEATURE_OPTIONS_NAMES)))
        self.addParameter(
            QgsProcessingParameterEnum(
                self.FEATURE_SELECTION,
                "📊 [1.4] Output Feature Stack",
                options=self.FEATURE_OPTIONS_NAMES,
                allowMultiple=True,
                defaultValue=default_feats,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.BAND_MATH_FORMULA,
                "🧮 [1.4] Custom Band Math Formula (e.g. (B2-B3)/(B2+B3) or Log_B2/Log_B3)",
                defaultValue="",
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.APPLY_DEEPWATER,
                "🌊 [1.5] Apply Deep Water Filter",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.DEEPWATER_METHOD,
                "🌊 [1.5] Deep Water Definition Method",
                options=self.OSW_METHODS_NAMES,
                defaultValue=2,
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.DEEPWATER_ROI,
                "🌊 [1.5] Deep Water ROI (Polygon) [Optional if Manual Mode]",
                types=[QgsProcessing.TypeVectorPolygon],
                optional=True,
            )
        )

        # ===================================================================
        # [2] Phase 02: Robust Filtering
        # ===================================================================
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_RANSAC,
                "━━━━━━━━━ 🛡️ [2] Enable Phase 02 Data Filtering ━━━━━━━━━",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_TRAIN, "📍 [2.1] Main Training Points (ICESat-2 / In-situ)"
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_DEPTH,
                "📏 [2.1] Depth Field",
                defaultValue="ortho_h",
                parentLayerParameterName=self.INPUT_TRAIN,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_WEIGHT,
                "⚖️ [2.1] Weight Field [optional]",
                defaultValue="confidence",
                parentLayerParameterName=self.INPUT_TRAIN,
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.FILTER_MODE,
                "🧹 [2.2] Filtering Strategy",
                options=self.FILTER_MODES_NAMES,
                defaultValue=2,
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.FILTER_NUMERATOR_BAND,
                "🧹 [2.2] Log-Ratio Numerator Band",
                defaultValue=2,
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.FILTER_DENOMINATOR_BAND,
                "🧹 [2.2] Log-Ratio Denominator Band",
                defaultValue=3,
            )
        )

        # ===================================================================
        # [3] Phase 03: Global Auto-ML & Feature Analysis
        # ===================================================================
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_MODELING,
                "━━━━━━━━━ 🤖 [3] Phase 03: Global Auto-ML SDB Modeling ━━━━━━━━━",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.SELECTED_ALGOS,
                "🤖 [3.1] Algorithms to Benchmark",
                options=self.MODEL_LIST_NAMES,
                allowMultiple=True,
                defaultValue=[3, 12, 13, 14, 15, 17],
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.OPTIMIZER_METHOD,
                "🤖 [3.2] Optimizer Method",
                options=self.OPTIMIZER_LIST_NAMES,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.FEATURE_CORR_METHOD,
                "🤖 [3.3] Feature Correlation Method",
                options=["Disabled", "Pearson (Linear)", "Spearman (Rank)", "Automatic-RANSAC", "Automatic-Random Forest"],
                defaultValue=3,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.FEATURE_CORR_THRESHOLD,
                "🤖 [3.3] Feature Correlation Threshold",
                options=self.FEATURE_CORR_THRESHOLDS,
                defaultValue=2,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.SPATIAL_CV_P3,
                "🌍 [3.4] Enable Spatial Block Cross-Validation",
                defaultValue=True,
            )
        )

        # ===================================================================
        # [4] Phase 04: Adaptive Refinement
        # ===================================================================
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_ADAPTIVE,
                "━━━━━━━━━ 🎯 [4] Phase 04: Adaptive Refinement ━━━━━━━━━",
                defaultValue=False,
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
        self.addParameter(
            QgsProcessingParameterEnum(
                self.RESIDUAL_INTERP_METHOD,
                "📍 [4] Spatial Residual Interpolation Method",
                options=["Standard KNN", "Robust KNN (Huber Weights)", "Gaussian Process / Kriging"],
                defaultValue=0,
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

        # ===================================================================
        # [5] Phase 05: Validation & Reporting
        # ===================================================================
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_VALIDATION, "━━━━━━━━━ 📉 [5] Phase 05: Validation & Reporting ━━━━━━━━━", defaultValue=False
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

        # --- [System] ---
        p_threads = QgsProcessingParameterNumber(
            self.NUM_THREADS,
            "⚙️ [System] Processing Threads (Multi-processing)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=4,
        )
        p_threads.setFlags(p_threads.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_threads)

        # --- [Phase 01] ---
        p_sg_pct = QgsProcessingParameterNumber(
            self.SUNGLINT_PERCENTILE,
            "☀️ [Phase 01] Sunglint Deep Water %",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=1.0,
        )
        p_sg_pct.setFlags(p_sg_pct.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_sg_pct)

        p_shrink = QgsProcessingParameterNumber(
            self.SHRINK_EDGE_DIST,
            "🗺️ [Phase 01] Water Edge Shrink (Map Units, e.g. -10)",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.0,
        )
        p_shrink.setFlags(p_shrink.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_shrink)

        p_man_th = QgsProcessingParameterNumber(
            self.MANUAL_THRESHOLD,
            "🏖️ [Phase 01] Manual Water Mask Threshold",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.0,
            optional=True,
        )
        p_man_th.setFlags(p_man_th.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_man_th)

        p_otsu = QgsProcessingParameterNumber(
            self.OTSU_ADJUSTMENT,
            "🏖️ [Phase 01] Otsu Threshold Adjustment",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.0,
        )
        p_otsu.setFlags(p_otsu.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_otsu)

        p_mask_k = QgsProcessingParameterNumber(
            self.MASK_KERNEL_SIZE,
            "🏖️ [Phase 01] Mask Cleanup Kernel Size",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=3,
        )
        p_mask_k.setFlags(p_mask_k.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_mask_k)

        p_nir_pct = QgsProcessingParameterNumber(
            self.NIR_PERCENTILE_OSW,
            "🌊 [Phase 01] NIR Percentile for Deep Water (e.g. 10%)",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=10.0,
        )
        p_nir_pct.setFlags(p_nir_pct.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_nir_pct)

        p_osw_med = QgsProcessingParameterNumber(
            self.OSW_MEDIAN_SIZE,
            "🌊 [Phase 01] OSW Mask Median Filter Size",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=3,
        )
        p_osw_med.setFlags(p_osw_med.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_osw_med)

        p_fill = QgsProcessingParameterBoolean(
            self.FILL_INTERNAL_HOLES,
            "🌊 [Phase 01] Fill All Internal Holes in OSW Mask",
            defaultValue=True,
        )
        p_fill.setFlags(p_fill.flags() | QgsProcessingParameterDefinition.FlagHidden)
        self.addParameter(p_fill)

        p_extract = QgsProcessingParameterBoolean(
            self.EXTRACT_POLYGON,
            "🌊 [Phase 01] Extract OSW Mask as Polygon",
            defaultValue=True,
        )
        p_extract.setFlags(p_extract.flags() | QgsProcessingParameterDefinition.FlagHidden)
        self.addParameter(p_extract)

        # --- [Phase 02] ---
        p_r_th = QgsProcessingParameterNumber(
            self.RANSAC_THRESHOLD,
            "🧹 [Phase 02] RANSAC Threshold / Sigma Multiplier",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=3.0,
        )
        p_r_th.setFlags(p_r_th.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_r_th)

        p_r_tr = QgsProcessingParameterNumber(
            self.RANSAC_MAX_TRIALS,
            "🧹 [Phase 02] RANSAC Maximum Trials",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=100,
        )
        p_r_tr.setFlags(p_r_tr.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_r_tr)

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

        # --- [Phase 03] ---
        p_col = QgsProcessingParameterEnum(
            self.COLLISION_HANDLING,
            "🤖 [Phase 03] In-Situ Points Collision Handling",
            options=self.COLLISION_LIST_NAMES,
            defaultValue=0,
        )
        p_col.setFlags(p_col.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_col)

        p_n_iter = QgsProcessingParameterNumber(
            self.N_ITERATIONS,
            "🤖 [Phase 03] Hyperparameter Optimization Iterations",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=20,
        )
        p_n_iter.setFlags(p_n_iter.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_n_iter)

        p_ens_size = QgsProcessingParameterNumber(
            self.ENSEMBLE_SIZE,
            "📊 [Phase 03] Ensemble Size (Top N Models to blend)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=3,
            minValue=2,
            maxValue=5,
        )
        p_ens_size.setFlags(p_ens_size.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_ens_size)

        p_cv = QgsProcessingParameterNumber(
            self.CV_FOLDS, "🎛️ [Phase 03] ML Cross-Validation Folds", type=QgsProcessingParameterNumber.Integer, defaultValue=5
        )
        p_cv.setFlags(p_cv.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_cv)

        p_uncert = QgsProcessingParameterNumber(
            self.UNCERT_TREES, "🎛️ [Phase 03] Uncertainty Model Estimators (Trees)", type=QgsProcessingParameterNumber.Integer, defaultValue=200
        )
        p_uncert.setFlags(p_uncert.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_uncert)

        p_split = QgsProcessingParameterNumber(
            self.TRAIN_TEST_SPLIT, "🎛️ [Phase 03 & 04] Training Data Ratio (e.g., 0.8 for 80%)", type=QgsProcessingParameterNumber.Double, defaultValue=0.8
        )
        p_split.setFlags(p_split.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_split)

        p_var_corr = QgsProcessingParameterBoolean(
            self.ENABLE_DEPTH_VARIANCE_CORR,
            "🤖 [Phase 03] Enable Depth Variance Correction",
            defaultValue=False,
        )
        p_var_corr.setFlags(p_var_corr.flags() | QgsProcessingParameterDefinition.FlagAdvanced | QgsProcessingParameterDefinition.FlagHidden)
        self.addParameter(p_var_corr)

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
            self.MAX_GPR_SAMPLES, "📍 [Phase 04] Max GPR Training Samples", type=QgsProcessingParameterNumber.Integer, defaultValue=1500
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
            options=["Disabled", "Pearson (Linear)", "Spearman (Rank)", "Automatic-RANSAC", "Automatic-Random Forest"],
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
        p_med = QgsProcessingParameterNumber(
            self.MEDIAN_SIZE,
            "🧽 [Phase 03 & 04] Output Depth Median Filter Size",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=5,
        )
        p_med.setFlags(p_med.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_med)

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

        # --- [General System & Ranking] ---
        p_rs = QgsProcessingParameterNumber(
            self.RANDOM_STATE, "⚙️ [General] Random State for ML Split", type=QgsProcessingParameterNumber.Integer, defaultValue=42
        )
        p_rs.setFlags(p_rs.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_rs)

        p_fmt = QgsProcessingParameterEnum(
            self.OUTPUT_FORMAT, "⚙️ [General] Output Raster Format (Bit Depth)", options=["float32", "float64", "uint16"], defaultValue=0
        )
        p_fmt.setFlags(p_fmt.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_fmt)

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

        # --- Hyperparameters Search Grids ---
        p_rf = QgsProcessingParameterString(self.PARAM_RF, "🎛️ [General ML] Random Forest Hyperparameters", defaultValue="'n_estimators':[100, 500], 'max_depth':[10, 30]")
        p_rf.setFlags(p_rf.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_rf)

        p_gb = QgsProcessingParameterString(self.PARAM_GB, "🎛️ [General ML] Gradient Boosting Hyperparameters", defaultValue="'n_estimators':[100, 300], 'learning_rate':[0.05, 0.1]")
        p_gb.setFlags(p_gb.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_gb)

        p_et = QgsProcessingParameterString(self.PARAM_ET, "🎛️ [General ML] Extra Trees Hyperparameters", defaultValue="'n_estimators':[100, 500], 'max_depth':[10, 30]")
        p_et.setFlags(p_et.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_et)

        p_svr = QgsProcessingParameterString(self.PARAM_SVR, "🎛️ [General ML] SVR Hyperparameters", defaultValue="'C':[1, 10, 100], 'kernel':['rbf'], 'cache_size':[1000], 'max_iter':[20000]")
        p_svr.setFlags(p_svr.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_svr)

        p_mlp = QgsProcessingParameterString(self.PARAM_MLP, "🎛️ [General ML] MLP Hyperparameters", defaultValue="'hidden_layer_sizes':[(100,), (50, 50)], 'max_iter':[500]")
        p_mlp.setFlags(p_mlp.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_mlp)

        p_ridge = QgsProcessingParameterString(self.PARAM_RIDGE, "🎛️ [General ML] Ridge Hyperparameters", defaultValue="'alpha':[0.1, 1.0]", optional=True)
        p_ridge.setFlags(p_ridge.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_ridge)

        p_lasso = QgsProcessingParameterString(self.PARAM_LASSO, "🎛️ [General ML] Lasso Hyperparameters", defaultValue="'alpha':[0.01, 0.1]", optional=True)
        p_lasso.setFlags(p_lasso.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_lasso)

        p_en = QgsProcessingParameterString(self.PARAM_ELASTICNET, "🎛️ [General ML] ElasticNet Hyperparameters", defaultValue="'l1_ratio':[0.5]", optional=True)
        p_en.setFlags(p_en.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_en)

        p_knn = QgsProcessingParameterString(self.PARAM_KNN, "🎛️ [General ML] KNN Hyperparameters", defaultValue="'n_neighbors':[5, 10]", optional=True)
        p_knn.setFlags(p_knn.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_knn)

        p_dt = QgsProcessingParameterString(self.PARAM_DT, "🎛️ [General ML] Decision Tree Hyperparameters", defaultValue="'max_depth':[5, 10]", optional=True)
        p_dt.setFlags(p_dt.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_dt)

        p_huber = QgsProcessingParameterString(self.PARAM_HUBER, "🎛️ [General ML] Huber Hyperparameters", defaultValue="'epsilon':[1.1, 1.35, 1.5]", optional=True)
        p_huber.setFlags(p_huber.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_huber)

        p_xgb = QgsProcessingParameterString(self.PARAM_XGB, "🎛️ [General ML] XGBoost Hyperparameters", defaultValue="'n_estimators':[100, 200], 'max_depth':[4, 6], 'learning_rate':[0.05, 0.1]", optional=True)
        p_xgb.setFlags(p_xgb.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_xgb)

        p_lgbm = QgsProcessingParameterString(self.PARAM_LGBM, "🎛️ [General ML] LightGBM Hyperparameters", defaultValue="'n_estimators':[100, 200], 'max_depth':[4, 6], 'learning_rate':[0.05, 0.1]", optional=True)
        p_lgbm.setFlags(p_lgbm.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_lgbm)

        p_cat = QgsProcessingParameterString(self.PARAM_CATBOOST, "🎛️ [General ML] CatBoost Hyperparameters", defaultValue="'iterations':[100, 200], 'depth':[4, 6], 'learning_rate':[0.05, 0.1]", optional=True)
        p_cat.setFlags(p_cat.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_cat)


    def processAlgorithm(self, parameters, context, feedback):
        image_root = self.parameterAsString(parameters, self.INPUT_IMAGE_ROOT, context)
        if image_root and os.path.isfile(image_root):
            image_root = os.path.dirname(image_root)
        global_train_layer = self.parameterAsVectorLayer(parameters, self.INPUT_TRAIN, context)
        out_folder = self.parameterAsString(parameters, self.OUTPUT_MASTER_FOLDER, context)
        
        os.makedirs(out_folder, exist_ok=True)
        log_file_path = os.path.join(out_folder, "SpatioSpectral_Master_Log.txt")
        
        class FileLoggingFeedback(QgsProcessingFeedback):
            def __init__(self, inner, log_path):
                super().__init__()
                self.inner = inner
                self.log_path = log_path

            def setProgressText(self, text):
                append_log(text, self.log_path, None)
                self.inner.setProgressText(text)

            def pushInfo(self, info):
                append_log(info, self.log_path, None)
                self.inner.pushInfo(info)

            def pushCommandInfo(self, info):
                append_log(info, self.log_path, None)
                self.inner.pushCommandInfo(info)

            def pushDebugInfo(self, info):
                append_log(info, self.log_path, None)
                self.inner.pushDebugInfo(info)

            def pushConsoleInfo(self, info):
                append_log(info, self.log_path, None)
                self.inner.pushConsoleInfo(info)

            def reportError(self, error, fatalError=False):
                append_log(f"ERROR: {error}", self.log_path, None)
                self.inner.reportError(error, fatalError)

            def isCanceled(self):
                return self.inner.isCanceled()

        custom_feedback = FileLoggingFeedback(feedback, log_file_path)

        masterflow_params = {}
        for k, v in parameters.items():
            masterflow_params[k] = v

        # Set default values for params that were removed/replaced
        masterflow_params["OUTPUT_FOLDER"] = out_folder

        runner = SpatioSpectralSDBRunner(out_folder)
        results = runner.run_spatiospectral_flow(image_root, masterflow_params, self, context, custom_feedback)
        
        if not results:
            results = {}
        results[self.OUTPUT_MASTER_FOLDER] = out_folder
        results["OUTPUT_FOLDER"] = out_folder

        primary_files = {
            "Aggregated Depth Map": results.get("AGGREGATED_DEPTH"),
            "Refined Depth Map": results.get("FINAL_REFINED_DEPTH"),
            "HTML Dashboard": os.path.join(out_folder, "SDB_Validation_Dashboard.html") if os.path.exists(os.path.join(out_folder, "SDB_Validation_Dashboard.html")) else None,
            "Master Execution Log": log_file_path,
        }
        try:
            log_module_completion(
                module_title="SDB SpatioSpectral Masterflow (Multi-Scene)",
                out_dir=out_folder,
                primary_files=primary_files,
                log_path=log_file_path,
                feedback=feedback,
            )
        except Exception:
            pass

        return results
