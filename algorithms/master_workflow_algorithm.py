from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer, QgsProcessingParameterVectorLayer,
    QgsProcessingParameterFile, QgsProcessingParameterField,
    QgsProcessingParameterString, QgsProcessingParameterNumber,
    QgsProcessingParameterBand, QgsProcessingParameterFolderDestination,
    QgsProcessingParameterBoolean, QgsRasterLayer, QgsProject, QgsProcessingUtils,
    QgsApplication
)
import os
import time
import numpy as np
import pandas as pd
import processing
import rasterio

try:
    from scipy.ndimage import median_filter, binary_opening, binary_closing
    scipy_installed = True
except ImportError:
    scipy_installed = False

class MasterWorkflowAlgorithm(QgsProcessingAlgorithm):
    # (تعريف المدخلات يبقى كما هو)
    INPUT_RASTER = 'INPUT_RASTER'; INPUT_TRAINING_VEC = 'INPUT_TRAINING_VEC'; INPUT_TESTING_VEC = 'INPUT_TESTING_VEC'
    DEPTH_FIELD_TRAINING = 'DEPTH_FIELD_TRAINING'; DEPTH_FIELD_TESTING = 'DEPTH_FIELD_TESTING'; N_ITERATIONS = 'N_ITERATIONS'
    OUTPUT_FOLDER = 'OUTPUT_FOLDER'; GREEN_BAND = 'GREEN_BAND'; NIR_BAND = 'NIR_BAND'; SWIR_BAND = 'SWIR_BAND'
    WATER_INDEX_THRESHOLD = 'WATER_INDEX_THRESHOLD'; RUN_RF = 'RUN_RF'; RUN_GB = 'RUN_GB'; RUN_ET = 'RUN_ET'
    RUN_MLP = 'RUN_MLP'; RUN_SVR = 'RUN_SVR'; RUN_KNN = 'RUN_KNN'; RUN_DT = 'RUN_DT'; RUN_ELASTIC = 'RUN_ELASTIC'
    RUN_RIDGE = 'RUN_RIDGE'; RUN_LASSO = 'RUN_LASSO'; RUN_LINEAR = 'RUN_LINEAR'; RUN_BANDRATIO = 'RUN_BANDRATIO'
    MEDIAN_FILTER_SIZE = 'MEDIAN_FILTER_SIZE'
    BANDRATIO_HIGH = 'BANDRATIO_HIGH'
    BANDRATIO_LOW = 'BANDRATIO_LOW'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_RASTER, 'Input Satellite Image'))
        self.addParameter(QgsProcessingParameterBand(self.GREEN_BAND, 'Water Masking - Green Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=3))
        self.addParameter(QgsProcessingParameterBand(self.NIR_BAND, 'Water Masking - NIR Band (for NDWI)', parentLayerParameterName=self.INPUT_RASTER, optional=True))
        self.addParameter(QgsProcessingParameterBand(self.SWIR_BAND, 'Water Masking - SWIR Band (for MNDWI - preferred)', parentLayerParameterName=self.INPUT_RASTER, optional=True))
        self.addParameter(QgsProcessingParameterNumber(self.WATER_INDEX_THRESHOLD, 'Water Index Threshold', type=QgsProcessingParameterNumber.Double, defaultValue=0.0))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_TRAINING_VEC, 'Training Points'))
        self.addParameter(QgsProcessingParameterField(self.DEPTH_FIELD_TRAINING, 'Depth Field (Training)', parentLayerParameterName=self.INPUT_TRAINING_VEC, type=QgsProcessingParameterField.Numeric))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_TESTING_VEC, 'Unseen Testing Points'))
        self.addParameter(QgsProcessingParameterField(self.DEPTH_FIELD_TESTING, 'Depth Field (Testing)', parentLayerParameterName=self.INPUT_TESTING_VEC, type=QgsProcessingParameterField.Numeric))
        
        self.addParameter(QgsProcessingParameterBoolean(self.RUN_RF, 'Run: RandomForest', defaultValue=True)); self.addParameter(QgsProcessingParameterBoolean(self.RUN_GB, 'Run: Gradient Boosting', defaultValue=True))
        self.addParameter(QgsProcessingParameterBoolean(self.RUN_ET, 'Run: Extra Trees', defaultValue=True)); self.addParameter(QgsProcessingParameterBoolean(self.RUN_MLP, 'Run: MLP (Neural Network)', defaultValue=False))
        self.addParameter(QgsProcessingParameterBoolean(self.RUN_SVR, 'Run: SVR (can be very slow)', defaultValue=False)); self.addParameter(QgsProcessingParameterBoolean(self.RUN_KNN, 'Run: KNN', defaultValue=True))
        self.addParameter(QgsProcessingParameterBoolean(self.RUN_DT, 'Run: Decision Tree', defaultValue=False)); self.addParameter(QgsProcessingParameterBoolean(self.RUN_ELASTIC, 'Run: ElasticNet', defaultValue=False))
        self.addParameter(QgsProcessingParameterBoolean(self.RUN_RIDGE, 'Run: Ridge', defaultValue=False)); self.addParameter(QgsProcessingParameterBoolean(self.RUN_LASSO, 'Run: Lasso', defaultValue=False))
        self.addParameter(QgsProcessingParameterBoolean(self.RUN_LINEAR, 'Run: Linear Regression', defaultValue=True))
        
        self.addParameter(QgsProcessingParameterBoolean(self.RUN_BANDRATIO, 'Run: Band Ratio (Stumpf)', defaultValue=True))
        self.addParameter(QgsProcessingParameterBand(self.BANDRATIO_HIGH, 'Band Ratio - High Reflectance Band (e.g., Green)', parentLayerParameterName=self.INPUT_RASTER, defaultValue=3))
        self.addParameter(QgsProcessingParameterBand(self.BANDRATIO_LOW, 'Band Ratio - Low Reflectance Band (e.g., Blue)', parentLayerParameterName=self.INPUT_RASTER, defaultValue=2))

        self.addParameter(QgsProcessingParameterNumber(self.N_ITERATIONS, 'Search Iterations (for complex models)', type=QgsProcessingParameterNumber.Integer, defaultValue=20, minValue=10))
        self.addParameter(QgsProcessingParameterNumber(self.MEDIAN_FILTER_SIZE, 'Apply Median Filter to ALL Results (0 to disable)', type=QgsProcessingParameterNumber.Integer, defaultValue=3, minValue=0))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, 'Main Output Folder'))

    def name(self): return 'sdb_master_workflow'
    def displayName(self): return 'SDB Master Workflow (Full Comparison)'
    def group(self): return 'SDB Tools'
    def groupId(self): return 'sdb_tools'
    def createInstance(self): return MasterWorkflowAlgorithm()
    
    def shortHelpString(self):
        return "This master tool automates the entire SDB process by running and comparing a selection of algorithms."
            
    def processAlgorithm(self, parameters, context, feedback):
        provider_id = 'sdb_tools'
        provider = QgsApplication.processingRegistry().providerById(provider_id)
        if not provider:
            raise RuntimeError(f"Could not find SDB Tools provider ('{provider_id}'). Is the plugin enabled and loaded correctly?")
        
        input_raster = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context).source()
        training_points = self.parameterAsVectorLayer(parameters, self.INPUT_TRAINING_VEC, context).source()
        testing_points = self.parameterAsVectorLayer(parameters, self.INPUT_TESTING_VEC, context).source()
        depth_field_training = self.parameterAsString(parameters, self.DEPTH_FIELD_TRAINING, context)
        depth_field_testing = self.parameterAsString(parameters, self.DEPTH_FIELD_TESTING, context)
        n_iterations = self.parameterAsInt(parameters, self.N_ITERATIONS, context)
        main_output_folder = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        green_band_idx = self.parameterAsInt(parameters, self.GREEN_BAND, context)
        nir_band_idx = self.parameterAsInt(parameters, self.NIR_BAND, context)
        swir_band_idx = self.parameterAsInt(parameters, self.SWIR_BAND, context)
        threshold = self.parameterAsDouble(parameters, self.WATER_INDEX_THRESHOLD, context)
        filter_size = self.parameterAsInt(parameters, self.MEDIAN_FILTER_SIZE, context)
        br_high_idx = self.parameterAsInt(parameters, self.BANDRATIO_HIGH, context)
        br_low_idx = self.parameterAsInt(parameters, self.BANDRATIO_LOW, context)
        start_time = time.time()

        if filter_size > 1 and not scipy_installed:
            raise RuntimeError("Required library 'scipy' is not installed, but it's needed for water mask cleaning or median filtering.")
        os.makedirs(main_output_folder, exist_ok=True)

        feedback.setProgress(0)
        feedback.pushInfo("Step 1: Creating and cleaning Water Mask (Progress: 0% -> 10%)...")
        temp_folder = QgsProcessingUtils.tempFolder()
        water_mask_path = os.path.join(temp_folder, 'water_mask.tif')
        
        with rasterio.open(input_raster) as src:
            profile = src.profile; green_band = src.read(green_band_idx).astype('float32')
            if swir_band_idx > 0:
                feedback.pushInfo("Using MNDWI (Green/SWIR) for water masking.")
                swir_band = src.read(swir_band_idx).astype('float32'); numerator, denominator = green_band - swir_band, green_band + swir_band
            elif nir_band_idx > 0:
                feedback.pushInfo("Using NDWI (Green/NIR) for water masking.")
                nir_band = src.read(nir_band_idx).astype('float32'); numerator, denominator = green_band - nir_band, green_band + nir_band
            else: raise RuntimeError("Please provide either a NIR or SWIR band for water masking.")
            denominator = np.where(denominator == 0, 1e-9, denominator); water_index = numerator / denominator
            water_mask_raw = (water_index > threshold)
            structure = np.ones((3,3), dtype=bool)
            mask_opened = binary_opening(water_mask_raw, structure=structure)
            mask_cleaned = binary_closing(mask_opened, structure=structure).astype('int16')
            mask_profile = profile.copy(); mask_profile.update(dtype=rasterio.int16, count=1, compress='lzw', nodata=0)
            with rasterio.open(water_mask_path, 'w', **mask_profile) as dst: dst.write(mask_cleaned, 1)

        feedback.pushInfo("Step 2: Applying water mask to the original image...")
        masked_image_path = os.path.join(main_output_folder, 'masked_image.tif')
        with rasterio.open(input_raster) as src:
            with rasterio.open(water_mask_path) as mask_src:
                out_profile = src.profile.copy(); out_profile['dtype'] = 'float32'; nodata_val = -9999.0; out_profile['nodata'] = nodata_val
                with rasterio.open(masked_image_path, 'w', **out_profile) as dst:
                    for i in range(1, src.count + 1):
                        band_data, mask_data = src.read(i).astype('float32'), mask_src.read(1)
                        band_data[mask_data == 0] = nodata_val; dst.write(band_data, i)
        feedback.pushInfo(f"Masked image created at: {masked_image_path}"); feedback.setProgress(10)

        all_algorithms = {
            self.RUN_RF: ('autopredict', f'{provider_id}:sdb_autopredict_rf', 'rf', 'RandomForest'), self.RUN_GB: ('autopredict', f'{provider_id}:sdb_autopredict_gb', 'gb', 'Gradient Boosting'),
            self.RUN_ET: ('autopredict', f'{provider_id}:sdb_autopredict_extratrees', 'extratrees', 'Extra Trees'), self.RUN_MLP: ('autopredict', f'{provider_id}:sdb_autopredict_mlp', 'mlp', 'MLP'),
            self.RUN_SVR: ('autopredict', f'{provider_id}:sdb_autopredict_svr', 'svr', 'SVR'), self.RUN_KNN: ('autopredict', f'{provider_id}:sdb_autopredict_knn', 'knn', 'KNN'),
            self.RUN_DT: ('autopredict', f'{provider_id}:sdb_autopredict_decisiontree', 'dt', 'Decision Tree'), self.RUN_ELASTIC: ('autopredict', f'{provider_id}:sdb_autopredict_elasticnet', 'elasticnet', 'ElasticNet'),
            self.RUN_RIDGE: ('autopredict', f'{provider_id}:sdb_autopredict_ridge', 'ridge', 'Ridge'), self.RUN_LASSO: ('autopredict', f'{provider_id}:sdb_autopredict_lasso', 'lasso', 'Lasso'),
            self.RUN_LINEAR: ('simple', f'{provider_id}:ml_linear', 'linear', 'Linear Regression'), 
            self.RUN_BANDRATIO: ('simple', f'{provider_id}:bandratio', 'bandratio', 'Band Ratio')
        }
        selected_algs_to_run = [info for param, info in all_algorithms.items() if self.parameterAsBool(parameters, param, context)]
        
        final_summary = []; total_algs = len(selected_algs_to_run)
        if total_algs == 0: raise RuntimeError("No algorithms were selected to run.")
        progress_per_alg = 85.0 / total_algs

        for i, (alg_type, alg_id, alg_name, display_name) in enumerate(selected_algs_to_run):
            if feedback.isCanceled(): break
            feedback.pushInfo(f"\n--- Running Algorithm {i+1}/{total_algs}: {display_name.upper()} ---")
            feedback.setProgress(int(10 + (i * progress_per_alg)))
            alg_output_folder = os.path.join(main_output_folder, alg_name)
            os.makedirs(alg_output_folder, exist_ok=True)
            params = {'INPUT_RASTER': masked_image_path, 'INPUT_SAMPLES_VEC': training_points, 'DEPTH_FIELD_VEC': depth_field_training, 'OUTPUT_FOLDER': alg_output_folder}
            
            if alg_type == 'autopredict':
                params['N_ITERATIONS'] = n_iterations
                predicted_raster_filename = f"autopredict_{alg_name}_depth.tif"; internal_report_filename = f"autopredict_{alg_name}_report.txt"
            else:
                if alg_id.endswith('bandratio'):
                    if br_high_idx == 0 or br_low_idx == 0: feedback.pushWarning("Band Ratio was selected but its bands were not specified. Skipping."); continue
                    params['BAND_HIGH_REF'] = br_high_idx; params['BAND_LOW_REF'] = br_low_idx
                predicted_raster_filename = f"{alg_name}_depth.tif"; internal_report_filename = f"{alg_name}_report.txt"
            
            try: processing.run(alg_id, params, context=context, feedback=feedback, is_child_algorithm=True)
            except Exception as e: feedback.pushWarning(f"Algorithm {display_name} failed. Skipping. Error: {e}"); continue
            
            predicted_raster_path_original = os.path.join(alg_output_folder, predicted_raster_filename)
            path_for_evaluation = predicted_raster_path_original
            if filter_size > 1 and os.path.exists(predicted_raster_path_original):
                feedback.pushInfo(f"Applying {filter_size}x{filter_size} Median Filter...")
                filtered_name = predicted_raster_filename.replace('.tif', f'_filtered.tif')
                predicted_raster_path_filtered = os.path.join(alg_output_folder, filtered_name)
                try:
                    with rasterio.open(predicted_raster_path_original) as src:
                        profile = src.profile; depth_raster = src.read(1); nodata_value = profile.get('nodata', np.nan)
                        valid_data_mask = ~np.isnan(depth_raster)
                        if not np.isnan(nodata_value): valid_data_mask &= (depth_raster != nodata_value)
                        filtered_data = median_filter(depth_raster, size=filter_size, mode='reflect')
                        filtered_raster = depth_raster.copy(); filtered_raster[valid_data_mask] = filtered_data[valid_data_mask]
                        with rasterio.open(predicted_raster_path_filtered, 'w', **profile) as dst: dst.write(filtered_raster, 1)
                        path_for_evaluation = predicted_raster_path_filtered
                except Exception as e: feedback.pushWarning(f"Could not apply median filter. Evaluating original result. Error: {e}")

            eval_report_path = os.path.join(alg_output_folder, "FINAL_EVALUATION_REPORT.txt")
            summary_entry = {'Algorithm': display_name.upper(), 'Internal_R2': np.nan, 'Internal_RMSE': np.nan, 'Unseen_Data_R2': np.nan, 'Unseen_Data_RMSE': np.nan, 'Raster_Path': path_for_evaluation}
            internal_report_path = os.path.join(alg_output_folder, internal_report_filename)
            if os.path.exists(internal_report_path):
                with open(internal_report_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if "R-squared (R2):" in line: summary_entry['Internal_R2'] = float(line.split(':')[1].strip())
                        if "Root Mean Squared Error (RMSE):" in line: summary_entry['Internal_RMSE'] = float(line.split(':')[1].strip())
            if os.path.exists(path_for_evaluation):
                eval_params = {'INPUT_PREDICTED_RASTER': path_for_evaluation, 'INPUT_UNSEEN_VEC': testing_points, 'DEPTH_FIELD_VEC': depth_field_testing, 'OUTPUT_FILE': eval_report_path}
                try:
                    processing.run('sdb_tools:sdb_evaluate_model', eval_params, context=context, feedback=feedback, is_child_algorithm=True)
                    if os.path.exists(eval_report_path):
                        with open(eval_report_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                if "R-squared (R2):" in line: summary_entry['Unseen_Data_R2'] = float(line.split(':')[1].strip())
                                if "Root Mean Squared Error (RMSE):" in line: summary_entry['Unseen_Data_RMSE'] = float(line.split(':')[1].strip())
                except Exception as e: feedback.pushWarning(f"Evaluation failed for {display_name}. Error: {e}")
            else: feedback.pushWarning(f"Predicted raster not found for {display_name}. Skipping evaluation.")
            final_summary.append(summary_entry)

        feedback.setProgress(95)
        feedback.pushInfo("\n--- WORKFLOW COMPLETE ---")
        summary_report_path = os.path.join(main_output_folder, "MASTER_SUMMARY_REPORT.txt")
        if not final_summary: feedback.pushWarning("No algorithms were successfully run."); return {}
        summary_df = pd.DataFrame(final_summary); summary_df.dropna(subset=['Unseen_Data_R2', 'Unseen_Data_RMSE'], inplace=True)
        if summary_df.empty:
            feedback.pushWarning("No models were successfully evaluated on unseen data.")
            with open(summary_report_path, 'w', encoding='utf-8') as f: f.write("No models were successfully evaluated on unseen data.")
            return {}
        r2_range = summary_df['Unseen_Data_R2'].max() - summary_df['Unseen_Data_R2'].min()
        rmse_range = summary_df['Unseen_Data_RMSE'].max() - summary_df['Unseen_Data_RMSE'].min()
        if r2_range == 0 or np.isnan(r2_range): r2_norm = pd.Series([1.0] * len(summary_df), index=summary_df.index)
        else: r2_norm = (summary_df['Unseen_Data_R2'] - summary_df['Unseen_Data_R2'].min()) / r2_range
        if rmse_range == 0 or np.isnan(rmse_range): rmse_norm = pd.Series([1.0] * len(summary_df), index=summary_df.index)
        else: rmse_norm = 1 - ((summary_df['Unseen_Data_RMSE'] - summary_df['Unseen_Data_RMSE'].min()) / rmse_range)
        summary_df['Final_Score'] = 0.7 * r2_norm + 0.3 * rmse_norm
        summary_df = summary_df.sort_values(by='Final_Score', ascending=False, na_position='last')
        column_order = ['Algorithm', 'Final_Score', 'Unseen_Data_R2', 'Unseen_Data_RMSE', 'Internal_R2', 'Internal_RMSE', 'Raster_Path']
        existing_columns = [col for col in column_order if col in summary_df.columns]
        summary_df = summary_df[existing_columns]
        with open(summary_report_path, 'w', encoding='utf-8') as f:
            f.write("====================================================\n"); f.write(" SDB Master Workflow - Full Comparison Report\n"); f.write("====================================================\n\n")
            f.write(f"Workflow completed on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"); f.write(f"Total time: {(time.time() - start_time)/60:.2f} minutes\n")
            f.write(f"Search iterations for complex models: {n_iterations}\n\n"); f.write("Internal R2/RMSE: Performance on the model's internal test set.\n")
            f.write("Unseen Data R2/RMSE: Performance on the separate, unseen validation points.\n"); f.write("Final_Score: Combined metric (70% R2, 30% RMSE) to rank models. Higher is better.\n\n")
            f.write(summary_df.to_string(index=False, float_format="%.4f"))
        feedback.pushInfo(f"All processes finished. Master summary report saved to: {summary_report_path}")
        best_result = summary_df.iloc[0]
        if pd.notna(best_result['Final_Score']) and os.path.exists(best_result['Raster_Path']):
            best_alg_name = best_result['Algorithm']; best_raster_path = best_result['Raster_Path']
            feedback.pushInfo(f"\nLoading the best result into QGIS: {best_alg_name} (Final Score = {best_result['Final_Score']:.4f})")
            layer_name = f"Best Result - {best_alg_name}"; rlayer = QgsRasterLayer(best_raster_path, layer_name)
            if rlayer.isValid(): QgsProject.instance().addMapLayer(rlayer)
            else: feedback.pushWarning(f"Could not load the best raster layer: {best_raster_path}")
        else: feedback.pushWarning("Could not determine or find the best result to load.")
        feedback.setProgress(100)
        return {}
