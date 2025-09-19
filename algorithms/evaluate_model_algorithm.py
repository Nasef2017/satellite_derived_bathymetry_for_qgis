from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer, QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField, QgsProcessingParameterFileDestination,
    QgsProject, QgsCoordinateTransform, QgsPointXY
)
import os
import time
import numpy as np
import pandas as pd

try:
    import rasterio
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    libs_installed = True
except ImportError:
    libs_installed = False

class EvaluateModelAlgorithm(QgsProcessingAlgorithm):
    INPUT_PREDICTED_RASTER = 'INPUT_PREDICTED_RASTER'
    INPUT_UNSEEN_VEC = 'INPUT_UNSEEN_VEC'
    DEPTH_FIELD_VEC = 'DEPTH_FIELD_VEC'
    OUTPUT_FILE = 'OUTPUT_FILE'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_PREDICTED_RASTER, 'Input Predicted Depth Raster'))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_UNSEEN_VEC, 'Unseen Validation Points (Vector)'))
        self.addParameter(QgsProcessingParameterField(self.DEPTH_FIELD_VEC, 'True Depth Field', parentLayerParameterName=self.INPUT_UNSEEN_VEC, type=QgsProcessingParameterField.Numeric))
        
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT_FILE, 'Output Evaluation Report', fileFilter='Text files (*.txt)'))

    def name(self): return 'sdb_evaluate_model'
    def displayName(self): return 'SDB: Evaluate Model on Unseen Data'
    def group(self): return 'SDB Tools'
    def groupId(self): return 'sdb_tools'
    def shortHelpString(self): return 'Evaluates a predicted depth raster against unseen validation points and generates an accuracy report.'
    def createInstance(self): return EvaluateModelAlgorithm()
    def _read_validation_points(self, vlayer, depth_field, feedback):
        if vlayer is None: raise RuntimeError('Validation vector layer not provided.')
        if not depth_field:
            numeric = [f.name() for f in vlayer.fields() if f.typeName().lower() in ('double','real','integer','int','integer64','float')]
            if len(numeric) == 1:
                depth_field = numeric[0]; feedback.pushInfo(f'Auto-detected depth field: {depth_field}')
            else: raise RuntimeError('Please specify the true depth field for the validation layer.')
        rows = []
        for feat in vlayer.getFeatures():
            geom = feat.geometry()
            if geom is None: continue
            pt = geom.centroid().asPoint() if geom.isMultipart() else geom.asPoint()
            try:
                d = float(feat[depth_field])
                rows.append((float(pt.x()), float(pt.y()), d))
            except (KeyError, TypeError, ValueError):
                continue
        if not rows: raise RuntimeError('No valid validation points read from vector layer.')
        return pd.DataFrame(rows, columns=['x','y','depth_true'])

    def _transform_point_if_needed(self, x, y, sample_crs, raster_crs, feedback):
        if sample_crs == raster_crs: return x, y
        try:
            tr = QgsCoordinateTransform(sample_crs, raster_crs, QgsProject.instance())
            p = tr.transform(QgsPointXY(x, y)); return float(p.x()), float(p.y())
        except Exception as e:
            feedback.pushWarning(f'CRS transform failed: {e}. Assuming coordinates match raster CRS.')
            return x, y
            
    def processAlgorithm(self, parameters, context, feedback):
        if not libs_installed: raise RuntimeError('Required libraries (scikit-learn, rasterio) are not installed.')
        
        predicted_raster = self.parameterAsRasterLayer(parameters, self.INPUT_PREDICTED_RASTER, context)
        validation_layer = self.parameterAsVectorLayer(parameters, self.INPUT_UNSEEN_VEC, context)
        depth_field = self.parameterAsString(parameters, self.DEPTH_FIELD_VEC, context)
        output_file = self.parameterAsFileOutput(parameters, self.OUTPUT_FILE, context)
        
        start_time = time.time()
        
        feedback.pushInfo("Step 1: Reading validation data and predicted raster...")
        if not predicted_raster: raise RuntimeError('Invalid input predicted raster.')
        if not validation_layer: raise RuntimeError('Invalid input validation points.')
        
        validation_df = self._read_validation_points(validation_layer, depth_field, feedback)
        validation_crs = validation_layer.crs()

        with rasterio.open(predicted_raster.source()) as src:
            transform = src.transform
            raster_crs = predicted_raster.crs()
            raster_band = src.read(1).astype('float32')

        feedback.pushInfo(f"Read {len(validation_df)} validation points.")
        
        # ---                         ---
        feedback.pushInfo("\nStep 2: Extracting predicted values and evaluating...")
        y_true_list = []
        y_pred_list = []
        
        for _, row in validation_df.iterrows():
            x0, y0, depth_true = row['x'], row['y'], row['depth_true']
            x2, y2 = self._transform_point_if_needed(x0, y0, validation_crs, raster_crs, feedback)
            col, row_idx = ~transform * (x2, y2)
            r_i, c_i = int(row_idx), int(col)
            
            if 0 <= r_i < raster_band.shape[0] and 0 <= c_i < raster_band.shape[1]:
                predicted_val = raster_band[r_i, c_i]
                if not np.isnan(predicted_val):
                    y_true_list.append(depth_true)
                    y_pred_list.append(predicted_val)

        if len(y_true_list) < 1:
            raise RuntimeError("No validation points were found within the raster extent.")
        
        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        
        feedback.pushInfo(f"Successfully evaluated {len(y_true)} points.")
        
        # ---                   ---
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        bias = np.mean(y_pred - y_true)

        feedback.pushInfo(f'Evaluation complete: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}')

        # --- 4.               ---
        feedback.pushInfo("\nStep 3: Generating final report...")
        end_time = time.time()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("====================================================\n")
            f.write(" SDB Model Evaluation Report (on Unseen Data)\n")
            f.write("====================================================\n\n")
            f.write(f"Evaluation performed on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Predicted Raster: {predicted_raster.name()}\n")
            f.write(f"Validation Points: {validation_layer.name()}\n")
            f.write(f"Number of points evaluated: {len(y_true)}\n")
            f.write(f"Total time: {end_time - start_time:.2f} seconds\n")
            
            f.write("\n--- Accuracy Metrics ---\n")
            f.write(f"R-squared (R2): {r2:.4f}\n")
            f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
            f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
            f.write(f"Bias (Mean Error): {bias:.4f}\n")

        feedback.pushInfo(f"Process complete! Evaluation report saved to: {output_file}")
        
        return {self.OUTPUT_FILE: output_file}