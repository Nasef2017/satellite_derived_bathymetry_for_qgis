from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer, QgsProcessingParameterVectorLayer,
    QgsProcessingParameterFile, QgsProcessingParameterField,
    QgsProcessingParameterString, QgsProcessingParameterNumber,
    QgsProcessingParameterFolderDestination,
    QgsProject, QgsCoordinateTransform, QgsPointXY
)
import os, time, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import rasterio
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from skopt import gp_minimize
    from skopt.space import Integer, Categorical
    from skopt.utils import use_named_args
    libs_installed = True
except ImportError:
    libs_installed = False

class RandomForestAutoPredictAlgorithm(QgsProcessingAlgorithm):
    INPUT_RASTER = 'INPUT_RASTER'; INPUT_SAMPLES_VEC = 'INPUT_SAMPLES_VEC'; INPUT_SAMPLES_FILE = 'INPUT_SAMPLES_FILE'
    DEPTH_FIELD_VEC = 'DEPTH_FIELD_VEC'; DEPTH_FIELD_FILE = 'DEPTH_FIELD_FILE'; N_ITERATIONS = 'N_ITERATIONS'; OUTPUT_FOLDER = 'OUTPUT_FOLDER'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_RASTER, 'Input raster (multiband)'))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_SAMPLES_VEC, 'Sample points (vector, optional)', optional=True))
        self.addParameter(QgsProcessingParameterFile(self.INPUT_SAMPLES_FILE, 'Sample points file (CSV/XYZ, optional)', optional=True, fileFilter='*.csv;*.xyz;*.txt'))
        self.addParameter(QgsProcessingParameterField(self.DEPTH_FIELD_VEC, 'Depth field (vector)', parentLayerParameterName=self.INPUT_SAMPLES_VEC, type=QgsProcessingParameterField.Numeric, optional=True))
        self.addParameter(QgsProcessingParameterString(self.DEPTH_FIELD_FILE, 'Depth column name (CSV/XYZ)', optional=True))
        self.addParameter(QgsProcessingParameterNumber(self.N_ITERATIONS, 'Number of Smart Search Iterations', type=QgsProcessingParameterNumber.Integer, defaultValue=30, minValue=10))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, 'Output Folder'))

    def name(self): 
        return 'sdb_autopredict_rf'
    
    def displayName(self): 
        return 'SDB Auto-Predict: RandomForest'
    
    def group(self): 
        return 'SDB Tools'
    
    def groupId(self): 
        return 'sdb_tools'
    
    def shortHelpString(self): 
        return 'Fully automated: Finds best parameters for RandomForest and generates all final outputs in one step.'
    
    def createInstance(self): 
        return RandomForestAutoPredictAlgorithm()
        
    def _read_samples_from_vector(self, vlayer, depth_field, feedback):
        if vlayer is None: raise RuntimeError('Vector layer not provided.')
        if not depth_field:
            numeric = [f.name() for f in vlayer.fields() if f.typeName().lower() in ('double','real','integer','int','integer64','float')]
            if len(numeric) == 1:
                depth_field = numeric[0]; feedback.pushInfo(f'Auto-detected depth field: {depth_field}')
            else: raise RuntimeError('Please specify depth field for vector layer.')
        rows = []
        for feat in vlayer.getFeatures():
            geom = feat.geometry()
            if geom is None: continue
            pt = geom.centroid().asPoint() if geom.isMultipart() else geom.asPoint()
            try: rows.append((float(pt.x()), float(pt.y()), float(feat[depth_field])))
            except (KeyError, TypeError, ValueError): continue
        if not rows: raise RuntimeError('No valid sample points read from vector layer.')
        return pd.DataFrame(rows, columns=['x','y','depth'])

    def _read_samples_from_file(self, filepath, depth_col, feedback):
        if not filepath or not os.path.exists(filepath): raise RuntimeError('Sample file not provided or not found.')
        try: df = pd.read_csv(filepath, sep=None, engine='python')
        except Exception: df = pd.read_csv(filepath, sep=r'\s+', engine='python', header=None)
        if df.shape[1] < 3: raise RuntimeError('CSV/XYZ file must have at least 3 columns.')
        if depth_col and depth_col in df.columns: depth_series = df[depth_col]
        else:
            numeric_cols = [c for c in df.columns[2:] if pd.api.types.is_numeric_dtype(df[c])]
            if not numeric_cols: raise RuntimeError('No numeric depth column found; please provide depth column name.')
            depth_col = numeric_cols[0]; feedback.pushInfo(f'Auto-using depth column: {depth_col}')
            depth_series = df[depth_col]
        return pd.DataFrame({'x': df.iloc[:,0].astype(float), 'y': df.iloc[:,1].astype(float), 'depth': depth_series.astype(float)})

    def _transform_point_if_needed(self, x, y, sample_crs, raster_crs, feedback):
        if sample_crs == raster_crs: return x, y
        try:
            tr = QgsCoordinateTransform(sample_crs, raster_crs, QgsProject.instance())
            p = tr.transform(QgsPointXY(x, y)); return float(p.x()), float(p.y())
        except Exception as e:
            feedback.pushWarning(f'CRS transform failed: {e}. Assuming coordinates match raster CRS.')
            return x, y
            
    def processAlgorithm(self, parameters, context, feedback):
        if not libs_installed: raise RuntimeError('Required libraries are not installed. Please run `pip install scikit-learn scikit-optimize rasterio` in OSGeo4W Shell.')
        
        raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        vlayer = self.parameterAsVectorLayer(parameters, self.INPUT_SAMPLES_VEC, context)
        sample_file = self.parameterAsFile(parameters, self.INPUT_SAMPLES_FILE, context)
        depth_field_vec = self.parameterAsString(parameters, self.DEPTH_FIELD_VEC, context)
        depth_field_file = self.parameterAsString(parameters, self.DEPTH_FIELD_FILE, context)
        n_iterations = self.parameterAsInt(parameters, self.N_ITERATIONS, context)
        out_folder = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        
        feedback.pushInfo("Step 1: Reading and processing data...")
        if not raster_layer: raise RuntimeError('Invalid input raster.')
        if vlayer is None and not sample_file: raise RuntimeError('Please provide sample points.')
        if vlayer:
            samples_df = self._read_samples_from_vector(vlayer, depth_field_vec, feedback)
            sample_crs = vlayer.crs()
        else:
            samples_df = self._read_samples_from_file(sample_file, depth_field_file, feedback)
            sample_crs = raster_layer.crs()
        with rasterio.open(raster_layer.source()) as src:
            profile = src.profile
            bands = [src.read(i + 1).astype('float32') for i in range(src.count)]
            transform = src.transform
            raster_crs = raster_layer.crs()
        
        processed_data = []
        for _, row in samples_df.iterrows():
            x0, y0, depth = row['x'], row['y'], row['depth']
            x2, y2 = self._transform_point_if_needed(x0, y0, sample_crs, raster_crs, feedback)
            col, row_idx = ~transform * (x2, y2); r_i, c_i = int(row_idx), int(col)
            if 0 <= r_i < bands[0].shape[0] and 0 <= c_i < bands[0].shape[1]:
                feat = [b[r_i, c_i] for b in bands]
                if not any(np.isnan(feat)):
                    processed_data.append([x0, y0, depth] + feat)
        
        num_bands = len(bands)
        columns = ['x_orig', 'y_orig', 'depth'] + [f'band_{i+1}' for i in range(num_bands)]
        processed_df = pd.DataFrame(processed_data, columns=columns)
        
        if len(processed_df) < 20: raise RuntimeError("Not enough valid samples for a reliable search.")

        features = processed_df.drop(columns=['x_orig', 'y_orig', 'depth'])
        target = processed_df['depth']
        coords = processed_df[['x_orig', 'y_orig']]

        scaler = StandardScaler().fit(features)
        X_scaled = scaler.transform(features)
        min_bounds, max_bounds = features.min(axis=0).values, features.max(axis=0).values
        
        X_train_full, X_test, y_train_full, y_test, coords_train, coords_test = train_test_split(X_scaled, target, coords, test_size=0.3, random_state=42)
        X_train_search, X_val_search, y_train_search, y_val_search = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)
        feedback.pushInfo(f"Data ready for search. Training on {len(y_train_search)}, validating on {len(y_val_search)}.")
        
        feedback.pushInfo("\nStep 2: Starting smart search for best parameters...")
        param_space = [
            Integer(50, 500, name='n_estimators'), Integer(10, 50, name='max_depth'),
            Integer(2, 10, name='min_samples_split'), Integer(1, 5, name='min_samples_leaf'),
            Categorical(['sqrt', 'log2', None], name='max_features')
        ]
        reg_search = RandomForestRegressor(random_state=42, n_jobs=1)
        
        @use_named_args(param_space)
        def objective_function(**params):
            if feedback.isCanceled(): return 1e9
            reg_search.set_params(**params); reg_search.fit(X_train_search, y_train_search); return -reg_search.score(X_val_search, y_val_search)
        
        progress_callback = lambda res: feedback.setProgress(int(100 * (len(res.x_iters) + 1) / n_iterations))
        search_results = gp_minimize(objective_function, param_space, n_calls=n_iterations, random_state=42, callback=progress_callback)
        if feedback.isCanceled(): return {}
        
        best_params = dict(zip([dim.name for dim in param_space], search_results.x))
        feedback.pushInfo(f"Search complete. Best parameters found: {best_params}")
        
        feedback.pushInfo("\nStep 3: Training final model...")
        final_model = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
        final_model.fit(X_train_full, y_train_full)
        
        feedback.pushInfo("\nStep 4: Evaluating and generating outputs...")
        y_pred = final_model.predict(X_test); rmse = np.sqrt(mean_squared_error(y_test, y_pred)); mae = mean_absolute_error(y_test, y_pred); r2 = r2_score(y_test, y_pred)
        feedback.pushInfo(f'Final Performance on internal test data: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}')

        os.makedirs(out_folder, exist_ok=True)
        report_path = os.path.join(out_folder, 'autopredict_rf_report.txt'); raster_out = os.path.join(out_folder, 'autopredict_rf_depth.tif')
        samples_csv = os.path.join(out_folder, 'autopredict_rf_samples.csv'); scatter_png = os.path.join(out_folder, 'autopredict_rf_scatter.png')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("--- SDB Auto-Predict: RandomForest - Final Report ---\n"); f.write(f"\nSearch iterations performed: {n_iterations}\n")
            f.write("\n--- Best Hyperparameters Found ---\n"); 
            for key, value in best_params.items(): f.write(f"{key}: {value}\n")
            f.write("\n--- Performance on Internal Test Set ---\n"); f.write(f"R-squared (R2): {r2:.4f}\n")
            f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"); f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
        
        # --- هذا هو الجزء المصحح ---
        results_df = pd.DataFrame({
            'x_orig': coords_test['x_orig'].values,
            'y_orig': coords_test['y_orig'].values,
            'depth_true': y_test.values,
            'depth_pred': y_pred
        })
        results_df.to_csv(samples_csv, index=False, float_format='%.4f')
        
        plt.figure(figsize=(6,6)); plt.scatter(y_test, y_pred, alpha=0.6)
        mmin, mmax = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        plt.plot([mmin, mmax], [mmin, mmax], 'r--'); plt.xlabel('Observed'); plt.ylabel('Predicted'); plt.title('Final RandomForest Performance'); plt.grid(True)
        plt.savefig(scatter_png, dpi=150); plt.close()
        
        feedback.pushInfo("Predicting full raster...")
        flatX = np.hstack([b.ravel()[:, np.newaxis] for b in bands]); valid_mask = ~np.isnan(flatX).any(axis=1)
        flat_preds = np.full(flatX.shape[0], np.nan, dtype='float32')
        if valid_mask.any(): 
            data_to_predict = flatX[valid_mask]
            data_to_predict_clipped = np.clip(data_to_predict, min_bounds, max_bounds)
            data_to_predict_scaled = scaler.transform(data_to_predict_clipped)
            flat_preds[valid_mask] = final_model.predict(data_to_predict_scaled)
        
        depth_raster = flat_preds.reshape(bands[0].shape)
        new_profile = profile.copy(); new_profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=np.nan)
        with rasterio.open(raster_out, 'w', **new_profile) as dst:
            dst.write(depth_raster, 1)
        
        return {}
