from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer, QgsProcessingParameterVectorLayer,
    QgsProcessingParameterFile, QgsProcessingParameterField,
    QgsProcessingParameterString, QgsProcessingParameterBand,
    QgsProcessingParameterFolderDestination,
    QgsProject, QgsCoordinateTransform, QgsPointXY
)
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import rasterio
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.linear_model import LinearRegression
    libs_installed = True
except ImportError:
    libs_installed = False

class BandRatioStumpfAlgorithm(QgsProcessingAlgorithm):
    INPUT_RASTER = 'INPUT_RASTER'; INPUT_SAMPLES_VEC = 'INPUT_SAMPLES_VEC'; INPUT_SAMPLES_FILE = 'INPUT_SAMPLES_FILE'
    DEPTH_FIELD_VEC = 'DEPTH_FIELD_VEC'; DEPTH_FIELD_FILE = 'DEPTH_FIELD_FILE'; OUTPUT_FOLDER = 'OUTPUT_FOLDER'
    BAND_HIGH_REF = 'BAND_HIGH_REF'
    BAND_LOW_REF = 'BAND_LOW_REF'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_RASTER, 'Input raster (multiband)'))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_SAMPLES_VEC, 'Sample points (vector, optional)', optional=True))
        self.addParameter(QgsProcessingParameterFile(self.INPUT_SAMPLES_FILE, 'Sample points file (CSV/XYZ, optional)', optional=True, fileFilter='*.csv;*.xyz;*.txt'))
        self.addParameter(QgsProcessingParameterField(self.DEPTH_FIELD_VEC, 'Depth field (vector)', parentLayerParameterName=self.INPUT_SAMPLES_VEC, type=QgsProcessingParameterField.Numeric, optional=True))
        self.addParameter(QgsProcessingParameterString(self.DEPTH_FIELD_FILE, 'Depth column name (CSV/XYZ)', optional=True))
        
        self.addParameter(QgsProcessingParameterBand(self.BAND_HIGH_REF, 'High Reflectance Band (e.g., Green)', parentLayerParameterName=self.INPUT_RASTER, defaultValue=3))
        self.addParameter(QgsProcessingParameterBand(self.BAND_LOW_REF, 'Low Reflectance Band (e.g., Blue)', parentLayerParameterName=self.INPUT_RASTER, defaultValue=2))

        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, 'Output Folder'))

    def name(self): 
        return 'bandratio'
        
    def displayName(self): 
        return 'ML: Band Ratio (Stumpf Log-Ratio)'
        
    def group(self): 
        return 'SDB Tools'
        
    def groupId(self): 
        return 'sdb_tools'
        
    def shortHelpString(self): 
        return 'Applies the Stumpf et al. (2003) log-ratio model, automatically finding best m0 and m1 for the selected bands.'
        
    def createInstance(self): 
        return BandRatioStumpfAlgorithm()
    
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
        if not libs_installed: raise RuntimeError('Required libraries (scikit-learn, rasterio) are not installed.')
        raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context); vlayer = self.parameterAsVectorLayer(parameters, self.INPUT_SAMPLES_VEC, context)
        sample_file = self.parameterAsFile(parameters, self.INPUT_SAMPLES_FILE, context); depth_field_vec = self.parameterAsString(parameters, self.DEPTH_FIELD_VEC, context)
        depth_field_file = self.parameterAsString(parameters, self.DEPTH_FIELD_FILE, context); out_folder = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        band_high_idx = self.parameterAsInt(parameters, self.BAND_HIGH_REF, context)
        band_low_idx = self.parameterAsInt(parameters, self.BAND_LOW_REF, context)

        if not raster_layer: raise RuntimeError('Invalid input raster.')
        if vlayer is None and not sample_file: raise RuntimeError('Please provide sample points.')
        if vlayer: samples_df = self._read_samples_from_vector(vlayer, depth_field_vec, feedback); sample_crs = vlayer.crs()
        else: samples_df = self._read_samples_from_file(sample_file, depth_field_file, feedback); sample_crs = raster_layer.crs()
        with rasterio.open(raster_layer.source()) as src:
            profile = src.profile; transform = src.transform; raster_crs = raster_layer.crs()
            b_high = src.read(band_high_idx).astype('float32')
            b_low = src.read(band_low_idx).astype('float32')
        
        eps = 1e-9
        with np.errstate(divide='ignore', invalid='ignore'):
            log_ratio = np.log(np.maximum(b_high, eps) / np.maximum(b_low, eps))
        
        X_vals, y_vals, sample_rows = [], [], []
        for _, row in samples_df.iterrows():
            x0, y0, d = row['x'], row['y'], row['depth']; x2, y2 = self._transform_point_if_needed(x0, y0, sample_crs, raster_crs, feedback)
            col, rowf = ~transform * (x2, y2); r_i, c_i = int(rowf), int(col)
            if 0 <= r_i < log_ratio.shape[0] and 0 <= c_i < log_ratio.shape[1]:
                val = log_ratio[r_i, c_i]
                if np.isfinite(val): X_vals.append([val]); y_vals.append(d); sample_rows.append((x0,y0,d))
        
        X, y = np.array(X_vals), np.array(y_vals)
        if len(y) < 10: raise RuntimeError("Not enough valid samples.")
        
        X_train, X_test, y_train, y_test, _, idx_test = train_test_split(X, y, np.arange(len(y)), test_size=0.3, random_state=42)
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test); rmse = np.sqrt(mean_squared_error(y_test, y_pred)); mae = mean_absolute_error(y_test, y_pred); r2 = r2_score(y_test, y_pred)

        m1, m0 = model.coef_[0], model.intercept_
        depth_raster = m1 * log_ratio + m0
        
        os.makedirs(out_folder, exist_ok=True)
        report_path = os.path.join(out_folder, 'bandratio_report.txt'); raster_out = os.path.join(out_folder, 'bandratio_depth.tif')
        samples_csv = os.path.join(out_folder, 'bandratio_samples_pred.csv'); scatter_png = os.path.join(out_folder, 'bandratio_scatter.png')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("--- Band Ratio (Stumpf) Report ---\n\n"); f.write(f"High Reflectance Band: {band_high_idx}, Low Reflectance Band: {band_low_idx}\n")
            f.write(f"Calculated m1 (slope): {m1:.4f}\nCalculated m0 (intercept): {m0:.4f}\n\n"); f.write("--- Performance on Test Set ---\n")
            f.write(f"R-squared (R2): {r2:.4f}\n"); f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"); f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
        
        test_coords = [sample_rows[i] for i in idx_test]
        results_df = pd.DataFrame({'x_orig': [c[0] for c in test_coords], 'y_orig': [c[1] for c in test_coords], 'depth_true': y_test, 'depth_pred': y_pred})
        results_df.to_csv(samples_csv, index=False, float_format='%.4f')
        
        plt.figure(figsize=(6,6)); plt.scatter(y_test, y_pred, alpha=0.6)
        mmin, mmax = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        plt.plot([mmin, mmax], [mmin, mmax], 'r--'); plt.xlabel('Observed'); plt.ylabel('Predicted')
        plt.title('Band Ratio Performance'); plt.grid(True); plt.savefig(scatter_png, dpi=150); plt.close()
        
        new_profile = profile.copy(); new_profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=np.nan)
        with rasterio.open(raster_out, 'w', **new_profile) as dst: dst.write(depth_raster.astype('float32'), 1)
        
        return {}
