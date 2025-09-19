from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer, QgsProcessingParameterVectorLayer,
    QgsProcessingParameterFile, QgsProcessingParameterField,
    QgsProcessingParameterString, QgsProcessingParameterFolderDestination,
    QgsProject, QgsCoordinateTransform, QgsPointXY
)
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # لمنع مشاكل الواجهة الرسومية في QGIS
import matplotlib.pyplot as plt

# external libs
try:
    import rasterio
    from rasterio.transform import Affine
except Exception:
    rasterio = None

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class LinearRegressionAlgorithm(QgsProcessingAlgorithm):
    INPUT_RASTER = 'INPUT_RASTER'
    INPUT_SAMPLES_VEC = 'INPUT_SAMPLES_VEC'
    INPUT_SAMPLES_FILE = 'INPUT_SAMPLES_FILE'
    DEPTH_FIELD_VEC = 'DEPTH_FIELD_VEC'
    DEPTH_FIELD_FILE = 'DEPTH_FIELD_FILE'
    OUTPUT_FOLDER = 'OUTPUT_FOLDER'

    def initAlgorithm(self, config=None):
        # inputs
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_RASTER, 'Input raster (multiband)'))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_SAMPLES_VEC, 'Sample points (vector, optional - point layer / shapefile)', optional=True))
        self.addParameter(QgsProcessingParameterFile(self.INPUT_SAMPLES_FILE, 'Sample points file (CSV/XYZ, optional)', optional=True, fileFilter='CSV/XYZ files (*.csv *.xyz *.txt)'))
        self.addParameter(QgsProcessingParameterField(self.DEPTH_FIELD_VEC, 'Depth field (when using vector samples)', parentLayerParameterName=self.INPUT_SAMPLES_VEC, type=QgsProcessingParameterField.Numeric, optional=True))
        self.addParameter(QgsProcessingParameterString(self.DEPTH_FIELD_FILE, 'Depth column name (when using CSV/XYZ file)', optional=True))

        # output
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, 'Output Folder'))

    def name(self):
        return 'ml_linear'

    def displayName(self):
        return 'ML: Linear Regression'

    def group(self):
        return 'SDB Tools'

    def groupId(self):
        return 'sdb_tools'

    def shortHelpString(self):
        return 'Predict bathymetry using Linear Regression. Outputs: report, predicted depth raster, sample predictions, scatter plot.'

    def createInstance(self):
        return LinearRegressionAlgorithm()
        # ---------- helpers (مطابقة للكود المرجعي) ----------
    def _read_samples_from_vector(self, vlayer, depth_field, feedback):
        if vlayer is None:
            raise RuntimeError('Vector layer not provided.')
        if not depth_field:
            numeric = [f.name() for f in vlayer.fields() if f.typeName().lower() in ('double','real','integer','int','integer64','float')]
            if len(numeric) == 1:
                depth_field = numeric[0]
                feedback.pushInfo(f'Auto-detected depth field: {depth_field}')
            else:
                raise RuntimeError('Please specify depth field for vector layer (or ensure exactly one numeric field exists).')

        rows = []
        for feat in vlayer.getFeatures():
            geom = feat.geometry()
            if geom is None: continue
            pt = geom.centroid().asPoint() if geom.isMultipart() else geom.asPoint()
            try:
                d = float(feat[depth_field])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append((float(pt.x()), float(pt.y()), d))
        if not rows:
            raise RuntimeError('No valid sample points read from vector layer.')
        return pd.DataFrame(rows, columns=['x','y','depth'])

    def _read_samples_from_file(self, filepath, depth_col, feedback):
        if not filepath or not os.path.exists(filepath):
            raise RuntimeError('Sample file not provided or not found.')
        try:
            df = pd.read_csv(filepath, sep=None, engine='python')
        except Exception:
            df = pd.read_csv(filepath, sep=r'\s+', engine='python', header=None)
        
        if df.shape[1] < 3:
            raise RuntimeError('CSV/XYZ file must have at least 3 columns: x, y, depth.')

        if depth_col and depth_col in df.columns:
            depth_series = df[depth_col]
        else:
            numeric_cols = [c for c in df.columns[2:] if pd.api.types.is_numeric_dtype(df[c])]
            if not numeric_cols:
                raise RuntimeError('No numeric depth column found; please provide depth column name.')
            depth_col = numeric_cols[0]
            feedback.pushInfo(f'Auto-using depth column: {depth_col}')
            depth_series = df[depth_col]
            
        return pd.DataFrame({
            'x': df.iloc[:,0].astype(float),
            'y': df.iloc[:,1].astype(float),
            'depth': depth_series.astype(float)
        })

    def _transform_point_if_needed(self, x, y, sample_crs, raster_crs, feedback):
        if sample_crs == raster_crs:
            return x, y
        try:
            tr = QgsCoordinateTransform(sample_crs, raster_crs, QgsProject.instance())
            p = tr.transform(QgsPointXY(x, y))
            return float(p.x()), float(p.y())
        except Exception as e:
            feedback.pushWarning(f'CRS transform failed: {e}. Assuming coordinates match raster CRS.')
            return x, y

    # ---------- main ----------
    def processAlgorithm(self, parameters, context, feedback):
        raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        vlayer = self.parameterAsVectorLayer(parameters, self.INPUT_SAMPLES_VEC, context)
        sample_file = self.parameterAsFile(parameters, self.INPUT_SAMPLES_FILE, context)
        depth_field_vec = self.parameterAsString(parameters, self.DEPTH_FIELD_VEC, context)
        depth_field_file = self.parameterAsString(parameters, self.DEPTH_FIELD_FILE, context)
        out_folder = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)

        if not raster_layer: raise RuntimeError('Invalid input raster.')
        if vlayer is None and not sample_file: raise RuntimeError('Please provide sample points.')
        if rasterio is None: raise RuntimeError('rasterio is required for this tool.')

        if vlayer is not None:
            samples_df = self._read_samples_from_vector(vlayer, depth_field_vec, feedback)
            sample_crs = vlayer.crs()
        else:
            samples_df = self._read_samples_from_file(sample_file, depth_field_file, feedback)
            sample_crs = raster_layer.crs()
            feedback.pushInfo('CSV/XYZ used: ensure its coordinates are in same CRS as raster.')
        feedback.pushInfo(f'Read {len(samples_df)} sample points.')

        with rasterio.open(raster_layer.source()) as src:
            profile = src.profile
            transform = src.transform
            bands = [src.read(i + 1).astype('float32') for i in range(src.count)]
            raster_crs = raster_layer.crs()

        X, y, sample_rows = [], [], []
        for _, row in samples_df.iterrows():
            x0, y0, depth = row['x'], row['y'], row['depth']
            x2, y2 = self._transform_point_if_needed(x0, y0, sample_crs, raster_crs, feedback)
            col, row_idx = ~transform * (x2, y2)
            r_i, c_i = int(row_idx), int(col)
            if 0 <= r_i < bands[0].shape[0] and 0 <= c_i < bands[0].shape[1]:
                feat = [b[r_i, c_i] for b in bands]
                if any(np.isnan(feat)): continue
                X.append(feat)
                y.append(depth)
                sample_rows.append((x0, y0, depth))

        if len(X) < 5: raise RuntimeError(f'Not enough valid samples on raster (found {len(X)}).')
        X, y = np.array(X, dtype='float32'), np.array(y, dtype='float32')
        feedback.pushInfo(f'Extracted features for {X.shape[0]} samples; feature dim = {X.shape[1]}')

        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, np.arange(len(y)), test_size=0.3, random_state=42
        )

        model = LinearRegression()
        feedback.pushInfo('Training Linear Regression model...')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        feedback.pushInfo(f'Training finished. RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}')

        os.makedirs(out_folder, exist_ok=True)
        report_path = os.path.join(out_folder, 'linear_report.txt')
        raster_out = os.path.join(out_folder, 'linear_depth.tif')
        samples_csv = os.path.join(out_folder, 'linear_samples_pred.csv')
        scatter_png = os.path.join(out_folder, 'linear_scatter.png')

        with open(report_path, 'w') as f:
            f.write('Algorithm: Linear Regression\n')
            f.write(f'samples_used={len(y)} (train={len(y_train)}, test={len(y_test)})\n')
            f.write(f'RMSE={rmse}\n')
            f.write(f'MAE={mae}\n')
            f.write(f'R2={r2}\n')

        pd.DataFrame({
            'x_orig': [sample_rows[i][0] for i in idx_test],
            'y_orig': [sample_rows[i][1] for i in idx_test],
            'depth_true': y_test,
            'depth_pred': y_pred
        }).to_csv(samples_csv, index=False)

        try:
            plt.figure(figsize=(6,6))
            plt.scatter(y_test, y_pred, alpha=0.6)
            mmin = min(y_test.min(), y_pred.min())
            mmax = max(y_test.max(), y_pred.max())
            plt.plot([mmin, mmax], [mmin, mmax], 'r--')
            plt.xlabel('Observed depth'); plt.ylabel('Predicted depth')
            plt.title('Linear Regression Observed vs Predicted'); plt.grid(True)
            plt.savefig(scatter_png, dpi=150)
            plt.close()
        except Exception as e:
            feedback.pushWarning(f'Failed to create scatter plot: {e}')

        feedback.pushInfo('Predicting full raster...')
        rows, cols = bands[0].shape
        flatX = np.hstack([b.ravel()[:, np.newaxis] for b in bands])
        
        valid_mask = ~np.isnan(flatX).any(axis=1)
        flat_preds = np.full(flatX.shape[0], np.nan, dtype='float32')
        if valid_mask.any():
            flat_preds[valid_mask] = model.predict(flatX[valid_mask])
        
        depth_raster = flat_preds.reshape(rows, cols)

        new_profile = profile.copy()
        new_profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=np.nan)
        with rasterio.open(raster_out, 'w', **new_profile) as dst:
            dst.write(depth_raster.astype('float32'), 1)

        feedback.pushInfo(f'Linear Regression outputs saved to: {out_folder}')
        return {}