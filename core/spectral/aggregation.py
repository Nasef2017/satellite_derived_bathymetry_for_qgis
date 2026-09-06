import os
import warnings
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

# --------------------------------------------------------------------------
# SPATIOSPECTRAL AGGREGATION OPTIONS REGISTRY
# --------------------------------------------------------------------------
AGGREGATION_METHODS = [
    "Median",
    "Mean",
    "Max (Deepest)",
    "Min (Shallowest)",
    "Weighted Median (R2/RMSE)",
    "Weighted Mean (R2/RMSE)",
    "Select Best Scene (High R2 / Low RMSE)",
]


def compute_r2_rmse_weight(r2, rmse):
    """Classic weight based on R2 and RMSE."""
    return max(0.001, float(r2)) / (max(0.001, float(rmse)) + 0.001)


def spatiospectral_aggregate(input_rasters, output_path, method="Median", weights=None, feedback=None):
    """
    Aggregates multiple raster files pixel-by-pixel using a chunked windowed approach
    to prevent memory overflow. Automatically handles spatial alignment (reprojection/resampling)
    by using the first raster as the reference grid.

    Args:
        input_rasters (list of str): List of paths to input raster files.
        output_path (str): Path to save the aggregated raster.
        method (str): Aggregation method ("Median", "Mean", "Max (Deepest)", "Min (Shallowest)", "Weighted Median (R2/RMSE)", "Weighted Mean (R2/RMSE)").
        weights (list of float, optional): Weights corresponding to each input raster.
        feedback (QgsProcessingFeedback, optional): Feedback object for logging.
    """
    if not input_rasters:
        raise ValueError("No input rasters provided for aggregation.")

    ref_raster_path = input_rasters[0]
    
    if feedback:
        feedback.pushInfo(f"   [Aggregation] Reference raster for alignment: {os.path.basename(ref_raster_path)}")
        feedback.pushInfo(f"   [Aggregation] Method: {method}. Chunked processing active.")

    with rasterio.open(ref_raster_path) as ref_src:
        profile = ref_src.profile
        ref_transform = ref_src.transform
        ref_crs = ref_src.crs
        ref_width = ref_src.width
        ref_height = ref_src.height
        ref_nodata = ref_src.nodata if ref_src.nodata is not None else -9999.0

        # Update profile for output
        profile.update(
            dtype=rasterio.float32,
            count=1,
            nodata=ref_nodata,
            compress='deflate'
        )

        with rasterio.open(output_path, 'w', **profile) as dst:
            # Create a 1024x1024 block processing window
            block_size = 1024
            
            for row_start in range(0, ref_height, block_size):
                if feedback and feedback.isCanceled():
                    return False
                    
                row_end = min(row_start + block_size, ref_height)
                for col_start in range(0, ref_width, block_size):
                    if feedback and feedback.isCanceled():
                        return False
                        
                    col_end = min(col_start + block_size, ref_width)
                    window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
                    
                    # Accumulate data from all rasters for this specific window
                    block_stack = []
                    
                    for r_path in input_rasters:
                        with rasterio.open(r_path) as src:
                            # If exactly the same dimensions, transform, and crs, read directly
                            if (src.transform == ref_transform and 
                                src.crs == ref_crs and 
                                src.width == ref_width and 
                                src.height == ref_height):
                                data = src.read(1, window=window)
                                nodata_val = src.nodata
                            else:
                                # Warp on the fly to match the reference raster using WarpedVRT
                                with WarpedVRT(src, crs=ref_crs, transform=ref_transform, 
                                               width=ref_width, height=ref_height, 
                                               resampling=Resampling.bilinear) as vrt:
                                    data = vrt.read(1, window=window)
                                    nodata_val = vrt.nodata
                            
                            data = data.astype(np.float32)
                            if nodata_val is not None:
                                if np.isnan(nodata_val):
                                    data[np.isnan(data)] = np.nan
                                else:
                                    data[np.isclose(data, nodata_val, atol=1e-3) | (data == nodata_val)] = np.nan
                            data[data <= -9000.0] = np.nan
                                
                            block_stack.append(data)
                    
                    if not block_stack:
                        continue
                        
                    # Stack arrays
                    stacked_array = np.stack(block_stack, axis=0)
                    
                    # Apply aggregation
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        if method == "Median":
                            agg_data = np.nanmedian(stacked_array, axis=0)
                        elif method == "Mean":
                            agg_data = np.nanmean(stacked_array, axis=0)
                        elif method == "Max (Deepest)":
                            abs_stack = np.abs(stacked_array)
                            valid_mask = ~np.isnan(abs_stack)
                            # Fill NaNs with -inf to safely use argmax
                            abs_stack_filled = np.where(valid_mask, abs_stack, -np.inf)
                            idx = np.argmax(abs_stack_filled, axis=0)
                            agg_data = np.take_along_axis(stacked_array, np.expand_dims(idx, axis=0), axis=0).squeeze(axis=0)
                            # Restore NaNs where all pixels were NaN
                            all_nan = ~np.any(valid_mask, axis=0)
                            agg_data[all_nan] = np.nan
                        elif method == "Min (Shallowest)":
                            abs_stack = np.abs(stacked_array)
                            valid_mask = ~np.isnan(abs_stack)
                            # Fill NaNs with +inf to safely use argmin
                            abs_stack_filled = np.where(valid_mask, abs_stack, np.inf)
                            idx = np.argmin(abs_stack_filled, axis=0)
                            agg_data = np.take_along_axis(stacked_array, np.expand_dims(idx, axis=0), axis=0).squeeze(axis=0)
                            all_nan = ~np.any(valid_mask, axis=0)
                            agg_data[all_nan] = np.nan
                        elif method == "Weighted Mean (R2/RMSE)" and weights is not None and len(weights) == stacked_array.shape[0]:
                            w = np.array(weights, dtype=np.float32)
                            mask = ~np.isnan(stacked_array)
                            w_expanded = np.broadcast_to(w[:, None, None], stacked_array.shape)
                            w_masked = np.where(mask, w_expanded, 0.0)
                            weighted_sum = np.nansum(stacked_array * w_masked, axis=0)
                            sum_weights = np.sum(w_masked, axis=0)
                            agg_data = np.divide(weighted_sum, sum_weights, out=np.full_like(weighted_sum, np.nan), where=sum_weights > 0)
                        elif method == "Weighted Median (R2/RMSE)" and weights is not None and len(weights) == stacked_array.shape[0]:
                            w = np.array(weights, dtype=np.float32)
                            mask = ~np.isnan(stacked_array)
                            w_expanded = np.broadcast_to(w[:, None, None], stacked_array.shape)
                            w_masked = np.where(mask, w_expanded, 0.0)
                            
                            # Sort data along Z axis
                            sort_idx = np.argsort(stacked_array, axis=0)
                            sorted_data = np.take_along_axis(stacked_array, sort_idx, axis=0)
                            sorted_weights = np.take_along_axis(w_masked, sort_idx, axis=0)
                            
                            cum_weights = np.cumsum(sorted_weights, axis=0)
                            total_weights = cum_weights[-1, :, :]
                            target = total_weights / 2.0
                            
                            # argmax on boolean array returns first True
                            median_idx = np.argmax(cum_weights >= target[None, :, :], axis=0)
                            agg_data = np.take_along_axis(sorted_data, median_idx[None, :, :], axis=0).squeeze(axis=0)
                            
                            all_nan = ~np.any(mask, axis=0) | (total_weights <= 0)
                            agg_data[all_nan] = np.nan
                        else:
                            agg_data = np.nanmedian(stacked_array, axis=0)
                    
                    # Replace NaNs with nodata
                    agg_data[np.isnan(agg_data)] = ref_nodata
                    
                    dst.write(agg_data.astype(np.float32), 1, window=window)
                    
    return True

def spatiospectral_mask_intersection(mask_rasters, output_path, feedback=None):
    """
    Computes the intersection of multiple mask rasters (where pixels must be valid across all images).
    """
    import warnings
    if not mask_rasters:
        return False
        
    ref_raster_path = mask_rasters[0]
    
    with rasterio.open(ref_raster_path) as ref_src:
        profile = ref_src.profile
        ref_transform = ref_src.transform
        ref_crs = ref_src.crs
        ref_width = ref_src.width
        ref_height = ref_src.height
        
        # Ensure it's byte
        profile.update(dtype=rasterio.uint8, count=1, nodata=0, compress='deflate')
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            block_size = 1024
            for row_start in range(0, ref_height, block_size):
                if feedback and feedback.isCanceled(): return False
                row_end = min(row_start + block_size, ref_height)
                for col_start in range(0, ref_width, block_size):
                    if feedback and feedback.isCanceled(): return False
                    col_end = min(col_start + block_size, ref_width)
                    window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
                    
                    intersect_mask = None
                    
                    for r_path in mask_rasters:
                        with rasterio.open(r_path) as src:
                            if (src.transform == ref_transform and src.crs == ref_crs and 
                                src.width == ref_width and src.height == ref_height):
                                data = src.read(1, window=window)
                            else:
                                with WarpedVRT(src, crs=ref_crs, transform=ref_transform, 
                                               width=ref_width, height=ref_height, 
                                               resampling=Resampling.nearest) as vrt:
                                    data = vrt.read(1, window=window)
                            
                            # Assuming mask 1 is valid, 0 is nodata
                            valid_pixels = (data > 0)
                            if intersect_mask is None:
                                intersect_mask = valid_pixels
                            else:
                                intersect_mask = intersect_mask & valid_pixels
                                
                    out_mask = np.zeros((row_end - row_start, col_end - col_start), dtype=np.uint8)
                    if intersect_mask is not None:
                        out_mask[intersect_mask] = 1
                        
                    dst.write(out_mask, 1, window=window)
                    
    return True
