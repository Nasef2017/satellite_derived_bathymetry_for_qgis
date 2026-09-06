<table>
<tr>
<td>

<img src="icon.png" width="100">

</td>
<td>

**Bathymetrix-AI v7.9: Advanced SDB Toolkit & Coastal Dynamics**

</td>
</tr>
</table>


> **Note:** Bathymetrix-AI v7.9 supports any satellite imagery provided that the imagery is atmospherically corrected (Surface Reflectance) and the raster values are stored as Float.

**Bathymetrix-AI (v7.9)** is a professional QGIS research toolkit for high-precision Satellite-Derived Bathymetry (SDB).

The toolkit integrates multispectral satellite imagery with ICESat-2 (ATL24) LiDAR bathymetry through a modular and adaptive Machine Learning framework. It is designed to automate the main SDB processing steps while maintaining control over data quality, model selection, spatial refinement, uncertainty, and scientific validation.

At the core of Bathymetrix-AI is the **SDB Single Masterflow** — the standard end-to-end workflow for producing a bathymetric map from a single satellite scene.

The toolkit also provides advanced Masterflows and standalone modules for more complex situations, including:
- **SDB SpatioSpectral Masterflow** — for evaluating and combining multiple satellite scenes of the same area.
- **SDB SpatioTemporal Masterflow** — for multi-year SDB modeling using time as part of the learning process.
- **Coastal Dynamics Analysis** — for analyzing bathymetric change, seabed stability, erosion, accretion, and shoreline dynamics.
- **ICESat-2 Downloader** — for acquiring and preparing ICESat-2 bathymetry data.
- **Tidal Datum Converter** — for converting bathymetric observations between tidal reference systems.

---

### 🌟 Key Capabilities & Innovations

* 🛡️ **Physics-Driven Feature Engineering (Zero-Bias Attenuation):**
  * Linearizes optical exponential attenuation according to the Beer-Lambert law ($L(\lambda) = L_\infty + C_b \cdot e^{-2K z}$) via Log-transformed spectral bands ($\ln(10000 \cdot R + 1)$), reducing depth systematic bias error to near-zero ($+0.0157\text{ m}$) and preventing volumetric distortion in sediment calculations.
* 🌊 **Advanced Multi-Method Optically Shallow Water (OSW) Engine:**
  * Features physics-based automated extinction detection (**Automated Knee-Point Extinction [Recommended]**, **Turbidity-Invariant Log-Ratio Extinction**, **Multi-Otsu / GMM 3D Spectral Clustering**, and **Connected-Component Topological Cleaning**) to strictly isolate shallow coastal zones from deep ocean and plume interference.
* 🎯 **Robust Model Selection & Winner Stability Simulation:**
  * Evaluates 15+ machine learning algorithms across **7 specialized selection strategies**, including **Monte Carlo Winner Stability** (testing candidate models across $N$ stochastic noise iterations) and auto-balanced multi-metric weighting.
* 🔄 **Decoupled Adaptive Spatial Refinement & IHO S-44 Assessment:**
  * Features independent, decoupled controls for **Depth Variance Correction (Datum Mean Shift)** and **Spatial Residual Error Modeling (Standard/Robust KNN, Gaussian Process / Kriging)**, generating uncertainty maps and evaluating compliance against **IHO S-44 Order 1a/2 Total Vertical Uncertainty (TVU)** standards with interactive HTML dashboards.
* 🏔️ **Interactive 3D Seabed Modeling & Scaled Metric Dimension Bars:**
  * Generates high-definition 3D Seabed Topography and Bathymetry models with dynamic deep-to-shallow camera alignment, balanced vertical exaggeration, baseline depth contours, and calibrated metric dimension scale bars (km / m axes) directly inside validation reports and dashboards.
* ⏱️ **Robust SpatioTemporal AI Modeling:**
  * Incorporates time/year directly into a Global Spatiotemporal AI model, with intelligent data validation that gracefully skips point-deficient years without workflow interruption while training and predicting across all valid temporal scenes.
* 🌐 **Automated Hydrodynamic Tidal Datum Engine:**
  * Integrated **NASA GSFC GOT4.10c** global ocean tide model for automated constituent caching and seamless vertical datum transformations ($WGS84 \leftrightarrow MSL \leftrightarrow LAT \leftrightarrow CD$).
* 🏖️ **Comprehensive Coastal Dynamics & Volumetric Tracking:**
  * Multi-year morphological analysis including Net Bathymetric Change, Morphological Stability Index (MSI), Statistical Level of Detection (StatCD), shoreline migration, and Target ROI volumetric sand tracking ($m^3$).

---

### 🔬 Scientific Methodology — Core 5-Phase Workflow

The **SDB Single Masterflow** provides the core processing architecture of Bathymetrix-AI.
It organizes the complete SDB process into five connected phases:

**Phase 01 → Phase 02 → Phase 03 → Phase 04 → Phase 05**

Each phase has a specific role, from preparing the satellite data and cleaning the training observations to building the AI model, refining the spatial prediction, and scientifically validating the final result.



#### Phase 01: Advanced Pre-processing
The first phase prepares the satellite imagery for bathymetric modeling by isolating the aquatic domain, reducing radiometric interference, and generating depth-sensitive spectral features.
- **Sun-Glint Removal:** Reduces surface reflection effects to improve the visibility of the seabed signal using the Hedley approach (Hedley et al., 2005). The implementation also handles infinite and NaN values to improve processing stability.
- **Water Segmentation:** Identifies the aquatic domain using NDWI, MNDWI, and NWI together with adaptive thresholding, supporting both automated masking and seamless Ready-made Water Mask polygon ingestion.
- **Deep Water OSW Filtering:** Removes deep-water areas that are unsuitable for optically derived bathymetry using the new multi-method engine (Automated Knee-Point Extinction, Turbidity-Invariant Log-Ratio Extinction, Multi-Otsu/GMM 3D Spectral Clustering, and Connected-Component Topological cleaning).
- **OSW Boundary Extraction:** Automatically generates the Optically Shallow Water boundary as a GeoPackage vector while preserving the CRS of the source imagery.
- **Log-Ratio Features & Indices:** Transforms spectral information into bathymetry-sensitive features based on light attenuation principles. This includes physics-based Log-Ratio features such as Blue/Green, together with additional spectral indices.

#### Phase 02: Robust Filtering
This phase focuses on improving the quality of the training dataset before machine-learning modeling.
The workflow identifies unreliable observations, outliers, and environmental noise in the depth measurements.
- **Noise Removal & Bypass:** Iteratively identifies high-confidence depth observations using Linear RANSAC, LS Variance Fit, or Huber Variance Fit. When disabled by the user, the input training dataset passes directly and untouched into Phase 03.
- **Dynamic Diagnostic Plotting:** Generates readable diagnostic plots using robust percentile-based visualization so that extreme noise does not dominate the displayed variance and trend patterns.

*The objective of Phase 02 is simple: Better training data → More reliable AI modeling.*

#### Phase 03: Global Auto-ML & Feature Analysis
This is the main machine-learning stage of the workflow.
Instead of relying on a single predefined algorithm, Bathymetrix-AI automatically evaluates multiple models and determines which approach is most suitable for the available dataset.
- **Feature Analysis:** Identifies weak, redundant, or highly correlated features using Pearson correlation, Spearman correlation, and automated selection approaches such as Automatic-RANSAC and Automatic-Random Forest.
- **Algorithm Benchmarking:** Evaluates more than 15 machine-learning algorithms, including models such as Random Forest, Gradient Boosting, XGBoost, CatBoost, SVR, MLP, and other supported regressors.
- **Model Selection & Winner Stability:** Evaluates models using 7 distinct selection strategies (Winner Stability via Monte Carlo Sensitivity Simulation, SDB Composite Score, Max R², Min RMSE, Min wMAPE, Min |Bias|, Min MAE) with auto-balanced metric weighting and custom syntax parsing.
- **Hyperparameter Optimization:** Automatically searches for suitable model parameters using Random Search, Grid Search, or Bayesian Optimization.
- **Spatial Cross-Validation:** Provides independent spatial block cross-validation to evaluate model performance while reducing the risk of overly optimistic results caused by spatial dependence.
- **Ensemble Blending:** Supports multiple prediction-combination strategies, including Standard Average, Median, Stacking, and **Uncertainty-Weighted Pixel Fusion**. The Uncertainty-Weighted approach gives greater influence to predictions with lower estimated residual uncertainty.
- **Memory-Efficient Prediction:** Large rasters are processed in chunks to reduce memory usage and avoid crashes during prediction.
- **Customization:** Advanced users can control model parameters and optimization settings for detailed experimentation and research.

*The output of Phase 03 is the initial global bathymetric prediction.*

#### Phase 04: Adaptive Refinement (Decoupled Architecture)
Global machine-learning models can perform well overall while still missing small-scale spatial patterns. Phase 04 is designed to address these remaining local errors with completely independent, decoupled controls.
- **Depth Variance Correction (Datum Mean Shift):** Calibrates and corrects global systematic datum offsets against adaptive control points.
- **Spatial Residual Modeling:** Analyzes local prediction errors and generates 2D spatial residual correction grids using Standard KNN, Robust KNN (Huber Weights), or Gaussian Process / Kriging.
- **Adaptive Re-training & Pixel Fusion:** Combines model information, Phase 03 depth maps, and spatial error grids to generate a refined bathymetric surface.
- **IHO Standards Assessment:** Evaluates results against applicable IHO Order 1a/2 Total Vertical Uncertainty (TVU) standards.

*The main purpose of Phase 04 is: Global prediction → Local error analysis → Spatial refinement → Improved final SDB.*

#### Phase 05: Validation & Reporting
The final phase evaluates the complete SDB result using independent validation information when available.
- **Independent Accuracy Assessment:** The final model can be tested against unseen validation points to provide an independent assessment of predictive performance. Typical metrics include **R²** (Coefficient of Determination), **RMSE** (Root Mean Square Error), and **wMAPE** (Weighted Mean Absolute Percentage Error). The workflow can also evaluate compliance with relevant IHO Order 1a/2 TVU criteria.
- **Interactive Validation Dashboard:** The system generates a structured HTML dashboard containing model leaderboards, validation metrics, diagnostic plots, 3D seabed visualizations with dimension bars, and summary information.

*The final objective is not only to generate a depth map, but also to provide a measurable and traceable assessment of its quality.*

---

### From Core Workflow to Advanced Analysis
The SDB Single Masterflow is the foundation of the Bathymetrix-AI processing architecture.
When the project becomes more complex, the toolkit provides additional workflows designed for specific objectives:
- **One satellite scene** → SDB Single Masterflow
- **Multiple scenes of the same area** → SDB SpatioSpectral Masterflow
- **Multiple years** → SDB SpatioTemporal Masterflow
- **Yearly SDB maps** → Coastal Dynamics Analysis

This structure allows Bathymetrix-AI to progress from single-scene bathymetric mapping to multi-scene optimization, and finally to multi-year bathymetric and coastal change analysis.

The detailed descriptions of these advanced Masterflows and standalone modules are provided in the sections below.

🚀 **Masterflows & Standalone Modules**

### 1️⃣ SDB Single Masterflow
What if you could turn a satellite image into a bathymetric map without manually choosing algorithms, tuning dozens of parameters, or building the workflow step by step? That is the idea behind the **SDB Single Masterflow**.

I designed the Single Masterflow as a complete end-to-end workflow for **Satellite-Derived Bathymetry (SDB)** from a single satellite scene. Instead of running separate tools one by one, the Masterflow organizes the complete process into a single workflow:
**Pre-processing → Data Filtering → AI Modeling → Adaptive Refinement → Scientific Validation**

<p align="center">
  <img src="SDB Single MasterFlow.drawio.png" width="80%">
</p>

#### What does it actually do?
- First, the satellite image is prepared for SDB. The workflow performs the required preprocessing, including atmospheric and water-related preparation, sun-glint correction, water masking, and spectral feature extraction.
- Then, the training data are examined and filtered to reduce the influence of unreliable or abnormal observations.
- After that comes the AI modeling stage. The workflow can evaluate multiple machine-learning algorithms and identify the model that performs best for the available dataset.
- But the process does not stop at producing a raw prediction. The resulting SDB is passed through an **Adaptive Refinement** stage, where spatial residual patterns can be analyzed and corrected to improve the final bathymetric surface.
- Finally, when independent validation data are available, the workflow evaluates the result using scientific performance metrics.

#### Why is this useful?
Traditional SDB processing can become a long chain of separate steps:
`Satellite image → preprocessing → masking → feature extraction → filtering → model training → prediction → correction → validation...`

For many users, the challenge is knowing **how they should be connected and how to keep the workflow consistent**. The Single Masterflow is designed to solve that problem. It provides a structured workflow where the processing stages are connected and executed as one complete SDB pipeline.

#### When should you use it?
The Single Masterflow is the right choice when you have:
- 📡 **One suitable satellite scene**
- 📍 **Training depth data**
- 🎯 **Optional independent validation data**

and your goal is to produce **one reliable SDB map for that scene**.

It is especially useful for standard SDB production, testing machine-learning models, processing a specific acquisition date, and establishing a consistent SDB workflow. Think of it as the **foundation of the Bathymetrix-AI Masterflows**.

---

### 2️⃣ SDB SpatioSpectral Masterflow
What if the problem is not your SDB model — but simply choosing the wrong satellite scene? A satellite image may look visually good, but that does not necessarily mean it will produce the best bathymetric result. Water clarity, sun-glint, waves, turbidity, atmospheric conditions, and other factors can strongly affect SDB performance.

That is why I developed the **SDB SpatioSpectral Masterflow**. Instead of asking the user: *“Which satellite image should I use?”*, the workflow asks: *“What can each available scene tell us about the bathymetry?”*

<p align="center">
  <img src="SDB SpatioSpectral Masterflow.drawio.png" width="80%">
</p>

#### How does it work?
The workflow starts with multiple satellite scenes covering the same area (e.g., Scene 1 + Scene 2 + Scene 3 + Scene 4). Each scene is processed **independently**.
- **Phase 1 (Pre-processing) & Phase 2 (Data Filtering)** are applied to each scene. The training data are evaluated for that specific scene, which is important because the same training point may behave differently from one image to another.
- **Phase 3 (AI Modeling)** models each scene independently using the available machine-learning algorithms. The result is a separate SDB map for each scene, together with its model performance and reliability information.

**Then comes the important part.** The system does **not simply merge all images together**. Instead, it compares the independent SDB results and allows different strategies to determine the most appropriate final representation:
- 🏆 **Select Best Scene:** Automatically choose the scene with the strongest model performance.
- 📊 **Weighted Mean:** Give stronger influence to scenes that produced more reliable results.
- 📐 **Weighted Median:** Use scene reliability while keeping the result more resistant to unusual or extreme predictions.
- 📈 **Median & Mean:** Use standard aggregations without assigning additional weights.

This means the workflow uses the **actual SDB performance of each scene** to help determine how the final bathymetry should be represented. Cloud percentage alone does not tell the whole story.

After aggregation, the resulting bathymetric map moves to **Phase 4 (Adaptive Refinement)** where remaining local spatial residual errors can be corrected. Finally, **Phase 5 (Scientific Validation)** evaluates the aggregated result and generates the same extensive **Interactive HTML Dashboards and detailed reports** as the Single Masterflow, ensuring strict **IHO S-44 Compliance**.

#### When should you use it?
Use it when:
- 📡 You have **multiple satellite scenes of the same area**
- ❓ You are **not sure which scene is best**
- 🤖 You want the system to **evaluate the scenes automatically**
- 🗺️ You want **one final bathymetric map** from multiple observations (for a single time period).

---

### 3️⃣ SDB SpatioTemporal Masterflow
What if an SDB map could do more than show the seabed — what if it could help explain how the seabed changes over time? While the Single Masterflow focuses on one satellite scene, and the SpatioSpectral Masterflow works with multiple scenes to produce one improved SDB map, the SpatioTemporal Masterflow is designed for a different question:

**How does bathymetry evolve through time?**

<p align="center">
  <img src="SDB SpatioTemporal Masterflow.drawio.png" width="80%">
</p>

#### How does it work?
Imagine having satellite imagery from 2019, 2021, 2023, and 2025. The goal is not simply to create four independent SDB maps. Instead, the workflow uses the information from all available years to build a **Global Spatiotemporal AI Model**.

For each year, the imagery is first processed independently through preprocessing and data filtering. Then the training information from the different years is combined into a global matrix containing: **Spectral Features + Depth + Year**.

The important difference is that **Year becomes a feature inside the AI model**. This allows the model to learn from the relationship between spectral information, depth, and time, rather than treating every year as a completely isolated problem. The global model can then generate consistent maps for each year (SDB 2019, 2021, 2023, 2025). 

After that, Adaptive Refinement is performed separately for each year to address local spatial errors that may differ from one year to another. The result is a **consistent series of bathymetric maps through time**.

#### Coastal Dynamics Analysis (Module 06)
The yearly SDB maps become the input for the **Coastal Dynamics Analysis** tool. Instead of looking at each map separately, the tool analyzes the changes between years. It investigates:
- **Long-term bathymetric trends & Net bathymetric change**
- **Morphological Stability Index (MSI)** (computes the volatility and stability of the seabed based on temporal depth variance).
- **Erosion and accretion & Shoreline movement**
- **Volumetric Sand Tracking & Target ROI Analytics:** Utilizes robust Linear Regression to track sediment volume (m³).

The system can also apply a dynamic noise threshold (Statistical Level of Detection - StatCD) so that very small apparent changes are not automatically interpreted as real seabed change.

#### Why is this important?
A single SDB map tells us: *"What is the seabed like?"*
Multiple SDB maps tell us: *"How has the seabed changed?"*

This creates a complete chain from satellite observations to temporal coastal interpretation:
**Multiple Years → Global Spatiotemporal AI → Yearly SDB Maps → Temporal Bathymetric Analysis → Morphological & Coastal Dynamics**

---

### 🛠️ Other Standalone Tools

**ICESat-2 Downloader**
A specialized standalone tool to query, filter, and download ICESat-2 (ATL24) LiDAR bathymetry data directly from NSIDC for integration into the MasterFlow pipeline.

**Tidal Datum Converter**
A dedicated standalone tool to seamlessly convert and correct bathymetric data across different tidal datums (e.g., Mean Sea Level to Chart Datum) integrating the NASA GSFC GOT4.10c automated global ocean tide model, ensuring accurate vertical alignment for precise charting and temporal comparisons.

📊 **Performance Metrics**

The tool evaluates results using three main standards:  
**R2** (Coefficient of Determination): Measures how well the model fits the data.  
**RMSE** (Root Mean Square Error): Measures the average vertical error in meters.  
**wMAPE** (Weighted Mean Absolute Percentage Error): Measures the relative error across different depth ranges.

🛠️ **Installation & Dependencies**  
Open **OSGeo4W Shell** (as Administrator) and run the following command to install all required libraries. This version is optimized for **QGIS 4.0 (Qt6)** and **NumPy 2.0** support:

```bash
pip install numpy pandas rasterio matplotlib seaborn scikit-learn>=1.5.0 scipy joblib scikit-optimize sliderule icepyx geopandas parquet netCDF4 xgboost lightgbm catboost optuna
```

📧 ***Contact & Citations***

**Author:** Mohamed Aly Nasef  
**Email:** Eng.m.nasef2017@gmail.com, Nasefm.aly@alexu.edu.eg  


🤖 **AI Acknowledgment**  
The development of the Bathymetrix-AI code, its logical structure, and the technical documentation were significantly enhanced and optimized using Google Gemini. The AI assisted in debugging complex workflows and ensuring the implementation follows best practices in data science.
