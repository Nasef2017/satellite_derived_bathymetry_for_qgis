SDB Master Workflow: The Ultimate User Guide


1. Introduction: What is this Tool?
The SDB Master Workflow is a powerful and integrated tool for QGIS designed to fully automate the process of Satellite-Derived Bathymetry (SDB). Instead of manually testing different algorithms, this tool acts as an "automated expert," intelligently running a comprehensive suite of machine learning models, comparing their performance, selecting the best one for your specific data, and producing a final, high-quality depth map—all in a single click.
This workflow follows the best practices recommended by international bodies like the International Hydrographic Organization (IHO) and empowers you to derive accurate bathymetry from satellite imagery efficiently and reliably.


3. The Automated Workflow Explained
The tool follows a logical, four-stage pipeline:
Pre-processing and Smart Water Masking: The tool automatically calculates a water index (MNDWI or NDWI) and applies morphological cleaning (opening and closing) to create a high-quality, "clean," water-only version of your satellite image. All subsequent analyses are performed exclusively on this masked image.
Algorithm Comparison and Hyperparameter Tuning: For each selected algorithm, the tool performs a Smart Search (Bayesian Optimization) to discover the optimal settings (hyperparameters) for your specific data before training the final model.

Post-processing with Median Filter (Optional): If enabled, the tool applies a Median Filter to every raw depth map to remove "salt-and-pepper" noise, resulting in smoother and more realistic outputs.
Evaluation and the Decision Engine: Each model is rigorously tested against your Unseen Test Points. A Final Score is calculated (70% weight for R², 30% for RMSE) to objectively rank the models. The tool generates a master report and automatically loads the raster from the winning algorithm into QGIS.


5. Further Reading and Scientific References
For a deeper understanding, we highly recommend the following resources:
IHO Publication B-13: Cookbook for Satellite-Derived Bathymetry: The essential global reference standard for SDB. Search for it on the IHO website: https://iho.int/
Stumpf, R. P., et al. (2003). The original paper for the Band Ratio model. *Limnology and Oceanography*.
Caballero, I., & Stumpf, R. P. (2019). A great paper comparing machine learning (like RandomForest) to the Band Ratio model. *Remote Sensing*.




7. Acknowledgements and Tool Development
The development of the SDB Tools plugin was significantly accelerated by leveraging advanced AI language models.
Google's Gemini Pro: Utilized for its strong capabilities in code generation, logical structuring of complex workflows, and debugging.
OpenAI's ChatGPT (GPT-3.5/4): Employed for initial code scaffolding and assisting in troubleshooting.

This is the official GitHub of Mohamed Nasef, developer of the "Satellite-Derived Bathymetry for QGIS" plugin.  
Contact: eng.m.nasef2017@gmail.com or Nasef.M.Aly@alexu.edu.eg

