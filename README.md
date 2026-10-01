# Revisiting travel demand using big data: an empirical comparison of explainable machine learning models

#### Songhua Hu, Chenfeng Xiong, Peng Chen, Paul Schonfeld

![F1](figures/framework.png "results")

Using the nationwide census block group (CBG)-level population inflow derived from Mobile device location data (MDLD) as the proxy of travel demand, 
this study examines its relations with various factors including socioeconomics, demographics, land use, and CBG attributes. 
A host of tree-based machine learning (ML) models and interpretation techniques (feature importance, partial dependence plot (PDP), accumulated local effect (ALE), SHapley Additive exPlanations (SHAP)) 
are extensively compared to determine the best model architecture and justify the interpretation robustness.

## Code structure
* Data used for model building is located at the folder `data`, which is computed via `1.0-Match_CBG_POI.py`, `1.1-Read_Data.py`, `1.2-Data_EDA.py`.
* `3.0-models-origin.py`, `3.1-models-transform.py` are used for model training and tuning. The first uses the original data while the second considers the data transformation.
* `4.1-Interpret models.py` is used to interpret the trained model.


## Results
#### Permutation importance of tree-based models (Shuffling vs. SHAP)
![F1](figures/importance.png "results")
#### Sensitivity analysis of impurity importance across different hyperparameters
![F1](figures/sensitivity.png "results")
#### PDPs of the 20 most important features in fine-tuned LightGBM
![F1](figures/PDP.png "results")
#### ALE plots of the 20 most important features in fine-tuned LightGBM
![F1](figures/ALE.png "results")
#### SHAP interaction plots of the 20 most important features in fine-tuned LightGBM
![F1](figures/SHAP.png "results")
