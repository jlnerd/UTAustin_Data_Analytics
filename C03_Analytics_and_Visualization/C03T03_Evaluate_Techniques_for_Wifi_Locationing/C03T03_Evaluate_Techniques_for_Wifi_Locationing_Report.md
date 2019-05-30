

# Technical Report: Evaluating Techniques for Wifi Location
__Author:__ John T. Leonard <br>
__Contact:__ jtleona01@gmail.com <br>
__Date:__ 2019.05.27 <br>

<img src="figures/Wifi-router-and-mobile.jpg" style="width: 300px;"/>

# Overview
In this technical report, we review an investigation in predicting Wifi location using signals from multiple wifi hotspots within a building. We describe the data set used for the analysis, a number of preprocessing and feature engineering steps taken, and the various prediction models evaluated. More specifically, in building the prediction models, we performed hyperparameter grid search routines, with parallel computing sessions, to attempt to find the best model for each different type of model evaluated. The model types evaluated include: (1) decision tree, (2) K-nearest neighbors (KNN), (3) support vector machine (SVM), (4) linear model. For each type of model, we developed a classification-based version, and a regression-based version, as some of the target labels were categorical, while other were continuous. Following training, we compared the performance metrics of each of the best models from the hyperparameter grid search and evaluated the prediction time for each model. Based on the model performance, including prediction time, we conclude the decision tree model is the most optimal for predicting Wifi Location.

# Notes on Reference Material
This investigation was performed using a Jupyter Notebook python 3 kernel. The complete notebook can be found at [C03T03_Evaluate_Techniques_for_Wifi_Locationing_Notebook](C03T03_Evaluate_Techniques_for_Wifi_Locationing_Notebook.ipynb). 

The dataset used for this study can be found at: [UJIndoorLoc](https://s3.amazonaws.com/gbstool/emails/2102/UJIndoorLoc.zip?AWSAccessKeyId=AKIAJBIZLMJQ2O6DKIAA&Expires=1558602000&Signature=%2BW7CPlk%2B61Ha6eF%2F9X%2BoG%2B1Mo%2Bk%3D)


# Objectives
Overall, the key objective of this study is to develop a general framework for dealing with large dataset of wifi location information. This primary objective can be broken into sub-objectives, including:
1. Develop visualization techniques to gain insight into the data set prior to developing machine-learning (ML) models
2. Define general rules to allow filtering of unnecessary feature/labels from the data set prior to training the ML models
3. Develop feature engineering techniques to improve the efficiency with which the ML model can find the relationship between the features and labels of interest
4. Compare the performance of various types of models & determine the optimal model for future deployment.
5. Perform hyperparameter grid search routines for each type of model to ensure the optimal set of parameters is chosen for a given type of model.

# Data Source & Structure
The data used for this study is contained in the 'trainingData.csv' file. This data set was used to compose the training set, validation set, and testing set throughout the study. We will refer to this data set as the 'DataFrame' (df). The df consists of 19937 rows and 529 columns. These columns can broken into two main fields: non-WAP columns and WAP columns. The meaning of WAP is unknown and is not really important in any case. The WAP columns simply refer to columns with the WAP prefix, following by some index. These columns are our many features of interest, containing the sensor data from various wifi hotspots for a given user. In total, the initial df contained 520 WAP columns and 9 non-WAP columns. The non-WAP columns and their respective data types are shown in the table below.
<img src="figures/df_non_WAP_headers_and_dtypes.png" style="width: 200px;"/>



As can be seen the table above, most of these non-WAP columns (headers) have integer data types. These correspond to categorical features/labels.

For training, we split the data set into features and labels. The preliminary set of features includes all the WAP columns (520 features) plus the 'USERID', 'PHONEID', and 'TIMESTAMP' (i.e. 523 features in total). These are all columns which we assume contain information which is available whenever one needs to make a prediction of their wifi coordinate/position. The labels we attempted to predict are: 'LATITUDE', 'LONGITUDE', 'RELATIVEPOSITION', 'FLOOR', 'BUILDINGID', 'SPACEID'. Of these labels, 'LATITUDE', 'LONGITUDE' are continuous values. Because the labels consisted of continuous and categorical data types, we trained both classification and regression based models for each different label type respectively (discussed in detail in a later section). 

# Data Visualization & Feature Engineering
Following the definition of the features and labels, we proceeded to visualize our data to gain more insights into it's structure and meaning, and also engineer our features/labels to eliminate any feature/labels that do not appear necessary for prediction.

## Visualizing Latitude & Longitude Maps
One of the most intuitive ways to understand this data set is to plot the latitude and longitude coordinates vs. some 3rd parameter/axis. To start, we build this 2D map vs. the number of counts for each position (i.e. a 2D histogram plot), shown below.

<img src="figures/2D_hist.png" style="width: 400px;"/>

The rough layout of the campus is shown in the 2D histogram (density) plot above. Based on the counts (shown on the color scale bar), we can see there is a large density of measurements/pings on the right side of the build or buildings in the top-left quadrant. Additionally, The building in the bottom right quadrant appears to have a large density of measurements, particularly in the wings of the building.

We can gain further understanding of the data set by expanding this 2D map to 3D to include the floor. The data set consists of measurements from 5 floors (floor 0 to 4). The 3D map is shown below.
<img src="figures/3D_floor_map.png" style="width: 400px;"/>

In the 3D scatter plot above, we can see the distribution in the measurements in more detail. Specifically, we can see that the building in the top left quadrant from our original 2D histogram shows a significant portion of the measurements on the right side of the building on each floor, however on the 3rd floor the density of the measurements appears more evenly distributed, while on the 4th floor it appears there are no measurements at all on the top half of the left side of the building. In contrast to the building in the top left quadrant, the building in the bottom right quadrant is much more uniformly distributed on all floors.

Finally, We visualize the 2D map vs. the other 3 labels we are trying to predict. These maps are shown below.
<img src="figures/2D_map_vs_relative_postion.png" style="width: 400px;"/>
<img src="figures/2D_map_vs_building_ID.png" style="width: 400px;"/>
<img src="figures/2D_map_vs_space_ID.png" style="width: 400px;"/>

The Building ID and relative position data above seems reasonably structured and understandable, however the Space ID data seems somewhat random. Because of this, and because we were given no information on what this label means (or even told what specific labels to predict), we drop space ID from the labels we are trying to predict, since it does not seem necessary to include it for location identification/tracking anyways.

## Visualizing Linear Correlation Coefficients

Now that we have a better grasp of the spatial distribution of the labels and their significance, we dive deeper into the features and their correlation with the labels we are trying to predict. Typically we would just generate multi-variate plots showing a 2D matrix of scatter plots of each feature column plotted vs. all the other columns. This would allows us to see nonlinear and linear trends between various features and labels. However, because there are hundreds of features, this approach is not really feasible. A simplified method for understanding the cross correlation between features and the correlation between features and labels is the just determine the linear correlation coefficient. This is done using a built in method for our pandas df of data. (df.corr()). The correlation matrix below plot the feature or label index on the x and y axis, while the correlation coefficient strength is represented by the color, with the scale bar showing the purple regions correspond to a highly negative correlation and the yellow regions corresponding to a highly positive correlation.
<img src="figures/corr_matrix_all_data.png" style="width: 300px;"/>

As can be seen, a number of columns have correlation coeffictions that are NaN, resulting in them being colored white. This is likely because there is a single unique value for that entire column, and the standard deviation is 0, which would result in a NaN correlation coeff.. Since these features clearly cannot teach us anything about correlations, we drop them from the data set, resulting in the number of features reducing from 523 to 468. The resulting correlation coeff. matrix is shown below

<img src="figures/corr_matrix_drop_0_stddev.png" style="width: 300px;"/>

In the correlation matrix above we can start to see some structure in the data. Next we slice this correlation matrix to just focus on the correlations between the features and the labels.

In the bar charts below we plot the correlation coefficients for a given label vs. all the feature (after the previously mentioned filtering). The bar plots on the left hand side show the unsorted correlations (as they are if you slice the data directly from the correlation matricies plotted above), while the plots on the right show the correlation coefficients after sorting. Note that the x-axis labels are not legible. This is because there are 468 features plotted, so it is not really reasonable to show the specific label for each feature.
<img src="figures/corr_pareto_before_filter.png" style="width: 800px;"/>


In the correlation factor bar charts above we can see many of the features have little to no correlation with the labels of interest. It is important to note that these correlation coefficients are linear correlation coefficients, however despite this fact, little to no linear correlation suggests the label probably is not important and will likely just introduce more noise to the model than actually help us predict something. That being said, by comparing the unsorted plots above (left side) to each other we can see that many of the features with weak correlation to one label have stronger correlation to another label. To ensure that all the present features have a reasonably strong correlation to at least one of the labels, we search through the correlation data and build a list of features with low correlation to all labels. In this analysis we chose to drop features with less than 10% (0.1) absolute value correlation coefficient to all the features. This resulted in 184 additional features being dropped, bringing our total number of relevant features to 272. The correlation matrix after these features are dropped is shown below.

<img src="figures/corr_matrix_after_low_corr_filter.png" style="width: 300px;"/>


## Visualizing Feature Distributions & Feature Engineering

To gain an understanding of how our relevant features are distributed, we plotted the histogram for each feature. a subset of the histograms is shown below for the first 5 features, which are visually representative of the general shape of the distributions observed for all features.

<img src="figures/feature_histograms.png" style="width: 800px;"/>

As can be seen, all the features have very extreme bimodal distributions. This is not necessarily an issue, however, it can make it hard for models generally to pick up on subtle/small changes in a feature and it's impact on a particular label. One method to mitigate this risk is by transforming the feature to a log10 scale. To do this effectively, we need to normalize the data, since many of the features have negative values. This normalization is done using a min-max scaler, which is a built-in function in the sklearn python library.

The histograms for the first 5 features, after min-max scaling and log10 transformation, are shown in the histograms below

<img src="figures/feature_histograms_after_log.png" style="width: 800px;"/>


In the histograms above, we can see that we are able to more clearly see the distribution by switching to a normalized log-scale axis. This transformation is a form of feature engineering and should give our model a better ability to differentiate and correlate the connections between the features and the labels.

The final feature engineering task is to drop the Time stamp data, since we can intuitiveley assume the time-stamp has no physical relevance to the predicted position

# Model Training

## Combine Categorical Labels, One Hot Encode, & Train-Test Split

To simplify our models, we combine our various categorical labels into a single label by concatenating them. Specifically, we concatenated relative position, floor ID, and building ID to form the column "combined_categorical_labels".

Once this new class is defined, we one-hot encode these classes into sparse vectors of 1s and 0s. In total our data set had 24 classes.

Following one-hot encoding, we perform a train-test split on the features, continuous labels, and categorical labels with a 33% test size split.

## Training

We evaluated 4 different types of models: (1) decision tree, (2) K-nearest neighbors (KNN), (3) support vector machine (SVM), (4) linear model. Each model type was trained to predict the categorical labels (i.e. trained as a classifier), as well as being trained, as a separate model, to predict the continuous labels (i.e. trained as a regressor). Since we have multiple continuous labels for a given data point, the regressors were trained as multi-output regressors.

To more fairly compare the performance of a given type of model, we performed a grid search across many of the hyperparameters defining a model. The dictionary defining the hyperparameters for each models grid search is shown below

__Grid Search Hyperparameters Dictionary:__
```
{'Decision Tree': {
    'classifier': {
        'max_depth': [None, 5, 10, 100],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 0.1, 0.2],
        'max_features': [None, 'auto', 'log2'],
        'max_leaf_nodes': [None, 5, 10, 100]},
    'regressor': {
       'max_depth': [None, 5, 10, 100],
       'min_samples_split': [2, 5, 10],
       'min_samples_leaf': [1, 0.1, 0.2],
       'max_features': [None, 'auto', 'log2'],
       'max_leaf_nodes': [None, 5, 10, 100]}},
'SVM': {
    'classifier': {
        'estimator__C': [1, 2, 5, 10],
        'estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'estimator__degree': [3],
        'estimator__tol': [0.1, 0.01, 0.001]},
    'regressor': {
        'estimator__C': [1, 2, 5, 10],
        'estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'estimator__degree': [3],
        'estimator__tol': [0.1, 0.01, 0.001]}},
'KNN': {
    'classifier': {
        'estimator__algorithm': ['ball_tree','kd_tree','brute'],
        'estimator__leaf_size': [30],
        'estimator__n_neighbors': [5],
        'estimator__p': [1, 2, 3, 5]},
    'regressor': {
        'estimator__algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'estimator__leaf_size': [30],
        'estimator__n_neighbors': [5],
        'estimator__p': [1, 2, 3, 5]}},
'linear': {
    'classifier': {
        'estimator__C': range(1, 2),
        'estimator__penalty': ['l1', 'l2'],
        'estimator__solver': ['liblinear']},
  'regressor': {}}}
 ```

To reduce the training time (and the model size for the case of KNN), we chose to sub-sample the training data by a factor of 4, resulting in 3339 rows of data being used for training.

For each model instance in the grid search, we performed 10-fold cross-validation and chose defined the best model as that which had the best cross-validation score, which for the regression models corresponded the to R2 score, while for the classificaiton models corresponded to the accuracy.

To improve the speed at which the grid search was performed for each model, we used multiprocessing, where our CPU allows 8 jobs to run in parallel. Overall for the decision tree, we evaluated 432 models with different hyperparameters for the classifier & regressor. The linear model only involved evaluated 2 different combinations of hyperparameters for the classifier, and 1 for the regressor. For the KNN, 12 candidates were evaluated for the classifier and regressor respectively. Finally, for the SVM, 48 combinations of hyperparameters were evaluated for both the classifier and regressor builds.

After finding the best model from each grid search, we evaluated the accuracy and kappa score for the classifier using the entire training and testing sets, and the mean-square-error (MSE) and R2 score for the regressor on the entire training and testing sets. For a metric describing the quality of the regressor and classifier combined, we simply added the R2 score and the accuracy and took the average. Additionally, we calculated the prediction time for each of these models.

The best estimator model results are shown in the table below

<img src="figures/metrics.png" style="width: 600px;"/>

Focusing on the performance metrics for the testing set, we can see the KNN model had the best average accuracy + R2 score, with a value of 0.9. The Decision tree classifier came in a close second, with a score of 0.89. We can see the specific metrics for the classification and regression models trained. As a reminder, the classifiers objective is to predict the 'RELATIVEPOSITION', 'FLOOR', and 'BUILDINGID' labels/categories, while the regressors objective is the predict the exact latitude and longitude coordinates. Focusing on the metrics for KNN and decision tree, we can see the KNN outperforms the decision tree for both the classification and regression model.

Aside from these prediction metrics, it is also important to note the prediction and training times, as some of the models make faster predictions and are more quickly trained than others. In the table above, we can see the training time in minutes (mins). This was the total training time for both the classifier and regressor models combined, as well as all the hyperparameter grid search operations. Here, we see the KNN took nearly 10 hrs to train and evaluate all the hyperparamter cases. Here, this training time is likely not completely accurate because we actually had to travel for ~8 hrs and pause the computation before the KNN finished, yet inspecting the run time for each of the hyperparameter search tasks for the KNN suggests the KNN models still took >1 hour of computation, making them the most computationally heavy models. For a specific model being trained, the training time could be reduced by using multiprocessing during the training task, which was not done here, as we were using multiprocessing for the hyperparameter grid-search operation.

Focusing on the prediction times in the table above, we can see the KNN model takes ~3mins to make a prediction, while the Decision tree takes less than 0.6 seconds.
    
Overall, since the Decision tree has performance metrics only slightly worse than the KNN, while the KNN takes ~3mins to make a prediction, it seems that the best model for deployment would be the decision tree, since it has the best trade-off between prediction performance and prediction time.

# Conclusions

Overall, the investigation described in this report demonstrated that one can predict a users location using signals from multiple wifi hotspots within a building. A number of preprocessing and feature engineering steps taken allowed us to significantly reduce the number of wifi hotpost signals we used for making predictions. In building our prediction models, we performed hyperparameter grid search routines, with parallel computing sessions, to attempt to find the best model for each different type of model evaluated ((1) decision tree, (2) K-nearest neighbors (KNN), (3) support vector machine (SVM), (4) linear model). For each type of model, we demonstrated that a classification-based version, and a regression-based version can be used to predict categorical information, such as build ID, as well as exact latitude and longitudinal location. We compared the performance metrics of each of the best models from the hyperparameter grid search and evaluated the prediction time for each model. The decision tree models were determined to be the most optimal, since they had performance metrics only slightly lower than that of the KNN, but their prediction times were orders of magnitude faster, which is quite important for this application. 

Though the current model demonstrations are promising, there is likely room for further improvement. One promising avenue that may be worth investigating is using a neural-network based approach to make predictions. This is considered a promising approach because neural networks have many more hyperparameters and computational structure possibilities than the more conventional models evaluated here, implying they may be able to discover more subtle relationships. Another very promising path would be to take a model ensemble approach, where the predictions of multiple different types of models are combined and voted-on to make a final prediction. This style of prediction has been shown to be very promising in many Kaggle competitions.


```python

```
