# Test3

### Authors: Tina Jin, Virginia Weston

#### Data Cleaning and EDA (See [this notebook](Cleaning_EDA.ipynb))
The first step in our data cleaning process was reading in the NewBioDeg csv with pandas and setting all of the headers to the feature name from a separate document. After this, we binarized our target class variable by setting "NRB" as 0 and "RB" as 1. We also noted that because there are almost twice as many 0s as 1s, it will be necessary to stratify the data when splitting train-test sets in our model. Our next step was to check the dataset for NaN values. We found that dropping any columns with missing data would result in only 787 examples out of an original 1055 examples. Therefore, we decided to impute the missing values instead of dropping all of the NaNs. To do so we replaced the discrete NaN features with mode (which all ended up being 0), and we imputed the continuous NaN features with their mean. After performing this data cleaning/imputation, we saved the cleaned data in a new csv file named "BioDegData". 

As for our EDA portion, we ran descriptive statistics on this dataset and found that the categroical data is right-skewed (mean>median). Therefore, normalization and standardization are needed. Another way we explored this dataset was through running a correlation heatmap on the features. From this heatmap we need to ensure that we handle highly correlated features that may be redundant and negatively affect our model.


#### Feature Selection: SBS and Random Forest Selection (See [this notebook](SBS.ipynb) and [this notebook](RF_Selection.ipynb))

To implement SBS on our features, we utilized the class code given to create an SBS class that constructs a KNN to fit, transform, and calculate each metric/score based on the KNN. The Sequential Backwards Selection essentially determined which features lead to the lowest performance loss and removed those features to improve the feature set. For this particuliar feature set, we set k_features equal to 1 so that the SBS class evaluates each feature individually. We also plotted accuracy versus number of features and found that the applied model performed best with between 25 to 35 features. The optimal model ended up returning the best 31 features in this feature set. These features include: ['J_Dz(e)', 'nHM', 'F01[N-N]', 'F04[C-N]', 'nCb-', 'C%', 'nCp', 'nO', 'F03[C-N]', 'SdssC', 'HyWi_B(m)', 'LOC', 'SM6_L', 'F03[C-O]', 'Me', 'Mi', 'nArNO2', 'nCRX3', 'SpPosA_B(p)', 'B01[C-Br]', 'Psi_i_1d', 'TI2_L', 'C-026', 'F02[C-N]', 'nHDon', 'SpMax_B(m)', 'Psi_i_A', 'nN', 'SM6_B(m)', 'nArCOOR', 'nX'].


The second portion of our feature selection process consisted of implementing a Random Forest selection to select the best features given a certain threshold. To do this, we imported the dataset with the header_list set equal to columns. We then applied a RandomForestRegressor to our data with an 80-20 train-test split. Printing out the features and their corresponding importance led us to choose a threshold of 0.03 to output the 12 most important features. The feature "SpMax_B(m)" has, by far, the highest importance of 0.27. With the RF Selection applied to our dataset, we find that the following 12 features are the most important and therefore will be most beneficial in our model: ['spMax_B(m)', 'SpMax_L', 'SpPosA_B(p)', 'Psi_i_A', 'Mi', 'F02[C-N]', 'SM6_B(m)', 'SdssC', 'nN', 'SpMax_A', 'SdO', 'J_Dz(e)'].


#### Feature Extraction: LDA and PCA (See [this notebook](LDA.ipynb))
For our LDA implementation, we first created a header_list to apply to the dataset as names so we could output the feature names if necesssary. Data must be labeled in order to run the LDA properly, so this first step was essential. Additionally, we set X to the feature columns and y to the target column. Finally, we split the data into 80-20 train-test sets. In the next step we applied a standard scaler to the split train-test data to later apply to the LDA. With the scaled data, we used scikit learn's LDA API call and set the n_components to 1 to perform LDA on this data. We then transformed the train and test data and applied it to a RFClassifier test to obtain an accuracy score for LDA. We found that the accuracy score for the LDA was decent, but did not perform as well for this dataset as PCA. Moreover, we stuck to implementing PCA as a tool for feature extraction in our SVM model. Another disadvantage when using LDA on a dataset like this is that it is biased towards a majority features and does not perform well with skewed data. The feature "SpMax_B(m)" has the highest importance by far, so it is possible that the LDA would overcompensate for that one feature. 

PCA is included in the main notebook building SVM.


#### Model Building (See [this notebook](SVM_Evaluation.ipynb))
We built 4 pipelines using dimensionality reduction methods RF selection, SBS, PCA and LDA. For each pipeline, we first split the data with 60-20-20 train-validation-test split, hoping to reduce the generalization error and adjust the hyperparameters using the validation data. Note that the data is stratified to account for the class imbalance, as discussed in the EDA portion. Before fitting the SVM, we applied StandardScaler and MinMaxScaler to the features. For the pipeline using RF selection and SBS, we simply used what they selected as the best subset of features. We included LDA and PCA in the pipeline for grid search, because we want to find the best values for 'solver' in LDA and 'n_components' in PCA. Then we ran a grid search to find the best hyperparameters for the SVM, namely, 'c value', 'kernel' and 'gamma' if the kernel is 'rbf'. After obtaining the best estimator through grid search, we ran a 10-fold cross validation on the validation data to see the performance of the computed set of hyperparameters. Finally, the best estimator was applied to the testing data that none of the models had seen before, to see if the results were generalizable.


#### Evaluation (Also in [this notebook](SVM_Evaluation.ipynb))
We used various metrics, including accuracy, precision, recall, F1 score, and confusion matrix to evaluate our models. Models using different dimensionality reduction methods had different performances. A summary of their performance is as follows.
![RF](/figs/RF_eval.png)
![SBS](/figs/SBS_eval.png)
![PCA](/figs/PCA_eval.png)
![LDA](/figs/LDA_eval.png)

Overall, we think that the pipeline using StandardScaler, MinMaxScaler, PCA and then SVM is the best choice. It has the best accuracy and few false positives and false negatives as seen in the confusion matrix. Moreover, the testing accuracy is higher than the cross validation result, meaning that the generalization error is low.

#### Websites/Works used in code
http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
https://stackabuse.com/implementing-lda-in-python-with-scikit-learn/
