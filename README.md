# Test3

### Authors: Tina Jin, Virginia Weston

#### Data Cleaning and EDA (See [this notebook](Cleaning_EDA.ipynb))
The first step in our data cleaning process was reading in the NewBioDeg csv with pandas and setting all of the headers to the feature name from a separate document. After this, we binarized our target class variable by setting "NRB" as 0 and "RB" as 1. We also noted that because there are almost twice as many 0s as 1s, it will be necessary to stratify the data when splitting train-test sets in our model. Our next step was to check the dataset for NaN values. We found that dropping any columns with missing data would result in only 787 examples out of an original 1055 examples. Moreover, we decided to impute the missing values instead of dropping all of the NaNs. To do so we replaced the discrete NaN features with mode (which all ended up being 0), and we imputed the continuous NaN features with their mean. After performing this data cleaning/imputation, we saved the cleaned data in a new csv file named "BioDegData". 

As for our EDA portion, we ran descriptive statistics on this dataset and found that the categroical data is right-skewed (mean>median). Therefore, normalization and standardization are needed for models like KNN; however, for our SVM we will continue to use MinMaxScaler and StandardScaler. Another way we explored this dataset was through running a correlation heatmap on the features. From this heatmap we need to ensure that we handle highly correlated features that may be redundant and negatively affect our model.


#### Feature Selection: SBS and Random Forest Selection (See [this notebook](SBS.ipynb) and [this notebook](RF_Selection.ipynb))




The second portion of our feature selection process consisted of implementing a Random Forest selection to select the best features given a certain threshold. To do this, we imported the dataset with the header_list set equal to columns. We then applied a RandomForestRegressor to our data with an 80-20 train-test split. Printing out the features and their corresponding importance led us to choose a threshold of 0.03 to output the 12 most important features. The feature "SpMax_B(m)" has, by far, the highest importance of 0.27. With the RF Selection applied to our dataset, we find that the following 12 features are the most important and therefore will be most beneficial in our model: ['spMax_B(m)', 'SpMax_L', 'SpPosA_B(p)', 'Psi_i_A', 'Mi', 'F02[C-N]', 'SM6_B(m)', 'SdssC', 'nN', 'SpMax_A', 'SdO', 'J_Dz(e)']



#### Feature Extraction: LDA (See [this notebook](LDA.ipynb))
For our LDA implementation, we first created a header_list to apply to the dataset as names so we could output the feature names if necesssary. Data must be labeled in order to run the LDA properly, so this first step was essential. Additionally, we set X to the feature columns and y to the target column. Finally, we split the data into 80-20 train-test sets. In the next step we applied a standard scaler to the split train-test data to later apply to the LDA. With the scaled data, we used scikit learn's LDA API call and set the n_components to 1 to perform LDA on this data. We then transformed the train and test data and applied it to a RFClassifier test to obtain an accuracy score for LDA. We found that the accuracy score for the LDA was decent, but did not perform as well for this dataset as PCA. Moreover, we stuck to implementing PCA as a tool for feature extraction in our SVM model. Another disadvantage when using LDA on a dataset like this is that it is biased towards a majority features and does not perform well with skewed data. The feature "SpMax_B(m)" has the highest importance by far, so it is possible that the LDA would overcompensate for that one feature. 


#### Model Building (See [this notebook](SVM_Evaluation.ipynb))



#### Evaluation (Also in [this notebook](SVM_Evaluation.ipynb))

#### Websites/Works used in code
http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
