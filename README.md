# Test3

### Authors: Tina Jin, Virginia Weston

#### Data Cleaning and EDA
The first step in our data cleaning process was reading in the NewBioDeg csv with pandas and setting all of the headers to the feature name from a separate document. After this, we binarized our target class variable by setting "NRB" as 0 and "RB" as 1. We also noted that because there are almost twice as many 0s as 1s, it will be necessary to stratify the data when splitting train-test sets in our model. Our next step was to check the dataset for NaN values. We found that dropping any columns with missing data would result in only 787 examples out of an original 1055 examples. Moreover, we decided to impute the missing values instead of dropping all of the NaNs. To do so we replaced the discrete NaN features with mode (which all ended up being 0), and we imputed the continuous NaN features with their mean. After performing this data cleaning/imputation, we saved the cleaned data in a new csv file named "BioDegData". 

As for our EDA portion, we ran descriptive statistics on this dataset and found that the categroical data is right-skewed (mean>median). Therefore, normalization and standardization are needed for models like KNN; however, for our SVM we will continue to use MinMaxScaler and StandardScaler. Another way we explored this dataset was through running a correlation heatmap on the features. From this heatmap we need to ensure that we handle highly correlated features that may be redundant and negatively affect our model.


#### Feature Selection: SBS and Random Forest Selection

#### Feature Extraction: LDA and PCA

#### Model Building

#### Evaluation

#### Websites/Works used in code
https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
