#!/usr/bin/env python
# coding: utf-8

# #                                Capstone Project 1- Heart Failure Prediction

# General introduction and defining the problem- 
# 
# Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease.
# People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.
# 

# # Importing the essential libraries

# In[1]:


import numpy as np 
import pandas as pd 
pd.set_option("display.max_rows",None) 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.neighbors import KNeighborsClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, linear_model, metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import warnings
warnings.simplefilter('ignore')


# In[2]:


# Read the CSV file into a DataFrame
df = pd.read_csv('heart_failure_clinical_records1.csv')
df.head(10)


# # Dataset Description

# In[3]:


# Attribute Description:
# Age: Age of the patient in years
# Sex: Gender of the patient (M: Male, F: Female)
# ChestPain: Type of chest pain (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic)
# RestingBP: Resting blood pressure in mm Hg
# Cholesterol: Serum cholesterol level in mg/dL
# FastingBS: Fasting blood sugar level in mg/dL
# RestingECG: Result of resting electrocardiogram (Normal, ST: ST-T wave abnormality, LVH: Left ventricular hypertrophy)
# MaxHR: Maximum heart rate achieved
# ExerciseAngina: Whether the patient experienced exercise-induced angina (Y: Yes, N: No)
# Oldpeak: ST depression induced by exercise relative to rest
# ST_Slope: Slope of the peak exercise ST segment (Up: Upsloping, Flat: Flat, Down: Downsloping)
# HeartDisease: Output class indicating the presence or absence of heart disease (1: heart disease, 0: normal)


# In[4]:


# info of the dataframe
df.info()


# We can see that in DataFrame following columns :
# 
# Age: Integer values representing the age of patients.
# 
# Sex: Categorical values indicating the gender of the patients.
# 
# ChestPainType: Categorical values representing the type of chest pain.
# 
# RestingBP: Integer values representing resting blood pressure.
# 
# Cholesterol: Integer values representing serum cholesterol level.
# 
# FastingBS: Integer values indicating fasting blood sugar level.
# 
# RestingECG: Categorical values representing results of resting electrocardiogram.
# 
# MaxHR: Integer values representing maximum heart rate achieved.
# 
# ExerciseAngina: Categorical values indicating whether the patient experienced exercise-induced angina.
# 
# Oldpeak: Float values representing ST depression induced by exercise relative to rest.
# 
# ST_Slope: Categorical values representing the slope of the peak exercise ST segment.
# 
# HeartDisease: Integer values indicating the presence (1) or absence (0) of heart disease.

# In[5]:


df.shape


# In[6]:


df.size


# In[7]:


# Definig a function for the description of dataset
def dataset_description(df):
    # Number of columns in the DataFrame
    num_columns = len(df.columns)
    print('No. of Columns in DataFrame:', num_columns)
    
    # Column attribute names
    print('\nColumn Attribute Names:', df.columns.tolist())
    
    # Numerical columns
    numerical_columns = df.describe().columns.tolist()
    print('\nNo. of Numerical Columns in DataFrame:', len(numerical_columns))
    print('\nNumerical Column Names:', numerical_columns)
    
    # Categorical columns
    categorical_columns = [col for col in df.columns if col not in numerical_columns]
    print('\nNo. of Categorical Columns in DataFrame:', len(categorical_columns))
    print('\nCategorical Column Names:', categorical_columns)

# Calling the function with DataFrame
dataset_description(df)


# In[8]:


#checking the null values present in the DataFrame
df.isnull().sum() 


# It appears that there are no null values present in any of the columns of DataFrame.

# In[9]:


#unique values found in the DataFrame of each column
df.nunique() 


# Here's a summary of the unique values for each column:
# 
# Age: 50 unique values
# 
# Sex: 2 unique values
# 
# ChestPainType: 4 unique values
# 
# RestingBP: 67 unique values
# 
# Cholesterol: 222 unique values
# 
# FastingBS: 2 unique values
# 
# RestingECG: 3 unique values
# 
# MaxHR: 119 unique values
# 
# ExerciseAngina: 2 unique values
# 
# Oldpeak: 53 unique values
# 
# ST_Slope: 3 unique values
# 
# HeartDisease: 2 unique values

# In[10]:


df.describe().T


# The output is the summary of statistics for the numerical columns of DataFrame:
# 
# Count: Number of non-null values in each column.
# 
# Mean: Average value of the column.
# 
# Std: Standard deviation, which measures the dispersion or spread of values around the mean.
# 
# Min: Minimum value observed in the column.
# 25%: 25th percentile, also known as the first quartile (Q1), which represents the value below which 25% of the observations fall.
# 
# 50%: 50th percentile, also known as the median or second quartile (Q2), which represents the middle value of the column.
# 
# 75%: 75th percentile, also known as the third quartile (Q3), which represents the value below which 75% of the observations fall.
# 
# Max: Maximum value observed in the column.

# In[11]:


df.corr()


# In[12]:


# Plotting  correlation heatmap
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate correlation matrix
correlation_matrix = df.corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


# The output represents the correlation between each pair of columns in DataFrame. Each cell contains the correlation coefficient between the corresponding pair of columns. Values close to 1 or -1 indicate strong positive or negative correlations, respectively, while values close to 0 indicate weak or no correlation.

# In[13]:


df.skew()


# The output you received represents the skewness for each numerical column in your DataFrame. Here's a breakdown of the skewness values:
# 
# Age: Skewness of approximately -0.196, indicating a slightly left-skewed distribution.
# 
# RestingBP: Skewness of approximately 0.180, indicating a slightly right-skewed distribution.
# 
# Cholesterol: Skewness of approximately -0.610, indicating a moderately left-skewed distribution.
# 
# FastingBS: Skewness of approximately 1.264, indicating a moderately right-skewed distribution.
# 
# MaxHR: Skewness of approximately -0.144, indicating a slightly left-skewed distribution.
# 
# Oldpeak: Skewness of approximately 1.023, indicating a moderately right-skewed distribution.
# 
# HeartDisease: Skewness of approximately -0.215, indicating a slightly left-skewed distribution.

# # Data Visualization

# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt

# Define the features to plot
features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']

# Set up the figure and subplots
plt.figure(figsize=(18, 18))

for i, feature in enumerate(features, start=1):
    plt.subplot(3, 2, i)
    plt.style.use('seaborn')
    plt.tight_layout()
    sns.set_context('talk')
    sns.histplot(data=df, x=feature, hue='HeartDisease', multiple='stack', palette='magma')
    plt.title(f'{feature} vs HeartDisease')

plt.show()


# This histograms compare different health metrics with the presence or absence of heart disease. Each plot represents a different variable and how it correlates with instances of heart disease (denoted as '1' for presence and '0' for absence). Let's break down each plot:
# 
# 1. **Age vs Heart Disease**: This plot shows that younger individuals (approximately age 30 to 50) more commonly do not have heart disease (purple bars). The prevalence of heart disease (red bars) increases with age, particularly noticeable from age 50 onwards.
# 
# 2. **Resting Blood Pressure (RestingBP) vs Heart Disease**: Higher resting blood pressure values appear slightly more frequently associated with heart disease. The distribution is fairly similar across different blood pressure levels, but there is a slight increase in heart disease cases as blood pressure increases, especially beyond 120 mmHg.
# 
# 3. **Cholesterol vs Heart Disease**: Cholesterol levels show a broad distribution with respect to heart disease. Both groups (with and without heart disease) have a wide range of cholesterol levels, but there's a noticeable cluster around 200-300 mg/dL where the presence of heart disease is more prevalent.
# 
# 4. **Fasting Blood Sugar (FastingBS) vs Heart Disease**: This plot shows that most individuals, regardless of their fasting blood sugar being below or above 0.8 (likely denoting 120 mg/dL as a threshold), do not have heart disease. However, those with higher fasting blood sugar (>0.8) show a notable increase in the incidence of heart disease.
# 
# 5. **Maximum Heart Rate (MaxHR) vs Heart Disease**: Individuals with higher maximum heart rates generally do not have heart disease, as indicated by the higher counts of non-heart disease cases across higher heart rate values. Lower maximum heart rates seem to have a more pronounced association with heart disease.
# 
# 6. **Oldpeak vs Heart Disease**: Oldpeak, likely a measure related to the depression of the ST segment in an ECG, shows that lower values (near 0) are common among individuals both with and without heart disease. However, higher values (especially around 2) are predominantly associated with heart disease.
# 
# Overall, these plots provide a visual correlation between certain physiological and health metrics with the presence of heart disease. Each metric offers some insight into how it might influence or relate to the risk of developing heart disease.

# In[15]:


#g = sns.FacetGrid(tips, col="time")
#g.map(sns.histplot, "tip")
sns.set()
df.hist(figsize=(10,10))
plt.show()


# These histograms representing the distribution of different variables related to heart health. Here is an interpretation of each histogram:
# 
# Age:
# 
# The distribution of age is approximately normal, with most individuals falling between 40 and 60 years old. The peak is around 55 years.
# 
# RestingBP (Resting Blood Pressure):
# 
# The resting blood pressure values are mostly concentrated around 120 mmHg, with a few outliers on both lower and higher ends.
# 
# Cholesterol:
# 
# Cholesterol levels show a wide range, with a peak around 200 mg/dL. There are several individuals with cholesterol levels above 300 mg/dL.
# 
# FastingBS (Fasting Blood Sugar):
# 
# The majority of individuals have a fasting blood sugar level of 0 (presumably indicating normal levels), while a smaller group has a level of 1 (indicating high fasting blood sugar).
# 
# MaxHR (Maximum Heart Rate Achieved):
# 
# The maximum heart rate achieved is normally distributed, with most values between 120 and 170 bpm. The peak is around 140 bpm.
# 
# Oldpeak:
# 
# Oldpeak values, which measure ST depression induced by exercise relative to rest, are mostly concentrated around 0, with a few higher values up to 5.
# 
# HeartDisease:
# 
# The distribution of heart disease presence is binary, with a roughly equal number of individuals with (1) and without (0) heart disease.
# 
# These histograms provide a visual summary of the data, showing the central tendency, spread, and distribution shape for each variable.

# In[16]:


sns.pairplot(df,hue='HeartDisease')


# The pairplot provides a comprehensive visualization of the relationships between multiple variables in the dataset, with the data points colored by the presence (1) or absence (0) of heart disease. Here is an interpretation of the pairplot:
# 
# Age:
# 
# The distribution of age shows that individuals with heart disease (orange) tend to be older compared to those without heart disease (blue).
# There is no clear linear relationship between age and other variables, but older individuals seem to have a higher incidence of heart disease.
# 
# RestingBP (Resting Blood Pressure):
# 
# Resting blood pressure does not show a strong correlation with other variables.
# Both groups (with and without heart disease) have a similar distribution of resting blood pressure.
# 
# Cholesterol:
# 
# Cholesterol levels are widely spread for both groups, with no clear distinction between those with and without heart disease.
# There is no strong correlation between cholesterol and other variables.
# 
# FastingBS (Fasting Blood Sugar):
# 
# Fasting blood sugar is binary (0 or 1). Individuals with a fasting blood sugar level of 1 (high) are more likely to have heart disease.
# 
# There is no clear relationship between fasting blood sugar and other continuous variables.
# 
# MaxHR (Maximum Heart Rate Achieved):
# 
# Individuals with heart disease tend to have lower maximum heart rates compared to those without heart disease.
# There is a noticeable separation in the distribution of maximum heart rate between the two groups.
# 
# Oldpeak:
# 
# Oldpeak values (ST depression) are higher in individuals with heart disease.
# There is a clear distinction in the distribution of oldpeak values between the two groups.
# 
# HeartDisease:
# 
# The pairplot shows the distribution of heart disease presence (1) and absence (0) across all variables.
# The color coding helps to identify patterns and relationships between heart disease and other variables.
# 
# Scatter Plots:
# 
# The scatter plots show the pairwise relationships between variables. For example:
# 
# Age vs. MaxHR: Older individuals with heart disease tend to have lower maximum heart rates.
# 
# Cholesterol vs. Age: There is no clear pattern, indicating that cholesterol levels are not strongly related to age.
# 
# Oldpeak vs. MaxHR: Higher oldpeak values are associated with lower maximum heart rates, especially in individuals with heart 
# disease.
# 
# Density Plots:
# 
# The diagonal plots show the distribution of each variable, with density plots for continuous variables.
# These plots highlight the differences in distributions between individuals with and without heart disease.
# 
# Overall, the pairplot reveals some key patterns:
# 
# Older age, lower maximum heart rate, higher oldpeak values, and higher fasting blood sugar levels are associated with a higher likelihood of heart disease.
# 
# Resting blood pressure and cholesterol do not show strong relationships with heart disease in this visualization.
# 
# 
# 

# In[17]:


df1= df.groupby('Sex').agg({'Age' : 'mean', "ChestPainType":'count','RestingBP':'mean','Cholesterol':'mean',
                            'FastingBS':'sum','RestingECG':'count','MaxHR':'mean','ExerciseAngina':'count','Oldpeak':'mean',
                            'ST_Slope':'count','HeartDisease':'sum'})
df1


# Age: Mean is appropriate to understand the average age.
# 
# ChestPainType: Count is appropriate to understand the frequency of each type of chest pain.
# 
# RestingBP: Mean is appropriate to understand the average resting blood pressure.
# 
# Cholesterol: Mean is appropriate to understand the average cholesterol level.
# 
# FastingBS: Sum is appropriate if you want to count how many people have a fasting blood sugar > 120 mg/dl.
# 
# RestingECG: Count is appropriate to understand the frequency of each type of ECG result.
# 
# MaxHR: Mean is appropriate to understand the average maximum heart rate.
# 
# ExerciseAngina: Count is appropriate to understand the frequency of exercise-induced angina.
# 
# Oldpeak: Mean is appropriate to understand the average ST depression induced by exercise.
# 
# ST_Slope: Count is appropriate to understand the frequency of each type of ST slope.
# 
# HeartDisease: Sum is appropriate if you want to count how many people have heart disease.
# 

# #### Using the same aggregation function for all variables would not be meaningful because the nature of these variables is different. The average age is same for both male and female.

# In[18]:


import plotly.express as px
px.bar(data_frame=df1, barmode='group', title = "Gender wise Analyzing",template="plotly_dark")


# # Data Preprocessing

# In[19]:


def outliers_graph(df_column):
    Q75, Q25 = np.percentile(df_column, [75 ,25]) 
    IQR = Q75 - Q25
    print('Q25: ',Q25)
    print('Q75: ',Q75)
    print('Inter Quartile Range: ',IQR)
    print('Outliers lie before', Q25-1.5*IQR, 'and beyond', Q75+1.5*IQR)
    print('Number of Rows with Left Extreme Outliers:', len(df[df_column <Q25-1.5*IQR]))
    print('Number of Rows with Right Extreme Outliers:', len(df[df_column>Q75+1.5*IQR]))


# In[20]:


outliers_graph(df['Age'])


# In[21]:


outliers_graph(df['RestingBP'])


# In[22]:


df=df[df.RestingBP>=90]
len(df)


# In[23]:


outliers_graph(df['Cholesterol'])


# In[24]:


df=df[df.Cholesterol<=450]  
len(df)


# In[25]:


outliers_graph(df['FastingBS'])


# In[26]:


outliers_graph(df['MaxHR'])


# In[27]:


df=df[df.MaxHR>=70]
len(df)


# In[28]:


outliers_graph(df['Oldpeak'])


# In[29]:


print('Mean: ',df['Cholesterol'].mean())
print('Median: ',df['Cholesterol'].median())


# In[30]:


mc=df[df['Cholesterol']>0].Cholesterol.mean() #mean value of Cholesterol without including the cholesterol=0
print('Mean of Cholesterol>0: ',mc)


# In[31]:


df.describe().T


# ## One Hot Encoding 

# In[32]:


def OneHotEncoding(dfcolumn):
  global df
  dfcolumn.nunique()
  len(df.columns)
  finallencol = (dfcolumn.nunique() - 1) + (len(df.columns)-1)
  dummies = pd.get_dummies(dfcolumn, drop_first=True, prefix=dfcolumn.name)
  df=pd.concat([df,dummies],axis='columns')
  df.drop(columns=dfcolumn.name,axis=1,inplace=True) 
  if(finallencol==len(df.columns)):
    print('OneHotEncoding is sucessfull') 
    print('')
  else:
    print('Unsucessfull')
  return df.head(5)


# In[33]:


OneHotEncoding(df['ChestPainType'])
OneHotEncoding(df['Sex'])
OneHotEncoding(df['RestingECG'])
OneHotEncoding(df['ExerciseAngina'])
OneHotEncoding(df['ST_Slope'])


# In[34]:


df.describe().columns.to_list()


# # Imputation

# In[35]:


#Zero value of cholesterol are filled usnig KNNImputer
from sklearn.impute import KNNImputer
df['Cholesterol'].replace(to_replace = 0, value =np.nan, inplace=True)
KNN_imputed = KNNImputer(n_neighbors=5)
I=KNN_imputed.fit_transform(df)
Cholesterol=[]
for i in range(0,len(df)):
  Cholesterol.append(I[i][2])
df['Cholesterol']=Cholesterol


# # Feature Scaling

# In[36]:


# altering the DataFrame
df = df[['Age',
 'RestingBP',
 'Cholesterol',
 'FastingBS',
 'MaxHR',
 'Oldpeak',
 'ChestPainType_ATA',
 'ChestPainType_NAP',
 'ChestPainType_TA',
 'Sex_M',
 'RestingECG_Normal',
 'RestingECG_ST',
 'ExerciseAngina_Y',
 'ST_Slope_Flat',
 'ST_Slope_Up',
 'HeartDisease',]]
 
# printing the altered DataFrame
df.head(5)


# In[37]:


scaler = StandardScaler()
scaler.fit(df.drop('HeartDisease',axis = 1))


# In[38]:


scaled_features = scaler.transform(df.drop('HeartDisease',axis = 1))
df_feat = pd.DataFrame(scaled_features,columns = df.columns[:-1])
df_feat.head()


# In[39]:


df.head(5)


# # Feature Selection

# In[40]:


col=df.describe().columns.to_list()
print(col)


# #All the features are important

# # Splitting the data

# In[41]:


X = df_feat
y = df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=20)


# # 5-Fold Cross validation and model comparison

# In[42]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'k-NN': KNeighborsClassifier(),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Naive Bayes': GaussianNB()
}

# Cross-validation and model evaluation
print("Cross-validation results:")
for model_name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f'{model_name} - Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}')


# I have selected Logistic regression, SVM and Random Forest for the further steps as these three found to have best accuracy among all the algorithms

# In[43]:


# Define hyperparameters for tuning
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4] 
}


param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

param_grid_lr = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],  # 'saga' is needed for 'elasticnet'
    'l1_ratio': [0.5] 
}


# In[44]:


# Initializing the Models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Initialize models
svm_model = SVC(probability=True)
rf_model = RandomForestClassifier()
lr_model = LogisticRegression()


# In[45]:


# Performing Grid Search CV
from sklearn.model_selection import GridSearchCV

# Initialize GridSearchCV
grid_search_svm = GridSearchCV(svm_model, param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_lr = GridSearchCV(lr_model, param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)

# Fit GridSearchCV
grid_search_svm.fit(X_train, y_train)
grid_search_rf.fit(X_train, y_train)
grid_search_lr.fit(X_train, y_train)


# In[46]:


# Evaluating the best model
from sklearn.metrics import accuracy_score, classification_report

# Best models
best_model_svm = grid_search_svm.best_estimator_
best_model_rf = grid_search_rf.best_estimator_
best_model_lr = grid_search_lr.best_estimator_

# Evaluate on test set
y_pred_svm = best_model_svm.predict(X_test)
y_pred_rf = best_model_rf.predict(X_test)
y_pred_lr = best_model_lr.predict(X_test)

# Accuracy scores
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

# Print results
print(f"SVM - Best Parameters: {grid_search_svm.best_params_}, Accuracy: {accuracy_svm:.4f}")
print(f"Random Forest - Best Parameters: {grid_search_rf.best_params_}, Accuracy: {accuracy_rf:.4f}")
print(f"Logistic Regression - Best Parameters: {grid_search_lr.best_params_}, Accuracy: {accuracy_lr:.4f}")

# Classification reports
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr))


# Model Performance Comparison
# 
# Before proceeding further, we evaluated the performance of several classification models using cross-validation. Here are the accuracy scores obtained for each model:
# 
# Logistic Regression: 0.8628
# 
# k-NN: 0.8598
# 
# SVM: 0.8716
# 
# Decision Tree: 0.7624
# 
# Random Forest: 0.8643
# 
# Gradient Boosting: 0.8495
# 
# Naive Bayes: 0.8584
# 
# After hyperparameter tuning using Grid Search CV, we re-evaluated the models and obtained the following accuracy scores:
# 
# Logistic Regression: 0.8546
# 
# SVM: 0.8767
# 
# Random Forest: 0.8811
# 
# As observed, hyperparameter tuning led to improvements in the accuracy of some models while others remained relatively unchanged. For instance, the accuracy of the Random Forest model increased from 0.8643 to 0.8811 after tuning, indicating a significant enhancement in predictive performance. On the other hand, the accuracy of the Logistic Regression model slightly decreased from 0.8628 to 0.8546 post-tuning.
# 
# In addition to accuracy, we also examined other performance metrics such as precision, recall, and F1-score before and after hyperparameter tuning to gain deeper insights into model performance.

# In[47]:


from sklearn.metrics import ConfusionMatrixDisplay

# Plotting confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

ConfusionMatrixDisplay.from_estimator(best_model_svm, X_test, y_test, ax=axes[0], cmap='Blues')
axes[0].set_title("SVM Confusion Matrix")

ConfusionMatrixDisplay.from_estimator(best_model_rf, X_test, y_test, ax=axes[1], cmap='Blues')
axes[1].set_title("Random Forest Confusion Matrix")

ConfusionMatrixDisplay.from_estimator(best_model_lr, X_test, y_test, ax=axes[2], cmap='Blues')
axes[2].set_title("Logistic Regression Confusion Matrix")

plt.show()


# In[48]:


from sklearn.metrics import roc_curve, auc

# Plot ROC curves
fig, ax = plt.subplots(figsize=(10, 8))

# SVM
y_prob_svm = best_model_svm.predict_proba(X_test)[:, 1]
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)
ax.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_svm:.2f})')

# Random Forest
y_prob_rf = best_model_rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')

# Logistic Regression
y_prob_lr = best_model_lr.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)
ax.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')

# Plot settings
ax.plot([0, 1], [0, 1], 'k--', lw=2)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc="lower right")

plt.show()


# We've successfully performed hyperparameter tuning and evaluated the models. Here are some observations:
# 
# Random Forest
# 
# Best Parameters: {'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 100}
# 
# Accuracy: 87.67%
# 
# Classification Report:
# 
# Class 0: Precision: 0.88, Recall: 0.82, F1-Score: 0.85
# 
# Class 1: Precision: 0.88, Recall: 0.92, F1-Score: 0.90
# 
# Overall: The model performs well with an overall accuracy of 87.67%. The precision and recall values for both classes are 
# balanced, indicating the model is effective in identifying both classes correctly.
# 
# SVM (Support Vector Machine)
# 
# Best Parameters: {'C': 1, 'degree': 2, 'gamma': 'scale', 'kernel': 'rbf'}
# 
# Accuracy: 87.67%
# 
# Classification Report:
# 
# Class 0: Precision: 0.88, Recall: 0.81, F1-Score: 0.84
# 
# Class 1: Precision: 0.87, Recall: 0.92, F1-Score: 0.90
# 
# Overall: The SVM model achieves the same accuracy as the Random Forest model. It has a slightly lower recall for Class 0 
# compared to Random Forest but performs equally well in terms of precision and F1-score.
# 
# Logistic Regression
# 
# Best Parameters: {'C': 0.1, 'l1_ratio': 0.5, 'penalty': 'l1', 'solver': 'liblinear'}
# 
# Accuracy: 85.46%
# 
# Classification Report:
# 
# Class 0: Precision: 0.83, Recall: 0.82, F1-Score: 0.82
# 
# Class 1: Precision: 0.87, Recall: 0.88, F1-Score: 0.88
# 
# Overall: Logistic Regression has a slightly lower accuracy compared to the other two models. However, it still maintains high precision and recall, especially for Class 1, indicating it is good at identifying positive cases.
# 
# Summary
# Random Forest and SVM: Both achieved an accuracy of 87.67%. Random Forest has slightly better recall for Class 1, while SVM has slightly lower recall for Class 0. Overall, both models perform similarly well.
# Logistic Regression: Achieved a slightly lower accuracy of 85.46%. It still performs well but is slightly outperformed by Random Forest and SVM.
# 
# Conclusion
# 
# Based on the results, Random Forest appears to be the best-performing model, closely followed by SVM. Both models achieve similar accuracy and exhibit balanced performance across precision, recall, and F1-score metrics. Logistic Regression, while performing well, falls slightly behind in overall accuracy. Moving forward, we can prioritize the Random Forest model for further development.

# # Hyperparameter Tuning for Random Forest:
#  

# In[49]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Instantiate the Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Perform Grid Search
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid, 
                              cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit the model
grid_search_rf.fit(X_train, y_train)

# Get the best parameters
best_params_rf = grid_search_rf.best_params_
print("Best Parameters for Random Forest: ", best_params_rf)

# Get the best model
best_rf_model = grid_search_rf.best_estimator_

# Evaluate on the test set
y_pred_rf = best_rf_model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred_rf))
print("Classification Report: \n", classification_report(y_test, y_pred_rf))


# #  Evaluating the Tuned Model

# In[50]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
y_prob_rf = best_rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob_rf)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Random Forest')
plt.legend(loc="lower right")
plt.show()


# Summary of Model Performance
# 
# Before Hyperparameter Tuning:
# 
# Random Forest Accuracy: 0.8643 ± 0.0192
# 
# After Hyperparameter Tuning:
# 
# Random Forest Accuracy: 0.8810

# The best parameters found for the Random Forest model were:
# 
# {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}

# # Ensemble Methods:

# In[51]:


# Voting classifier
from sklearn.ensemble import VotingClassifier

# Instantiate individual models
log_reg = LogisticRegression(C=0.1, solver='newton-cg', random_state=42)
svm = SVC(C=1, kernel='rbf', gamma='scale', probability=True, random_state=42)
rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=1, min_samples_split=2, random_state=42)

# Create a Voting Classifier
voting_clf = VotingClassifier(estimators=[('lr', log_reg), ('svc', svm), ('rf', rf)], voting='soft')
voting_clf.fit(X_train, y_train)

# Evaluate the ensemble model
y_pred_voting = voting_clf.predict(X_test)
print("Ensemble Model Accuracy: ", accuracy_score(y_test, y_pred_voting))
print("Ensemble Model Classification Report: \n", classification_report(y_test, y_pred_voting))

# Confusion Matrix for Ensemble Model
conf_matrix_voting = confusion_matrix(y_test, y_pred_voting)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_voting, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Ensemble Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve for Ensemble Model
y_prob_voting = voting_clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob_voting)
roc_auc_voting = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_voting)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Ensemble Model')
plt.legend(loc="lower right")
plt.show()


# In[52]:


from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Create a BaggingClassifier with RandomForest
bagging_clf = BaggingClassifier(estimator=RandomForestClassifier(), n_estimators=100, random_state=42)
bagging_clf.fit(X_train, y_train)

# Evaluate the Bagging model
y_pred_bagging = bagging_clf.predict(X_test)
print("Bagging Model Accuracy: ", accuracy_score(y_test, y_pred_bagging))
print("Bagging Model Classification Report: \n", classification_report(y_test, y_pred_bagging))

# Confusion Matrix for Bagging Model
conf_matrix_bagging = confusion_matrix(y_test, y_pred_bagging)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_bagging, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Bagging Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve for Bagging Model
y_prob_bagging = bagging_clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob_bagging)
roc_auc_bagging = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_bagging)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Bagging Model')
plt.legend(loc="lower right")
plt.show()


# In[53]:


# Boosting
from sklearn.ensemble import AdaBoostClassifier

# Create an AdaBoostClassifier
ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_clf.fit(X_train, y_train)

# Evaluate the AdaBoost model
y_pred_ada = ada_clf.predict(X_test)
print("AdaBoost Model Accuracy: ", accuracy_score(y_test, y_pred_ada))
print("AdaBoost Model Classification Report: \n", classification_report(y_test, y_pred_ada))

# Confusion Matrix for AdaBoost Model
conf_matrix_ada = confusion_matrix(y_test, y_pred_ada)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_ada, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for AdaBoost Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve for AdaBoost Model
y_prob_ada = ada_clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob_ada)
roc_auc_ada = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_ada)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for AdaBoost Model')
plt.legend(loc="lower right")
plt.show()


# In[54]:


from sklearn.ensemble import GradientBoostingClassifier

# Create a GradientBoostingClassifier
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_clf.fit(X_train, y_train)

# Evaluate the Gradient Boosting model
y_pred_gb = gb_clf.predict(X_test)
print("Gradient Boosting Model Accuracy: ", accuracy_score(y_test, y_pred_gb))
print("Gradient Boosting Model Classification Report: \n", classification_report(y_test, y_pred_gb))

# Confusion Matrix for Gradient Boosting Model
conf_matrix_gb = confusion_matrix(y_test, y_pred_gb)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_gb, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Gradient Boosting Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve for Gradient Boosting Model
y_prob_gb = gb_clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob_gb)
roc_auc_gb = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_gb)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Gradient Boosting Model')
plt.legend(loc="lower right")
plt.show()


# ## Model Comparison and Final Selection
# 
# ### Hyperparameter Tuning for Random Forest
# After hyperparameter tuning, the Random Forest model achieved the following performance metrics:
# 
# - **Best Parameters:** `{'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}`
# - **Accuracy:** 0.8811
# - **Classification Report:**
#     ```plaintext
#                precision    recall  f1-score   support
# 
#            0       0.88      0.83      0.85        94
#            1       0.88      0.92      0.90       133
# 
#     accuracy                           0.88       227
#    macro avg       0.88      0.87      0.88       227
# weighted avg       0.88      0.88      0.88       227
#     ```
# 
# ### Ensemble Methods
# We also evaluated different ensemble methods including Voting Classifier, Bagging Classifier, AdaBoost, and Gradient Boosting.
# 
# #### Voting Classifier
# - **Accuracy:** 0.8811
# - **Classification Report:**
#     ```plaintext
#                precision    recall  f1-score   support
# 
#            0       0.88      0.83      0.85        94
#            1       0.88      0.92      0.90       133
# 
#     accuracy                           0.88       227
#    macro avg       0.88      0.87      0.88       227
# weighted avg       0.88      0.88      0.88       227
#     ```
# 
# #### Bagging Classifier
# - **Accuracy:** 0.8811
# - **Classification Report:**
#     ```plaintext
#                precision    recall  f1-score   support
# 
#            0       0.88      0.83      0.85        94
#            1       0.88      0.92      0.90       133
# 
#     accuracy                           0.88       227
#    macro avg       0.88      0.87      0.88       227
# weighted avg       0.88      0.88      0.88       227
#     ```
# 
# #### AdaBoost
# - **Accuracy:** 0.8326
# - **Classification Report:**
#     ```plaintext
#                precision    recall  f1-score   support
# 
#            0       0.77      0.84      0.81        94
#            1       0.88      0.83      0.85       133
# 
#     accuracy                           0.83       227
#    macro avg       0.83      0.83      0.83       227
# weighted avg       0.84      0.83      0.83       227
#     ```
# 
# #### Gradient Boosting
# - **Accuracy:** 0.8678
# - **Classification Report:**
#     ```plaintext
#                precision    recall  f1-score   support
# 
#            0       0.86      0.81      0.84        94
#            1       0.87      0.91      0.89       133
# 
#     accuracy                           0.87       227
#    macro avg       0.87      0.86      0.86       227
# weighted avg       0.87      0.87      0.87       227
#     ```
# 
# ### Conclusion
# Based on the above results, the **Random Forest model** with hyperparameter tuning provides the best accuracy and balanced performance across all metrics. Therefore, we recommend using the Random Forest model for this classification task.
# 
# 

# ## Confusion Matrix and ROC Curve for Final Random Forest Model

# In[55]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Confusion Matrix for Random Forest
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve for Random Forest
y_prob_rf = best_rf_model.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Random Forest')
plt.legend(loc="lower right")
plt.show()


# # Final Model Evaluation and Fine-tuning

# In[56]:


# Fine-tuning the final model
final_model = RandomForestClassifier(max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=300)
final_model.fit(X_train, y_train)

# Evaluate the final model
y_pred_final = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred_final)
print("Final Model Accuracy:", final_accuracy)

# Other evaluation metrics and visualizations
# Confusion Matrix
conf_matrix_final = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_final, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Final Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
y_prob_final = final_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob_final)
roc_auc_final = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_final)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Final Model')
plt.legend(loc="lower right")
plt.show()


# # Feature Importance Analysis:
# 

# In[57]:


importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importance")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()


# # Cross-Validation and Model Stability

# In[58]:


from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(best_rf_model, X, y, cv=10, scoring='accuracy')
print("Cross-Validation Scores: ", cv_scores)
print("Mean CV Score: ", np.mean(cv_scores))


# Cross-Validation and Model Stability
# To ensure the robustness and stability of our model, we performed cross-validation. This method provides a more comprehensive evaluation by splitting the data into multiple training and testing sets and evaluating the model on each split. Below are the results from our cross-validation:
# 
# Cross-Validation Scores:
# 0.9121
# ,
# 0.9560
# ,
# 0.7802
# ,
# 0.8901
# ,
# 0.8791
# ,
# 0.8444
# ,
# 0.9111
# ,
# 0.7889
# ,
# 0.7667
# ,
# 0.7667
# 0.9121,0.9560,0.7802,0.8901,0.8791,0.8444,0.9111,0.7889,0.7667,0.7667
# 
# Mean Cross-Validation Score:
# 0.8495
# 0.8495
# 
# Analysis
# The cross-validation scores indicate how well the model performs across different subsets of the data. Here are some observations:
# 
# High Variability: The scores range from 0.7667 to 0.9560, indicating some variability in model performance across different folds.
# Mean Performance: The mean cross-validation score of 0.8495 suggests that, on average, the model performs well, but there are instances where performance drops below the mean.
# Outliers: The scores of 0.7802 and 0.7667 are notably lower than others. This could indicate certain data splits where the model struggled, possibly due to class imbalance or specific characteristics of the data.

# # Saving Model 

# In[59]:


# Train the model before saving it
best_rf_model.fit(X_train, y_train)  

# Save the trained model
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(best_rf_model, f)  # Save the trained model

print("Model trained and saved successfully!")



# In[60]:


# Loading the saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


# In[ ]:





# # Summary:

# In this project, we conducted a comprehensive analysis of a heart disease dataset to build a classification model for predicting the presence or absence of heart disease in patients. Here's a summary of the key steps and findings:
# 
# Data Preprocessing:
# 
# The dataset underwent preprocessing steps, including handling missing values, encoding categorical variables, and splitting into features and target variable.
# Feature Selection:
# 
# All features were considered important for the analysis based on the initial examination of the dataset.
# Model Selection and Evaluation:
# 
# We evaluated various classification models, including Logistic Regression, k-NN, SVM, Decision Tree, Random Forest, Gradient Boosting, and Naive Bayes, using 5-fold cross-validation.
# Among these models, SVM, Random Forest, and Logistic Regression showed the best accuracy scores.
# Hyperparameter Tuning:
# 
# Hyperparameter tuning was performed for SVM, Random Forest, and Logistic Regression models using GridSearchCV to optimize their performance.
# Ensemble Methods:
# 
# We explored ensemble methods such as Voting Classifier, Bagging Classifier, AdaBoost, and Gradient Boosting to further enhance model performance.
# Final Model Selection:
# 
# After comparing the performance of tuned models and ensemble methods, we selected the Random Forest model as it exhibited the highest accuracy and balanced performance.
# Model Evaluation and Fine-tuning:
# 
# The final Random Forest model was evaluated using accuracy metrics, confusion matrix, and ROC curve.
# Cross-validation was conducted to assess model stability and variability.
# Feature Importance Analysis:
# 
# Feature importance analysis was performed to understand the contribution of each feature in predicting heart disease.
# Model Saving:
# 
# The trained Random Forest model was saved using the pickle module for future use.
# Overall, this analysis provides valuable insights into building an accurate and reliable classification model for heart disease prediction, facilitating informed decision-making in clinical settings.
# 
# 

# # Submitted By- Shweta Kanungo (B-6)

# In[ ]:




