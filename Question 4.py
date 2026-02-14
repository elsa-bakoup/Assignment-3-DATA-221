#Question 4
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score



kidney_disease_data = pd.read_csv('kidney_disease.csv')
kidney_disease_data.replace("?", pd.NA, inplace=True)

#Changing the type of numerical data that are type string
numeric_fix = ['pcv', 'wc', 'rc']

for col in numeric_fix:
    kidney_disease_data[col] = pd.to_numeric(kidney_disease_data[col], errors='coerce')

feature_matrix = kidney_disease_data.drop(columns=['classification','id'])
labels = kidney_disease_data['classification']


#Separate column by categories
num_cols = feature_matrix.select_dtypes(include='number').columns
cat_cols = feature_matrix.select_dtypes(exclude='number').columns


#datatypes = feature_matrix.dtypes
#print(datatypes)

# Split the data into training and testing sets
feature_matrix_train, feature_matrix_test, labels_train, labels_test = train_test_split(
feature_matrix, labels,
test_size=0.3, # 30% of data for testing
random_state=42, # Ensures reproducibility
shuffle=True # Shuffles data before splitting
)


#Creating the pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),    #Replacing numerical missing values by the median of the values of that column
            ('scaler', StandardScaler())
        ]), num_cols),

        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),     #Replacing categorical missing values by the most frequent value in that column 
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ]
)

model = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=5))  # Setting the number of neighbors to k = 5
])


model.fit(feature_matrix_train, labels_train)

#making the prediction (either ckd or notckd)
ckd_pred = model.predict(feature_matrix_test)


#computing the confusion matrix
cm = confusion_matrix(labels_test, ckd_pred)
confusion_matrix = pd.DataFrame(cm, index=['Actual CKD', 'Actual not CKD'], columns=['Predicted CKD', 'Predicted not CKD'])
print(confusion_matrix)


# computing and printing all the metrics
accuracy = accuracy_score(labels_test, ckd_pred)
print("\nAccuracy:", accuracy)
precision = precision_score(labels_test, ckd_pred, pos_label='ckd')
print("Precision:", precision)
recall = recall_score(labels_test, ckd_pred, pos_label='ckd')
print("Recall:", recall)
f1 = f1_score(labels_test, ckd_pred, pos_label='ckd')
print("F1_score:", f1)


# True Positive : prediction of a kidney disease & patient has a kidney disease.
# False Positive : prediction of a kidney disease & patient does NOT have a kidney disease.
# True Negative : prediction of no kidney disease & patient does NOT have a kidney disease.
# False Negative : prediction of no kidney disease & patient has a kidney disease.

# Accuracy can be misleading because it treats all predictions as equal and does not take into account
# cases of false positive. For example a naive model that predict "notckd" 100% of the time on a dataset
# where 99% of the patient don't have a kidney  will get a 99% percent accuracy whereas it has not
# learned anything and blindly predict the same outcome for every patient.

# If missing a kidney disease case is very serious, the most important metric is the Recall as it
# measures TP/(TP + FN) because in that case we want the FN be as close to 0 as possible for very
# few sick patient to be missed.



