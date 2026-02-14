#Question 5
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


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
            ('imputer', SimpleImputer(strategy='most_frequent')),   #Replacing categorical missing values by the most frquent value of that column
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ]
)

accuracy_scores = []  # Creating a list to store all the accuracy scores

k_values = [1, 3, 5, 7, 9]   #list of required k values


# Creating a loop to iterate through each given value of k
for k in k_values:
    model = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=k))])
    model.fit(feature_matrix_train, labels_train)
    ckd_pred = model.predict(feature_matrix_test)

    accuracy = accuracy_score(labels_test, ckd_pred)   # computing an accuracy score for every value of k
    accuracy_scores.append(accuracy)   # appending it to the list to store all the accuracy scores

data = { 'Values of k' : k_values, 'Accuracy': accuracy_scores}  # creating a dictionary with all the data to convert it to a dataframe

table = pd.DataFrame(data)
print(table)

# Changing the value of k changes the number k of the closest data points the model will use to make predictions.
# In this case we can clearly see that it affects the accuracy of the model and make it vary depending on k.

# When k is very small, the model is hyper-sensitive to the specific location of every single data point and
# fits the training data perfectly --> it will likely fail on new, unseen data because it captured the
# noise rather than the general trend.

# When k is very large, the model becomes too "lazy" and ignores local patterns. By looking at too many neighbors,
# the model washes out the distinct boundaries between classes --> it fails to capture the actual relationship
# between features and classes and will most likely always predict the class that represents the majority of the data points.


