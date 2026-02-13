#Question 3
import pandas as pd
from sklearn.model_selection import train_test_split

kidney_disease_data = pd.read_csv('kidney_disease.csv')
feature_matrix = kidney_disease_data.drop(columns=['classification'])
labels = kidney_disease_data['classification']

# Split the data into training and testing sets
feature_matrix_train, feature_matrix_test, labels_train, labels_test = train_test_split(
feature_matrix, labels,
test_size=0.3, # 30% of data for testing
random_state=42, # Ensures reproducibility
shuffle=True # Shuffles data before splitting
)

# Training and testing a model on the same dataset is not a good practice because the objective of testing
# is to estimate how well the model will perform predictions on data that it has not seen.
# If the model is trained on the same data it has already seen, it will memorize the training data and fail
# to understand its structure. This can lead to overfitting, where the model learns the noise and details
# in the training data, negatively impacting its performance on new data.


# A testing set is used to evaluate the performance of a model on unseen data. It used to determine how accurate and
# precise a model is in general as well as ensuring that the model isn't overfitting to the training data.

