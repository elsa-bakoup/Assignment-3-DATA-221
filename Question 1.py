#Question 1
import numpy as np
import pandas as pd

df = pd.read_csv('crime.csv')

mean = np.mean(df['ViolentCrimesPerPop'])
print("Mean:", mean)
median = np.median(df['ViolentCrimesPerPop'])
print("Median:", median)
standard_deviation = np.std(df['ViolentCrimesPerPop'])
print("Standard deviation:", standard_deviation)
max_value = np.max(df['ViolentCrimesPerPop'])
print("Max value:", max_value)
min_value = np.min(df['ViolentCrimesPerPop'])
print("Min value:", min_value)

# The mean is bigger than the median so we can conclude that the data is skewed as there might be some unusually large
# values pulling the average upward while the median stays consistent.

# If there are extreme values the mean is more affected than the median as it is computed using every data point
# whereas the median only depends on the order of values. An extreme outlier directly shifts the total sum (and thus the mean)
# but rarely changes the middle position.
