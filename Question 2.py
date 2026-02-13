#Question 2
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('crime.csv')

plt.hist(df['ViolentCrimesPerPop'],edgecolor = 'white')
plt.title('Distribution of Violent Crimes Per Population')
plt.xlabel('Violent Crimes Per Population')
plt.ylabel('Frequency')
plt.show()

plt.boxplot(df['ViolentCrimesPerPop'])
plt.title('Distribution of Violent Crimes Per Population')
plt.xlabel('Violent Crimes Per Population')
plt.ylabel('Frequency')
plt.show()


# The histogram is positively/right skewed which shows that
# the values are not heavenly spread as they are more clustered on
# the left side.

# The boxplot shows that the median is closer to Q1 which is
# also an indicator of a right skewed dataset.

# The boxplot does not suggest the presence of outliers as
# there is no individual points beyond the whiskers.


