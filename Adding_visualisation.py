#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[20]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


# ## Load Datasets

# #### Define the path to the dataset and load it into a pandas DataFrame

# In[21]:


path = "/Users/aakritigautam/Downloads/climate_change_impact_on_agriculture_2024.csv"


# In[22]:


climate_impact_df = pd.read_csv(path)
climate_impact_df


# ## Data Exploration and Cleaning

# #### Display column names:
# This provides an overview of the available data and helps understand the structure of the dataset.

# In[23]:


climate_impact_df.columns


# The dataset is well-structured with 15 columns capturing various aspects of climate change and agriculture.

# #### Display DataFrame information:
# The '.info()' method provides details about the dataset, such as the number of entries, column data types, and non-null values.

# In[24]:


climate_impact_df.info()


# #### Describe the data 
# This gives a quick look at key statistics (mean, standard deviation, min, max, quartiles) of numeric columns.

# In[25]:


climate_impact_df.describe()


# The dataset spans a period from 1990 to 2024, with diverse information on climate conditions, crop yield, and economic impact. Descriptive statistics hint at potential outliers (like negative temperatures) which may require further investigation.

# #### Find any missing values:
# Replace common placeholders for missing data (like '-', 'N/A', 'na') with NaN, then check for any null values.

# In[26]:


climate_impact_df.replace(['', ' ', '-', 'N/A', 'na', '?'], np.nan, inplace=True)
climate_impact_df.isnull().sum()


# No missing values were identified, which simplifies further analysis.

# ## Data Visualization

# In[27]:


plt.figure(figsize=(14, 8))
plt.suptitle("Impact of Climate Change on Agriculture Over the Years", fontsize=16, fontweight="bold", y=1.05)

# Average Temperature over the years
plt.subplot(2, 2, 1)
sns.lineplot(x="Year", y="Average_Temperature_C", data=climate_impact_df, marker="o")
plt.title("Average Temperature Over Years", fontweight="bold")
plt.xlabel("Year")
plt.ylabel("Avg Temperature (°C)")
plt.annotate("Shows the trend of yearly average temperatures.", 
             xy=(0.5, -0.25), xycoords="axes fraction", ha="center", fontsize=9)

# Precipitation over the years
plt.subplot(2, 2, 2)
sns.lineplot(x= "Year", y= "Total_Precipitation_mm", data= climate_impact_df, marker= "o")
plt.title("Total Precipitation Over Years", fontweight= "bold")
plt.xlabel("Year")
plt.ylabel("Precipitation (mm)")
plt.annotate("Tracks yearly precipitation levels.", 
             xy= (0.5, -0.25), xycoords= "axes fraction", ha= "center", fontsize=9)

# CO2 Emissions over the years
plt.subplot(2, 2, 3)
sns.lineplot(x= "Year", y= "CO2_Emissions_MT", data= climate_impact_df, marker= "o")
plt.title("CO2 Emissions Over Years", fontweight= "bold")
plt.xlabel("Year")
plt.ylabel("CO2 Emissions (MT)")
plt.annotate("Monitors CO2 emissions per year.", 
             xy= (0.5, -0.25), xycoords= "axes fraction", ha= "center", fontsize=9)

# Crop Yield over the years
plt.subplot(2, 2, 4)
sns.lineplot(x= "Year", y= "Crop_Yield_MT_per_HA", data= climate_impact_df, marker= "o")
plt.title( "Crop Yield Over Years", fontweight= "bold")
plt.xlabel("Year")
plt.ylabel("Crop Yield (MT/HA)")
plt.annotate("Displays annual crop yield trends.", 
             xy=(0.5, -0.25), xycoords= "axes fraction", ha= "center", fontsize=9)

# Adjust spacing between plots and from the top title
plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.5, wspace=0.4)
plt.show()


# The above graph illustrates the impact of climate change on agriculture from 1990 to 2025, focusing on temperature, precipitation, CO2 emissions, and crop yield. The first graph shows an upward trend in average temperatures, fluctuating between 13°C and 17°C, indicating global warming over time. The second graph tracks total precipitation, which varies between 1450 mm and 1750 mm, showing no clear long-term trend but highlighting erratic rainfall patterns. The third graph depicts CO2 emissions, fluctuating between 13 and 17 million metric tons, with a slight increase, reflecting rising human-induced emissions. The fourth graph shows crop yield trends, ranging from 2.0 to 2.5 metric tons per hectare, with no consistent increase or decrease, suggesting the unpredictable effects of climate change on agricultural productivity. Together, the graphs demonstrate the complex and variable relationship between climate factors and agricultural outcomes.

# In[ ]:
