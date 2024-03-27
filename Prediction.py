import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlrd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#Data set naming
data_set='Life_Expectancy.xls'

#----------------------------------------------------------------------------------------------------------------
#This section focuses on creating the dataframes from the different sheets in the dataset 
#----------------------------------------------------------------------------------------------------------------

# Reads the excel file into the pandas dataframe, requires XLRD
df = pd.read_excel(data_set, sheet_name='UK at birth', skiprows=7)

# Because within the dataset there are two life expectancy columns they had to be selected manually then seperated later 
selected_columns = [0, 1, 5, 6]
df_selected = df.iloc[:, selected_columns]

#Renaming the columns
df_selected.columns = ['Year(M)', 'Life expectancy (Males)', 'Year(F)', 'Life expectancy (Females)']
print(df_selected)

#seperating the dataframe into two data frames of male and female 
df_males = df_selected[['Year(M)', 'Life expectancy (Males)']]
df_females = df_selected[['Year(F)', 'Life expectancy (Females)']]
print(df_females,"\n",df_males)

#----------------------------------------------------------------------------------------------------------------
#This section focuses on the regional male and female dataframes
#----------------------------------------------------------------------------------------------------------------
#Female section

# Reads the excel file into the dataframe  starting from row 8 and up to row 19
df_region_f= pd.read_excel(data_set, sheet_name='Regions at birth - F', skiprows=7, nrows=10)

# Selecting necessary columns (excluding 'Area code') Due to extra data surrounding the table this had to be done manually
df_selected_f = df_region_f[['Region', '1991-1993', '1992-1994', '1993-1995', '1994-1996', '1995-1997', 
                         '1996-1998', '1997-1999', '1998-2000', '1999-2001', '2000-2002', '2001-2003', 
                         '2002-2004', '2003-2005', '2004-2006', '2005-2007', '2006-2008', '2007-2009', 
                         '2008-2010', '2009-2011', '2010-2012']]
# Remove rows with all null values
df_selected_f = df_selected_f.dropna(how='all')

# Reset the index
df_selected_f = df_selected_f.reset_index(drop=True)

# Print the resulting DataFrame
print("Female data\n",df_selected_f)

#Male section

# Reads the excel file into the dataframe  starting from row 8 and up to row 19
df_region_m = pd.read_excel(data_set, sheet_name='Regions at birth - M', skiprows=7, nrows=10)

# Selecting necessary columns (excluding 'Area code') Due to extra data surrounding the table this had to be done manually
df_selected_m = df_region_m[['Region', '1991-1993', '1992-1994', '1993-1995', '1994-1996', '1995-1997', 
                         '1996-1998', '1997-1999', '1998-2000', '1999-2001', '2000-2002', '2001-2003', 
                         '2002-2004', '2003-2005', '2004-2006', '2005-2007', '2006-2008', '2007-2009', 
                         '2008-2010', '2009-2011', '2010-2012']]
# Remove rows with all null values
df_selected_m = df_selected_m.dropna(how='all')

# Reset the index
df_selected_m = df_selected_m.reset_index(drop=True)


print(df_selected_m)

#----------------------------------------------------------------------------------------------------------------
#This section focuses on plotting the data for historical analysis from the regional data
#----------------------------------------------------------------------------------------------------------------


# transposing the dataframes
df_selected_m_transposed = df_selected_m.set_index('Region').T
df_selected_f_transposed = df_selected_f.set_index('Region').T

# dictionary containing colours
region_colors = {
    'North East': '#1f77b4',   # dark blue
    'North West': '#2ca02c',   # dark green
    'Yorkshire and The Humber': '#d62728',   # dark red
    'East Midlands': '#ff7f0e',   # dark orange
    'West Midlands': '#9467bd',   # dark purple
    'East': '#8c564b',   # dark brown
    'London': '#e377c2',   # dark pink
    'South East': '#17becf',   # dark cyan
    'South West': '#bcbd22'   # dark yellow
}

# plotting the transposed dataframes
ax = None
for region in df_selected_m_transposed.columns:
    # removes extra spaces from region names
    region_cleaned = region.strip()
    if ax is None: #initalizing the plot 
        ax = df_selected_m_transposed[region].plot(kind='line', marker='s', color=region_colors[region_cleaned], figsize=(12, 6), label=region_cleaned + ' (Male)')
        df_selected_f_transposed[region].plot(kind='line', marker='o', color=region_colors[region_cleaned], ax=ax, label=region_cleaned + ' (Female)')
    else: #plot on the existing plot
        df_selected_m_transposed[region].plot(kind='line', marker='s', color=region_colors[region_cleaned], ax=ax, label=region_cleaned + ' (Male)')
        df_selected_f_transposed[region].plot(kind='line', marker='o', color=region_colors[region_cleaned], ax=ax, label=region_cleaned + ' (Female)')

# adjusting x-axis labels
ax.set_xticks(range(len(df_selected_m_transposed.index)))
ax.set_xticklabels(df_selected_m_transposed.index, rotation=45)
#x axis labels and titles
plt.xlabel('Year')
plt.ylabel('Life Expectancy')
plt.title('Life Expectancy Over Time by Region')
plt.grid(True)
#changing the legend between male and female
plt.legend(title='Region', loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()
#----------------------------------------------------------------------------------------------------------------
#This section focuses on plotting the data and calculating the mean difference from the life expectancy of female and male 
#----------------------------------------------------------------------------------------------------------------



# This section calculated the differences in ages mean 



# Calculating the Difference in Life Expectancy
differences = df_selected['Life expectancy (Females)'] - df_selected['Life expectancy (Males)']

# Calculating the Average Difference
average_difference = differences.mean()

print("Average difference in female and male life expectancy:", average_difference)



#----------------------------------------------------------------------------------------------------------------
# This section focuses on plotting the data of male and female life expectancy 
#----------------------------------------------------------------------------------------------------------------


# Plotting male life expectancy
plt.plot(df_males['Year(M)'], df_males['Life expectancy (Males)'], label='Male')

# Plotting female life expectancy
plt.plot(df_females['Year(F)'], df_females['Life expectancy (Females)'], label='Female')

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('Life Expectancy')
plt.title('Life Expectancy Over Time for Males and Females')

# adds the legend into the graph
plt.legend()

# Set y-axis ticks to display every integer
plt.yticks(range(int(min(df_males['Life expectancy (Males)'].min(), df_females['Life expectancy (Females)'].min())), 
                int(max(df_males['Life expectancy (Males)'].max(), df_females['Life expectancy (Females)'].max())) + 1))


plt.show()
#----------------------------------------------------------------------------------------------------------------
#This section focuses on predicting future life expectancy 
#----------------------------------------------------------------------------------------------------------------

# Resets the index for df_males and df_females and extracts the start years of each
df_males['Year'] = df_males['Year(M)'].str.split('-', expand=True)[0].astype(int)
df_females['Year'] = df_females['Year(F)'].str.split('-', expand=True)[0].astype(int)

# Defines the x and y axis for males and females
X_males = df_males['Year'].values.reshape(-1, 1)
y_males = df_males['Life expectancy (Males)']
X_females = df_females['Year'].values.reshape(-1, 1)
y_females = df_females['Life expectancy (Females)']

# trains linear regression models
model_males = LinearRegression().fit(X_males, y_males)
model_females = LinearRegression().fit(X_females, y_females)

# Generate future years and make predictions
future_years = np.arange(1999, 2041).reshape(-1, 1)
future_predictions_males = model_males.predict(future_years)
future_predictions_females = model_females.predict(future_years)


# Plots actual and predicted values for males and females on the same plot
plt.figure(figsize=(12, 6))

# Plots actual values for males
plt.plot(df_males['Year'], df_males['Life expectancy (Males)'], label='Actual (Males)', color='blue')

# Plots predicted values for males
plt.plot(future_years, future_predictions_males, label='Predicted (Males)', linestyle='--', color='blue')

# Plots actual values for females
plt.plot(df_females['Year'], df_females['Life expectancy (Females)'], label='Actual (Females)', color='red')

# Plots predicted values for females
plt.plot(future_years, future_predictions_females, label='Predicted (Females)', linestyle='--', color='red')

plt.xlabel('Year')
plt.ylabel('Life Expectancy')
plt.title('Life Expectancy Prediction for Males and Females')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()