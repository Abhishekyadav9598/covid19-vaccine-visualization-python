import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 #read csv file by file location
df=pd.read_csv("C:\\Users\\Dell\\OneDrive\\Desktop\\covid19vaccinesbyzipcode_test.csv")
print(df)


# Checking Null Values
print(df.isnull().sum())

# Fill missing values with the average of that column (if any)
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)
df['local_health_jurisdiction'] = df['local_health_jurisdiction'].fillna("No Data")
df['county'] = df['county'].fillna("No Country")

# Save cleaned file without dropping any columns
df.to_csv("covid19vaccines_cleaned.csv", index=False)

# Display remaining nulls (if any)
print("Remaining nulls in the DataFrame:\n", df.isnull().sum())



#Printing head(10) and tail(10)
print(df.head(10))
print(df.tail(10))


# Describe And Info
print(df.info())
print(df.describe())



#Selecting DataFrame Having age12_plus_population Greater than 5000
df1=df[df['age12_plus_population']>5000]
print(df1)


#Numaric column


numeric_cols = [
    'age12_plus_population',
    'age5_plus_population',
    'tot_population',
    'persons_fully_vaccinated',
    'persons_partially_vaccinated',
    'percent_of_population_fully_vaccinated',
    'percent_of_population_partially_vaccinated',
    'percent_of_population_with_1_plus_dose'
]


#Box Plot

for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(y=df[col], color='skyblue')
    plt.title(f'Box Plot of {col}')
    plt.tight_layout()
    plt.show()


# Sort by zip_code for better line continuity
df_sorted = df.sort_values(by='zip_code_tabulation_area')

#LinePlot


    
plt.figure(figsize=(12, 6))
sns.lineplot(
    x='zip_code_tabulation_area', 
    y='percent_of_population_fully_vaccinated', 
    data=df_sorted,
    marker='o',
    color='green'
)
plt.title('Percent of Fully Vaccinated Population by ZIP Code')
plt.xlabel('ZIP Code')
plt.ylabel('Percent Fully Vaccinated')
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()




# Grouping the data
county_vax = df.groupby('county')['persons_fully_vaccinated'].sum().sort_values(ascending=False).dropna()

# BarChart
plt.figure(figsize=(12, 6))
sns.barplot(x=county_vax.index, y=county_vax.values, hue=county_vax.index, palette='viridis', legend=False)
plt.title('Total Fully Vaccinated People by County')
plt.xlabel('County')
plt.ylabel('Persons Fully Vaccinated')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Histogtam


sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.histplot(df['percent_of_population_fully_vaccinated'], bins=20, kde=True, color='skyblue')
plt.title("Histogram: % Fully Vaccinated")
plt.xlabel("Percent Fully Vaccinated")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()



# Scatter

sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='tot_population', y='persons_fully_vaccinated', color='orange')
plt.title("Scatter: Total Population vs. Fully Vaccinated")
plt.xlabel("Total Population")
plt.ylabel("Persons Fully Vaccinated")
plt.tight_layout()
plt.show()



# PieChart

top_counties = df.groupby("county")["tot_population"].sum().nlargest(5)

plt.figure(figsize=(6, 6))
plt.pie(top_counties, labels=top_counties.index, autopct='%1.1f%%', startangle=140)
plt.title("Pie Chart: Top 5 Counties by Population")
plt.tight_layout()
plt.show()



# Select only numerical columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Compute correlation matrix
corr_matrix = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title("Heatmap: Correlation Matrix of Numeric Features")
plt.tight_layout()
plt.show()



# Create pair plot
sns.pairplot(df[numeric_cols], diag_kind='kde', corner=True)
plt.suptitle("Pair Plot of Vaccination Data", y=1.02)
plt.show()
