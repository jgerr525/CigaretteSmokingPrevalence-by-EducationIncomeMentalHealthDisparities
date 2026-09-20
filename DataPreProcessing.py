import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths and corresponding disparity labels
DF_income = pd.read_excel("Income-Related_Disparities_in_Cigarette_Smoking_Among_Adults_20250217.xlsx")
DF_health = pd.read_excel("Mental_Health-Related_Disparities_in_Cigarette_Smoking_Among_Adults_20250217.xlsx")
DF_education = pd.read_excel("Education_Attained_Disparities_in_Cigarette_Smoking_Among_Adults_20250217.xlsx")


# Initialize an empty DataFrame
main_df = pd.DataFrame()

main_df['Year'] = DF_income.iloc[:, 0]  # First column
main_df['State'] = DF_income.iloc[:, 1]  # Second column

main_df['Education Demographic'] = DF_education.iloc[:, 4]
main_df['Education Use %'] = DF_education.iloc[:, 5]

main_df['Income Demographic'] = DF_income.iloc[:, 4]
main_df['Income Use %'] = DF_income.iloc[:, 5]

main_df['Health Demographic'] = DF_health.iloc[:, 4]
main_df['Health Use %'] = DF_health.iloc[:, 5]

combined_df = main_df[main_df["Education Use %"] != main_df["Education Use %"].shift()]

# Ensure the relevant columns are numeric and coerce any non-numeric values to NaN
combined_df[['Education Use %', 'Income Use %', 'Health Use %']] = combined_df[['Education Use %', 'Income Use %', 'Health Use %']].apply(pd.to_numeric, errors='coerce')

# Check for any NaN values after conversion
print(combined_df[['Education Use %', 'Income Use %', 'Health Use %']].isna().sum())

# Drop rows with NaN values in the relevant columns
combined_df = combined_df.dropna(subset=['Education Use %', 'Income Use %', 'Health Use %'])

# Compute the average smoking prevalence for each state based on Education, Income, and Health
combined_df.loc[:, 'Cigarette Use Prevalence (%)'] = combined_df[['Education Use %', 'Income Use %', 'Health Use %']].mean(axis=1)


# Show result and save file
print(combined_df.head())
combined_df.to_excel("Cigarette_Smoking_Data.xlsx")