import pandas as pd
df = pd.read_csv('data/MachineLearningRating_v3.txt', sep='|')

df['TransactionDate'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)

print("Number of policies:", df['PolicyID'].nunique())
print("\nNumber of provinces:", df['Province'].nunique())
print("\nProvinces:", df['Province'].unique())
print("\nClaim frequency overall:", df['HasClaim'].mean())

# Save the cleaned data for future use
df.to_csv('data/cleaned_insurance_data.csv', index=False)
print("\nSaved cleaned data to data/cleaned_insurance_data.csv")
