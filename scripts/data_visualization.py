import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/cleaned_insurance_data.csv')

# Analyze claims by province
province_stats = df.groupby('Province').agg(
    TotalPolicies=('PolicyID', 'nunique'),
    TotalClaims=('TotalClaims', 'sum'),
    ClaimFrequency=('HasClaim', 'mean'),
    AverageClaim=('TotalClaims', lambda x: x[x > 0].mean() if (x > 0).any() else 0),
    AveragePremium=('TotalPremium', 'mean')
).reset_index()

# Sort by claim frequency
province_stats = province_stats.sort_values('ClaimFrequency', ascending=False)

# Display the results
print("\nProvince-wise Statistics:")
print(province_stats.to_string())

# Visualize claim frequency by province
plt.figure(figsize=(12, 6))
sns.barplot(x='Province', y='ClaimFrequency', data=province_stats)
plt.title('Claim Frequency by Province')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/claim_frequency_by_province.png')
print("\nSaved visualization to visualizations/claim_frequency_by_province.png")
