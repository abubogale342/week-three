import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skewnorm, poisson

# Create a figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Three Key Data Landscapes in Insurance Analytics', fontsize=16)

# --- 1. Normal Distribution --- 
data_normal = norm.rvs(loc=1500, scale=200, size=10000, random_state=42)
mean_normal = np.mean(data_normal)
median_normal = np.median(data_normal)
ax1.hist(data_normal, bins=50, density=True, alpha=0.7, color='blue')
ax1.axvline(mean_normal, color='red', linestyle='dashed', linewidth=2, label=f'Mean: ${mean_normal:,.0f}')
ax1.axvline(median_normal, color='green', linestyle='dashed', linewidth=2, label=f'Median: ${median_normal:,.0f}')
ax1.set_title('1. The Bell Curve (Normal Distribution)')
ax1.set_xlabel('Claim Processing Time (Hours)')
ax1.set_ylabel('Probability Density')
ax1.legend()
ax1.text(0.95, 0.85, 'Use Case: Operational Data\n(e.g., call times, processing times)\nMean and Median are nearly identical.', transform=ax1.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8))

# --- 2. Skewed Distribution (Log-Normal) ---
data_skewed = np.exp(norm.rvs(loc=7, scale=1.0, size=10000, random_state=42))
mean_skewed = np.mean(data_skewed)
median_skewed = np.median(data_skewed)
ax2.hist(data_skewed, bins=100, density=True, alpha=0.7, color='orange', range=[0, 15000])
ax2.axvline(mean_skewed, color='red', linestyle='dashed', linewidth=2, label=f'Mean: ${mean_skewed:,.0f}')
ax2.axvline(median_skewed, color='green', linestyle='dashed', linewidth=2, label=f'Median: ${median_skewed:,.0f}')
ax2.set_title('2. The Long Tail (Right-Skewed)')
ax2.set_xlabel('Claim Amount ($)')
ax2.set_ylabel('Probability Density')
ax2.legend()
ax2.text(0.95, 0.85, 'Use Case: Financial Data\n(e.g., claim amounts, income)\nMean is pulled higher by outliers.\nMedian is a better measure of \'typical\'.', transform=ax2.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', fc='moccasin', alpha=0.8))

# --- 3. Poisson Distribution ---
lambda_val = 0.3 # Average 0.3 claims per year
data_poisson = poisson.rvs(mu=lambda_val, size=10000, random_state=42)
mean_poisson = np.mean(data_poisson)
ax3.hist(data_poisson, bins=np.arange(data_poisson.min(), data_poisson.max() + 2) - 0.5, density=True, alpha=0.7, color='purple', rwidth=0.8)
ax3.axvline(mean_poisson, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_poisson:.2f}')
ax3.set_title('3. The Count (Poisson Distribution)')
ax3.set_xlabel('Number of Claims per Policyholder per Year')
ax3.set_ylabel('Probability')
ax3.legend()
ax3.set_xticks(np.arange(0, data_poisson.max() + 1))
ax3.text(0.95, 0.85, 'Use Case: Event Frequency\n(e.g., # claims, # calls)\nModels discrete, whole numbers.', transform=ax3.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', fc='thistle', alpha=0.8))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('distributions_plot.png')
