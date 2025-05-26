
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

I    = 1.22e-2      # Moment of inertia [kg·m²]
I_err = 0.09e-2     # Uncertainty in I
r    = 0.0135       # Cylinder radius [m]
r_err = 0.0005      # Uncertainty in r

csv_files = {
    "20.1grami.csv": 20.1,
    "30.1grami.csv": 30.1,
    "40.2grami.csv": 40.2,
    "50grami.csv": 50.0,
    "60grami.csv": 60.0,
    "70.2grami.csv": 70.2,
}

def extract_alpha_from_corrupt_csv(file_path):
    df = pd.read_csv(file_path)
    if len(df) > 100:
        df = df.iloc[50:-50]
    values = []

    for col in df.columns:
        try:
            col_data = df[col].astype(str).str.split(';').explode()
            col_data = pd.to_numeric(col_data, errors='coerce').dropna()
            values.extend(col_data.values)
        except Exception as e:
            continue
    return np.array(values)

alpha_by_mass = {}
means, stds, masses = [], [], []

for filepath, mass in csv_files.items():
    alphas = extract_alpha_from_corrupt_csv(filepath)
    alpha_by_mass[mass] = alphas
    mu = np.mean(alphas)
    sigma = np.std(alphas)
    masses.append(mass)
    means.append(mu)
    stds.append(sigma)
    
    plt.figure(figsize=(6, 4))
    count, bins, _ = plt.hist(alphas, bins=15, density=True, alpha=0.6, label='Histogram')
    x = np.linspace(min(bins), max(bins), 100)
    plt.plot(x, norm.pdf(x, mu, sigma), 'r--', 
             label=f'Gaussian Fit\nμ={mu:.2f}, σ={sigma:.2f}')
    plt.title(f'Histogram - {mass}g')
    plt.xlabel('Angular Acceleration (α)')
    plt.ylabel('Density')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Histogram_{mass}g.png", dpi=300)
    plt.close()

masses = np.array(masses)
means  = np.array(means)
stds   = np.array(stds)

weights = 1 / np.maximum(stds, 1e-6)
p, cov = np.polyfit(masses, means, 1, w=weights, cov=True)
slope, intercept = p
slope_err = np.sqrt(cov[0, 0]) 
mass_range = np.linspace(min(masses), max(masses), 100)
alpha_fit  = slope * mass_range + intercept

plt.figure(figsize=(8, 6))
plt.errorbar(masses, means, yerr=stds, fmt='o', capsize=5, label='Measured Data')
plt.plot(mass_range, alpha_fit, 'r-', 
         label=f'Fit: α = {slope:.2f}·m + {intercept:.2f}')
plt.xlabel('Mass (g)')
plt.ylabel('Angular Acceleration (α)')
plt.title('Angular Acceleration vs. Mass')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Alpha_vs_Mass.png", dpi=300)
plt.close()


g_calculated = slope * 1000 * I / r
g_err = g_calculated * np.sqrt((slope_err/slope)**2 + (I_err/I)**2 + (r_err/r)**2)

plt.figure(figsize=(8, 6))  

values = [g_calculated / 1000, 9.81] 
errors = [g_err, 0]  
labels = ["Calculated g", "Accepted g"]

plt.bar(labels, values, yerr=errors, capsize=8, color=['skyblue', 'lightcoral'], 
        edgecolor='black', linewidth=1.2)
plt.ylim(0, 15)
plt.ylabel('Gravitational Acceleration g (m/s²)', fontsize=12)
plt.title('Comparison of Calculated and Accepted g', fontsize=14, pad=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, (value, error) in enumerate(zip(values, errors)):
    plt.text(i, value + error + 0.2, f"{value:.2f} m/s²", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig("Calculated_vs_Accepted_g_Extended.png", dpi=300)
plt.close()

print("=== Regression Results ===")
print(f"Slope (A)         : {slope:.4f} rad/s² per gram")
print(f"Intercept (B)     : {intercept:.4f} rad/s²")
print(f"Slope Uncertainty : {slope_err:.4f} rad/s² per gram")
print("\n=== Calculated Gravitational Acceleration ===")
print(f"g = {g_calculated:.2f} ± {g_err:.2f} m/s²")
