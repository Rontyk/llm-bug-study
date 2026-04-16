import numpy as np
from scipy.stats import chi2_contingency

# Observed frequencies: [Pass, Fail]
observed = np.array([
    [407, 257],  # Llama 3.1
    [513, 151],  # Llama 3.3
    [562, 100],  # DeepSeek V3
    [619, 25]    # GPT-OSS
])

# Perform Chi-square test of independence
chi2, p, dof, expected = chi2_contingency(observed)

print("--- Chi-Square Test Results ---")
print(f"Chi-square statistic (χ2): {chi2:.2f}")
print(f"P-value: {p:.10e}") 
print(f"Degrees of freedom (df): {dof}")

print("\n--- Expected Frequencies (E) ---")
models = ["Llama 3.1", "Llama 3.3", "DeepSeek V3", "GPT-OSS"]
for i, model in enumerate(models):
    print(f"{model}: Pass={expected[i][0]:.2f}, Fail={expected[i][1]:.2f}")

# Significance level
alpha = 0.001
if p < alpha:
    print(f"\nVerdict: The difference is statistically significant (p < {alpha}). Model scale significantly impacts performance.")
else:
    print("\nVerdict: The difference is NOT statistically significant. Models perform similarly.")
