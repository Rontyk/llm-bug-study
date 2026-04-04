import numpy as np
from scipy.stats import chi2_contingency

observed = np.array([
    [407, 257],  # Llama 3.1
    [513, 151],  # Llama 3.3
    [562, 100],  # DeepSeek V3
    [619, 25]    # GPT-OSS
])


chi2, p, dof, expected = chi2_contingency(observed)


print("--- Результаты теста Хи-квадрат ---")
print(f"Значение Chi-square (χ2): {chi2:.2f}")
print(f"P-value: {p:.10e}") 
print(f"Степени свободы (df): {dof}")

print("\n--- Ожидаемые значения (E) ---")
models = ["Llama 3.1", "Llama 3.3", "DeepSeek V3", "GPT-OSS"]
for i, model in enumerate(models):
    print(f"{model}: Pass={expected[i][0]:.2f}, Fail={expected[i][1]:.2f}")


alpha = 0.001
if p < alpha:
    print("\nВердикт: Разница статистически значима (p < 0.001). Масштаб модели влияет на результат.")
else:
    print("\nВердикт: Разница НЕ значима. Модели работают одинаково.")
