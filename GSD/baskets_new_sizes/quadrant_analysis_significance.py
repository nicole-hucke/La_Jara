from scipy.stats import binom_test

# Number of "successes" (points in correct quadrants)
k = 21
n = 33
p_expected = 0.5

# One-sided test: is the observed proportion > 0.5?
p_value = binom_test(k, n, p_expected, alternative='greater')
print(f"P-value: {p_value}")
