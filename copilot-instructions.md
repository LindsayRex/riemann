# GitHub Copilot Instructions for Expert SageMath Software Architect and Mathematician

This document provides detailed instructions for GitHub Copilot to assist
an expert SageMath software architect and mathematician in writing high-level 
SageMath experiments, specifically tailored for numerical experiments in analytic number theory, 
such as those investigating the energy functional for L-function zeros and critical line stability. 
The instructions are designed to guide Copilot in generating accurate, efficient, 
and mathematically rigorous SageMath code, aligning with the level of sophistication 
seen in the provided Experiment 1 results and 
leveraging the SageMath documentation ([Reference Manual](https://doc.sagemath.org/html/en/reference/),
 [Parallel Computing](https://doc.sagemath.org/html/en/reference/parallel/index.html)).

---

## Context and Objectives

### Role Description
You are assisting an **expert SageMath software architect and mathematician** with extensive experience in:
- Analytic number theory, particularly L-functions, Riemann Hypothesis (RH), and Weil’s explicit formula.
- Variational methods and energy functionals for studying critical points.
- Numerical experiments using SageMath for high-precision computations.
- Parallel computing in SageMath to optimize performance.
- Statistical analysis (e.g., polynomial fitting, bootstrap resampling) integrated with mathematical modeling.

The user writes experiments at the level of **Experiment 1: Single-Zero Perturbation Analysis**, which involves:
- Defining energy functionals for L-function zero sets.
- Perturbing zeros to measure energy differences (\( \Delta E(\delta) \)).
- Fitting polynomial models to test hypotheses (e.g., \( \Delta E(\delta) = C_1 \delta^2 + C_2 \delta^3 \)).
- Using robust statistical methods (least squares, bootstrap, AIC) to validate results.
- Producing publication-quality visualizations and reports.

### Objectives for Copilot
- Generate **SageMath code** that is mathematically precise, computationally efficient, and adheres to best practices.
- Suggest code structures that leverage SageMath’s strengths (e.g., symbolic computation, numerical precision, parallelization).
- Provide inline comments explaining mathematical and computational rationale.
- Anticipate common pitfalls in numerical number theory (e.g., floating-point errors, convergence issues).
- Offer suggestions for visualizations, statistical analysis, and parallel computing when appropriate.
- Ensure code is modular, reusable, and compatible with the SageMath 10.3 environment.

---

## SageMath-Specific Guidelines

### Core SageMath Features to Utilize
Refer to the [SageMath Reference Manual](https://doc.sagemath.org/html/en/reference/) for detailed functionality:
- **Symbolic and Numerical Computation**:
  - Use `sage.rings.real_mpfr` for arbitrary-precision arithmetic (`RealField(prec)`).
  - Leverage `sage.functions` for special functions (e.g., `zeta`, `gamma`, `digamma`).
  - Employ `sage.symbolic` for symbolic expressions when deriving analytical forms.
- **Linear Algebra**:
  - Use `sage.matrix` for efficient matrix operations (e.g., computing Hessians or quadratic forms).
  - Utilize `sage.vector` for discrepancy operator computations.
- **Numerical Methods**:
  - Apply `sage.numerical.optimize` for gradient descent or minimization tasks.
  - Use `sage.numerical.least_squares` for polynomial fitting.
- **Plotting**:
  - Generate plots with `sage.plot` (e.g., `plot`, `scatter_plot`) for visualizing energy landscapes or residuals.
  - Ensure publication-quality output with customized options (e.g., `figsize`, `fontsize`).
- **Data Structures**:
  - Use `sage.data_structures` for efficient handling of zero sets or test function bases.
  - Store results in `pandas` DataFrames for statistical analysis (via `sage.interfaces`).

### Parallel Computing
Refer to the [Parallel Computing Documentation](https://doc.sagemath.org/html/en/reference/parallel/index.html):
- Use `@parallel` decorator for parallelizing independent computations (e.g., energy calculations across \( \delta \)-values).
- Employ `ParallelMap` or `p_iter_fork` for distributing tasks across multiple cores.
- Ensure thread safety by avoiding shared mutable state in parallelized functions.
- Optimize for small-scale parallelism (4–16 cores) typical of numerical experiments.

### Mathematical Context
- **Energy Functional**: Defined as \( E[S] = \sum_j w_j (D_S(\varphi_j))^2 \), where \( D_S(\varphi) = \sum_{\rho \in S} \varphi(\Im \rho) - P(\varphi) \).
- **Perturbation Model**: Single-zero perturbations \( \rho(\delta) = \frac{1}{2} + \delta + i \gamma \), with \( \Delta E(\delta) = E[\{\rho(\delta)\}] - E[\{\rho(0)\}] \).
- **Hypothesis**: \( \Delta E(\delta) = C_1 \delta^2 + C_2 \delta^3 + O(\delta^4) \), with \( C_1 > 0 \).
- **Test Functions**: Orthonormal Gaussian or Fourier basis, e.g., \( \varphi_j(x) = e^{-x^2/2\sigma^2} \cos(jx) \).
- **Statistical Goals**: Fit polynomials, compute confidence intervals via bootstrap, and test hypotheses (e.g., \( C_1 > 0 \)).

---

## Copilot Instructions

### General Coding Style
- **Language**: Write SageMath code compatible with version 10.3, using Python 3 syntax.
- **Modularity**: Structure code into functions or classes for reusability (e.g., `EnergyFunctional`, `PerturbationExperiment`).
- **Comments**: Include:
  - Mathematical explanations (e.g., why a formula is used).
  - Computational notes (e.g., precision choices, parallelization rationale).
  - References to SageMath documentation for non-standard functions.
- **Naming Conventions**:
  - Use descriptive names (e.g., `compute_energy`, `fit_polynomial`).
  - Follow mathematical notation where possible (e.g., `delta` for \( \delta \), `C1` for \( C_1 \)).
- **Error Handling**:
  - Check for numerical stability (e.g., division by zero, overflow).
  - Validate inputs (e.g., ensure \( \delta \)-ranges are small for perturbation validity).
- **Performance**:
  - Precompute constant values (e.g., test function evaluations).
  - Use vectorized operations over loops where possible.
  - Suggest parallelization for computationally intensive tasks.

### Specific Tasks for Copilot

1. **Defining the Energy Functional**
   - Generate code to compute \( E[S] \) for a given zero set \( S \).
   - Implement \( D_S(\varphi) \) using a sum over zeros and a prime/archimedean term \( P(\varphi) \).
   - Example:
     ```sage
     from sage.all import *
     def discrepancy_operator(S, phi, primes_bound=1000):
         """
         Compute D_S(phi) = sum_{rho in S} phi(Im(rho)) - P(phi).
         Parameters:
             S: List of complex zeros.
             phi: Test function (callable).
             primes_bound: Upper bound for prime summation.
         Returns:
             Real value of discrepancy.
         """
         zero_sum = sum(phi(rho.imag) for rho in S)
         prime_sum = sum(log(p) * (phi(log(p^m)) + phi(-log(p^m)))
                         for p in prime_range(primes_bound)
                         for m in range(1, 10))
         return zero_sum - prime_sum
     ```
   - Suggest test functions (e.g., Gaussian: \( \varphi(x) = e^{-x^2/2} \)).
   - Ensure symmetry: \( D_S(s) = D_S(1-s) \).

2. **Perturbation Experiments**
   - Write functions to perturb a single zero and compute \( \Delta E(\delta) \).
   - Example:
     ```sage
     def compute_delta_energy(gamma, delta_range, n_points, test_functions):
         """
         Compute Delta E(delta) for single-zero perturbation.
         Parameters:
             gamma: Imaginary part of zero.
             delta_range: Tuple (-delta_max, delta_max).
             n_points: Number of delta points.
             test_functions: List of test functions.
         Returns:
             List of (delta, Delta E) pairs.
         """
         R = RealField(256)  # High precision
         S_c = [R(0.5) + I * gamma]
         E_c = compute_energy(S_c, test_functions)
         deltas = srange(delta_range[0], delta_range[1], step=(delta_range[1] - delta_range[0])/n_points, include_endpoint=True)
         results = []
         for delta in deltas:
             S_delta = [R(0.5 + delta) + I * gamma]
             E_delta = compute_energy(S_delta, test_functions)
             results.append((delta, E_delta - E_c))
         return results
     ```
   - Suggest parallelization for multiple \( \delta \)-values:
     ```sage
     @parallel(ncpus=4)
     def compute_energy_parallel(S, test_functions):
         return compute_energy(S, test_functions)
     ```

3. **Polynomial Fitting and Statistical Analysis**
   - Generate code for least-squares fitting to \( \Delta E(\delta) = C_1 \delta^2 + C_2 \delta^3 \).
   - Use `sage.numerical.least_squares` or `numpy.polyfit` (via Sage’s Python interface).
   - Example:
     ```sage
     import numpy as np
     def fit_polynomial(data, degree=3):
         """
         Fit polynomial to (delta, Delta E) data.
         Parameters:
             data: List of (delta, Delta E) pairs.
             degree: Polynomial degree (e.g., 3 for cubic).
         Returns:
             Coefficients [C_n, ..., C_0], standard errors, R^2.
         """
         x, y = zip(*data)
         coeffs = np.polyfit(x, y, degree)
         poly = np.poly1d(coeffs)
         residuals = np.array(y) - poly(x)
         R2 = 1 - sum(residuals**2) / sum((y - np.mean(y))**2)
         return coeffs[::-1], residuals.std(), R2
     ```
   - Implement bootstrap resampling for confidence intervals:
     ```sage
     def bootstrap_ci(data, n_resamples=10000, degree=3):
         """
         Compute bootstrap confidence intervals for polynomial coefficients.
         Returns:
             Dictionary of 95% CIs for each coefficient.
         """
         coeffs_list = []
         for _ in range(n_resamples):
             sample = [(x, y) for x, y in data if random() < 0.8]
             coeffs, _, _ = fit_polynomial(sample, degree)
             coeffs_list.append(coffs)
         return {f'C{i}': (percentile(coeffs_list[:,i], 2.5), percentile(coeffs_list[:,i], 97.5))
                 for i in range(degree+1)}
     ```

4. **Visualization**
   - Create plots for \( \Delta E \) vs. \( \delta \), \( \Delta E \) vs. \( \delta^2 \), and residuals.
   - Example:
     ```sage
     def plot_energy(data, fitted_coeffs):
         """
         Plot Delta E vs delta and Delta E vs delta^2.
         """
         x, y = zip(*data)
         p1 = plot(lambda x: sum(fitted_coeffs[i] * x^i for i in range(len(fitted_coeffs))),
                   (min(x), max(x)), color='blue', legend_label='Fitted')
         p1 += scatter_plot(data, color='red', marker='o', legend_label='Data')
         p2 = plot(lambda x: sum(fitted_coeffs[i] * x^(i/2) for i in range(len(fitted_coeffs))),
                   (min(x)^2, max(x)^2), color='blue', legend_label='Fitted')
         p2 += scatter_plot([(x^2, y) for x, y in data], color='red', marker='o')
         show(graphics_array([[p1, p2]]), figsize=(10, 5))
     ```
   - Ensure plots are publication-ready (high resolution, clear labels).

5. **Parallel Computing Suggestions**
   - Identify tasks suitable for parallelization (e.g., computing \( E[S] \) for multiple \( S \)).
   - Suggest `@parallel` or `p_iter_fork` for loops over \( \delta \)-values or configurations.
   - Example:
     ```sage
     def parallel_delta_energy(gamma, delta_range, n_points, test_functions):
         deltas = srange(delta_range[0], delta_range[1], step=(delta_range[1] - delta_range[0])/n_points)
         inputs = [([0.5 + delta + I * gamma], test_functions) for delta in deltas]
         results = list(compute_energy_parallel(inputs))
         E_c = compute_energy([0.5 + I * gamma], test_functions)
         return [(deltas[i], res[1] - E_c) for i, res in enumerate(results)]
     ```

### Common Pitfalls to Avoid
- **Numerical Precision**: Use `RealField(256)` or higher for L-function computations to avoid floating-point errors.
- **Convergence**: Check for convergence in sums (e.g., prime sums in \( P(\varphi) \)) by setting appropriate bounds.
- **Symmetry**: Ensure test functions and zero sets respect \( s \mapsto 1-s \) symmetry.
- **Parallelization**: Avoid shared state in parallel functions to prevent race conditions.
- **Overfitting**: Use AIC or cross-validation to prevent overfitting in polynomial fits.

### Suggestions for Copilot
- **Proactive Suggestions**:
  - Suggest alternative test functions (e.g., Hermite polynomials) if Gaussian/Fourier bases are used.
  - Recommend parallelization for large loops or multiple configurations.
  - Propose additional statistical tests (e.g., Shapiro-Wilk for residual normality).
- **Context Awareness**:
  - Recognize mathematical terms (e.g., “energy functional,” “discrepancy operator”) and map to appropriate SageMath constructs.
  - Understand the goal of testing \( C_1 > 0 \) and prioritize code that computes second derivatives or fits quadratic terms.
- **Error Messages**:
  - If errors occur (e.g., `ZeroDivisionError`, `OverflowError`), suggest increasing precision or adjusting bounds.
  - Provide debug-friendly code with intermediate outputs for verification.

---

## Example Prompt and Expected Output

**Prompt**: “Write a SageMath function to compute the energy difference \( \Delta E(\delta) \) for a single-zero perturbation at height \( \gamma = 14.135 \), with \( \delta \in [-0.1, 0.1] \), using 20 Gaussian test functions.”

**Expected Output**:
```sage
from sage.all import *
import numpy as np

def gaussian_test_function(x, j, sigma=1):
    """Gaussian test function phi_j(x) = exp(-x^2/(2*sigma^2)) * cos(j*x)."""
    return exp(-x^2/(2*sigma^2)) * cos(j*x)

def compute_energy(S, test_functions, weights=None, primes_bound=1000):
    """
    Compute energy E[S] = sum_j w_j (D_S(phi_j))^2.
    Parameters:
        S: List of complex zeros.
        test_functions: List of callable test functions.
        weights: List of weights (default: 1 for all).
        primes_bound: Upper bound for prime summation.
    Returns:
        Real value of energy.
    """
    if weights is None:
        weights = [1] * len(test_functions)
    R = RealField(256)
    energy = 0
    for phi, w in zip(test_functions, weights):
        zero_sum = sum(phi(rho.imag) for rho in S)
        prime_sum = sum(R(log(p)) * (phi(log(p^m)) + phi(-log(p^m)))
                        for p in prime_range(primes_bound)
                        for m in range(1, 10))
        discrepancy = zero_sum - prime_sum
        energy += w * discrepancy^2
    return energy

@parallel(ncpus=4)
def compute_energy_parallel(args):
    """Parallel wrapper for compute_energy."""
    S, test_functions = args
    return compute_energy(S, test_functions)

def single_zero_experiment(gamma=14.135, delta_range=(-0.1, 0.1), n_points=41, n_test_funcs=20):
    """
    Compute Delta E(delta) for single-zero perturbation.
    Parameters:
        gamma: Imaginary part of zero.
        delta_range: Tuple (-delta_max, delta_max).
        n_points: Number of delta points.
        n_test_funcs: Number of Gaussian test functions.
    Returns:
        List of (delta, Delta E) pairs, fitted coefficients, plot.
    """
    R = RealField(256)
    # Define test functions
    test_functions = [lambda x, j=j: gaussian_test_function(x, j) for j in range(n_test_funcs)]
    weights = [1] * n_test_funcs  # Equal weights for simplicity
    
    # Base configuration
    S_c = [R(0.5) + I * gamma]
    E_c = compute_energy(S_c, test_functions)
    
    # Parallel computation of Delta E
    deltas = srange(delta_range[0], delta_range[1], step=(delta_range[1] - delta_range[0])/(n_points-1), include_endpoint=True)
    inputs = [([R(0.5 + delta) + I * gamma], test_functions) for delta in deltas]
    results = list(compute_energy_parallel(inputs))
    data = [(deltas[i], res[1] - E_c) for i, res in enumerate(results)]
    
    # Polynomial fitting
    coeffs, std_err, R2 = fit_polynomial(data, degree=3)
    
    # Visualization
    p1 = scatter_plot(data, color='red', marker='o', legend_label='Data')
    p1 += plot(lambda x: sum(coeffs[i] * x^i for i in range(len(coeffs))),
               (delta_range[0], delta_range[1]), color='blue', legend_label='Fit')
    show(p1, figsize=(8, 6), title='Delta E vs delta')
    
    return data, coeffs, R2

# Run experiment
data, coeffs, R2 = single_zero_experiment()
print(f"Fitted coefficients: C1 = {coeffs[2]:.4f}, C2 = {coeffs[3]:.4f}")
print(f"R^2 = {R2:.6f}")
```

**Notes**:
- The code uses high precision (`RealField(256)`).
- Parallelizes energy computations across \( \delta \)-values.
- Includes visualization and polynomial fitting.
- Comments explain mathematical and computational choices.

---

## Additional Notes
- **SageMath Version**: Assume SageMath 10.3 unless specified otherwise.
- **Environment**: Code should run in a standard SageMath environment with Python libraries (`numpy`, `pandas`, `matplotlib`) available.
- **Documentation**: Always reference relevant SageMath manual sections (e.g., [Numerical Methods](https://doc.sagemath.org/html/en/reference/numerical/index.html)) in comments.
- **Future Prompts**: Expect prompts for multi-zero experiments, gradient descent, or Hessian computations, requiring similar rigor.

---

**Document Information**:
- **Generated**: June 18, 2025, 03:48 PM AEST
- **Author**: Expert SageMath Software Architect and Mathematician
- **Purpose**: Guide GitHub Copilot for SageMath experiment development
- **Version**: 1.0