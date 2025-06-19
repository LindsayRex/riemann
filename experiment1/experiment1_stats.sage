# ############################################################################
#
# EXPERIMENT 1: SINGLE-ZERO PERTURBATION - STATISTICAL ANALYSIS MODULE
# ======================================================================
#
# This module provides comprehensive statistical analysis for Experiment 1
# results including:
# - Polynomial fitting: ΔE(δ) = C₁δ² + C₂δ³ + higher order terms
# - Coefficient estimation with confidence intervals and standard errors
# - Hypothesis testing: H₀: C₁ > 0 (local stability test)
# - Bootstrap resampling for robust uncertainty quantification
# - Residual analysis and model validation
# - Publication-quality statistical reporting
# - HDF5 data storage integration
#
# ############################################################################

import h5py
import numpy as np
import csv
import time
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit
from sage.all import *

class Experiment1Statistics:
    """Statistical analysis module for single-zero perturbation experiment with HDF5 support."""
    
    def __init__(self, hdf5_file, confidence_level=0.95, bootstrap_samples=10000):
        """
        Initialize statistical analysis module.
        
        Args:
            hdf5_file: Path to HDF5 data file
            confidence_level: Confidence level for intervals and tests
            bootstrap_samples: Number of bootstrap resamples for uncertainty
        """
        self.hdf5_file = hdf5_file
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.bootstrap_samples = bootstrap_samples
        
        print(f"Experiment 1 Statistics initialized:")
        print(f"  HDF5 file: {hdf5_file}")
        print(f"  Confidence level: {confidence_level*100}%")
        print(f"  Bootstrap samples: {bootstrap_samples}")
    
    def polynomial_model_quadratic(self, delta, C1):
        """
        Quadratic model: ΔE(δ) = C₁δ²
        
        Args:
            delta: Perturbation values
            C1: Quadratic coefficient
            
        Returns:
            Model predictions
        """
        return C1 * delta**2
    
    def polynomial_model_cubic(self, delta, C1, C2):
        """
        Cubic model: ΔE(δ) = C₁δ² + C₂δ³
        
        Args:
            delta: Perturbation values
            C1: Quadratic coefficient  
            C2: Cubic coefficient
            
        Returns:
            Model predictions
        """
        return C1 * delta**2 + C2 * delta**3
    
    def polynomial_model_quartic(self, delta, C1, C2, C3):
        """
        Quartic model: ΔE(δ) = C₁δ² + C₂δ³ + C₃δ⁴
        
        Args:
            delta: Perturbation values
            C1: Quadratic coefficient
            C2: Cubic coefficient
            C3: Quartic coefficient
            
        Returns:
            Model predictions
        """
        return C1 * delta**2 + C2 * delta**3 + C3 * delta**4
    
    def fit_polynomial_models(self, delta_values, delta_E_values):
        """
        Fit multiple polynomial models to the ΔE(δ) data.
        
        Args:
            delta_values: Array of δ perturbation values
            delta_E_values: Array of corresponding ΔE values
            
        Returns:
            dict: Fitting results for different polynomial orders
        """
        fitting_results = {}
        
        # Remove the point at δ = 0 if it exists (should be exactly 0)
        mask = np.abs(delta_values) > 1e-12
        delta_fit = delta_values[mask]
        delta_E_fit = delta_E_values[mask]
        
        try:
            # Quadratic fit: ΔE = C₁δ²
            popt_quad, pcov_quad = curve_fit(
                self.polynomial_model_quadratic, 
                delta_fit, delta_E_fit,
                p0=[1.0],  # Initial guess for C₁
                maxfev=10000
            )
            
            # Compute goodness of fit
            pred_quad = self.polynomial_model_quadratic(delta_fit, *popt_quad)
            residuals_quad = delta_E_fit - pred_quad
            ss_res_quad = np.sum(residuals_quad**2)
            ss_tot = np.sum((delta_E_fit - np.mean(delta_E_fit))**2)
            r_squared_quad = 1 - (ss_res_quad / ss_tot)
            
            # Standard errors from covariance matrix
            param_errors_quad = np.sqrt(np.diag(pcov_quad))
            
            fitting_results['quadratic'] = {
                'coefficients': popt_quad,
                'covariance': pcov_quad,
                'standard_errors': param_errors_quad,
                'residuals': residuals_quad,
                'r_squared': r_squared_quad,
                'aic': len(delta_fit) * np.log(ss_res_quad / len(delta_fit)) + 2 * 1,
                'C1': popt_quad[0],
                'C1_stderr': param_errors_quad[0]
            }
            
        except Exception as e:
            print(f"Warning: Quadratic fit failed: {e}")
            fitting_results['quadratic'] = None
        
        try:
            # Cubic fit: ΔE = C₁δ² + C₂δ³
            popt_cubic, pcov_cubic = curve_fit(
                self.polynomial_model_cubic,
                delta_fit, delta_E_fit,
                p0=[1.0, 0.0],  # Initial guess for C₁, C₂
                maxfev=10000
            )
            
            pred_cubic = self.polynomial_model_cubic(delta_fit, *popt_cubic)
            residuals_cubic = delta_E_fit - pred_cubic
            ss_res_cubic = np.sum(residuals_cubic**2)
            r_squared_cubic = 1 - (ss_res_cubic / ss_tot)
            param_errors_cubic = np.sqrt(np.diag(pcov_cubic))
            
            fitting_results['cubic'] = {
                'coefficients': popt_cubic,
                'covariance': pcov_cubic,
                'standard_errors': param_errors_cubic,
                'residuals': residuals_cubic,
                'r_squared': r_squared_cubic,
                'aic': len(delta_fit) * np.log(ss_res_cubic / len(delta_fit)) + 2 * 2,
                'C1': popt_cubic[0],
                'C2': popt_cubic[1],
                'C1_stderr': param_errors_cubic[0],
                'C2_stderr': param_errors_cubic[1]
            }
            
        except Exception as e:
            print(f"Warning: Cubic fit failed: {e}")
            fitting_results['cubic'] = None
        
        try:
            # Quartic fit: ΔE = C₁δ² + C₂δ³ + C₃δ⁴
            popt_quartic, pcov_quartic = curve_fit(
                self.polynomial_model_quartic,
                delta_fit, delta_E_fit,
                p0=[1.0, 0.0, 0.0],  # Initial guess
                maxfev=10000
            )
            
            pred_quartic = self.polynomial_model_quartic(delta_fit, *popt_quartic)
            residuals_quartic = delta_E_fit - pred_quartic
            ss_res_quartic = np.sum(residuals_quartic**2)
            r_squared_quartic = 1 - (ss_res_quartic / ss_tot)
            param_errors_quartic = np.sqrt(np.diag(pcov_quartic))
            
            fitting_results['quartic'] = {
                'coefficients': popt_quartic,
                'covariance': pcov_quartic,
                'standard_errors': param_errors_quartic,
                'residuals': residuals_quartic,
                'r_squared': r_squared_quartic,
                'aic': len(delta_fit) * np.log(ss_res_quartic / len(delta_fit)) + 2 * 3,
                'C1': popt_quartic[0],
                'C2': popt_quartic[1],
                'C3': popt_quartic[2],
                'C1_stderr': param_errors_quartic[0],
                'C2_stderr': param_errors_quartic[1],
                'C3_stderr': param_errors_quartic[2]
            }
            
        except Exception as e:
            print(f"Warning: Quartic fit failed: {e}")
            fitting_results['quartic'] = None
        
        return fitting_results
    
    def bootstrap_coefficient_analysis(self, delta_values, delta_E_values, model_type='cubic'):
        """
        Bootstrap analysis for coefficient uncertainty estimation.
        
        Args:
            delta_values: Array of δ values
            delta_E_values: Array of ΔE values
            model_type: Which polynomial model to use ('quadratic', 'cubic', 'quartic')
            
        Returns:
            dict: Bootstrap statistics for coefficients
        """
        # Remove δ = 0 point
        mask = np.abs(delta_values) > 1e-12
        delta_fit = delta_values[mask]
        delta_E_fit = delta_E_values[mask]
        n_points = len(delta_fit)
        
        # Choose model function
        if model_type == 'quadratic':
            model_func = self.polynomial_model_quadratic
            n_params = 1
            param_names = ['C1']
        elif model_type == 'cubic':
            model_func = self.polynomial_model_cubic
            n_params = 2
            param_names = ['C1', 'C2']
        elif model_type == 'quartic':
            model_func = self.polynomial_model_quartic
            n_params = 3
            param_names = ['C1', 'C2', 'C3']
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Bootstrap resampling
        np.random.seed(42)  # For reproducibility
        bootstrap_coefficients = []
        
        print(f"Performing bootstrap analysis with {self.bootstrap_samples} samples...")
        
        for i in range(self.bootstrap_samples):
            # Resample with replacement
            indices = np.random.choice(n_points, size=n_points, replace=True)
            delta_boot = delta_fit[indices]
            delta_E_boot = delta_E_fit[indices]
            
            try:
                # Fit model to bootstrap sample
                if model_type == 'quadratic':
                    popt, _ = curve_fit(model_func, delta_boot, delta_E_boot, p0=[1.0])
                elif model_type == 'cubic':
                    popt, _ = curve_fit(model_func, delta_boot, delta_E_boot, p0=[1.0, 0.0])
                elif model_type == 'quartic':
                    popt, _ = curve_fit(model_func, delta_boot, delta_E_boot, p0=[1.0, 0.0, 0.0])
                
                bootstrap_coefficients.append(popt)
                
            except:
                # Skip failed fits
                continue
            
            if (i + 1) % 1000 == 0:
                print(f"  Completed {i + 1}/{self.bootstrap_samples} bootstrap samples")
        
        if len(bootstrap_coefficients) == 0:
            print("Warning: All bootstrap fits failed!")
            return None
        
        bootstrap_coefficients = np.array(bootstrap_coefficients)
        
        # Compute bootstrap statistics
        bootstrap_stats = {}
        
        for j, param_name in enumerate(param_names):
            param_values = bootstrap_coefficients[:, j]
            
            # Basic statistics
            mean_val = np.mean(param_values)
            std_val = np.std(param_values, ddof=1)
            
            # Confidence intervals (percentile method)
            ci_lower = np.percentile(param_values, 100 * self.alpha / 2)
            ci_upper = np.percentile(param_values, 100 * (1 - self.alpha / 2))
            
            bootstrap_stats[param_name] = {
                'mean': mean_val,
                'std': std_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'values': param_values.copy()
            }
        
        bootstrap_stats['successful_samples'] = len(bootstrap_coefficients)
        bootstrap_stats['model_type'] = model_type
        
        return bootstrap_stats
    
    def hypothesis_testing(self, fitting_results, bootstrap_stats=None):
        """
        Perform hypothesis tests for local stability and model adequacy.
        
        Args:
            fitting_results: Results from fit_polynomial_models()
            bootstrap_stats: Bootstrap statistics (optional)
            
        Returns:
            dict: Hypothesis testing results
        """
        hypothesis_tests = {}
        
        # Test 1: Local Stability Test - H₀: C₁ ≤ 0 vs H₁: C₁ > 0
        if 'cubic' in fitting_results and fitting_results['cubic'] is not None:
            cubic_fit = fitting_results['cubic']
            C1 = cubic_fit['C1']
            C1_stderr = cubic_fit['C1_stderr']
            
            # One-sided t-test for C₁ > 0
            t_statistic = C1 / C1_stderr
            df = len(cubic_fit['residuals']) - 2  # n - p (2 parameters)
            
            # P-value for one-sided test (H₁: C₁ > 0)
            p_value_stability = 1 - scipy_stats.t.cdf(t_statistic, df)
            
            hypothesis_tests['local_stability'] = {
                'null_hypothesis': 'C1 <= 0 (unstable)',
                'alternative_hypothesis': 'C1 > 0 (stable)',
                'test_statistic': t_statistic,
                'p_value': p_value_stability,
                'degrees_freedom': df,
                'significant': p_value_stability < 0.05,
                'C1_estimate': C1,
                'C1_stderr': C1_stderr
            }
        
        # Test 2: Cubic Term Significance - H₀: C₂ = 0 vs H₁: C₂ ≠ 0
        if 'cubic' in fitting_results and fitting_results['cubic'] is not None:
            cubic_fit = fitting_results['cubic']
            C2 = cubic_fit['C2']
            C2_stderr = cubic_fit['C2_stderr']
            df = len(cubic_fit['residuals']) - 2
            
            # Two-sided t-test for C₂ ≠ 0
            t_statistic_C2 = C2 / C2_stderr
            p_value_cubic = 2 * (1 - scipy_stats.t.cdf(abs(t_statistic_C2), df))
            
            hypothesis_tests['cubic_significance'] = {
                'null_hypothesis': 'C2 = 0 (no cubic term)',
                'alternative_hypothesis': 'C2 ≠ 0 (cubic term present)',
                'test_statistic': t_statistic_C2,
                'p_value': p_value_cubic,
                'degrees_freedom': df,
                'significant': p_value_cubic < 0.05,
                'C2_estimate': C2,
                'C2_stderr': C2_stderr
            }
        
        # Test 3: Model Comparison - Quadratic vs Cubic using F-test
        if ('quadratic' in fitting_results and fitting_results['quadratic'] is not None and
            'cubic' in fitting_results and fitting_results['cubic'] is not None):
            
            quad_fit = fitting_results['quadratic']
            cubic_fit = fitting_results['cubic']
            
            # F-test for nested models
            ss_res_quad = np.sum(quad_fit['residuals']**2)
            ss_res_cubic = np.sum(cubic_fit['residuals']**2)
            
            n = len(cubic_fit['residuals'])
            f_statistic = ((ss_res_quad - ss_res_cubic) / 1) / (ss_res_cubic / (n - 2))
            p_value_f = 1 - scipy_stats.f.cdf(f_statistic, 1, n - 2)
            
            hypothesis_tests['model_comparison'] = {
                'null_hypothesis': 'Quadratic model adequate',
                'alternative_hypothesis': 'Cubic model significantly better',
                'f_statistic': f_statistic,
                'p_value': p_value_f,
                'significant': p_value_f < 0.05,
                'quad_r_squared': quad_fit['r_squared'],
                'cubic_r_squared': cubic_fit['r_squared']
            }
        
        # Test 4: Bootstrap-based stability test (if available)
        if bootstrap_stats is not None and 'C1' in bootstrap_stats:
            C1_bootstrap = bootstrap_stats['C1']['values']
            fraction_positive = np.mean(C1_bootstrap > 0)
            
            hypothesis_tests['bootstrap_stability'] = {
                'null_hypothesis': 'C1 <= 0 (unstable)',
                'alternative_hypothesis': 'C1 > 0 (stable)',
                'fraction_positive': fraction_positive,
                'bootstrap_p_value': 1 - fraction_positive,
                'significant': fraction_positive > 0.95,
                'C1_bootstrap_mean': bootstrap_stats['C1']['mean'],
                'C1_bootstrap_ci': [bootstrap_stats['C1']['ci_lower'], 
                                   bootstrap_stats['C1']['ci_upper']]
            }
        
        return hypothesis_tests
    
    def residual_analysis(self, fitting_results, delta_values, delta_E_values):
        """
        Perform residual analysis for model validation.
        
        Args:
            fitting_results: Results from polynomial fitting
            delta_values: Original δ values
            delta_E_values: Original ΔE values
            
        Returns:
            dict: Residual analysis results
        """
        residual_analysis = {}
        
        # Remove δ = 0 point
        mask = np.abs(delta_values) > 1e-12
        delta_fit = delta_values[mask]
        delta_E_fit = delta_E_values[mask]
        
        for model_name, fit_result in fitting_results.items():
            if fit_result is None:
                continue
                
            residuals = fit_result['residuals']
            n = len(residuals)
            
            # Basic residual statistics
            residual_mean = np.mean(residuals)
            residual_std = np.std(residuals, ddof=1)
            residual_range = [np.min(residuals), np.max(residuals)]
            
            # Normality test (Shapiro-Wilk)
            if n >= 3 and n <= 5000:
                shapiro_stat, shapiro_p = scipy_stats.shapiro(residuals)
            else:
                shapiro_stat, shapiro_p = None, None
            
            # Heteroscedasticity test (simple)
            # Split residuals into two groups and compare variances
            mid_idx = n // 2
            residuals_sorted_by_x = residuals[np.argsort(np.abs(delta_fit))]
            var1 = np.var(residuals_sorted_by_x[:mid_idx], ddof=1)
            var2 = np.var(residuals_sorted_by_x[mid_idx:], ddof=1)
            
            f_ratio = max(var1, var2) / min(var1, var2)
            
            residual_analysis[model_name] = {
                'mean': residual_mean,
                'std': residual_std,
                'range': residual_range,
                'shapiro_statistic': shapiro_stat,
                'shapiro_p_value': shapiro_p,
                'normality_ok': shapiro_p > 0.05 if shapiro_p is not None else None,
                'variance_ratio': f_ratio,
                'homoscedasticity_ok': f_ratio < 3.0,  # Rule of thumb
                'residuals': residuals.copy()
            }
        
        return residual_analysis
    
    def comprehensive_analysis(self, delta_values, delta_E_values):
        """
        Perform comprehensive statistical analysis of experiment 1 results.
        
        Args:
            delta_values: Array of perturbation values
            delta_E_values: Array of energy differences
            
        Returns:
            dict: Complete statistical analysis results
        """
        print("Performing comprehensive statistical analysis for Experiment 1...")
        
        start_time = time.time()
        
        # 1. Polynomial fitting
        print("  Fitting polynomial models...")
        fitting_results = self.fit_polynomial_models(delta_values, delta_E_values)
        
        # 2. Bootstrap analysis
        print("  Bootstrap analysis...")
        bootstrap_stats = self.bootstrap_coefficient_analysis(
            delta_values, delta_E_values, model_type='cubic'
        )
        
        # 3. Hypothesis testing
        print("  Hypothesis testing...")
        hypothesis_tests = self.hypothesis_testing(fitting_results, bootstrap_stats)
        
        # 4. Residual analysis
        print("  Residual analysis...")
        residual_analysis = self.residual_analysis(fitting_results, delta_values, delta_E_values)
        
        # 5. Model selection
        print("  Model selection...")
        best_model = self._select_best_model(fitting_results)
        
        analysis_time = time.time() - start_time
        
        # Compile comprehensive results
        comprehensive_results = {
            'fitting_results': fitting_results,
            'bootstrap_analysis': bootstrap_stats,
            'hypothesis_testing': hypothesis_tests,
            'residual_analysis': residual_analysis,
            'best_model': best_model,
            'analysis_time': analysis_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"  Statistical analysis completed in {analysis_time:.2f} seconds")
        
        return comprehensive_results
    
    def _select_best_model(self, fitting_results):
        """
        Select best model based on AIC and other criteria.
        
        Args:
            fitting_results: Polynomial fitting results
            
        Returns:
            dict: Best model selection results
        """
        valid_models = {k: v for k, v in fitting_results.items() if v is not None}
        
        if not valid_models:
            return None
        
        # Select based on lowest AIC
        aic_values = {k: v['aic'] for k, v in valid_models.items()}
        best_model_name = min(aic_values, key=aic_values.get)
        
        best_model_info = {
            'model_name': best_model_name,
            'aic': aic_values[best_model_name],
            'r_squared': valid_models[best_model_name]['r_squared'],
            'coefficients': valid_models[best_model_name]['coefficients'],
            'aic_comparison': aic_values
        }
        
        return best_model_info
    
    def analyze_configuration(self, config_name):
        """
        Perform comprehensive statistical analysis for a single configuration.
        
        Args:
            config_name: Name of configuration to analyze
            
        Returns:
            dict: Complete statistical analysis results
        """
        try:
            with h5py.File(self.hdf5_file, 'a') as f:
                if config_name not in f:
                    print(f"Configuration {config_name} not found in HDF5 file")
                    return None
                
                config_group = f[config_name]
                
                # Check for different data structure formats
                if 'delta_values' in config_group and 'delta_E_values' in config_group:
                    # New format
                    delta_values = config_group['delta_values'][:]
                    delta_E_values = config_group['delta_E_values'][:]
                elif 'perturbation_analysis' in config_group:
                    # Existing format
                    pert_group = config_group['perturbation_analysis']
                    if 'delta' in pert_group and 'delta_E' in pert_group:
                        delta_values = pert_group['delta'][:]
                        delta_E_values = pert_group['delta_E'][:]
                    else:
                        print(f"Required fields not found in perturbation_analysis for {config_name}")
                        return None
                else:
                    print(f"No suitable data structure found for {config_name}")
                    return None
                
                print(f"Analyzing configuration: {config_name}")
                print(f"  Data points: {len(delta_values)}")
                print(f"  Delta range: [{delta_values.min():.6f}, {delta_values.max():.6f}]")
                print(f"  Energy range: [{delta_E_values.min():.6e}, {delta_E_values.max():.6e}]")
                
                # Perform comprehensive analysis
                analysis_results = self.comprehensive_analysis(delta_values, delta_E_values)
                
                # Store results back to HDF5
                self._store_stats_to_hdf5(config_name, analysis_results)
                
                return analysis_results
                
        except Exception as e:
            print(f"Error analyzing configuration {config_name}: {e}")
            return None
    
    def _store_stats_to_hdf5(self, config_name, analysis_results):
        """Store comprehensive statistical results to HDF5 file."""
        try:
            with h5py.File(self.hdf5_file, 'a') as f:
                config_group = f[config_name]
                
                # Create or update statistics subgroup
                if 'statistics' in config_group:
                    del config_group['statistics']
                
                stats_group = config_group.create_group('statistics')
                
                # Store fitting results
                if 'fitting_results' in analysis_results:
                    fitting_group = stats_group.create_group('fitting_results')
                    for model_name, fit_result in analysis_results['fitting_results'].items():
                        if fit_result is not None:
                            model_group = fitting_group.create_group(model_name)
                            for key, value in fit_result.items():
                                if isinstance(value, (int, float, np.number)):
                                    model_group.create_dataset(key, data=value)
                                elif isinstance(value, np.ndarray):
                                    model_group.create_dataset(key, data=value)
                
                # Store hypothesis testing results
                if 'hypothesis_testing' in analysis_results:
                    hyp_group = stats_group.create_group('hypothesis_testing')
                    for test_name, test_result in analysis_results['hypothesis_testing'].items():
                        test_subgroup = hyp_group.create_group(test_name)
                        for key, value in test_result.items():
                            if isinstance(value, (int, float, bool, np.number)):
                                test_subgroup.create_dataset(key, data=value)
                            elif isinstance(value, str):
                                test_subgroup.create_dataset(key, data=value.encode('utf-8'))
                            elif isinstance(value, list):
                                test_subgroup.create_dataset(key, data=np.array(value))
                
                # Store bootstrap results
                if 'bootstrap_analysis' in analysis_results and analysis_results['bootstrap_analysis'] is not None:
                    bootstrap_group = stats_group.create_group('bootstrap_analysis')
                    bootstrap_stats = analysis_results['bootstrap_analysis']
                    for param_name, param_stats in bootstrap_stats.items():
                        if isinstance(param_stats, dict):
                            param_group = bootstrap_group.create_group(param_name)
                            for key, value in param_stats.items():
                                if isinstance(value, (int, float, np.number)):
                                    param_group.create_dataset(key, data=value)
                                elif isinstance(value, np.ndarray):
                                    param_group.create_dataset(key, data=value)
                        else:
                            bootstrap_group.create_dataset(param_name, data=param_stats)
                
                # Store timestamp and other metadata
                stats_group.create_dataset('analysis_time', data=analysis_results.get('analysis_time', 0))
                stats_group.create_dataset('timestamp', data=analysis_results.get('timestamp', '').encode('utf-8'))
                
        except Exception as e:
            print(f"Warning: Could not store statistics to HDF5: {e}")
    
    def print_detailed_report(self, analysis_results):
        """
        Print comprehensive statistical report to console.
        
        Args:
            analysis_results: Results from comprehensive_analysis()
        """
        print("\n" + "="*80)
        print("EXPERIMENT 1: STATISTICAL ANALYSIS REPORT")
        print("="*80)
        
        # Model fitting summary
        print("\nPOLYNOMIAL MODEL FITTING RESULTS:")
        print("-" * 40)
        
        fitting_results = analysis_results['fitting_results']
        for model_name, fit_result in fitting_results.items():
            if fit_result is None:
                print(f"  {model_name.upper()} MODEL: Failed to converge")
                continue
                
            print(f"  {model_name.upper()} MODEL:")
            print(f"    R² = {fit_result['r_squared']:.6f}")
            print(f"    AIC = {fit_result['aic']:.2f}")
            
            if model_name == 'quadratic':
                print(f"    C₁ = {fit_result['C1']:.6e} ± {fit_result['C1_stderr']:.2e}")
            elif model_name == 'cubic':
                print(f"    C₁ = {fit_result['C1']:.6e} ± {fit_result['C1_stderr']:.2e}")
                print(f"    C₂ = {fit_result['C2']:.6e} ± {fit_result['C2_stderr']:.2e}")
            elif model_name == 'quartic':
                print(f"    C₁ = {fit_result['C1']:.6e} ± {fit_result['C1_stderr']:.2e}")
                print(f"    C₂ = {fit_result['C2']:.6e} ± {fit_result['C2_stderr']:.2e}")
                print(f"    C₃ = {fit_result['C3']:.6e} ± {fit_result['C3_stderr']:.2e}")
        
        # Hypothesis testing summary
        print("\nHYPOTHESIS TESTING RESULTS:")
        print("-" * 30)
        
        hyp_tests = analysis_results['hypothesis_testing']
        
        if 'local_stability' in hyp_tests:
            stability = hyp_tests['local_stability']
            print(f"  LOCAL STABILITY TEST (C₁ > 0):")
            print(f"    t-statistic: {stability['test_statistic']:.4f}")
            print(f"    p-value: {stability['p_value']:.6f}")
            print(f"    Result: {'STABLE' if stability['significant'] else 'INCONCLUSIVE'}")
        
        if 'cubic_significance' in hyp_tests:
            cubic = hyp_tests['cubic_significance']
            print(f"  CUBIC TERM SIGNIFICANCE (C₂ ≠ 0):")
            print(f"    t-statistic: {cubic['test_statistic']:.4f}")
            print(f"    p-value: {cubic['p_value']:.6f}")
            print(f"    Result: {'SIGNIFICANT' if cubic['significant'] else 'NOT SIGNIFICANT'}")
        
        if 'bootstrap_stability' in hyp_tests:
            bootstrap = hyp_tests['bootstrap_stability']
            print(f"  BOOTSTRAP STABILITY TEST:")
            print(f"    Fraction C₁ > 0: {bootstrap['fraction_positive']:.4f}")
            print(f"    Bootstrap mean: {bootstrap['C1_bootstrap_mean']:.6e}")
            print(f"    95% CI: [{bootstrap['C1_bootstrap_ci'][0]:.6e}, {bootstrap['C1_bootstrap_ci'][1]:.6e}]")
            print(f"    Result: {'STABLE' if bootstrap['significant'] else 'INCONCLUSIVE'}")
        
        # Best model
        if 'best_model' in analysis_results and analysis_results['best_model'] is not None:
            best = analysis_results['best_model']
            print(f"\nBEST MODEL SELECTION:")
            print(f"  Selected: {best['model_name'].upper()}")
            print(f"  AIC: {best['aic']:.2f}")
            print(f"  R²: {best['r_squared']:.6f}")
        
        print("\n" + "="*80)
    
    def generate_summary_report(self, output_file):
        """Generate comprehensive summary report for all configurations."""
        try:
            with h5py.File(self.hdf5_file, 'r') as f:
                config_names = list(f.keys())
                
                with open(output_file, 'w') as report:
                    report.write("EXPERIMENT 1: COMPREHENSIVE STATISTICAL ANALYSIS SUMMARY\n")
                    report.write("=" * 70 + "\n\n")
                    report.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    report.write(f"Total configurations: {len(config_names)}\n\n")
                    
                    # Summary table header
                    report.write("CONFIGURATION SUMMARY\n")
                    report.write("-" * 50 + "\n")
                    report.write(f"{'Config':<20} {'C₁ (x10⁻⁶)':<15} {'C₂ (x10⁻⁹)':<15} {'R²':<10} {'Status':<15}\n")
                    report.write("-" * 75 + "\n")
                    
                    for config_name in config_names:
                        if 'statistics' in f[config_name]:
                            stats_group = f[config_name]['statistics']
                            
                            # Extract key results
                            if 'fitting_results' in stats_group and 'cubic' in stats_group['fitting_results']:
                                cubic_group = stats_group['fitting_results']['cubic']
                                C1 = float(cubic_group['C1'][()])
                                C2 = float(cubic_group['C2'][()])
                                R2 = float(cubic_group['r_squared'][()])
                                
                                # Check stability
                                status = "Unknown"
                                if 'hypothesis_testing' in stats_group and 'local_stability' in stats_group['hypothesis_testing']:
                                    stability_group = stats_group['hypothesis_testing']['local_stability']
                                    p_value = float(stability_group['p_value'][()])
                                    is_significant = p_value < 0.05
                                    status = "Stable" if is_significant else "Unstable"
                                
                                report.write(f"{config_name:<20} {C1*1e6:<15.3f} {C2*1e9:<15.3f} {R2:<10.6f} {status:<15}\n")
                            else:
                                report.write(f"{config_name:<20} {'N/A':<15} {'N/A':<15} {'N/A':<10} {'Failed':<15}\n")
                        else:
                            report.write(f"{config_name:<20} {'N/A':<15} {'N/A':<15} {'N/A':<10} {'No Stats':<15}\n")
                    
                    report.write("\n\nDETAILED ANALYSIS BY CONFIGURATION\n")
                    report.write("=" * 50 + "\n\n")
                    
                    for config_name in config_names:
                        report.write(f"Configuration: {config_name}\n")
                        report.write("-" * (len(config_name) + 15) + "\n")
                        
                        if 'statistics' in f[config_name]:
                            stats_group = f[config_name]['statistics']
                            
                            # Fitting results
                            if 'fitting_results' in stats_group:
                                report.write("Polynomial Fitting Results:\n")
                                fitting_group = stats_group['fitting_results']
                                
                                for model_name in ['quadratic', 'cubic', 'quartic']:
                                    if model_name in fitting_group:
                                        model_group = fitting_group[model_name]
                                        R2 = float(model_group['r_squared'][()])
                                        AIC = float(model_group['aic'][()])
                                        report.write(f"  {model_name.capitalize()}: R² = {R2:.6f}, AIC = {AIC:.2f}\n")
                                        
                                        if model_name == 'cubic':
                                            C1 = float(model_group['C1'][()])
                                            C2 = float(model_group['C2'][()])
                                            C1_err = float(model_group['C1_stderr'][()])
                                            C2_err = float(model_group['C2_stderr'][()])
                                            report.write(f"    C₁ = {C1:.6e} ± {C1_err:.2e}\n")
                                            report.write(f"    C₂ = {C2:.6e} ± {C2_err:.2e}\n")
                                
                                report.write("\n")
                            
                            # Hypothesis testing
                            if 'hypothesis_testing' in stats_group:
                                report.write("Hypothesis Testing:\n")
                                hyp_group = stats_group['hypothesis_testing']
                                
                                if 'local_stability' in hyp_group:
                                    stability_group = hyp_group['local_stability']
                                    t_stat = float(stability_group['test_statistic'][()])
                                    p_val = float(stability_group['p_value'][()])
                                    significant = p_val < 0.05
                                    
                                    report.write(f"  Local Stability (C₁ > 0): t = {t_stat:.4f}, p = {p_val:.6f}")
                                    report.write(f" -> {'STABLE' if significant else 'INCONCLUSIVE'}\n")
                                
                                if 'cubic_significance' in hyp_group:
                                    cubic_group = hyp_group['cubic_significance']
                                    t_stat = float(cubic_group['test_statistic'][()])
                                    p_val = float(cubic_group['p_value'][()])
                                    significant = p_val < 0.05
                                    
                                    report.write(f"  Cubic Term (C₂ ≠ 0): t = {t_stat:.4f}, p = {p_val:.6f}")
                                    report.write(f" -> {'SIGNIFICANT' if significant else 'NOT SIGNIFICANT'}\n")
                                
                                report.write("\n")
                        else:
                            report.write("  No statistical analysis available\n\n")
                        
                        report.write("\n")
                    
                    print(f"Comprehensive summary report written to: {output_file}")
                        
        except Exception as e:
            print(f"Error generating summary report: {e}")
    
    def analyze_all_configurations(self):
        """Analyze all configurations in the HDF5 file."""
        try:
            with h5py.File(self.hdf5_file, 'r') as f:
                config_names = list(f.keys())
                
            print(f"Found {len(config_names)} configurations to analyze")
            
            for i, config_name in enumerate(config_names):
                print(f"\n[{i+1}/{len(config_names)}] Processing {config_name}...")
                self.analyze_configuration(config_name)
            
            print(f"\nCompleted analysis of all {len(config_names)} configurations")
            
        except Exception as e:
            print(f"Error analyzing all configurations: {e}")
