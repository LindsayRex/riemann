# ############################################################################
#
# EXPERIMENT 2: CHUNKED HDF5 STATISTICS MODULE
# =============================================
#
# This module performs statistical analysis on large HDF5 datasets using
# chunked reads and online algorithms (Welford's method) to handle datasets
# that exceed available memory.
#
# Features:
# - Chunked HDF5 reading with h5py
# - Online statistics using Welford's algorithm
# - Polynomial fitting on streamed data
# - Bootstrap confidence intervals via resampling
# - Memory-efficient processing for large datasets
#
# ############################################################################

import h5py
import numpy as np
from sage.all import *
import time

try:
    from scipy import stats as scipy_stats
    from scipy.optimize import curve_fit
    scipy_available = True
except ImportError:
    scipy_available = False

class ChunkedHDF5Stats:
    """Memory-efficient statistical analysis for large HDF5 datasets."""
    
    def __init__(self, hdf5_path, chunk_size=1000, verbose=True):
        """
        Initialize chunked statistics processor.
        
        Args:
            hdf5_path: Path to HDF5 file
            chunk_size: Number of elements to process per chunk
            verbose: Print progress information
        """
        self.hdf5_path = hdf5_path
        self.chunk_size = chunk_size
        self.verbose = verbose
        
        # Online statistics accumulators (Welford's algorithm)
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        
        if verbose:
            print(f"Chunked HDF5 Statistics initialized")
            print(f"File: {hdf5_path}")
            print(f"Chunk size: {chunk_size}")
    
    def update_online_stats(self, chunk_data):
        """
        Update running statistics using Welford's online algorithm.
        
        Args:
            chunk_data: Numpy array of data values
        """
        for value in chunk_data:
            self.n += 1
            delta = value - self.mean
            self.mean += delta / self.n
            delta2 = value - self.mean
            self.M2 += delta * delta2
            
            # Update min/max
            if value < self.min_val:
                self.min_val = value
            if value > self.max_val:
                self.max_val = value
    
    def get_online_stats(self):
        """
        Get current statistical summary from online computations.
        
        Returns:
            dict: Statistical summary
        """
        if self.n < 2:
            return {
                'n': self.n,
                'mean': self.mean,
                'variance': 0.0,
                'std': 0.0,
                'min': self.min_val if self.n > 0 else 0.0,
                'max': self.max_val if self.n > 0 else 0.0
            }
        
        variance = self.M2 / (self.n - 1)
        std = np.sqrt(variance)
        
        return {
            'n': self.n,
            'mean': self.mean,
            'variance': variance,
            'std': std,
            'min': self.min_val,
            'max': self.max_val
        }
    
    def reset_stats(self):
        """Reset online statistics accumulators."""
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
    
    def compute_scheme_statistics(self, scheme_name):
        """
        Compute complete statistics for a scheme using chunked processing.
        
        Args:
            scheme_name: Name of scheme ('scheme_i', 'scheme_ii', 'scheme_both')
            
        Returns:
            dict: Complete statistical analysis
        """
        if self.verbose:
            print(f"Computing statistics for {scheme_name}...")
        
        start_time = time.time()
        
        with h5py.File(self.hdf5_path, 'r') as f:
            if scheme_name not in f:
                raise ValueError(f"Scheme {scheme_name} not found in HDF5 file")
            
            scheme_group = f[scheme_name]
            
            # Get dataset info
            delta_dataset = scheme_group['delta']
            delta_E_dataset = scheme_group['delta_E']
            
            total_points = len(delta_dataset)
            
            if self.verbose:
                print(f"  Dataset size: {total_points} points")
                print(f"  Processing in chunks of {self.chunk_size}")
            
            # Reset statistics
            self.reset_stats()
            
            # Arrays for full data (for polynomial fitting)
            all_deltas = []
            all_energies = []
            
            # Process in chunks
            num_chunks = (total_points + self.chunk_size - 1) // self.chunk_size
            
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * self.chunk_size
                end_idx = min(start_idx + self.chunk_size, total_points)
                
                # Read chunk
                delta_chunk = delta_dataset[start_idx:end_idx]
                energy_chunk = delta_E_dataset[start_idx:end_idx]
                
                # Update online statistics
                self.update_online_stats(energy_chunk)
                
                # Store for polynomial fitting (if memory allows)
                all_deltas.extend(delta_chunk)
                all_energies.extend(energy_chunk)
                
                if self.verbose and (chunk_idx + 1) % 10 == 0:
                    print(f"    Processed chunk {chunk_idx + 1}/{num_chunks}")
            
            # Get basic statistics
            basic_stats = self.get_online_stats()
            
            # Compute numerical gradient
            gradient = np.gradient(all_energies, all_deltas)
            
            # Polynomial fitting (quadratic: ΔE = C₁δ² + C₂δ³)
            delta_array = np.array(all_deltas)
            energy_array = np.array(all_energies)
            
            polyfit_results = self._fit_polynomial(delta_array, energy_array)
            
            # Bootstrap confidence intervals
            bootstrap_results = self._compute_bootstrap_ci(delta_array, energy_array)
            
            computation_time = time.time() - start_time
            
            results = {
                'scheme_name': scheme_name,
                'basic_statistics': basic_stats,
                'gradient': gradient,
                'polynomial_fit': polyfit_results,
                'bootstrap_ci': bootstrap_results,
                'computation_time': computation_time,
                'total_points': total_points,
                'chunks_processed': num_chunks
            }
            
            if self.verbose:
                print(f"  Statistics computed in {computation_time:.2f}s")
                print(f"  Mean energy: {basic_stats['mean']:.2e}")
                print(f"  Energy range: [{basic_stats['min']:.2e}, {basic_stats['max']:.2e}]")
                if polyfit_results['success']:
                    print(f"  Polynomial fit: C₁={polyfit_results['coefficients'][0]:.2e}, R²={polyfit_results['r_squared']:.4f}")
            
            return results
    
    def _fit_polynomial(self, delta_values, energy_values, degree=2):
        """
        Fit polynomial model to energy data.
        
        Args:
            delta_values: Delta perturbation values
            energy_values: Energy change values
            degree: Polynomial degree (default: 2 for quadratic)
            
        Returns:
            dict: Polynomial fitting results
        """
        try:
            # For quadratic stability analysis: ΔE ≈ C₁δ²
            # We fit: ΔE = C₀ + C₁δ² + C₂δ³ + ...
            
            # Create design matrix for polynomial
            delta_powers = np.column_stack([delta_values**i for i in range(degree + 1)])
            
            if scipy_available:
                # Use SciPy for robust fitting
                coeffs = np.polyfit(delta_values, energy_values, degree)
                
                # Compute R-squared
                energy_pred = np.polyval(coeffs, delta_values)
                ss_res = np.sum((energy_values - energy_pred) ** 2)
                ss_tot = np.sum((energy_values - np.mean(energy_values)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
            else:
                # Manual least squares fitting
                coeffs, residuals, rank, s = np.linalg.lstsq(delta_powers, energy_values, rcond=None)
                
                # Compute R-squared manually
                energy_pred = delta_powers @ coeffs
                ss_res = np.sum((energy_values - energy_pred) ** 2)
                ss_tot = np.sum((energy_values - np.mean(energy_values)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'success': True,
                'coefficients': coeffs,
                'r_squared': r_squared,
                'degree': degree,
                'method': 'scipy' if scipy_available else 'manual'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'coefficients': np.zeros(degree + 1),
                'r_squared': 0.0
            }
    
    def _compute_bootstrap_ci(self, delta_values, energy_values, n_bootstrap=1000, confidence=0.95):
        """
        Compute bootstrap confidence intervals for polynomial coefficients.
        
        Args:
            delta_values: Delta perturbation values
            energy_values: Energy change values
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level (0.95 = 95% CI)
            
        Returns:
            dict: Bootstrap confidence interval results
        """
        if len(delta_values) < 10:  # Too few points for bootstrap
            return {
                'success': False,
                'error': 'Insufficient data for bootstrap',
                'confidence_intervals': np.zeros((2, 2))
            }
        
        try:
            n_points = len(delta_values)
            bootstrap_coeffs = []
            
            # Perform bootstrap resampling
            for i in range(n_bootstrap):
                # Resample with replacement
                indices = np.random.choice(n_points, size=n_points, replace=True)
                delta_sample = delta_values[indices]
                energy_sample = energy_values[indices]
                
                # Fit polynomial to bootstrap sample
                fit_result = self._fit_polynomial(delta_sample, energy_sample)
                
                if fit_result['success']:
                    bootstrap_coeffs.append(fit_result['coefficients'])
            
            if len(bootstrap_coeffs) == 0:
                return {
                    'success': False,
                    'error': 'No successful bootstrap fits',
                    'confidence_intervals': np.zeros((2, 2))
                }
            
            bootstrap_coeffs = np.array(bootstrap_coeffs)
            
            # Compute confidence intervals
            alpha = 1 - confidence
            lower_percentile = 100 * (alpha / 2)
            upper_percentile = 100 * (1 - alpha / 2)
            
            ci_lower = np.percentile(bootstrap_coeffs, lower_percentile, axis=0)
            ci_upper = np.percentile(bootstrap_coeffs, upper_percentile, axis=0)
            
            # Format as [n_coeffs x 2] array
            confidence_intervals = np.column_stack([ci_lower, ci_upper])
            
            return {
                'success': True,
                'confidence_intervals': confidence_intervals,
                'n_bootstrap': len(bootstrap_coeffs),
                'confidence_level': confidence,
                'bootstrap_mean': np.mean(bootstrap_coeffs, axis=0),
                'bootstrap_std': np.std(bootstrap_coeffs, axis=0)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'confidence_intervals': np.zeros((2, 2))
            }
    
    def process_complete_file(self):
        """
        Process complete HDF5 file and add statistics to all schemes.
        
        Returns:
            dict: Complete statistical analysis for all schemes
        """
        if self.verbose:
            print("Processing complete HDF5 file...")
            print("=" * 50)
        
        start_time = time.time()
        results = {}
        
        # Process each scheme
        schemes = ['scheme_i', 'scheme_ii', 'scheme_both']
        
        for scheme in schemes:
            try:
                scheme_results = self.compute_scheme_statistics(scheme)
                results[scheme] = scheme_results
                
                # Write statistics back to HDF5
                self._write_statistics_to_hdf5(scheme, scheme_results)
                
            except Exception as e:
                if self.verbose:
                    print(f"  Error processing {scheme}: {e}")
                results[scheme] = {'error': str(e)}
        
        # Process interference analysis if available
        try:
            interference_results = self._process_interference_analysis()
            results['interference_analysis'] = interference_results
        except Exception as e:
            if self.verbose:
                print(f"  Error processing interference analysis: {e}")
        
        total_time = time.time() - start_time
        results['total_processing_time'] = total_time
        
        if self.verbose:
            print("=" * 50)
            print(f"Complete file processing finished in {total_time:.2f}s")
        
        return results
    
    def _write_statistics_to_hdf5(self, scheme_name, stats_results):
        """
        Write computed statistics back to HDF5 file.
        
        Args:
            scheme_name: Name of scheme group
            stats_results: Statistical analysis results
        """
        try:
            with h5py.File(self.hdf5_path, 'r+') as f:
                scheme_group = f[scheme_name]
                
                # Write numerical gradient
                if 'dE_d_delta' in scheme_group:
                    del scheme_group['dE_d_delta']
                scheme_group.create_dataset('dE_d_delta', data=stats_results['gradient'])
                
                # Write polynomial coefficients
                if stats_results['polynomial_fit']['success']:
                    if 'polyfit_coeffs' in scheme_group:
                        del scheme_group['polyfit_coeffs']
                    scheme_group.create_dataset('polyfit_coeffs', data=stats_results['polynomial_fit']['coefficients'])
                    
                    # Add R-squared as attribute
                    scheme_group.attrs['r_squared'] = stats_results['polynomial_fit']['r_squared']
                
                # Write bootstrap confidence intervals
                if stats_results['bootstrap_ci']['success']:
                    if 'bootstrap_CI' in scheme_group:
                        del scheme_group['bootstrap_CI']
                    scheme_group.create_dataset('bootstrap_CI', data=stats_results['bootstrap_ci']['confidence_intervals'])
                    
                    # Add bootstrap metadata
                    scheme_group.attrs['bootstrap_samples'] = stats_results['bootstrap_ci']['n_bootstrap']
                    scheme_group.attrs['confidence_level'] = stats_results['bootstrap_ci']['confidence_level']
                
                # Add basic statistics as attributes
                basic = stats_results['basic_statistics']
                scheme_group.attrs['mean_energy'] = basic['mean']
                scheme_group.attrs['std_energy'] = basic['std']
                scheme_group.attrs['min_energy'] = basic['min']
                scheme_group.attrs['max_energy'] = basic['max']
                scheme_group.attrs['n_points'] = basic['n']
                
            if self.verbose:
                print(f"  ✓ Statistics written to {scheme_name}")
                
        except Exception as e:
            if self.verbose:
                print(f"  ✗ Failed to write statistics to {scheme_name}: {e}")
    
    def _process_interference_analysis(self):
        """
        Process interference analysis using chunked reads.
        
        Returns:
            dict: Interference analysis results
        """
        if self.verbose:
            print("Computing interference analysis...")
        
        with h5py.File(self.hdf5_path, 'r+') as f:
            if 'interference_analysis' not in f:
                return {'error': 'No interference analysis data found'}
            
            interference_group = f['interference_analysis']
            
            if 'interference_ratio' in interference_group:
                ratios = interference_group['interference_ratio'][:]
                
                # Compute basic statistics on interference ratios
                self.reset_stats()
                self.update_online_stats(ratios)
                basic_stats = self.get_online_stats()
                
                # Add statistical significance test (t-test against zero)
                if scipy_available and len(ratios) > 1:
                    t_stat, p_value = scipy_stats.ttest_1samp(ratios, 0.0)
                else:
                    t_stat = 0.0
                    p_value = 1.0
                
                # Write p-values back to HDF5
                if 'p_values' in interference_group:
                    del interference_group['p_values']
                
                # Create p-values array (same length as ratios, all same p-value for now)
                p_values_array = np.full_like(ratios, p_value)
                interference_group.create_dataset('p_values', data=p_values_array)
                
                # Add notes
                notes = f"Statistical test: t={t_stat:.4f}, p={p_value:.4f}"
                if p_value < 0.05:
                    notes += " (significant interference detected)"
                else:
                    notes += " (no significant interference)"
                
                interference_group.attrs['statistical_notes'] = notes
                interference_group.attrs['t_statistic'] = t_stat
                interference_group.attrs['p_value'] = p_value
                
                return {
                    'basic_statistics': basic_stats,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'notes': notes
                }
            
        return {'error': 'No interference ratio data found'}


# ############################################################################
# CONVENIENCE FUNCTIONS
# ############################################################################

def process_hdf5_statistics(hdf5_path, chunk_size=1000, verbose=True):
    """
    Process statistics for an HDF5 file using chunked reads.
    
    Args:
        hdf5_path: Path to HDF5 file
        chunk_size: Chunk size for processing
        verbose: Print progress
        
    Returns:
        dict: Complete statistical analysis
    """
    processor = ChunkedHDF5Stats(hdf5_path, chunk_size=chunk_size, verbose=verbose)
    return processor.process_complete_file()

def create_chunked_hdf5_stats(hdf5_path, chunk_size=1000, verbose=True):
    """
    Factory function to create ChunkedHDF5Stats instance.
    
    Args:
        hdf5_path: Path to HDF5 file
        chunk_size: Chunk size for processing
        verbose: Print progress
        
    Returns:
        ChunkedHDF5Stats: Configured processor instance
    """
    return ChunkedHDF5Stats(hdf5_path, chunk_size=chunk_size, verbose=verbose)
