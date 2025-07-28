"""
Created on Wed Oct 30 15:30:51 2024

@author: Goltin
"""

import customtkinter as ctk
import pandas as pd
import tkinter as tk
import numpy as np
from scipy.optimize import least_squares
from typing import Callable, Tuple, List, Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from tkinter import filedialog,messagebox
from io import BytesIO
import itertools
import threading
import csv
from datetime import datetime

# === Simple Helper Functions ===

"""
    Detects which common delimiter (tab, semicolon, comma, pipe) is used in the first line of a file.
    Returns the first found delimiter. This is useful for robustly importing text-based calibration or measurement data.
"""

def detect_delimiter(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        for delimiter in ['\t', ';', ',', '|']:
            if delimiter in first_line:
                return delimiter
    return '\t'

"""
    Converts various special or non-ASCII characters into simple ASCII alternatives.
    This prevents encoding issues and increases portability of labels, outputs, and file names.
"""

def sanitize_text(text: str) -> str:
    replacements = {
        '–': '-', '—': '-', '…': '...'
    }
    for orig, replacement in replacements.items():
        text = text.replace(orig, replacement)
    return text

"""
    Checks if the calibration dataset is sufficiently large for stable regression.
    Enforces the minimal requirement: (2 × number of parameters - 1) ≤ number of data points.
    This avoids overfitting and ensures mathematically meaningful model estimates.
    Raises ValueError if this condition is not met.
"""

def check_data_sufficiency(cal_data: np.ndarray, num_parameters: int):
    num_points = cal_data.shape[0]
    if (2 * num_parameters - 1) > num_points:
        raise ValueError(
            f"Insufficient calibration data: For a stable parameter estimation, the following must hold:\n"
            f"(2 × number of parameters - 1) ≤ number of data points\n"
            f"Current: (2 × {num_parameters} - 1 = {2 * num_parameters - 1}) ≥ {num_points}"
        )
        
"""
    Builds the covariance matrix for the x-data, initially diagonal (from ux), 
    but allows insertion of manual off-diagonal covariances between specific data points.
    Essential for correct error propagation in cases of correlated calibration points.
"""

def build_sigma_x(ux: np.ndarray,
                  extra: Optional[List[Tuple[int, int, float]]] = None) -> np.ndarray:
    Σx = np.diag(ux**2)
    if extra:
        for i, j, cov in extra:
            Σx[i, j] += cov
            Σx[j, i] += cov
    return Σx

"""
    Converts a user-supplied string like '1/2 0.05, 2/3 -0.01' into a list of 
    index-pair/covariance tuples for manual error correlation input.
    Intended for expert use where known systematic dependencies exist in calibration data.
"""

def parse_manual_covariances(input_str: str) -> List[Tuple[int, int, float]]:
    covariances = []
    if not input_str.strip():
        return covariances

    entries = input_str.split(",")
    for entry in entries:
        try:
            pair_part, value_part = entry.strip().split()
            i_str, j_str = pair_part.strip().split("/")
            i, j = int(i_str) - 1, int(j_str) - 1
            cov_value = float(value_part.strip())
            covariances.append((i, j, cov_value))
        except Exception as e:
            raise ValueError(f"Invalid covariance entry format: '{entry}'. Expected format: 'i/j value'") from e
    
    return covariances

# === Model Functions ===

"""
    Calculates the value, derivative, and parameter partials of a polynomial model for given inputs.
    Used for standard polynomial calibration, e.g. linear, quadratic, cubic models.
"""

def compute_polynomial(y: np.ndarray, coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    x_calc = sum([coeff * y**i for i, coeff in enumerate(coeffs)])
    dx_dy = sum([i * coeffs[i] * y**(i-1) for i in range(1, len(coeffs))])
    partials = [y**i for i in range(len(coeffs))]
    return x_calc, dx_dy, partials

"""
    Standard exponential regression model: x = b0 + b1 * exp(b2 * y).
    Widely used for exponential growth/decay calibration or sensor transfer functions.
    Computes value, derivative, and parameter partials.
"""

def compute_exponential(y, b):
    exp_term = np.exp(np.clip(b[2] * y, -700, 700))
    x = b[0] + b[1] * exp_term
    dx_dy = b[1] * b[2] * exp_term
    partials = [
        np.ones_like(y),
        exp_term,
        b[1] * y * exp_term
    ]
    return x, dx_dy, partials

"""
    Exponential model with offset: x = b0 + b1 * (exp(b2 * y) - 1).
    Useful for systems with nonlinear baseline and gain, but zero at y=0.
"""

def compute_exponential_offset(y, b):
    exp_term = np.exp(np.clip(b[2] * y, -700, 700))
    x = b[0] + b[1] * (exp_term - 1)
    dx_dy = b[1] * b[2] * exp_term
    partials = [
        np.ones_like(y),
        exp_term - 1,
        b[1] * y * exp_term
    ]
    return x, dx_dy, partials

"""
    Power-law calibration model: x = b0 + b1 * y^(1 + b2).
    Suitable for sensor characteristics with nonlinear but monotonic response.
"""

def compute_power(y: np.ndarray, coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    b0, b1, b2 = coeffs
    y_safe = np.where(y <= 0, 1e-10, y)
    exponent = 1 + b2
    y_exp = np.power(y_safe, exponent)
    x_calc = b0 + b1 * y_exp
    dx_dy = b1 * exponent * np.power(y_safe, exponent - 1)
    partials = [
        np.ones_like(y),
        y_exp,
        b1 * y_exp * np.log(y_safe)
    ]
    return x_calc, dx_dy, partials

"""
    Rational function model: x = (a + b*y) / (1 + c*y).
    Useful for sensor or process models with asymptotic behavior.
"""

def compute_rational(y: np.ndarray, coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    a, b, c = coeffs
    denom = 1 + c * y
    denom_safe = np.where(denom == 0, 1e-10, denom)
    x_calc = (a + b * y) / denom_safe
    dx_dy = (b * denom_safe - (a + b * y) * c) / denom_safe**2
    partials = [
        1 / denom_safe,
        y / denom_safe,
        -(a + b * y) * y / denom_safe**2
    ]
    return x_calc, dx_dy, partials

"""
    Logarithmic model: x = a + b * ln(c * y).
    Applied in many fields for calibration curves with fast initial response and flattening at high y.
"""

def compute_logarithmic(y: np.ndarray, coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    a, b, c = coeffs
    y_safe = np.clip(y, 1e-10, None)
    ln_term = np.log(np.clip(c * y_safe, 1e-10, None))
    x_calc = a + b * ln_term
    dx_dy = b * c / (c * y_safe)
    partials = [
        np.ones_like(y),
        ln_term,
        b * y_safe / (c * y_safe)
    ]
    return x_calc, dx_dy, partials

"""
    Hyperbolic model: x = a + b / (y + c).
    Characteristic for some sensor saturation phenomena and transfer curves.
"""

def compute_hyperbolic(y: np.ndarray, coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    a, b, c = coeffs
    y_safe = np.where((y + c) == 0, 1e-10, y + c)
    x_calc = a + b / y_safe
    dx_dy = -b / (y_safe ** 2)
    partials = [
        np.ones_like(y),
        1 / y_safe,
        -b / (y_safe ** 2)
    ]
    return x_calc, dx_dy, partials

"""
    Mixed log-linear model: x = a + b * ln(y) + c * y.
    Allows both logarithmic and linear response features, flexible for diverse calibration needs.
"""

def compute_log_linear(y: np.ndarray, coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    a, b, c = coeffs
    y_safe = np.clip(y, 1e-10, None)
    log_term = np.log(y_safe)
    x_calc = a + b * log_term + c * y_safe
    dx_dy = b / y_safe + c
    partials = [
        np.ones_like(y),
        log_term,
        y_safe
    ]
    return x_calc, dx_dy, partials

"""
    Gompertz function: x = a * exp(-b * exp(-c * y)).
    Models sigmoidal growth/decay, commonly used in biology and aging studies.
"""

def compute_gompertz(y: np.ndarray, coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    a, b, c = coeffs
    exp_term = np.exp(-c * y)
    inner = b * exp_term
    x_calc = a * np.exp(-inner)
    dx_dy = a * inner * c * exp_term * np.exp(-inner)
    partials = [
        np.exp(-inner),
        -a * exp_term * np.exp(-inner),
        a * b * y * exp_term**2 * np.exp(-inner)
    ]
    return x_calc, dx_dy, partials

"""
    Square-root model: x = a + b * sqrt(y).
    Describes sublinear (e.g. diffusion-limited) effects, e.g. certain detector or optical systems.
"""

def compute_square_root(y: np.ndarray, coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    a, b = coeffs
    y_safe = np.clip(y, 0, None)
    sqrt_y = np.sqrt(y_safe)
    x_calc = a + b * sqrt_y
    dx_dy = b / (2 * sqrt_y + 1e-10)
    partials = [
        np.ones_like(y),
        sqrt_y
    ]
    return x_calc, dx_dy, partials

"""
    Logistic sigmoid model: x = L / (1 + exp(-k * (y - y0))).
    Classic S-shaped calibration curve for systems with both lower and upper saturation.
"""

def compute_sigmoid(y: np.ndarray, coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    L, k, y0 = coeffs
    exp_term = np.exp(np.clip(-k * (y - y0), -700, 700))
    denom = 1 + exp_term
    x_calc = L / denom
    dx_dy = (L * k * exp_term) / (denom ** 2)
    partials = [
        1 / denom,
        (L * (y - y0) * exp_term) / (denom**2),
        (-L * k * exp_term) / (denom**2)
    ]
    return x_calc, dx_dy, partials

"""
    Simple exponential: x = b0 * exp(b1 * y).
    Used for monotonic exponential scaling, e.g. in radiometric or chemical systems.
"""

def compute_simple_exponential(y: np.ndarray, coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    b0, b1 = coeffs
    exp_term = np.exp(np.clip(b1 * y, -700, 700))
    x_calc = b0 * exp_term
    dx_dy = b0 * b1 * exp_term
    partials = [
        exp_term,
        b0 * y * exp_term
    ]
    return x_calc, dx_dy, partials

# === Parameter Initialization / Initial Guesses ===

"""
    Generates starting parameter values for various nonlinear models using simple
    heuristics or linearizations. This ensures robust convergence of nonlinear fits.
    Handles all supported model types and provides fallback defaults.
"""

def estimate_initial_params(y: np.ndarray, x: np.ndarray, func: Callable, order: int = 2) -> np.ndarray:
    if func == compute_polynomial:
        return np.polyfit(y, x, order)[::-1]
    elif func == compute_exponential:
        x_safe = np.clip(x, 1e-6, np.inf)
        y_range = np.ptp(y) if np.ptp(y) != 0 else 1.0
        a = np.min(x_safe) * 0.8
        b = (np.max(x_safe) - a)
        c = np.log((np.max(x_safe) - a + 1e-6) / (np.min(x_safe) - a + 1e-6)) / (y_range + 1e-6)
        return np.array([a, b, c])
    elif func == compute_exponential_offset:
        x_safe = np.clip(x, 1e-6, np.inf)
        y_safe = np.clip(y, -np.inf, np.inf)
        y_range = np.ptp(y_safe) if np.ptp(y_safe) != 0 else 1.0
        a = np.min(x_safe)
        c = 1.0 / y_range
        exp_cy = np.exp(c * y_safe)
        b = (np.max(x_safe) - a) / (np.max(exp_cy - 1) + 1e-6)
        return np.array([a, b, c])
    elif func == compute_power:
        x_safe = np.clip(x, 1e-6, np.inf)
        y_safe = np.clip(y, 1e-6, np.inf)
        b0 = np.min(x_safe)
        b1 = (np.max(x_safe) - b0) / np.max(y_safe)
        b2 = 0.0
        return np.array([b0, b1, b2])
    elif func == compute_rational:
        a = np.min(x)
        b = (np.max(x) - a) / (np.max(y) + 1e-6)
        c = 0.01
        return np.array([a, b, c])
    elif func == compute_logarithmic:
        y_safe = np.clip(y, 1e-3, None)
        x_safe = np.clip(x, 1e-3, None)
        ln_y = np.log(y_safe)
        b, a = np.polyfit(ln_y, x_safe, 1)
        c = 1.0
        return np.array([a, b, c])
    elif func == compute_hyperbolic:
        y_safe = np.where(y == 0, 1e-6, y)
        inv_y = 1 / y_safe
        b, a = np.polyfit(inv_y, x, 1)
        c = 0.0
        return np.array([a, b, c])
    elif func == compute_log_linear:
        y_safe = np.clip(y, 1e-6, None)
        log_y = np.log(y_safe)
        A = np.vstack([np.ones_like(y), log_y, y_safe]).T
        coeffs, *_ = np.linalg.lstsq(A, x, rcond=None)
        return coeffs
    elif func == compute_gompertz:
        a = np.max(x)
        b = 1.0
        c = 1.0 / (np.max(y) - np.min(y) + 1e-6)
        return np.array([a, b, c])
    elif func == compute_square_root:
        sqrt_y = np.sqrt(np.clip(y, 1e-6, None))
        b, a = np.polyfit(sqrt_y, x, 1)
        return np.array([a, b])
    elif func == compute_sigmoid:
        L = np.max(x)
        k = 1.0 / (np.ptp(y) + 1e-6)
        y0 = np.median(y)
        return np.array([L, k, y0])
    elif func == compute_simple_exponential:
        x_safe = np.clip(x, 1e-6, np.inf)
        y_safe = np.clip(y, 1.0, np.inf)
        y_mean = np.mean(y_safe)
        y_centered = y_safe - y_mean
        log_x = np.log(x_safe)
        slope, intercept = np.polyfit(y_centered, log_x, 1)
        b1 = slope
        b0 = np.exp(intercept - b1 * y_mean)
        b0 = max(b0, 1e-8)
        b1 = np.clip(b1, 1e-8, 1e-3)
        return np.array([b0, b1])
    else:
        raise ValueError("Unknown function")

# === Fitting, Optimization, and Uncertainty Propagation ===

"""
    Computes the normalized and whitened residuals (using the covariance structure)
    for nonlinear least-squares optimization.
    Allows inclusion of off-diagonal manual covariances for correlated calibration data.
    Used internally in the core model fitting routines.
"""

def residuals_general(params_scaled, scale, cal_data, func, extra_cov=None):
    n = cal_data.shape[0]
    params = params_scaled * scale
    y_fit   = params[:n]
    coeffs  = params[n:]
    x, ux, y, uy = cal_data[:, 0], cal_data[:, 1], cal_data[:, 2], cal_data[:, 3]
    x_pred, dx_dy, _ = func(y_fit, coeffs)
    Σx = build_sigma_x(ux)
    if extra_cov:
        for i, j, cov_x in extra_cov:
            Σx[i, j] += cov_x
            Σx[j, i] += cov_x
    Lx       = np.linalg.cholesky(Σx)
    res_x    = np.linalg.solve(Lx, x_pred - x)
    res_y = (y_fit - y) / uy
    return np.concatenate([res_x, res_y])

"""
    Calculates the full covariance matrix of the fitted model parameters,
    considering both measurement uncertainties and (if present) additional covariances.
    This matrix is crucial for uncertainty propagation to predictions.
"""

def covariance_matrix(cal_data, y_hat, coeffs, func, extra_cov=None):
    x, ux, y, uy = cal_data.T
    _, dx_dy, partials = func(y_hat, coeffs)
    u_comb = np.sqrt(ux**2 + (dx_dy * uy)**2)
    A0 = np.vstack(partials).T / u_comb[:, None]
    B0 = np.diag(ux / u_comb)
    C0 = np.diag(dx_dy * uy / u_comb)
    if extra_cov:
        Σx = build_sigma_x(ux, extra_cov)
        L  = np.linalg.cholesky(Σx)
        Linv = np.linalg.inv(L)
        A = Linv @ A0
        B = Linv @ B0
        C = Linv @ C0
    else:
        A, B, C = A0, B0, C0
    ZZt = B @ B.T + C @ C.T
    AtA_inv = np.linalg.inv(A.T @ A)
    Σ_b = AtA_inv @ (A.T @ ZZt @ A) @ AtA_inv
    return Σ_b

"""
    The main nonlinear fitting routine for arbitrary calibration models.
    Handles arbitrary covariance structures and supports constraints
    via parameter scaling and robust initialization.
    Returns fitted parameters, their covariance, and fit residuals.
"""

def b_least_general(
    cal_data: np.ndarray,
    func: Callable,
    order: int = 1,
    extra_cov: Optional[List[Tuple[int, int, float]]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, ux, y, uy = cal_data.T
    n = len(x)
    coeffs0 = estimate_initial_params(y, x, func, order)
    check_data_sufficiency(cal_data, num_parameters=len(coeffs0))
    params0        = np.concatenate([y.copy(), coeffs0])
    scale          = np.where(params0 == 0, 1.0, params0)
    params0_scaled = params0 / scale
    res = least_squares(
        residuals_general,
        params0_scaled,
        args=(scale, cal_data, func, None),
        method='lm'
    )
    params_opt = res.x * scale
    y_opt      = params_opt[:n]
    coeffs_opt = params_opt[n:]
    residuals_all = residuals_general(
        res.x, scale, cal_data, func, extra_cov)
    cov_matrix = covariance_matrix(
        cal_data,
        y_opt,
        coeffs_opt,
        func,
        extra_cov=extra_cov)
    return coeffs_opt, cov_matrix, residuals_all

"""
    High-level fitting interface: wraps the general fit for a given model type.
    Forwards to b_least_general and returns the same outputs.
"""

def model_fit(cal_data: np.ndarray, func: Callable, order: int = 2, extra_cov=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return b_least_general(cal_data, func, order, extra_cov)

"""
    Applies the fitted model to new measurement data and propagates all uncertainties
    (including those from model calibration) to the prediction results.
    This function enables proper error estimation for arbitrary user measurement scenarios.
"""

def evaluate_model(measurement_data, coeffs, coeffs_cov, func):
    measurement_data = np.atleast_2d(measurement_data)
    y, uy = measurement_data[:, 0], measurement_data[:, 1]
    x_pred, dx_dy, partials = func(y, coeffs)
    dx_dy = dx_dy.flatten()
    jacobian = np.column_stack((np.diag(dx_dy), np.column_stack(partials)))
    y_cov_matrix = np.diag(uy ** 2)
    ny, nb = y.size, coeffs.size
    cov_matrix_input = np.zeros((ny + nb, ny + nb))
    cov_matrix_input[:ny, :ny] = y_cov_matrix
    cov_matrix_input[ny:, ny:] = coeffs_cov
    x_cov_matrix = jacobian @ cov_matrix_input @ jacobian.T
    return x_pred, x_cov_matrix

"""
    End-to-end calibration and evaluation routine. Fits the model to the calibration data,
    then evaluates it for new measurements, returning all relevant outputs
    (parameters, covariances, residuals, predictions, and their uncertainties).
    Intended as the main workflow function for automated or interactive calibration analysis.
"""

def run_analysis(cal_data: np.ndarray, meas_data: np.ndarray, func: Callable, order: int = 2, extra_cov=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    params, cov_matrix, residuals = model_fit(cal_data, func, order, extra_cov)
    x_pred, x_cov = evaluate_model(meas_data, params, cov_matrix, func)
    return params, cov_matrix, residuals, x_pred, x_cov

# === Display and Result Formatting Functions ===

def display_cal_data(cal_data: np.ndarray, manual_covariances: List[Tuple[int, int, float]] = None) -> str:
    header = "x\t|\tu(x)\t|\ty\t|\tu(y)\n"
    data_lines = "\n".join(["\t|\t".join(map(str, row)) for row in cal_data])
    result = header + data_lines
    if manual_covariances:
        result += "\n\nManual Covariances Between Calibration Points:\n"
        result += "Point i\t|\tPoint j\t|\tCovariance\n"
        for i, j, cov in manual_covariances:
            if 0 <= i < len(cal_data) and 0 <= j < len(cal_data):
                x_i = cal_data[i, 0]
                x_j = cal_data[j, 0]
                result += f"{i+1} ({x_i:.4g})\t|\t{j+1} ({x_j:.4g})\t|\t{cov:.6g}\n"
            else:
                result += f"{i+1}\t|\t{j+1}\t|\t{cov:.6g}  [⚠ index out of bounds]\n"
    return result

def display_results(params: np.ndarray, cov_matrix: np.ndarray, residuals: np.ndarray) -> str:
    result = (
        f"Covariance Matrix (Analysis Function):\n{cov_matrix}\n\n"
    )
    return result

def display_measurement_results(x: np.ndarray, x_cov: np.ndarray, meas_data: np.ndarray, decimal_places: int = 4) -> str:
    decimal_format = f"{{:.{decimal_places}e}}"
    header = "x\t|\tu(x)\t|\ty\t|\tu(y)\n"
    ux = np.sqrt(np.diag(x_cov))
    data_lines = "\n".join([
        f"{decimal_format.format(x[i])}\t|\t"
        f"{decimal_format.format(ux[i])}\t|\t"
        f"{meas_data[i, 0]}\t|\t"
        f"{meas_data[i, 1]}"
        for i in range(len(x))
    ])
    return header + data_lines

def display_polynomial_equation_with_uncertainty(coeffs: np.ndarray, uncertainties: np.ndarray) -> str:
    terms = []
    table_rows = []
    for i, (coeff, uncertainty) in enumerate(zip(coeffs, uncertainties)):
        if abs(coeff) < 1e-3:
            coeff_str = f"({coeff:.4e} ± {uncertainty:.4e})"
        else:
            coeff_str = f"({coeff:.4f} ± {uncertainty:.4f})"
        term = f"{coeff_str} * x^{i}"
        terms.append(term)
        param_label = f"b{i}"
        table_rows.append(f"{param_label:<10}| {coeff:.9g}       | {uncertainty:.9g}")
    equation = " + ".join(terms)
    table_str = (
        "\nParameter  | Value              | Uncertainty\n" +
        "\n".join(table_rows)
    )
    return f"{equation}\n\n{table_str}"

def display_exponential_equation_with_uncertainty(coeffs: np.ndarray, uncertainties: np.ndarray) -> tuple[str, pd.DataFrame]:
    a, b, c = coeffs
    ua, ub, uc = uncertainties
    def format_term(value, uncertainty):
        if abs(value) < 1e-3:
            return f"({value:.4e} ± {uncertainty:.4e})"
        else:
            return f"({value:.4f} ± {uncertainty:.4f})"
    term_a = format_term(a, ua)
    term_b = format_term(b, ub)
    term_c = format_term(c, uc)
    equation = f"{term_a} + {term_b} * exp({term_c} * y)"
    table_str = (
        "\nParameter   | Value               | Uncertainty\n"
        f"b0          | {a:.9g}       | {ua:.9g}\n"
        f"b1        | {b:.9g}       | {ub:.9g}\n"
        f"b        | {c:.9g}       | {uc:.9g}"
    )
    return f"{equation}\n{table_str}"

def display_exponential_offset_equation_with_uncertainty(coeffs: np.ndarray, uncertainties: np.ndarray) -> str:
    a, b, c = coeffs
    ua, ub, uc = uncertainties
    term_a = f"({a:.4f} ± {ua:.4f})"
    term_b = f"({b:.4f} ± {ub:.4f})"
    term_c = f"({c:.4f} ± {uc:.4f})"
    equation = f"{term_a} + {term_b} · [exp({term_c} · y) - 1]"
    table_str = (
        "\nParameter  | Value              | Uncertainty\n"
        f"b0         | {a:.9g}       | {ua:.9g}\n"
        f"b1         | {b:.9g}       | {ub:.9g}\n"
        f"b2         | {c:.9g}       | {uc:.9g}"
    )
    return f"{equation}\n{table_str}"

def display_power_equation_with_uncertainty(coeffs: np.ndarray, uncertainties: np.ndarray) -> str:
    b0, b1, b2 = coeffs
    ub0, ub1, ub2 = uncertainties
    term_b0 = f"({b0:.4e} ± {ub0:.4e})"
    term_b1 = f"({b1:.4e} ± {ub1:.4e})"
    term_b2 = f"({b2:.4e} ± {ub2:.4e})"
    equation = f"{term_b0} + {term_b1} * y^(1 + {term_b2})"
    table_str = (
        "\nParameter  | Value              | Uncertainty\n"
        f"b0        | {b0:.9g}       | {ub0:.9g}\n"
        f"b1        | {b1:.9g}       | {ub1:.9g}\n"
        f"b2        | {b2:.9g}       | {ub2:.9g}"
    )
    return f"{equation}\n{table_str}"

def display_logarithmic_equation_with_uncertainty(coeffs: np.ndarray, uncertainties: np.ndarray) -> str:
    a, b, c = coeffs
    ua, ub, uc = uncertainties
    term_a = f"({a:.4f} ± {ua:.4f})"
    term_b = f"({b:.4f} ± {ub:.4f})"
    term_c = f"({c:.4f} ± {uc:.4f})"
    equation = f"{term_a} + {term_b} · ln({term_c} · y)"
    table_str = (
        "\nParameter  | Value              | Uncertainty\n"
        f"b0         | {a:.9g}       | {ua:.9g}\n"
        f"b1         | {b:.9g}       | {ub:.9g}\n"
        f"b2         | {c:.9g}       | {uc:.9g}"
    )
    return f"{equation}\n{table_str}"

def display_hyperbolic_equation_with_uncertainty(coeffs: np.ndarray, uncertainties: np.ndarray) -> str:
    a, b, c = coeffs
    ua, ub, uc = uncertainties
    term_a = f"({a:.4f} ± {ua:.4f})"
    term_b = f"({b:.4f} ± {ub:.4f})"
    term_c = f"({c:.4f} ± {uc:.4f})"
    equation = f"{term_a} + {term_b} / (y + {term_c})"
    table_str = (
        "\nParameter  | Value              | Uncertainty\n"
        f"b0         | {a:.9g}       | {ua:.9g}\n"
        f"b1         | {b:.9g}       | {ub:.9g}\n"
        f"b2         | {c:.9g}       | {uc:.9g}"
    )
    return f"{equation}\n{table_str}"

def display_log_linear_equation_with_uncertainty(coeffs: np.ndarray, uncertainties: np.ndarray) -> str:
    a, b, c = coeffs
    ua, ub, uc = uncertainties
    term_a = f"({a:.4f} ± {ua:.4f})"
    term_b = f"({b:.4f} ± {ub:.4f})"
    term_c = f"({c:.4f} ± {uc:.4f})"
    equation = f"{term_a} + {term_b} · ln(y) + {term_c} · y"
    table_str = (
        "\nParameter  | Value              | Uncertainty\n"
        f"b0         | {a:.9g}       | {ua:.9g}\n"
        f"b1         | {b:.9g}       | {ub:.9g}\n"
        f"b2         | {c:.9g}       | {uc:.9g}"
    )
    return f"{equation}\n{table_str}"

def display_gompertz_equation_with_uncertainty(coeffs: np.ndarray, uncertainties: np.ndarray) -> str:
    a, b, c = coeffs
    ua, ub, uc = uncertainties
    term_a = f"({a:.4f} ± {ua:.4f})"
    term_b = f"({b:.4f} ± {ub:.4f})"
    term_c = f"({c:.4f} ± {uc:.4f})"
    equation = f"{term_a} · exp(-{term_b} · exp(-{term_c} · y))"
    table_str = (
        "\nParameter  | Value              | Uncertainty\n"
        f"b0         | {a:.9g}       | {ua:.9g}\n"
        f"b1         | {b:.9g}       | {ub:.9g}\n"
        f"b2         | {c:.9g}       | {uc:.9g}"
    )
    return f"{equation}\n{table_str}"

def display_square_root_equation_with_uncertainty(coeffs: np.ndarray, uncertainties: np.ndarray) -> str:
    a, b = coeffs
    ua, ub = uncertainties
    term_a = f"({a:.4f} ± {ua:.4f})"
    term_b = f"({b:.4f} ± {ub:.4f})"
    equation = f"{term_a} + {term_b} · sqrt(y)"
    table_str = (
        "\nParameter  | Value              | Uncertainty\n"
        f"b0         | {a:.9g}       | {ua:.9g}\n"
        f"b1         | {b:.9g}       | {ub:.9g}"
    )
    return f"{equation}\n{table_str}"

def display_sigmoid_equation_with_uncertainty(coeffs: np.ndarray, uncertainties: np.ndarray) -> str:
    L, k, y0 = coeffs
    uL, uk, uy0 = uncertainties
    term_L = f"({L:.4f} ± {uL:.4f})"
    term_k = f"({k:.4f} ± {uk:.4f})"
    term_y0 = f"({y0:.4f} ± {uy0:.4f})"
    equation = f"{term_L} / [1 + exp(-{term_k} · (y - {term_y0}))]"
    table_str = (
        "\nParameter  | Value              | Uncertainty\n"
        f"b0         | {L:.9g}       | {uL:.9g}\n"
        f"b1         | {k:.9g}       | {uk:.9g}\n"
        f"b2      | {y0:.9g}       | {uy0:.9g}"
    )
    return f"{equation}\n{table_str}"

def display_simple_exponential_equation_with_uncertainty(coeffs: np.ndarray, uncertainties: np.ndarray) -> str:
    b0, b1 = coeffs
    ub0, ub1 = uncertainties
    term_b0 = f"({b0:.4f} ± {ub0:.4f})"
    term_b1 = f"({b1:.4f} ± {ub1:.4f})"
    equation = f"{term_b0} · exp({term_b1} · y)"
    table_str = (
        "\nParameter  | Value              | Uncertainty\n"
        f"b0        | {b0:.9g}       | {ub0:.9g}\n"
        f"b1        | {b1:.9g}       | {ub1:.9g}"
    )
    return f"{equation}\n{table_str}"



if __name__ == "__main__":
    class CalibrationApp:
        def __init__(self, root_window, main_frame):
            root_window.title("B_Least - Calibration Analysis")
            self.root = main_frame
            
            self.control_frame = ctk.CTkFrame(main_frame, width=200)
            self.control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")
            
            self.customer_id_entry = ctk.CTkEntry(self.control_frame, width=300, placeholder_text="Customer ID")
            self.customer_id_entry.grid(row=1, column=0, pady=5, padx=5)
            
            self.additional_info_label = ctk.CTkLabel(self.control_frame, text="Additional Information")
            self.additional_info_label.grid(row=2, column=0, pady=5, padx=5)
            self.additional_info_entry = ctk.CTkTextbox(self.control_frame, width=300, height=100)
            self.additional_info_entry.grid(row=3, column=0, pady=5, padx=5)
            
            self.function_options = [
                "Linear Function", "Polynomial Function", "Exponential Function",
                "Exponential Offset Function", "Simple Exponential Function", "Power Function", "Rational Function",
                "Logarithmic Function", "Hyperbolic Function", "Log-Linear Function",
                "Gompertz Function", "Square Root Function", "Sigmoid Function"
                ]
            
            self.iso_function_options = [
                "Linear Function", "Polynomial Function", "Exponential Function", "Power Function"
                ]
            
            self.iso_filter_var = tk.BooleanVar()
            self.iso_filter_checkbox = ctk.CTkCheckBox(
                self.control_frame,
                text="Only ISO 6143-compliant models",
                variable=self.iso_filter_var,
                command=self.update_filter_views
                )
            self.iso_filter_checkbox.grid(row=5, column=0, pady=5, padx=5, sticky="w")
            
            self.function_label = ctk.CTkLabel(self.control_frame, text="Function Type")
            self.function_label.grid(row=4, column=0, pady=5, padx=5)
            
            self.function_dropdown = ctk.CTkOptionMenu(self.control_frame, values=self.function_options, command=self.update_function_selection)
            self.function_dropdown.grid(row=6, column=0, pady=5, padx=5)
            
            self.order_entry = ctk.CTkEntry(self.control_frame, placeholder_text="Which Order?")
            self.order_entry.grid(row=7, column=0, pady=5, padx=5)
            self.order_entry.grid_remove()
            
            self.cal_file_button = ctk.CTkButton(self.control_frame, text="Load Calibration Data", command=self.load_calibration_data)
            self.cal_file_button.grid(row=8, column=0, pady=5, padx=5)
            self.cal_file_path = ctk.CTkEntry(self.control_frame, state="disabled")
            self.cal_file_path.grid(row=9, column=0, pady=5, padx=5)
            
            self.covariance_input_entry = ctk.CTkEntry(self.control_frame, width=300, placeholder_text="Not Implemented !! Covariances (e.g. 4/8 0.0016, 6/8 0.00009)")
            self.covariance_input_entry.grid(row=10, column=0, pady=5, padx=5, sticky="w")
            
            self.meas_file_button = ctk.CTkButton(self.control_frame, text="Load Measurement Data", command=self.load_measurement_data)
            self.meas_file_button.grid(row=11, column=0, pady=5, padx=5)
            self.meas_file_path = ctk.CTkEntry(self.control_frame, state="disabled")
            self.meas_file_path.grid(row=12, column=0, pady=5, padx=5)
            
            self.additional_analysis_frame = ctk.CTkFrame(self.control_frame)
            self.additional_analysis_frame.grid(row=13, column=0, pady=10, padx=5, sticky="w")
            
            self.additional_analysis_label = ctk.CTkLabel(self.additional_analysis_frame, text="Additional Evaluations")
            self.additional_analysis_label.grid(row=0, column=0, pady=5, padx=5, sticky="w")
            
            self.residuals_checkbox = ctk.CTkCheckBox(self.additional_analysis_frame, text="Weighted Residuals")
            self.residuals_checkbox.grid(row=3, column=0, pady=5, padx=5, sticky="w")
            
            self.sum_sqaured_residuals_checkbox = ctk.CTkCheckBox(self.additional_analysis_frame, text="Residual sum of squared (weighted) deviations") 
            self.sum_sqaured_residuals_checkbox.grid(row=4, column=0, pady=5, padx=5, sticky="w")
            
            self.goodness_of_fit_checkbox = ctk.CTkCheckBox(self.additional_analysis_frame, text="Goodness-of-Fit (Gamma, Γ)")
            self.goodness_of_fit_checkbox.grid(row=5, column=0, pady=5, padx=5, sticky="w")
            
            self.rmse_checkbox = ctk.CTkCheckBox(self.additional_analysis_frame, text="Root Mean Squared Error")
            self.rmse_checkbox.grid(row=11, column=0, pady=5, padx=5, sticky="w")
            
            self.start_button = ctk.CTkButton(self.control_frame, text="Start Analysis", command=self.start_analysis, fg_color="green")
            self.start_button.grid(row=15, column=0, pady=5, padx=5)
            
            self.save_pdf_button = ctk.CTkButton(self.control_frame, text="Save as PDF with graph", command=self.save_results_as_pdf)
            self.save_pdf_button.grid(row=16, column=0, pady=5, padx=5)
            
            self.save_csv_button = ctk.CTkButton(self.control_frame, text="Save Results as CSV", command=self.save_results_as_csv)
            self.save_csv_button.grid(row=17, column=0, pady=5, padx=5)
            
            self.include_model_compare_checkbox = ctk.CTkCheckBox(self.control_frame, text="Include Model Comparison in PDF")
            self.include_model_compare_checkbox.grid(row=18, column=0, pady=5, padx=5, sticky="w")
            
            self.result_text = ctk.CTkTextbox(main_frame, width=650, height=835)
            self.result_text.configure(state="disabled")
            self.result_text.grid(row=0, column=1, padx=10, pady=10, sticky="n")
            
            self.result_text.tag_config("bold_tag", underline=1)
            self.result_text.tag_config("highlight_tag", background="#e0e0e0")
            
            self.graph_frame = ctk.CTkFrame(main_frame, height=835, width=850)
            self.graph_frame.grid(row=0, column=3, padx=10, pady=10, sticky="nw")
            self.graph_frame.grid_propagate(False)
            
            dpi = 100
            width_px = 850
            height_px = 835
            fig_width_inch = width_px / dpi
            fig_height_inch = height_px / dpi
            
            self.figure, (self.ax, self.ax_residuals) = plt.subplots(
                2, 1,
                figsize=(fig_width_inch, fig_height_inch),
                dpi=dpi,
                gridspec_kw={'height_ratios': [3, 1]}
            )
            
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_frame)
            canvas_widget = self.canvas.get_tk_widget()
            canvas_widget.grid(row=0, column=0, sticky="nsew")
            
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.graph_frame)
            self.toolbar.grid(row=1, column=0, sticky="ew")
            
            self.graph_frame.grid_rowconfigure(0, weight=1)
            self.graph_frame.grid_columnconfigure(0, weight=1)
            
            self.model_compare_frame = ctk.CTkFrame(main_frame)
            self.model_compare_frame.grid(row=1, column=1, padx=10, pady=(0, 10), sticky="nw")
            
            self.model_compare_label = ctk.CTkLabel(self.model_compare_frame, text="Model Comparison", anchor="w")
            self.model_compare_label.pack(anchor="w", padx=5, pady=(5, 0))
            
            self.model_compare_text = ctk.CTkTextbox(self.model_compare_frame, height=180, width=650)
            self.model_compare_text.pack(padx=5, pady=5)
            self.model_compare_text.configure(state="disabled")
            
            self.function_list_frame = ctk.CTkFrame(main_frame)
            self.function_list_frame.grid(row=1, column=3, padx=10, pady=(0, 10), sticky="nw")
            
            self.function_list_label = ctk.CTkLabel(self.function_list_frame, text="Available Functions", anchor="w")
            self.function_list_label.pack(anchor="w", padx=5, pady=(5, 0))
            
            self.function_list_text = ctk.CTkTextbox(self.function_list_frame, height=180, width=850)
            self.function_list_text.pack(padx=5, pady=5)
            self.function_list_text.configure(state="disabled")
            
            
            self.calibration_file_path = None
            self.measurement_file_path = None
            self.selected_function = "Linear Function"
            self.order = 2
            
            self.cal_data = None
            self.meas_data = None
            self.params = None
            self.cov_matrix = None
            self.residuals = None
            self.x_pred = None
            self.x_cov = None
            
            self.display_function_list()
            
        def update_function_selection(self, selection):
            self.selected_function = selection
            if selection == "Polynomial Function":
                self.order_entry.grid()
            else:
                self.order_entry.grid_remove()
                
        def update_function_dropdown(self):
            if self.iso_filter_var.get():
                options = self.iso_function_options
            else:
                options = self.all_function_options
                
            self.function_dropdown.configure(values=options)
            self.function_dropdown.set(options[0])
            
        def update_filter_views(self):
            if self.iso_filter_var.get():
                self.function_dropdown.configure(values=self.iso_function_options)
                self.function_dropdown.set(self.iso_function_options[0])
                self.display_function_list()
            else:
                self.function_dropdown.configure(values=self.function_options)
                self.function_dropdown.set(self.function_options[0])
                self.display_function_list()
                
        def load_calibration_data(self):
            file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")])
            if file_path:
                self.calibration_file_path = file_path
                self.cal_file_path.configure(state="normal")
                self.cal_file_path.delete(0, "end")
                self.cal_file_path.insert(0, file_path)
                self.cal_file_path.configure(state="disabled")
                
        def load_measurement_data(self):
            file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")])
            if file_path:
                self.measurement_file_path = file_path
                self.meas_file_path.configure(state="normal")
                self.meas_file_path.delete(0, "end")
                self.meas_file_path.insert(0, file_path)
                self.meas_file_path.configure(state="disabled")
                
        def display_function_list(self):
            function_groups = {
                "Standardized Models (ISO 6143)": [
                    ("Linear", "x = b0 + b1·y",
                     "Model for describing a proportional or linear relationship.\n"
                     ""),
                    
                    ("Polynomial (n)", "x = b0 + b1·y + … + bn·yⁿ",
                     "Model for describing smooth, continuous nonlinearities.\n"
                     ""),
                    
                    ("Exponential", "x = b0 + b1·exp(b2·y)",
                     "Model for describing rapidly increasing signals with growth-like behavior.\n"
                     ""),
                    
                    ("Power Function", "x = b0 + b1·y^(1 + b2)",
                     "Model for describing nonlinear relationships with increasing slope.\n"
                     "")
                ],  
                
                "Monotonically Increasing Functions": [
                    ("Simple Exponential Function", "x = b0·exp(b1·y)",
                     "Model for describing steep, continuously accelerating trends.\n"
                     ""),
                    
                    ("Exponential Function with Offset", "x = b0 + b1·(exp(b2·y) - 1)",
                     "Model for describing exponential growth from a non-zero baseline.\n"
                     ""),
                    
                    ("Root Function", "x = b0 + b1·√y",
                     "Model for describing diffusion- or transport-limited behavior.\n"
                     "")
                ],
                
                "Saturation and Limiting Functions": [
                    ("Rational Function", "x = (b0 + b1·y) / (1 + b2·y)",
                     "Model for describing saturation effects with asymptotic behavior.\n"
                     ""),
                    
                    ("Hyperbolic Function", "x = b0 + b1 / (y + b2)",
                     "Model for describing decreasing rates of change with increasing y.\n"
                     "")
                ],
                
                "Logarithmic and Combined Models": [
                    ("Logarithmic", "x = b0 + b1·ln(b2·y)",
                     "Model for describing strong initial response with flattening.\n"
                     ""),
                    
                    ("Log-Linear", "x = b0 + b1·ln(y) + b2·y",
                     "Model for describing combined logarithmic and linear growth.\n"
                     "")
                ],
                
                "Nonlinear Growth Models": [
                    ("Gompertz Function", "x = b0·exp(-b1·exp(-b2·y))",
                     "Model for describing S-shaped growth with an inflection point.\n"
                     ""),
                    
                    ("Sigmoid Function", "x = b0 / [1 + exp(-b1·(y - b2))]",
                     "Model for describing threshold behavior with saturation.\n"
                     "")
                ]
            }
            
            show_only_iso = self.iso_filter_var.get()
            
            self.function_list_text.configure(state="normal")
            self.function_list_text.delete("0.0", "end")
            
            for group_name, functions in function_groups.items():
                if show_only_iso and "ISO 6143" not in group_name:
                    continue
                
                self.function_list_text.insert("end", f"{group_name.upper()}\n")
                self.function_list_text.insert("end", "────────────────────────────────────────\n")
                
                for name, formula, description in functions:
                    self.function_list_text.insert("end", f"  • {name}\n")
                    self.function_list_text.insert("end", f"    Formula: {formula}\n")
                    self.function_list_text.insert("end", f"    Use Case: {description}\n\n")
                    
                self.function_list_text.insert("end", "\n")
                
            self.function_list_text.configure(state="disabled")
        
        def evaluate_all_models_display_only(self, cal_data):
            try:
                results = []
                
                functions = [
                    ("Linear", compute_polynomial, 1),
                    ("Polynomial (order 2)", compute_polynomial, 2),
                    ("Polynomial (order 3)", compute_polynomial, 3),
                    ("Exponential", compute_exponential, None),
                    ("Exponential Offset", compute_exponential_offset, None),
                    ("Simple Exponential", compute_simple_exponential, None),
                    ("Power", compute_power, None),
                    ("Rational", compute_rational, None),
                    ("Logarithmic", compute_logarithmic, None),
                    ("Hyperbolic", compute_hyperbolic, None),
                    ("Log-Linear", compute_log_linear, None),
                    ("Gompertz", compute_gompertz, None),
                    ("Square Root", compute_square_root, None),
                    ("Sigmoid", compute_sigmoid, None),
                ]
                
                for name, func, order in functions:
                    try:
                        if func == compute_polynomial:
                            params, cov, residuals = model_fit(cal_data, func, order)
                        else:
                            params, cov, residuals = model_fit(cal_data, func)
                            
                        ssd = np.sum(residuals**2)
                        gamma = np.max(np.abs(residuals))
                        rmse = np.sqrt(np.mean(residuals**2))
                        
                        results.append((name, ssd, gamma, rmse))
                    except Exception:
                        results.append((name, "Error", "Error", "Error"))
                        
                self.model_compare_text.configure(state="normal")
                self.model_compare_text.delete("0.0", "end")
                
                for name, ssd, gamma, rmse in results:
                    self.model_compare_text.insert("end", f"{name}\n")
                    self.model_compare_text.insert("end", f"  SSD: {ssd}\n")
                    self.model_compare_text.insert("end", f"  Goodness-of-Fit (Gamma): {gamma}\n")
                    self.model_compare_text.insert("end", f"  RMSE: {rmse}\n\n")
                    
                self.model_compare_text.configure(state="disabled")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error during model comparison:\n{e}")
                
        def start_analysis(self):
            if self.calibration_file_path and self.measurement_file_path:
               manual_cov_input = self.covariance_input_entry.get()
               try:
                   self.manual_cal_covariances = parse_manual_covariances(manual_cov_input)
                   print("Manuelle Kovarianzen:", self.manual_cal_covariances)
               except Exception as e:
                   messagebox.showerror("Invalid Covariance Input", str(e))
                   return
               
               try:
                   delimiter = detect_delimiter(self.calibration_file_path)
                   self.cal_data = np.genfromtxt(self.calibration_file_path, delimiter = delimiter)
                   
                   delimiter = detect_delimiter(self.measurement_file_path)
                   self.meas_data = np.atleast_2d(np.genfromtxt(self.measurement_file_path, delimiter=delimiter))
                   self.meas_data = np.atleast_2d(self.meas_data)
                   
                   if self.selected_function == 'Exponential Function':
                       self.params, self.cov_matrix, self.residuals, self.x_pred, self.x_cov = run_analysis(
                           self.cal_data, self.meas_data, compute_exponential, extra_cov=self.manual_cal_covariances
                        )
                   elif self.selected_function == 'Exponential Offset Function':
                       self.params, self.cov_matrix, self.residuals, self.x_pred, self.x_cov = run_analysis(
                           self.cal_data, self.meas_data, compute_exponential_offset, extra_cov=self.manual_cal_covariances
                           )
                   elif self.selected_function == 'Linear Function':
                        self.params, self.cov_matrix, self.residuals, self.x_pred, self.x_cov = run_analysis(
                            self.cal_data, self.meas_data, compute_polynomial, order=1, extra_cov=self.manual_cal_covariances
                            )
                   elif self.selected_function == 'Polynomial Function':
                           self.order = int(self.order_entry.get()) if self.order_entry.get().isdigit() else 2
                           self.params, self.cov_matrix, self.residuals, self.x_pred, self.x_cov = run_analysis(
                                self.cal_data, self.meas_data, compute_polynomial, self.order,  extra_cov=self.manual_cal_covariances
                                )
                   elif self.selected_function == 'Power Function':
                           self.params, self.cov_matrix, self.residuals, self.x_pred, self.x_cov = run_analysis(
                           self.cal_data, self.meas_data, compute_power, extra_cov=self.manual_cal_covariances
                           )
                   
                   elif self.selected_function == 'Rational Function':
                       self.params, self.cov_matrix, self.residuals, self.x_pred, self.x_cov = run_analysis(
                           self.cal_data, self.meas_data, compute_rational, extra_cov=self.manual_cal_covariances
                           )
                       
                   elif self.selected_function == 'Logarithmic Function':
                       self.params, self.cov_matrix, self.residuals, self.x_pred, self.x_cov = run_analysis(
                           self.cal_data, self.meas_data, compute_logarithmic, extra_cov=self.manual_cal_covariances
                           )
                       
                   elif self.selected_function == 'Hyperbolic Function':
                       self.params, self.cov_matrix, self.residuals, self.x_pred, self.x_cov = run_analysis(
                           self.cal_data, self.meas_data, compute_hyperbolic, extra_cov=self.manual_cal_covariances
                           )
                   elif self.selected_function == 'Log-Linear Function':
                       self.params, self.cov_matrix, self.residuals, self.x_pred, self.x_cov = run_analysis(
                           self.cal_data, self.meas_data, compute_log_linear, extra_cov=self.manual_cal_covariances
                           )
                       
                   elif self.selected_function == 'Gompertz Function':
                       self.params, self.cov_matrix, self.residuals, self.x_pred, self.x_cov = run_analysis(
                           self.cal_data, self.meas_data, compute_gompertz, extra_cov=self.manual_cal_covariances
                           )
                   
                   elif self.selected_function == 'Square Root Function':
                       self.params, self.cov_matrix, self.residuals, self.x_pred, self.x_cov = run_analysis(
                           self.cal_data, self.meas_data, compute_square_root, extra_cov=self.manual_cal_covariances
                           )
                       
                   elif self.selected_function == 'Sigmoid Function':
                       self.params, self.cov_matrix, self.residuals, self.x_pred, self.x_cov = run_analysis(
                           self.cal_data, self.meas_data, compute_sigmoid, extra_cov=self.manual_cal_covariances
                           )
                       
                   elif self.selected_function == 'Simple Exponential Function':
                       self.params, self.cov_matrix, self.residuals, self.x_pred, self.x_cov = run_analysis(
                           self.cal_data, self.meas_data, compute_simple_exponential, extra_cov=self.manual_cal_covariances
                           )
                       
                   else:
                       raise ValueError("Invalid function")
                       
                   self.display_results(self.cal_data, self.meas_data, self.params, self.cov_matrix, self.residuals, self.x_pred, self.x_cov)
                   self.update_graph()
                   
               except Exception as e:
                    self.result_text.configure(state="normal")
                    self.result_text.delete("0.0", "end")
                    self.result_text.insert("0.0", f"Error during analysis: {str(e)}\n")
                    self.result_text.configure(text_color="red")
                    self.result_text.configure(state="disabled")
                    self.result_text.update()
            else:
               self.result_text.configure(state="normal")
               self.result_text.delete("0.0", "end")
               self.result_text.insert("0.0", "Please load all required files first.\n")
               self.result_text.configure(text_color="red")
               self.result_text.configure(state="disabled")
               self.result_text.update()
               
            self.evaluate_all_models_display_only(cal_data=self.cal_data)
                    
        def save_results_as_csv(self):
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if not file_path:
                return
            
            try:
                with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile, delimiter=";")
                    
                    writer.writerow(["[Analysis Metadata]"])
                    writer.writerow(["Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                    writer.writerow(["Customer ID", self.customer_id_entry.get()])
                    writer.writerow(["Additional Information", self.additional_info_entry.get("1.0", "end").strip()])
                    writer.writerow(["Selected Function", self.selected_function])
                    writer.writerow(["Calibration File Path", self.calibration_file_path])
                    writer.writerow(["Measurement File Path", self.measurement_file_path])
                    writer.writerow([])
                    
                    writer.writerow(["[Calibration Data]"])
                    writer.writerow(["x", "u(x)", "y", "u(y)"])
                    for row in self.cal_data:
                        writer.writerow([f"{val:.9g}" for val in row])
                    writer.writerow([])
                    
                    uncertainties = np.sqrt(np.diag(self.cov_matrix))
                    
                    writer.writerow([])
                    
                    writer.writerow(["[Calibration Parameters]"])
                    writer.writerow(["Parameter", "Value", "Uncertainty"])
                    for i, (val, unc) in enumerate(zip(self.params, uncertainties)):
                        writer.writerow([f"b{i}", f"{val:.9g}", f"{unc:.9g}"])
                    writer.writerow([])
                    
                    writer.writerow(["[Covariance Matrix (Parameters)]"])
                    writer.writerow(["Row", "Col", "Covariance"])
                    n_params = len(self.params)
                    for i in range(n_params):
                        for j in range(n_params):
                            writer.writerow([i, j, f"{self.cov_matrix[i, j]:.9g}"])
                    writer.writerow([])
                    
                    writer.writerow(["[Covariances Between Coefficients]"])
                    writer.writerow(["b_i", "b_j", "Covariance"])
                    for i in range(n_params):
                        for j in range(i + 1, n_params):
                            writer.writerow([f"b{i}", f"b{j}", f"{self.cov_matrix[i, j]:.6e}"])
                    writer.writerow([])
                    
                    writer.writerow(["[Sample Results]"])
                    writer.writerow(["x", "u(x)", "y", "u(y)"])
                    ux = np.sqrt(np.diag(self.x_cov))
                    for i in range(len(self.x_pred)):
                        writer.writerow([
                            f"{self.x_pred[i]:.9g}",
                            f"{ux[i]:.9g}",
                            f"{self.meas_data[i, 0]:.9g}",
                            f"{self.meas_data[i, 1]:.9g}"
                        ])
                    writer.writerow([])
                    
                    writer.writerow(["[Covariance Matrix (Samples)]"])
                    writer.writerow(["Row", "Col", "Covariance"])
                    n = len(self.x_pred)
                    for i in range(n):
                        for j in range(n):
                            writer.writerow([i, j, f"{self.x_cov[i, j]:.9g}"])
                    writer.writerow([])
                    
                    res_x = self.residuals[:len(self.cal_data)]
                    res_y = self.residuals[len(self.cal_data):]
                    
                    writer.writerow(["[Residuals]"])
                    header = ["Type"] + [f"R{i+1}" for i in range(len(res_x))]
                    writer.writerow(header)
                    writer.writerow(["x"] + [f"{v:.4f}" for v in res_x])
                    writer.writerow(["y"] + [f"{v:.4f}" for v in res_y])
                    writer.writerow([])
                    
                    ssr = np.sum(self.residuals**2)
                    gamma = np.max(np.abs(self.residuals))
                    rmse = np.sqrt(np.mean(self.residuals**2))
                    status_ssr = "OK" if ssr <= len(self.cal_data)/2 else "NOT OK"
                    status_gamma = "OK" if gamma < 2 else "NOT OK"
                    writer.writerow(["[Summary Statistics]"])
                    writer.writerow(["Residual Sum of Squares", f"{ssr:.4f}", status_ssr])
                    writer.writerow(["Goodness-of-Fit (Gamma)", f"{gamma:.4f}", status_gamma])
                    writer.writerow(["Root Mean Square Error (RMSE)", f"{rmse:.4f}"])
                    writer.writerow([])
                    
                messagebox.showinfo("Success", "CSV with complete results saved successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving as CSV:\n{e}")
                
        def save_results_as_pdf(self):
            self.stop_runner = False
            file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
            if not file_path:
                return
            
            spinner_window = tk.Toplevel()
            spinner_window.title("Creating PDF...")
            spinner_window.geometry("250x130")
            spinner_window.resizable(False, False)
            
            info_label = tk.Label(spinner_window, text="Creating PDF...\nPlease do not interrupt.", font=("Courier", 10))
            info_label.pack(pady=5)
            
            runner_label = tk.Label(spinner_window, text="", font=("Courier", 14), justify="center")
            runner_label.pack()
            
            runner_frames = [
                " o \n/|\\\n/ \\",
                "\\o \n |\\\n/ \\",
                " o/\n/| \n/ \\",
                "\\o/\n | \n/ \\",
                ]
            runner_iter = itertools.cycle(runner_frames)
            
            def update_runner():
                if not self.stop_runner:
                    runner_label.config(text=next(runner_iter))
                    spinner_window.after(200, update_runner)
                
            update_runner()
            
 
            def generate_pdf():
                try:
                    
                    buf = BytesIO()
                    self.figure.savefig(
                        buf,
                        format="png",
                        dpi=300,
                        bbox_inches="tight",
                        facecolor="white"
                    )
                    buf.seek(0)
                    img = ImageReader(buf)
                    
                    c = pdf_canvas.Canvas(file_path, pagesize=A4)
                    width, height = A4
                    c.setFont("Courier", 10)
            
                    content = self.result_text.get("1.0", "end").strip()
                    y_pos = height - 40
            
                    for line in content.splitlines():
                        if y_pos < 40:
                            c.showPage()
                            c.setFont("Courier", 10)
                            y_pos = height - 40
            
                        if line.strip() == "":
                            y_pos -= 5
                        elif line.startswith("+"):
                            c.setFont("Courier-Bold", 10)
                            c.drawString(40, y_pos, line.strip())
                            c.setFont("Courier", 10)
                            y_pos -= 12
                        elif ":" in line and not line.startswith(" "):
                            key, value = line.split(":", 1)
                            c.setFont("Courier-Bold", 10)
                            c.drawString(40, y_pos, f"{key.strip()}:")
                            c.setFont("Courier", 10)
                            c.drawString(140, y_pos, value.strip())
                            y_pos -= 12
                        elif "|" in line:
                            cells = [cell.strip() for cell in line.split("|")]
                            col_width = (width - 80) / len(cells)
                            x_pos = 40
                            for cell in cells:
                                c.drawString(x_pos + 2, y_pos, cell)
                                x_pos += col_width
                            y_pos -= 14
                        else:
                            c.drawString(40, y_pos, line.strip())
                            y_pos -= 12
            
                    c.showPage()
                    c.setFont("Courier-Bold", 12)
                    c.drawCentredString(width / 2, height - 40, "Calibration Diagram")
            
                    max_w = width - 80
                    iw, ih = img.getSize()
                    scale = max_w / iw
                    c.drawImage(
                        img,
                        40,
                        100,
                        width=iw * scale,
                        height=ih * scale,
                        preserveAspectRatio=True,
                    )
                    
                    if self.include_model_compare_checkbox.get():
                        c.showPage()
                        c.setFont("Courier-Bold", 12)
                        c.drawString(40, height - 40, "Model Comparison")
                    
                        c.setFont("Courier", 10)
                        y_pos = height - 60
                        mc_content = self.model_compare_text.get("1.0", "end").strip()
                    
                        for line in mc_content.splitlines():
                            if y_pos < 40:
                                c.showPage()
                                c.setFont("Courier", 10)
                                y_pos = height - 40
                    
                            c.drawString(40, y_pos, line.strip())
                            y_pos -= 12
            
                    c.save()
                    self.root.after(0, lambda: messagebox.showinfo("Success", "PDF saved successfully."))
            
                except Exception as e:
                    self.root.after(0, lambda err=e: messagebox.showerror("Error", str(err)))
                finally:
                    self.stop_runner = True
                    self.root.after(0, spinner_window.destroy)

            threading.Thread(target=generate_pdf).start()
            
        def display_results(self, cal_data, meas_data, params, cov_matrix, residuals, x_pred, x_cov):
            self.result_text.configure(state="normal")
            self.result_text.delete("0.0", "end")
            
            customer_id = self.customer_id_entry.get()
            additional_info = self.additional_info_entry.get("1.0", "end").strip()
            
            self.result_text.insert("end", "Analysis Information\n", "highlight_tag")
            self.result_text.insert("end", "\nCustomer ID:\n", "bold_tag")
            self.result_text.insert("end", f"{customer_id}\n")
            self.result_text.insert("end", "\nAdditional Information:\n", "bold_tag")
            self.result_text.insert("end", f"{additional_info}\n\n")
            self.result_text.insert("end", "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")
            
            self.result_text.insert("end", "Calibration File\n", "highlight_tag")
            self.result_text.insert("end", f"\nPath:\n {self.calibration_file_path}\n\n")
            
            self.result_text.insert("end", "Calibration Data:\n","bold_tag")
            self.result_text.insert("end", display_cal_data(cal_data, self.manual_cal_covariances) + "\n\n")
            self.result_text.insert("end", "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")
            
            uncertainties = np.sqrt(np.diag(cov_matrix))
            self.result_text.insert("end", "User-oriented Analysis:\n", "highlight_tag")
            self.result_text.insert("end", "\nAnalysis Function:\n", "bold_tag")
            if self.selected_function == 'Exponential Function':
                self.result_text.insert("end", display_exponential_equation_with_uncertainty(params, uncertainties) + "\n\n")
            elif self.selected_function == 'Exponential Offset Function':
                self.result_text.insert("end", display_exponential_offset_equation_with_uncertainty(params, uncertainties) + "\n\n")
            elif self.selected_function == 'Power Function':
                self.result_text.insert("end", display_power_equation_with_uncertainty(params, uncertainties) + "\n\n")
            elif self.selected_function == 'Rational Function':
                self.result_text.insert("end",
                                        f"(({params[0]:.9f} ± {uncertainties[0]:.9f}) + ({params[1]:.9f} ± {uncertainties[1]:.9f})·y) / "
                                        f"(1 + ({params[2]:.9f} ± {uncertainties[2]:.9f})·y)\n\n")
            elif self.selected_function == 'Logarithmic Function':
                self.result_text.insert("end", display_logarithmic_equation_with_uncertainty(params, uncertainties) + "\n\n")
            elif self.selected_function == 'Hyperbolic Function':
                self.result_text.insert("end", display_hyperbolic_equation_with_uncertainty(params, uncertainties) + "\n\n")
            elif self.selected_function == 'Log-Linear Function':
                self.result_text.insert("end", display_log_linear_equation_with_uncertainty(params, uncertainties) + "\n\n")
            elif self.selected_function == 'Gompertz Function':
                self.result_text.insert("end", display_gompertz_equation_with_uncertainty(params, uncertainties) + "\n\n")
            elif self.selected_function == 'Square Root Function':
                self.result_text.insert("end", display_square_root_equation_with_uncertainty(params, uncertainties) + "\n\n")
            elif self.selected_function == 'Sigmoid Function':
                self.result_text.insert("end", display_sigmoid_equation_with_uncertainty(params, uncertainties) + "\n\n")
            elif self.selected_function == 'Simple Exponential Function':
                self.result_text.insert("end", display_simple_exponential_equation_with_uncertainty(params, uncertainties) + "\n\n")
            else:
                self.result_text.insert("end", display_polynomial_equation_with_uncertainty(params, uncertainties) + "\n\n")
            
            self.result_text.insert("end", display_results(params, cov_matrix, residuals))
            self.result_text.insert("end", "Covariances Between Coefficients:\n", "bold_tag")
            for i in range(len(params)):
                for j in range(i + 1, len(params)):
                    covariance_value = cov_matrix[i, j]
                    self.result_text.insert("end", f"Covariance between Coefficient b{i} and b{j}:\n {covariance_value:.6e}\n")
            
            self.result_text.insert("end", "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")
            self.result_text.insert("end", "Samples\n","highlight_tag")
            
            self.result_text.insert("end", f"\nPath:\n {self.measurement_file_path}\n\n")
            
            self.result_text.insert("end", "Content in Samples:\n","bold_tag")
            self.result_text.insert("end", display_measurement_results(x_pred, x_cov, meas_data) + "\n")
            self.result_text.insert("end", "\nCovariance Matrix (Sample Results):\n")
            self.result_text.insert("end", f"{x_cov}\n\n")
            
            self.result_text.insert("end", "Covariances Between Sample Results:\n", "bold_tag")
            n_samples = len(x_pred)
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    cov_value = x_cov[i, j]
                    self.result_text.insert("end", f"Covariance between Sample x{i + 1} and x{j + 1}:\n {cov_value:.6e}\n")
            self.result_text.insert("end", "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")
            self.result_text.configure(text_color="black")
            self.result_text.configure(state="disable")

            if self.residuals_checkbox.get() or self.sum_sqaured_residuals_checkbox.get() or self.goodness_of_fit_checkbox.get() or self.rmse_checkbox.get():
                self.result_text.configure(state="normal")
                self.result_text.insert("end", "Additional Evaluations (Calibration)\n","highlight_tag")
                
            if self.residuals_checkbox.get():
                self.result_text.configure(state="normal")
                res_x = residuals[:len(cal_data)]
                res_y = residuals[len(cal_data):]
                
                res_x_str = ", ".join([f"{val:.4f}" for val in res_x])
                res_y_str = ", ".join([f"{val:.4f}" for val in res_y])
                
                self.result_text.insert("end", "\nWeighted Residuals (normalized deviations):\n\n", "bold_tag")
                self.result_text.insert("end", "Residuals in x-direction:\n")
                self.result_text.insert("end", f"{res_x_str}\n\n")
                self.result_text.insert("end", "Residuals in y-direction:\n")
                self.result_text.insert("end", f"{res_y_str}\n")
                self.result_text.configure(state="disable")
                
            if self.sum_sqaured_residuals_checkbox.get():
                self.result_text.configure(state="normal")
                n_points = len(cal_data)
                n_params = len(params)
                acceptance_limit = 2 * (n_points - n_params)
                ssr = np.sum(residuals**2)
                
                if ssr <= acceptance_limit:
                    ssr_status = "OK"
                else:
                    ssr_status = "NOT OK"
                        
                self.result_text.insert("end", "\nResidual sum of squared (weighted) deviations:\n", "bold_tag")
                self.result_text.insert("end", f"{ssr:.4f} -> {ssr_status} (Acceptance Limit = 2(n - N - 1) = {acceptance_limit:.0f})\n")
                self.result_text.configure(state="disable")
                
            if self.goodness_of_fit_checkbox.get():
                self.result_text.configure(state="normal")
                try:
                        if self.selected_function in ['Exponential Function','Polynomial Function','Linear Function','Exponential Offset Function','Power Function','Rational Function','Logarithmic Function','Hyperbolic Function','Log-Linear Function','Gompertz Function','Square Root Function','Sigmoid Function']:
                            res_all = self.residuals
                            gamma = np.max(np.abs(res_all))
                            
                        if gamma < 2:
                           status = "OK"
                        else:
                           status = "NOT OK"
                                
                        self.result_text.insert("end", "\nGoodness-of-Fit (Gamma):\n", "bold_tag")
                        self.result_text.insert("end", f"{gamma:.4f} -> {status}\n")
                        self.result_text.configure(state="disable")
                        
                except Exception as e:
                    self.result_text.insert("end", f"Error in Γ calculation: {str(e)}\n")
                    self.result_text.configure(state="disable")
                    
            if self.rmse_checkbox.get():
                self.result_text.configure(state="normal")
                rmse = np.sqrt(np.mean(residuals**2))
                self.result_text.insert("end", "\nRoot Mean Square Error (RMSE):\n", "bold_tag")
                self.result_text.insert("end", f"{rmse:.4f}\n\n")
                self.result_text.configure(state="disabled")
                self.result_text.configure(state="disabled")
                
        def update_graph(self):
            # 0) Daten & Parameter
            cal_data   = self.cal_data
            meas_data  = self.meas_data
            params     = self.params
            cov_matrix = self.cov_matrix
            func_name  = self.selected_function
            
            # 1) Welches Model wählen?
            if func_name == 'Exponential Function':
                func = compute_exponential
            elif func_name == 'Exponential Offset Function':
                func = compute_exponential_offset
            elif func_name == 'Power Function':
                func = compute_power
            elif func_name == 'Rational Function':
                func = compute_rational
            elif func_name == 'Logarithmic Function':
                func = compute_logarithmic
            elif func_name == 'Hyperbolic Function':
                func = compute_hyperbolic
            elif func_name == 'Log-Linear Function':
                func = compute_log_linear
            elif func_name == 'Gompertz Function':
                func = compute_gompertz
            elif func_name == 'Square Root Function':
                func = compute_square_root
            elif func_name == 'Sigmoid Function':
                func = compute_sigmoid
            elif func_name == 'Simple Exponential Function':
                func = compute_simple_exponential
            elif func_name in ('Polynomial Function', 'Linear Function'):
                func = compute_polynomial
            else:
                raise ValueError(f"Unbekannte Funktion: {func_name}")
                
            # 2) Plot-Bereiche leeren
            self.ax.clear()
            self.ax_residuals.clear()

            # 3) Kalibrationspunkte mit Fehlerbalken
            self.ax.errorbar(
                cal_data[:, 2], cal_data[:, 0],
                xerr=cal_data[:, 3], yerr=cal_data[:, 1],
                fmt='o', label='Calibration Points', color='blue'
            )
            
            # 4) Fit-Kurve zeichnen
            y_values = np.linspace(cal_data[:, 2].min(),
                                   cal_data[:, 2].max(),
                                   200)
            x_values, _, _ = func(y_values, params)
            # Sicherstellen, dass x_values 1D sind
            x_values = np.array(x_values).flatten()
            self.ax.plot(y_values, x_values,
                         label='Analysis Function',
                         color='red', linewidth=2)
            
            # 5) 95 %-Konfidenzband berechnen und zeichnen
            #    Setze u(y)=0, damit nur Parameter‑Unsicherheit reinspielt
            meas_grid = np.column_stack([y_values,
                                         np.zeros_like(y_values)])
            x_pred_grid, x_cov_grid = evaluate_model(
                meas_grid, params, cov_matrix, func
            )
            sigma = np.sqrt(np.diag(x_cov_grid))
            upper = x_pred_grid + 1.96 * sigma
            lower = x_pred_grid - 1.96 * sigma
            
            self.ax.fill_between(
                y_values, lower, upper,
                alpha=0.3, label='95 % Konfidenzband'
            )
            
            # 6) Messpunkte mit Fehlerbalken
            ux = np.sqrt(np.diag(self.x_cov))
            self.ax.errorbar(
                meas_data[:, 0], self.x_pred,
                xerr=meas_data[:, 1], yerr=ux,
                fmt='s', label='Measurement Points',
                color='green'
            )

            # 7) Achsen, Titel, Legende
            self.ax.set_xlabel('y')
            self.ax.set_ylabel('x')
            self.ax.set_title('Calibration and Measurement Data')
            self.ax.legend(loc='upper left')
            
            # 8) Residuenplot
            n = len(cal_data)
            res_x = self.residuals[:n]
            res_y = self.residuals[n:]
            self.ax_residuals.plot(
                   cal_data[:, 2], res_x, 'x',
                   color='blue', label='Res. x-dir'
            )
            self.ax_residuals.plot(
                cal_data[:, 2], res_y, 'o',
                color='blue', label='Res. y-dir'
            )
            self.ax_residuals.axhline(0,
                              color='red',
                              linestyle='--')
            self.ax_residuals.set_xlabel('y')
            self.ax_residuals.set_ylabel('Weighted Residuals')
            self.ax_residuals.set_title('Calibration Residuals')
            self.ax_residuals.legend(loc='upper left', bbox_to_anchor=(0, -0.3), borderaxespad=0)
            
            # 9) Layout und neu zeichnen
            self.figure.subplots_adjust(hspace=0.3)
            self.canvas.draw()

    if __name__ == "__main__":
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        
        root = ctk.CTk()
        root.geometry("1500x1000")
        root.state("zoomed")
        
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        
        canvas = tk.Canvas(root)
        canvas.grid(row=0, column=0, sticky="nsew")
        
        scrollbar_y = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        
        scrollbar_x = tk.Scrollbar(root, orient="horizontal", command=canvas.xview)
        scrollbar_x.grid(row=1, column=0, sticky="ew")
        
        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        main_frame_a = ctk.CTkFrame(canvas)
        canvas.create_window((0, 0), window=main_frame_a, anchor="nw")
        
        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        main_frame_a.bind("<Configure>", on_configure)
        
        app = CalibrationApp(root, main_frame_a)
        root.mainloop()


