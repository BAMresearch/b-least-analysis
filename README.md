# b-least-calibration 
*Analysis according to ISO 6143 and beyond*

A comprehensive Python application for calibration modeling — both linear and nonlinear — with full uncertainty propagation, visual inspection, automated reporting (PDF/CSV), and model comparison. Supports a wide range of functions from polynomials and exponentials to sigmoids, and complies with ISO 6143 calibration methodology.

## Features
- ISO 6143-compliant calibration analysis

- Supports 14 different calibration models

- Full uncertainty propagation (incl. off-diagonal covariances)

- Automatic parameter estimation and robust fitting

- Intuitive GUI using Tkinter + CustomTkinter

- Visual output via Matplotlib (with confidence intervals)

- Export results as PDF (with graphs) or CSV

- Model diagnostics: RMSE, residual sum, Gamma (Γ)

- Weighted residual analysis

- Extensible architecture – easy to add new models or logic

## GUI Overview
The main interface consists of three panels:

**Left Panel:** Data loading, model selection, and analysis controls

**Center Panel:** Results, fitted equations, uncertainties

**Right Panel:** Plot with calibration points, fitted curve, confidence band

## Supported Models
| Category                      | Model Name                  | Equation                                             |
|------------------------------|-----------------------------|------------------------------------------------------|
| ISO 6143 Compatible           | Linear                      | `x = b0 + b1·y`                                      |
|                              | Polynomial (order n)        | `x = b0 + b1·y + … + bn·yⁿ`                          |
|                              | Exponential                 | `x = b0 + b1·exp(b2·y)`                              |
|                              | Power Function              | `x = b0 + b1·y^(1 + b2)`                             |
| Exponential Growth            | Simple Exponential          | `x = b0·exp(b1·y)`                                   |
|                              | Exponential w/ Offset       | `x = b0 + b1·(exp(b2·y) - 1)`                        |
| Saturation Models             | Rational Function           | `x = (b0 + b1·y) / (1 + b2·y)`                       |
|                              | Hyperbolic Function         | `x = b0 + b1 / (y + b2)`                             |
| Logarithmic Models            | Logarithmic Function        | `x = b0 + b1·ln(b2·y)`                               |
|                              | Log-Linear Function         | `x = b0 + b1·ln(y) + b2·y`                           |
| Biological Curves             | Gompertz Function           | `x = b0·exp(-b1·exp(-b2·y))`                         |
|                              | Sigmoid Function            | `x = b0 / [1 + exp(-b1·(y - b2))]`                   |
| Diffusion-like Models         | Square Root Function        | `x = b0 + b1·√y`                                     |

## Installation

### Requirements

- Python **3.9** or newer
- Recommended: use a virtual environment (`venv` or `conda`)

### Install libraries
- [numpy](https://pypi.org/project/numpy/)
- [pandas](https://pypi.org/project/pandas/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [reportlab](https://pypi.org/project/reportlab/)
- [customtkinter](https://pypi.org/project/customtkinter/)

## How to Use

```bash
python main.py
```
The GUI launches in full screen mode.

### Workflow
**1. Load Calibration Data:** CSV/TXT with columns x, u(x), y, u(y)

**2. Load Measurement Data:** CSV/TXT with columns y, u(y)

**3. Select Function Model (e.g., polynomial, exponential, etc.)**

**4.** Manually enter known covariances between calibration points *(currently in beta)*  
 Format: `2/4 0.0012, 3/4 0.0005`  
 ⚠️ Note: This feature is under development and may produce incorrect results in some cases.

**5. Run Analysis**

**6. Review Results:**

- Function equation + uncertainty per coefficient

- Residual analysis

- Fit quality metrics

**7. Export:**

- PDF Report with graph & optional model comparison

- CSV with full results and metadata

## Example File Format
### Calibration Data
```
x;u(x);y;u(y)
10.2;0.1;0.5;0.01
11.8;0.1;1.0;0.01
13.5;0.1;1.5;0.01
```
### Measurement Data
```
y;u(y)
0.8;0.01
1.4;0.01
```

## Analysis Options
- Weighted Residuals: both x- and y-direction

- Sum of Squared Residuals (SSR): compared to ISO acceptance threshold

- Goodness-of-Fit Γ: maximum normalized deviation

- Root Mean Squared Error (RMSE)

## Export Contents
### CSV Includes:

- Metadata (timestamp, selected function, user input)

- Calibration + Measurement Data

- Parameter estimates with uncertainties

- Covariance matrices (model parameters + predictions)

- Residuals + Diagnostics

### PDF Includes:
- Summary report in fixed-width layout

- Full analysis equation + parameter table

- Model graph with:

    - Calibration points (± uncertainty)

    - Measurement predictions

    - 95% confidence band

- Optional model comparison (all models + RMSE/Γ/SSR)

## Extensibility
The code is modular and extensible:

  - Add your own models via `compute_<model>()` functions

  - Extend GUI options with minimal changes

  - All visualizations and exports are handled via standard libraries (matplotlib, reportlab)

## License
Licensed under the [MIT License](LICENSE).

## Author
Tino Golub - Bundesanstalt für Materialforschung und -prüfung

Last updated: July 2025
 
Feel free to open issues or submit pull requests with suggestions or contributions.
