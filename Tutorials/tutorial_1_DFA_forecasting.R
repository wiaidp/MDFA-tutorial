# ════════════════════════════════════════════════════════════════════
# TUTORIAL 1: INTRODUCTION TO MDFA
# (Multivariate Direct Filter Approach)
# ════════════════════════════════════════════════════════════════════

# The MDFA (Multivariate Direct Filter Approach) provides a unified framework for solving 
# general prediction problems while simultaneously accommodating specific research priorities and objectives.

# ── THEORETICAL FOUNDATIONS ───────────────────────────────────────
# The theoretical foundations and principles underlying MDFA are
# documented in the following references:

# ── I. BOOKS ──────────────────────────────────────────────────────
#
#   Wildi, M. (2005)
#     Signal Extraction: Efficient Estimation, 'Unit Root'-Tests
#     and Early Detection of Turning Points.
#     Lecture Notes in Economics and Mathematical Systems, Springer.
#     https://doi.org/10.1007/b138291
#
#   Wildi, M. & McElroy, T. (in preparation)
#     A new book on MDFA is currently in preparation.

# ── II. ARTICLES ──────────────────────────────────────────────────
#
#   Wildi, M. & McElroy, T. (2019)
#     The trilemma between accuracy, timeliness and smoothness in
#     real-time signal extraction.
#     International Journal of Forecasting, Vol. 35, Issue 3.
#
#   Wildi, M. & McElroy, T. (2016)
#     Optimal real-time filters for linear prediction problems.
#     Journal of Time Series Econometrics, Vol. 8, Issue 2.
#
#   Wildi, M. & McElroy, T. (2020)
#     The multivariate linear prediction problem: Model-based and
#     direct filtering solutions.
#     Econometrics and Statistics, Vol. 14.
#
#   Quast, J., van Norden, S. & Wildi, M. (2026)
#     Credit cycles and credit crises: Some measurement issues
#     and implications.
#     Working paper submitted to the 2026 SNB Research Conference
#     (October 2–3, 2026).

# ── MDFA, M-SSA AND DFP/PCS PREDICTORS: A COMPARATIVE OVERVIEW ─────────────
#
# Historical context:
#   • DFA/MDFA   → origins in 2002 research and culminates in new MDFA book coauthored with Tucker McElroy (MDFA tutorials repository on github)
#   • M-SSA      → developed from early 2020 (M-SSA tutorials repository on github)
#   • DFP/PCS    → developed from mid 2020 (I'm on it)
#
# Common ground:
#   All three prediction frameworks are organized around the forecast trilemma,
#   jointly addressing Accuracy, Timeliness, and Smoothness —
#   albeit with practically relevant differences in formulation
#   and interpretation.
#
# Key distinctions:
#
#   • Domain
#       → MDFA operates in the frequency domain
#       → M-SSA and DFP/PCS are formulated in the time domain
#
#   • Trilemma decomposition in MDFA
#       → MSE is decomposed into amplitude and phase contributions,
#         which define the smoothness and timeliness terms
#         respectively — see cited literature for details
#
#   • Smoothness in M-SSA
#       → Measured as the mean duration between consecutive
#         sign changes of a zero-mean predictor (holding-time)
#       → Measured as the mean duration between consecutive
#         sign changes of a zero-mean predictor (holding-time)
#       → Yields more direct and intuitive interpretation
#         than the MDFA amplitude-based formulation
#       → Extends to max-monotonic and min-curvature 
#         predictors for integrated processes.
#
#   • Timeliness in DFP/PCS
#       → Quantified via the effective time-shift of the predictor
#         (rather than phase in the frequency domain for MDFA)
#       → Yields more direct and intuitive interpretation
#         than the MDFA phase-based formulation
# ════════════════════════════════════════════════════════════════════
# This introductory tutorial to the MDFA covers the univariate DFA. It consists of 6 exercises covering the
# foundations of (M)DFA.

# ── PURPOSE ───────────────────────────────────────────────────────
# The (univariate) Direct Filter Approach (DFA) can replicate classical MSE-based
# one- and multi-step ahead forecasting by:
#
#   • Specifying a corresponding target (allpass filter) and
#     forecast horizon (Lag)
#   • Specifying a corresponding spectral estimate
#
# A central goal of this tutorial is to show how classical
# (e.g., ARIMA-based) forecasting can be fully replicated within the DFA
# framework.
#   → Once grounded in this familiar territory, subsequent tutorials
#     deploy the additional flexibility of DFA beyond classical MSE (customization)

# ── NON-PARAMETRIC DFA ────────────────────────────────────────────
# The tutorial concludes by illustrating the relevance of a
# non-parametric DFA approach, based on:
#   • The Discrete Fourier Transform (DFT)
#   • Burg's maximum-entropy spectral estimate
#
# Key findings:
#
#   1. Out-of-sample forecast performance of (non-parametric) DFT-based DFA is
#      generally close to (or indistinguishable) from the universally best approach
#      (which assumes full knowledge of the true data-generating
#      process). Some care must be taken to control for overfitting (as shown along the tutorials).
#
#   2. Forecast performances of DFT-based DFA and max-entropy-based
#      DFA are mutually indistinguishable
#
# Implication:
#   → These results support the use of the non-parametric
#     (DFT-based) DFA approach across a wide range of applications —
#     especially when model misspecification is a concern,
#     as is invariably the case with real-world data

# ── FUNCTIONS AND PARAMETERS ──────────────────────────────────────
# Throughout this tutorial, we rely on the function MDFA_mse:
#   → The MSE-norm wrapper to the generic MDFA estimation routine
#
# By omitting advanced features (covered in later tutorials),
# this wrapper allows the focus to remain on the core problem
# structure.
#
# Specifically, we learn to handle the parameters of MDFA_mse
# by replicating a classical one-step ahead forecast framework,
# establishing a direct bridge to standard time series approaches.
# ─────────────────────────────────────────────────────────────────


rm(list=ls())

library(xts)
#install.packages("devtools")
library(devtools)
# Load MDFA package from github
devtools::install_github("wiaidp/MDFA")
# MDFA package: EURUSD is now part of the data in the package
library(MDFA)


# Brief overview of wrappers and main MDFA function
head(MDFA_mse)
head(MDFA_mse_constraint)
head(MDFA_cust)
head(MDFA_cust_constraint)
head(MDFA_reg)
head(MDFA_reg_constraint)
# Main estimation function
head(mdfa_analytic)

#-----------------------------------------------------------------------------------------------
# Source common functions

source("Common functions/plot_func.r")
source("Common functions/arma_spectrum.r")


#==================================================================================
# Example 1: One-Step Ahead Forecasting
#==================================================================================
#
# Overview:
#   DFA requires two user-supplied inputs:
#     1. A 'target' filter (Gamma): defines the desired output in the frequency domain
#     2. A 'spectrum': characterizes the stochastic properties of the input data
#   DFA then computes an optimal filter estimate under one of two criteria:
#     - MSE (Minimum Mean-Square Error)
#     - ATS-trilemma ('beyond MSE': balances Accuracy, Timeliness, and Smoothness)
#
# This example demonstrates how to configure a classic MSE one-step (or multi-step) ahead forecast using DFA.
#
# Design choices made in this example:
#   - Criterion    : MSE (Mean-Square Error)
#   - Model type   : Univariate (single input series)
#   - Spectrum type: White noise (flat, constant power across all frequencies)
#-------------------------------------------------------------------------------------------------

# Frequency grid setup:
#   DFA operates in the frequency domain; see McElroy/Wildi (references in the Literature folder).
#   - The frequency grid consists of equally spaced points: omega_j = j * pi / K, j = 0, 1, ..., K
#   - K controls the grid resolution (density):
#       * Larger K → finer grid → better MSE approximation (assuming true white noise process)
#       * Larger K → increased computation time
#   - Here we partition [0, pi] into K = 600 equally spaced frequency points
K <- 600

# Spectrum definition (white noise assumption):
#   - weight_func is a (K+1) x 2 matrix:
#       * Column 1: spectrum of the target series
#       * Column 2: spectrum of the explanatory (input) series
#   - In a univariate design, both columns are identical (same series used for target and input)
#   - White noise has a flat spectrum: all frequencies carry equal power
#   - Any strictly positive constant is a valid white noise spectrum value
weight_func <- matrix(rep(1, 2 * (K + 1)), ncol = 2)
colnames(weight_func) <- c("spectrum target", "spectrum explanatory")
head(weight_func)

# Plot the white noise spectrum:
#   - A flat (horizontal) line confirms that all frequencies contribute equally
plot(weight_func[, 1], type = "l",
     main = paste("White noise spectrum, denseness =", K, sep = ""),
     axes = F, xlab = "Frequency", ylab = "Amplitude", col = "black")
mtext(colnames(weight_func)[1], line = -1, col = "black")
axis(1, at = c(0, 1:6 * K / 6 + 1),
     labels = c("0", "pi/6", "2pi/6", "3pi/6", "4pi/6", "5pi/6", "pi"))
axis(2)
box()

# Target filter definition (Gamma):
#   - The target Gamma specifies the desired frequency response of the estimated filter
#   - Two common target types:
#       * Forecasting (all frequencies): Gamma = 1 for all frequencies → 'allpass' filter
#       * Trend extraction/nowcasting  : Gamma ≈ 1 for low frequencies, Gamma ≈ 0 for high frequencies
#       * Cyle extraction/nowcasting  : Gamma ≈ 1 for frequencies in a passband without zero frequency, Gamma ≈ 0 for high frequencies
#   - Important distinction from the spectrum:
#       * The spectrum values are arbitrary positive constants (negative or complex-valued spectra are `rotated' to positive real numbers)
#       * Gamma is normalized: ideally passes each frequency component with unit gain (= 1) when forecasting
#   - Here we use an allpass target (Gamma = 1), appropriate for general forecasting
Gamma <- rep(1, K + 1)

# Plot the allpass target filter:
#   - A flat line at amplitude = 1 confirms all frequencies are treated equally
plot(Gamma, type = "l",
     main = paste("Allpass forecast target, denseness =", K, sep = ""),
     axes = F, xlab = "Frequency", ylab = "Amplitude", col = "black")
mtext("Target", line = -1, col = "black")
axis(1, at = c(0, 1:6 * K / 6 + 1),
     labels = c("0", "pi/6", "2pi/6", "3pi/6", "4pi/6", "5pi/6", "pi"))
axis(2)
box()

# Forecast horizon:
#   - Lag = -1 → one-step ahead forecast (negative Lag = forecasting)
#   - Lag =  0 → nowcast (current period estimate)
#   - Lag > 0  → backcast (smoothing past values)
Lag <- -1

# Filter length:
#   - L defines the number of filter coefficients (weights) used in the convolution
#   - Larger L → more flexible filter, but higher risk of overfitting for small samples
L <- 10

# Estimate the optimal MSE filter using the MDFA-MSE wrapper:
#   - Inputs : filter length (L), frequency-domain spectrum (weight_func),
#              forecast horizon (Lag), and target filter (Gamma)
#   - Output : mdfa_obj contains the estimated filter coefficients and diagnostics
mdfa_obj <- MDFA_mse(L, weight_func, Lag, Gamma)$mdfa_obj

#-------------------------------------------------------------------------------------------------
# Diagnostic Plot 1: Filter Coefficients
#   - Under a true white noise process, all optimal filter coefficients are theoretically zero
#     (a white noise process is unpredictable; no filter can improve on the unconditional mean)
#   - The estimated coefficients are close to, but not exactly, zero because:
#       * The frequency grid has finite resolution (K = 600), introducing a small approximation error
#       * Increasing K drives estimates closer to zero, at the cost of longer computation
#   - This slight deviation from zero represents mild numerical overfitting,
#     which is negligible in practice when K / L > 10 (as verified below)
#-------------------------------------------------------------------------------------------------
par(mfrow = c(1, 1))
b <- mdfa_obj$b
colo <- rainbow(ncol(b))
plot(b[, 1], type = "l",
     main = "Filter Coefficients",
     axes = F, xlab = "Lag", ylab = "Coefficient", ylim = c(-1, 1), col = "black")
axis(1, at = 1:L, labels = 1:L)
axis(2)
box()

#-------------------------------------------------------------------------------------------------
# Diagnostic Plot 2: Full Filter Summary (via plot_estimate_func)
#   This function produces a three-panel summary of the estimated filter:
#     1. Filter coefficients (same as above)
#     2. Time-shift (phase delay) across frequencies
#     3. Amplitude (gain) function across frequencies
#
# Interpretation:
#   - Amplitude: the estimated gain (black line) is close to zero across all frequencies,
#     as expected for a white noise process (no forecastable signal exists)
#     * Small deviations from zero are due to the finite frequency grid (K < ∞)
#   - Time-shift: not meaningful here since the amplitude is effectively zero
#     (zero-amplitude components carry no information, so their phase is irrelevant)
#   - Rule of thumb: when K / L > 10, the finite-grid approximation error is negligible
#-------------------------------------------------------------------------------------------------
plot_estimate_func(mdfa_obj, weight_func, Gamma)




#==================================================================================
# Example 2: Classic One- and Multi-Step Ahead Forecasting for ARMA Processes
#            (Replicates State-of-the-Art ARMA Model Forecasts)
#==================================================================================

#-----------------------------------
# Frequency Grid Setup
# Partition [0, pi] into K equally spaced frequency points: omega_j = j*pi/K, j = 0, 1, ..., K
K <- 600

# Target Filter (Gamma):
#   - For forecasting, all frequency components are of equal interest
#   - Gamma is an allpass filter: passes all frequencies with unit gain
#   - Being forward-looking (Lag < 0), it defines the desired future signal
Gamma <- rep(1, K + 1)

#-----------------------------------
# ARMA Process Specification
# Experiment with different ARMA(p, q) configurations by setting AR and MA coefficients:
#   - AR(1): set a1 (AR coefficient) and b1 <- NULL (no MA component)
#   - MA(1): set b1 (MA coefficient) and a1 <- NULL (no AR component)
a1 <- 0.9   # AR(1) coefficient (comment out for pure MA)
b1 <- NULL  # MA coefficient (set to NULL for pure AR)
a1 <- NULL  # Comment out the AR coefficient to switch to pure MA
b1 <- 0.6   # MA(1) coefficient

# Toggle spectral plot: set to TRUE to visualize the ARMA spectrum
plot_T <- TRUE

# Compute the ARMA spectrum over the frequency grid [0, pi]
# Output: spectral density evaluated at each of the K+1 frequency points
par(mfrow=c(1,1))
spec <- arma_spectrum_func(a1, b1, K, plot_T)$arma_spec

#-----------------------------------
# Spectral Weight Matrix (weight_func)
# For univariate DFA, the target and explanatory series are identical,
# so both columns of weight_func share the same spectrum
weight_func <- abs(cbind(spec, spec))
colnames(weight_func) <- c("spectrum target", "spectrum explanatory")
head(weight_func)

#-----------------------------------
# Forecast Horizon (Lag)
# Lag controls the type and horizon of the estimation problem:
#   1. k-step ahead forecast : Lag = -k (negative integer)
#      Fractional Lag (e.g., Lag = -1.5): interpolates between two future time points
#   2. Nowcast (current period): Lag = 0
#   3. k-step backcast         : Lag = +k (positive integer)
#      Fractional Lag (e.g., Lag = +1.5): interpolates between two past time points
Lag <- -1   # One-step ahead forecast

# Filter Length (L):
# Number of filter coefficients (filter weights) used in the DFA estimator
# Rule of thumb: K/L > 10 ensures negligible discretization error
L <- 10

#-----------------------------------
# DFA Estimation via MDFA-MSE Wrapper
# Solves for the MSE-optimal causal filter given the spectrum, target, and lag
mdfa_obj <- MDFA_mse(L, weight_func, Lag, Gamma)$mdfa_obj

# Inspect available output components of the MDFA object
names(mdfa_obj)

# Display estimated filter coefficients (one-step ahead predictor weights)
mdfa_obj$b

#-----------------------------------
# Visualization
# Plot the estimated filter's:
#   1. Coefficients (filter weights in the time domain)
#   2. Time-shift   (phase delay across frequencies)
#   3. Amplitude    (gain function across frequencies; black = one-step ahead predictor)
plot_estimate_func(mdfa_obj, weight_func, Gamma)

#-----------------------------------
# Theoretical Benchmarks and Interpretation
#
# 1. Forecasting (Lag < 0, integer-valued):
#
#    AR(1) process:
#      - Lag = -1: optimal filter weights = (a1,   0,   0, ..., 0)
#      - Lag = -2: optimal filter weights = (a1^2, 0,   0, ..., 0)
#      - Lag = -k: optimal filter weights = (a1^k, 0,   0, ..., 0)
#      - DFA recovers these weights as K increases (denser frequency grid)
#      - For K/L > 10, the discretization error is negligible
#
#    MA(1) process:
#      - Lag = -1: optimal filter weights = (b1, -b1^2, b1^3, -b1^4, ...)
#                  (geometrically decaying, alternating-sign series)
#      - Lag = -2: optimal filter weights = (0, 0, 0, ..., 0)
#                  (MA(1) has no memory beyond one step: future beyond h=1 is unpredictable)
#      - DFA recovers these weights as K increases
#      - For K/L > 10, the discretization error is negligible
#
# 2. Nowcasting (Lag = 0):
#    - Optimal filter for any ARMA process: weights = (1, 0, 0, ..., 0)
#    - Trivial for an allpass target; becomes non-trivial for lowpass/bandpass/highpass targets
#
# 3. Backcasting (Lag > 0):
#    - Optimal filter places weight 1 at the lag specified by Lag, and 0 elsewhere
#    - Trivial for an allpass target; becomes non-trivial for lowpass/bandpass/highpass targets
#
# 4. Fractional Lag:
#    - Non-integer Lag values yield a filter that smoothly interpolates the signal
#      between two consecutive time points (past or future)
#
# 5. Time-Shift Interpretation:
#    - A negative time-shift indicates a lead (phase advancement): desirable for forecasting
#    - Ideal predictor: time-shift equals Lag (and amplitude equals one) uniformly across all frequencies 
#    - Deviations from this ideal indicate frequency-dependent forecasting distortions

# 6. Amplitude Interpretation:
#    - weight assigned by filter to a component with frequency omega: >1 (amplify), <1 (damp), =1 (pass `as is') 
#    - Ideal predictor: amplitude equals one uniformly across all frequencies 
#    - Deviations from this ideal indicate frequency-dependent forecasting distortions

# 7. The plot suggests that time-shift (!=Lag) and amplitude (!=1) differ from ideal
#    -MSE signifies that the deviations from ideal are such that the (squared) forecast error is minimized 
#    -But we could want to assign more weight to amplitude fitting: closer to ideal value 1 
#    -Or we could assign more weight to time-shift fitting: closer to ideal value -1
#    -DFA allows for such tweaking (customization) of the predictor

# 8. Replication of classic MSE by DFA
#    -DFA has replicated classic forecast approach (up to negligible finite sample deviations)


#==================================================================================
# Example 3: Comparing DFA with Classic ARIMA Forecasting in R
#==================================================================================
#
# Objective:
#   Demonstrate that DFA replicates the classic arima function in R, at arbitrary forecast horizons h>0.
#   For sufficiently large K (frequency resolution), the two forecasts become virtually indistinguishable.
#   DFA generates `direct' multi-step ahead forecasts vs. iterated forecasts for R-function predict

#-----------------------------------
# Simulation Setup

# Forecast horizon: number of steps ahead to forecast
h <- 5

# Sample size of the simulated time series
len <- 300

# Simulate an ARMA(1,1) process with known AR and MA coefficients
# AR coefficient: a1 = 0.6 (autoregressive memory)
# MA coefficient: b1 = 0.7 (moving average innovation loading)
a1 <- 0.6
b1 <- 0.7
set.seed(0)   # Set seed for reproducibility
x <- arima.sim(n = len, list(ar = a1, ma = b1))

#-----------------------------------
# Classic ARIMA Estimation and Forecasting

# Fit an ARIMA(1,0,1) model (no differencing, no intercept) to the simulated data
# This recovers estimates of a1 and b1 from the data
arima_obj <- arima(x, order = c(1, 0, 1), include.mean = FALSE)

# Diagnostic check: inspect residual autocorrelations, p-values, and standardized residuals
# A well-specified model should show no significant residual autocorrelation
tsdiag(arima_obj)

# Compute h-step ahead forecasts using the fitted ARIMA model (iterated one-step rule)
arima_pred <- predict(arima_obj, n.ahead = h)$pred
arima_pred
#-----------------------------------
# DFA Setup

# Frequency grid: partition [0, pi] into K equally spaced points
# Larger K = denser grid = better MSE approximation, at the cost of longer computation
K <- 600

# Target Filter (Gamma):
#   - Allpass filter: all frequency components are equally relevant for forecasting
#   - Forward-looking: the negative Lag below defines the forecast horizon
Gamma <- rep(1, K + 1)

#-----------------------------------
# ARMA Spectrum for DFA
# Extract the estimated AR and MA coefficients from the fitted ARIMA model
# These are used to construct the spectral weight function for DFA,
# ensuring both methods operate under the same model assumptions
a1 <- arima_obj$coef["ar1"]
b1 <- arima_obj$coef["ma1"]

# Toggle spectral plot: TRUE displays the estimated ARMA spectral density
plot_T <- TRUE

par(mfrow=c(1,1))
# Compute the ARMA spectral density over the K+1 frequency grid points
spec <- arma_spectrum_func(a1, b1, K, plot_T)$arma_spec

# Construct spectral weight matrix:
#   - Column 1: spectrum of the target series
#   - Column 2: spectrum of the explanatory series
#   - In a univariate design both columns are identical
weight_func <- abs(cbind(spec, spec))
colnames(weight_func) <- c("spectrum target", "spectrum explanatory")
head(weight_func)

#-----------------------------------
# Filter Length
# L controls the number of filter coefficients (lags) used in the DFA estimator
#   - For stationary ARMA processes (short memory), L = 10 is typically sufficient
#   - Exception: seasonal data may require L to span at least one full seasonal cycle
#     (e.g., L = 13 for monthly data with annual seasonality)
L <- 10

#-----------------------------------
# DFA Multi-Step Ahead Forecasting (Direct Approach)
# Compute forecasts for horizons h = 1, 2, ..., h using DFA
#
# Key distinction from ARIMA:
#   - ARIMA forecasts are 'iterated': the one-step model is applied recursively
#   - DFA forecasts are 'direct': a separate optimal filter is estimated for each horizon
#   - The two approaches are equivalent when DFA relies on the model-based spectrum,
#     but may differ when using alternative (e.g.,non-parametric) spectra in DFA: see example 4 below

dfa_forecast <- rep(NA, h)

for (i in 1:h)  # Loop over forecast horizons i = 1, ..., h
{
  Lag <- -i   # Negative Lag = i-step ahead forecast horizon
  
  # Estimate the MSE-optimal DFA filter for this horizon
  mdfa_obj <- MDFA_mse(L, weight_func, Lag, Gamma)$mdfa_obj
  
  # Extract filter coefficients
  b <- mdfa_obj$b
  
  # Apply the filter to the most recent L observations of x
  # Inner product of filter weights with the last L data points gives the forecast
  dfa_forecast[i] <- t(b) %*% x[length(x):(length(x) - L + 1)]
}

#-----------------------------------
# Forecast Comparison: ARIMA vs. DFA
#
# Both forecasts should be virtually identical when the frequency grid K is sufficiently dense (K/L > 10)
# Any residual difference vanishes as K -> infinity

# Construct comparison matrix:
#   - Column 1: ARIMA forecasts (preceded by NA padding for plotting alignment)
#   - Column 2: DFA forecasts   (preceded by NA padding for plotting alignment)
#   - Column 3: Observed data   (last 21 observations + NA placeholder for forecast period)
forecast_comparison <- cbind(
  c(rep(NA, 21), arima_pred),          # ARIMA h-step ahead forecasts
  c(rep(NA, 21), dfa_forecast),         # DFA h-step ahead forecasts
  c(x[(len - 20):len], rep(NA, h))      # Recent observations for visual context
)

# Plot: observed data (black), ARIMA forecasts (red), DFA forecasts (green dashed)
# Vertical line at t = 21 marks the boundary between observed data and forecast period
ts.plot(forecast_comparison[, 1],
        ylim = c(min(forecast_comparison, na.rm = TRUE),
                 max(forecast_comparison, na.rm = TRUE)),
        main = "Data (black); Classic ARIMA forecast (red) and DFA (green)",
        col = "red")
lines(forecast_comparison[, 2], col = "green", lty = 2)
lines(c(x[(len - 20):len], rep(NA, h)), col = "black")
abline(v = 21)  # Vertical line separating in-sample data from forecast horizon









#==========================================================================================
# Example 4: Non-Parametric DFA Forecasting via the Discrete Fourier Transform (DFT)
#==========================================================================================
#
# Objective:
#   Replace the model-based ARMA spectrum (used in Example 3) with a non-parametric
#   spectral estimate derived directly from the data via the Discrete Fourier Transform (DFT).
#
# Motivation:
#   In practice, the true data-generating process is rarely known.
#   The DFT provides a model-free spectral estimate that can be plugged directly
#   into DFA without assuming any parametric ARMA structure.
#
# Key trade-off:
#   - Model-based spectrum (Example 3): smooth, efficient, but requires correct model specification
#   - DFT-based spectrum (this example): flexible and assumption-free, but noisy and
#     susceptible to overfitting, especially for large filter lengths L

#-----------------------------------
# Non-Parametric Spectrum: Discrete Fourier Transform (DFT)

# Compute the DFT of x and use it as the spectral weight function
# per() returns the periodogram/DFT of the series x
#   - Setting the second argument to TRUE plots the DFT
# For a univariate design, both columns (target and explanatory) share the same spectrum
weight_func <- abs(cbind(per(x, TRUE)$DFT, per(x, TRUE)$DFT))
colnames(weight_func) <- c("spectrum target", "spectrum explanatory")

par(mfrow = c(1, 1))

# Visualize the DFT-based spectrum
# Note: the DFT is a noisy, sample-based approximation of the smooth ARMA spectrum
# from Example 3. The irregular spikes reflect sampling variability, not true spectral features.
ts.plot(weight_func, main = "DFT-Based Spectral Estimate of ARMA Process")

#-----------------------------------
# Frequency Grid
# The frequency resolution of the DFT is determined by the sample size:
#   K + 1 = number of DFT frequency points (length of weight_func rows)
#   omega_j = j * pi / K, j = 0, 1, ..., K
# Unlike Example 3 (where K = 600 was user-specified), K here is data-driven
K <- nrow(weight_func) - 1

# Target Filter (Gamma):
#   - Allpass filter: all frequency components contribute equally to the forecast
#   - Forward-looking: the negative Lag below defines the forecast horizon
Gamma <- rep(1, K + 1)

#-----------------------------------
# Filter Length
# L controls the number of filter coefficients used in DFA
#   - Most economic time series exhibit short memory (e.g., stationary AR or MA processes),
#     so large L is rarely beneficial and may increase overfitting risk
#   - Examples of short-memory series:
#       * Price series: approximately random walks (after differencing)
#       * Log-returns:  approximately white noise
#   - Overfitting under non-parametric spectra is more pronounced than under model-based
#     spectra, and grows with L (addressed in a later tutorial on regularization)
L <- 10

#-----------------------------------
# DFA Multi-Step Ahead Forecasting (Direct Approach)
# Compute direct h-step ahead forecasts for horizons i = 1, 2, ..., h
# A separate MSE-optimal filter is estimated independently for each horizon

dfa_forecast <- rep(NA, h)

for (i in 1:h)  # Loop over forecast horizons
{
  Lag <- -i   # Negative Lag = i-step ahead forecast horizon
  
  # Estimate the MSE-optimal DFA filter for horizon i
  # DFA now uses the DFT-based (non-parametric) spectrum instead of the ARMA spectrum
  mdfa_obj <- MDFA_mse(L, weight_func, Lag, Gamma)$mdfa_obj
  
  # Extract filter coefficients
  b <- mdfa_obj$b
  
  # Apply the filter: inner product of filter weights with the L most recent observations
  dfa_forecast[i] <- t(b) %*% x[length(x):(length(x) - L + 1)]
}

#-----------------------------------
# Forecast Comparison: ARIMA vs. DFA (Non-Parametric)
#
# Construct comparison matrix for plotting:
#   - Column 1: ARIMA forecasts     (NA-padded for alignment)
#   - Column 2: DFA forecasts       (NA-padded for alignment)
#   - Column 3: Observed data       (last 21 observations + NA for forecast period)

par(mfrow = c(1, 1))

forecast_comparison <- cbind(
  c(rep(NA, 21), arima_pred),        # Classic ARIMA h-step ahead forecasts
  c(rep(NA, 21), dfa_forecast),       # Non-parametric DFA h-step ahead forecasts
  c(x[(len - 20):len], rep(NA, h))    # Recent observations for visual context
)

# Plot: observed data (black), ARIMA forecasts (red), DFA forecasts (green dashed)
# Vertical line at t = 21 marks the boundary between observed data and forecast horizon
ts.plot(forecast_comparison[, 1],
        ylim = c(min(forecast_comparison, na.rm = TRUE),
                 max(forecast_comparison, na.rm = TRUE)),
        main = "Data (black); Classic ARIMA Forecast (red) and DFA Non-Parametric (green)",
        col = "red")
lines(forecast_comparison[, 2], col = "green", lty = 2)
lines(c(x[(len - 20):len], rep(NA, h)), col = "black")
abline(v = 21)  # Vertical line separating observed data from the forecast horizon

#-----------------------------------
# Interpretation and Limitations
#
# 1. Forecast similarity:
#    - Non-parametric DFA forecasts are close to, but visually slightly different from,
#      the model-based ARIMA forecasts
#    - The difference arises because the DFT is a noisy spectral estimate:
#      it captures sample-specific fluctuations rather than the `true' smooth spectrum
#    -Model-based spectrum is also an approximation of the true spectrum. In constrast to DFT it is smooth. 
#
# 2. Overfitting risk:
#    - The DFT-based spectrum introduces estimation noise into the filter optimization
#    - This noise is amplified for larger L (more filter coefficients to estimate)
#    - In contrast, the smooth ARMA spectrum in Example 3 is less prone to overfitting by DFA
#    - But the ARMA-parameters are also overfitted (they differ from true values)
#
# 3. Regularization (upcoming tutorial):
#    - Overfitting under non-parametric spectra can be mitigated via regularization,
#      which penalizes filter complexity and smooths the estimated coefficients
#    - This will be the focus of a subsequent tutorial in this series



# =============================================================================
# Example 5: Out-of-Sample Comparison of Non-Parametric DFA vs. Best Possible Forecast
# =============================================================================
# Goal:
#   For each simulated realization, we compute one-step-ahead out-of-sample
#   forecasts using two approaches:
#     (1) Non-parametric DFA (based on the Discrete Fourier Transform, DFT)
#     (2) ARMA-based forecast assuming true model orders (no model identification)
#   We then compare their Mean Squared Forecast Errors (MSFE).
#
# Key observations:
#   1. The non-parametric DFA requires NO a priori knowledge of the underlying
#      model structure — it is entirely data-driven via the DFT.
#   2. The ARMA-based forecast represents an idealized benchmark:
#      - We assume knowledge of the true model orders.
#      - Under these conditions, ARMA (via MLE) is the best possible forecast.
#      - This makes it a strong, favorable benchmark for the DFA approach.
#   3. Interpretation: if DFA performs comparably to this ideal benchmark,
#      it validates the DFT as a meaningful and efficient statistic for forecasting.
# =============================================================================

# -----------------------------------------------------------------------------
# Process Specifications (choose one or define your own)
# -----------------------------------------------------------------------------

# Example 5.1: Pure MA(1) process
a1 <- 0.0
b1 <- 0.7

# Example 5.2: ARMA(1,1) process with positive autocorrelation
a1 <- 0.6
b1 <- 0.7

# Example 5.3: AR(1) process with strong negative autocorrelation
a1 <- -0.9
b1 <- 0.0

# Additional processes can be defined here by modifying a1 and b1 accordingly.

# -----------------------------------------------------------------------------
# Simulation Setup
# -----------------------------------------------------------------------------

set.seed(1)           # Set seed for reproducibility
len     <- 300        # Total length of each simulated time series
anzsim  <- 500        # Number of Monte Carlo simulation replications

# Initialize vectors to store squared forecast errors across replications
mse_true_arma <- NULL   # Squared errors for the ARMA benchmark forecast
mse_dfa       <- NULL   # Squared errors for the non-parametric DFA forecast

# Initialize a progress bar to track simulation progress
pb <- txtProgressBar(min = 1, max = anzsim, style = 3)

# -----------------------------------------------------------------------------
# Main Simulation Loop
# -----------------------------------------------------------------------------
# For each replication:
#   - Simulate an ARMA time series
#   - Split into in-sample (estimation) and out-of-sample (evaluation) portions
#   - Compute forecasts from both the (true) ARMA model and the (empirical) DFA filter
#   - Record squared forecast errors

for (i in 1:anzsim)
{
  # Simulate one realization of the ARMA(1,1) process of length 'len'
  x <- arima.sim(n = len, list(ar = a1, ma = b1))
  
  # Use all observations except the last as the in-sample estimation window
  x_insample <- x[1:(len - 1)]
  
  # ----------------------------
  # Approach 1: ARMA Benchmark
  # ----------------------------
  # Fit an ARMA(1,1) model to the in-sample data using MLE (via arima).
  # We assume perfect knowledge of the true model orders — this is the
  # best achievable parametric forecast under correct model specification.
  arima_true_obj  <- arima(x_insample, order = c(1, 0, 1), include.mean = FALSE)
  
  # Generate the one-step-ahead forecast from the fitted ARMA model
  arima_true_pred <- predict(arima_true_obj, n.ahead = 1)$pred
  
  # ----------------------------
  # Approach 2: Non-Parametric DFA
  # ----------------------------
  # Construct the weight function using the periodogram (DFT-based spectral estimate)
  # of the in-sample data. Both target and explanatory spectra are set to the same
  # periodogram, reflecting a standard MSE-optimal filter design.
  weight_func <- cbind(per(x_insample, F)$DFT, per(x_insample, F)$DFT)
  colnames(weight_func) <- c("spectrum target", "spectrum explanatory")
  
  # Number of Fourier frequencies (K+1 total, from 0 to pi)
  K <- nrow(weight_func) - 1
  
  # Allpass target: we want to replicate the signal without any frequency selectivity
  Gamma <- rep(1, K + 1)
  
  # Filter length: L=10 is typically sufficient for most seasonally adjusted economic data
  L <- 10
  
  # Forecast horizon: Lag = -1 corresponds to a one-step-ahead forecast
  Lag <- -1
  
  # Compute the MSE-optimal DFA filter coefficients
  mdfa_obj <- MDFA_mse(L, weight_func, Lag, Gamma)$mdfa_obj
  b <- mdfa_obj$b
  
  # Apply the DFA filter to the most recent L observations to produce the forecast
  dfa_forecast <- t(b) %*% x_insample[length(x_insample):(length(x_insample) - L + 1)]
  
  # ----------------------------
  # Record Squared Forecast Errors
  # ----------------------------
  # The true out-of-sample value is the last observation: x[len]
  mse_true_arma <- c(mse_true_arma, (x[length(x)] - arima_true_pred)^2)
  mse_dfa       <- c(mse_dfa,       (x[length(x)] - dfa_forecast)^2)
  
  # Update progress bar
  setTxtProgressBar(pb, i)
}

# =============================================================================
# Results: Root Mean Squared Forecast Error (RMSFE) Ratio
# =============================================================================
# We compute the ratio:  RMSFE(ARMA) / RMSFE(DFA)
#
# Interpretation:
#   - A ratio < 1 means DFA performs worse than the ideal ARMA benchmark.
#   - Asymptotically (for large anzsim), the ratio cannot exceed 1 under correct ARMA specification,
#     since MLE-based ARMA is the theoretically optimal forecast in this setting.
#
# Observed results:
#   - L = 10  → ratio ≈ 98%: DFA nearly matches the best possible forecast.
#               This confirms that L=10 is a well-calibrated, efficient choice
#               for most typical (seasonally adjusted) economic time series.
#   - L = 100 → ratio ≈ 80%: significant performance degradation due to
#               overfitting — analogous to fitting an AR(100) or using
#               the Burg maximum-entropy spectral estimator.
# =============================================================================

sqrt(mean(mse_true_arma) / mean(mse_dfa))


# =============================================================================
# Example 6: Comparing Two Spectral Estimates as DFA Weighting Functions
# =============================================================================
# Goal:
#   Evaluate whether the choice of spectral estimate matters when used as a
#   weighting function in the DFA framework. Specifically, we compare:
#
#     (1) Non-parametric DFT-based spectrum:
#           - Computed directly from the data via the periodogram.
#           - Requires no model assumptions.
#
#     (2) Burg's Maximum-Entropy AR-based spectrum:
#           - Derived by fitting a high-order AR(p) model to the data,
#             where p is set to be "sufficiently large" (here p = L).
#           - The AR model implicitly captures the spectral shape of the data.
#           - No explicit model identification is required — we simply let
#             a large AR order approximate the true spectral structure.
#
# Key question:
#   Does the smoother, model-based AR spectrum yield better DFA filter
#   coefficients than the noisier, non-parametric DFT periodogram?
# =============================================================================

# -----------------------------------------------------------------------------
# Process Specifications (choose one or define your own)
# -----------------------------------------------------------------------------

# Example 6.1: Pure MA(1) process
a1 <- 0.0
b1 <- 0.7

# Example 6.2: ARMA(1,1) process with positive autocorrelation
a1 <- 0.6
b1 <- 0.7

# Example 6.3: AR(1) process with strong negative autocorrelation
a1 <- -0.9
b1 <- 0.0

# Additional processes can be defined here by modifying a1 and b1 accordingly.

# -----------------------------------------------------------------------------
# Simulation Setup
# -----------------------------------------------------------------------------

set.seed(1)           # Set seed for reproducibility
len    <- 300         # Total length of each simulated time series
anzsim <- 500         # Number of Monte Carlo simulation replications

# Initialize vectors to store squared forecast errors across replications
mse_burg <- NULL   # Squared errors for the Burg (AR-based) DFA forecast
mse_dfa  <- NULL   # Squared errors for the non-parametric (DFT-based) DFA forecast

# Initialize a progress bar to track simulation progress
pb <- txtProgressBar(min = 1, max = anzsim, style = 3)

# -----------------------------------------------------------------------------
# Main Simulation Loop
# -----------------------------------------------------------------------------
# For each replication:
#   - Simulate an ARMA time series
#   - Split into in-sample (estimation) and out-of-sample (evaluation) portions
#   - Compute DFA forecasts using both spectral estimation approaches
#   - Record squared forecast errors for each approach

for (i in 1:anzsim)
{
  # Simulate one realization of the specified ARMA process of length 'len'
  x <- arima.sim(n = len, list(ar = a1, ma = b1))
  
  # Use all observations except the last as the in-sample estimation window
  x_insample <- x[1:(len - 1)]
  
  # ===========================================================================
  # Approach 1: Non-Parametric DFA (DFT-based Periodogram Spectrum)
  # ===========================================================================
  
  # Construct the weighting function from the periodogram of the in-sample data.
  # Both target and explanatory columns are set to the same periodogram,
  # corresponding to a standard univariate MSE-optimal filter design.
  weight_func <- cbind(per(x_insample, F)$DFT, per(x_insample, F)$DFT)
  colnames(weight_func) <- c("spectrum target", "spectrum explanatory")
  
  # Number of Fourier frequency ordinates (K+1 total, from frequency 0 to pi)
  K <- nrow(weight_func) - 1
  
  # Allpass target: no frequency selectivity — we aim to replicate the full signal
  Gamma <- rep(1, K + 1)
  
  # Filter length: L=10 is typically well-suited for seasonally adjusted economic data
  L <- 10
  
  # Forecast horizon: Lag = -1 specifies a one-step-ahead forecast
  Lag <- -1
  
  # Compute the MSE-optimal DFA filter coefficients using the DFT-based spectrum
  mdfa_obj <- MDFA_mse(L, weight_func, Lag, Gamma)$mdfa_obj
  b <- mdfa_obj$b
  
  # Apply the DFA filter to the most recent L in-sample observations
  dfa_forecast <- t(b) %*% x_insample[length(x_insample):(length(x_insample) - L + 1)]
  
  # ===========================================================================
  # Approach 2: Burg's Maximum-Entropy DFA (AR-based Spectrum)
  # ===========================================================================
  # Rather than using the noisy periodogram, we fit a high-order AR(L) model
  # and derive a smooth, model-based spectral estimate from its coefficients.
  # This is equivalent to the Burg maximum-entropy spectral estimator.
  # No model identification is needed — a large AR order acts as a flexible
  # nonparametric approximation to the true spectrum.
  
  # Fit an AR(L) model to the in-sample data using MLE (via arima)
  arima_burg_obj <- arima(x_insample, order = c(L, 0, 0), include.mean = FALSE)
  ar_burg <- arima_burg_obj$coef   # Estimated AR coefficients
  ma_burg <- NULL                  # No MA component in this AR(L) model
  
  # Resolution of the AR-based spectrum: can be set freely since the spectrum
  # is derived analytically from the AR model (not constrained by sample size)
  K_burg  <- 600
  plot_T  <- FALSE   # Suppress intermediate spectral plot
  
  # Derive the smooth AR(L)-based spectrum from the estimated coefficients
  weight_func_burg <- arma_spectrum_func(ar_burg, ma_burg, K_burg, plot_T)$arma_spec
  
  # Format as a two-column weighting matrix (target and explanatory spectra)
  weight_func_burg <- cbind(weight_func_burg, weight_func_burg)
  colnames(weight_func_burg) <- c("spectrum target", "spectrum explanatory")
  
  # Allpass target for the Burg-based filter (consistent with DFT approach)
  Gamma_burg <- rep(1, K_burg + 1)
  
  # Filter length for the Burg-based DFA.
  # In principle, this could be set larger than L above, because the AR-based
  # spectrum is smoother and less noisy than the raw periodogram, potentially
  # allowing longer filters without severe overfitting.
  L_burg <- 10
  
  # Forecast horizon: one-step-ahead (consistent with DFT approach)
  Lag <- -1
  
  # Compute the MSE-optimal DFA filter coefficients using the AR-based spectrum
  mdfa_burg_obj <- MDFA_mse(L_burg, weight_func_burg, Lag, Gamma_burg)$mdfa_obj
  b_burg <- mdfa_burg_obj$b
  
  # Apply the Burg DFA filter to the most recent L in-sample observations
  dfa_burg_forecast <- t(b_burg) %*% x_insample[length(x_insample):(length(x_insample) - L + 1)]
  
  # ----------------------------
  # Record Squared Forecast Errors
  # ----------------------------
  # The true out-of-sample target value is the last observation: x[len]
  mse_burg <- c(mse_burg, (x[length(x)] - dfa_burg_forecast)^2)
  mse_dfa  <- c(mse_dfa,  (x[length(x)] - dfa_forecast)^2)
  
  # Update the progress bar
  setTxtProgressBar(pb, i)
}

# =============================================================================
# Results: Root Mean Squared Forecast Error (RMSFE) Ratio
# =============================================================================
# We compute the ratio:  RMSFE(Burg DFA) / RMSFE(DFT DFA)
#
# Interpretation:
#   - A ratio close to 1 indicates that both spectral estimates yield
#     virtually identical DFA forecast performance.
#   - A ratio > 1 would suggest that the Burg AR-based spectrum leads to
#     worse forecasts than the simple DFT periodogram.
#
# Observed results:
#   - For L = 10, the ratio is virtually 1:
#     Both approaches perform indistinguishably (differences are within
#     the range of random Monte Carlo sampling error).
#   - Conclusion: the non-parametric DFT periodogram is just as effective
#     as the Burg maximum-entropy spectral estimate when used as a DFA
#     weighting function — validating the simpler, assumption-free approach.
# =============================================================================

sqrt(mean(mse_burg) / mean(mse_dfa))


# =============================================================================
# Wrap-Up: Key Takeaways from Examples 5 and 6
# =============================================================================
# The DFA framework is a flexible and powerful alternative to classical
# model-based forecasting. Specifically, we have demonstrated that:
#
#   1. DFA can replicate classical MSE-optimal one- and multi-step-ahead
#      forecasts by:
#        - Setting an allpass target (Gamma = 1) to recover the full signal.
#        - Specifying the appropriate forecast horizon via Lag < 0.
#        - Plugging in a suitable spectral estimate as the weighting function.
#
#   2. The non-parametric DFT-based DFA performs nearly as well as the
#      ideal ARMA benchmark (which assumes perfect knowledge of the true
#      data-generating process), provided overfitting is controlled via a
#      moderate filter length (e.g., L = 10).
#
#   3. The choice of spectral estimate (DFT periodogram vs. Burg AR-based)
#      does not materially affect forecast performance at L = 10:
#        - Both yield virtually identical RMSFE.
#        - The DFT, being assumption-free, is therefore the preferred choice
#          for its simplicity and generality.
#
#   4. Overall conclusion: a carefully applied non-parametric DFA based on
#      the DFT is a robust, model-free forecasting tool that competes with
#      the best parametric approaches — without requiring model identification.
#
#   5. The main purpose of DFA is not replication of classic forecasting.
#      Instead we wish to modify forecasts according to alternative priorities 
#      (than MSE optimality), see exercise 2 above. 
# =============================================================================






