# =============================================================================
# Tutorial 3: Direct Filter Approach (DFA) for Signal Extraction (Nowcasting)
# =============================================================================
#
# OVERVIEW:
# This tutorial applies DFA to signal extraction, with a focus on nowcasting
# an ideal lowpass filter output. It builds on the foundations established in
# the first two tutorials:
#
#   Tutorial 1: Applied DFA to forecasting and replicated classic state-of-the-
#               art (SOTA) time series approaches.
#
#   Tutorial 2: Covered how to specify the target filter (Gamma) and the
#               weighting function (spectrum). Topics included:
#               - Exploring the DFA interface (see main_DFA_interface.r)
#               - Interpreting amplitude and time-shift functions
#               - Diagnosing overfitting via rippled amplitude, unstable shifts,
#                 irregular coefficients, and degraded out-of-sample performance
#
#   Tutorial 3 (THIS FILE):
#               - Applies DFA to signal extraction, specifically nowcasting of
#                 an ideal lowpass filter target
#               - Examples can be straightforwardly extended to arbitrary targets
#                 and/or forecasting/backcasting scenarios
#
# -----------------------------------------------------------------------------
# KEY COMPARISONS AND FINDINGS:
#
#   1. MSE Performance Benchmark:
#      Compares the best possible one-sided filter (derived under the assumption
#      that the true data-generating process (DGP) is known) against the
#      DFA-based target filter.
#
#   2. Spectral Estimation Methods:
#      Compares the best possible one-sided DFA against two empirical DFA
#      variants — one based on the Discrete Fourier Transform (DFT) and one
#      based on Burg's Maximum Entropy spectral estimate — both in-sample and
#      out-of-sample.
#
#   3. DFT-Based DFA vs. True Model:
#      Demonstrates that DFT-based DFA for lowpass nowcasting performs nearly
#      as well as the oracle approach (i.e., assuming knowledge of the true DGP)
#      in terms of MSE, provided that:
#        - Elementary precautions are taken to avoid overfitting
#        - Even heavily over-parameterized designs incur only a modest efficiency
#          loss (approximately 10%) compared to the best possible approach
#
#   4. DFT vs. Burg Spectrum:
#      Shows that DFA based on the DFT and DFA based on Burg's Maximum Entropy
#      spectral estimate yield equivalent performance, suggesting that either
#      spectral estimator is suitable for real-time signal extraction
#      (lowpass nowcasting) in practice.
#
# -----------------------------------------------------------------------------
# SCOPE AND LIMITATIONS:
#
#   - This tutorial focuses exclusively on univariate, unconstrained MSE designs
#   - Multivariate extensions will be covered in Tutorial 4
#   - Customization (e.g., regularization, constraint imposition) will be
#     addressed in dedicated subsequent tutorials
#
# -----------------------------------------------------------------------------
# FUNCTIONS AND PARAMETERS:
#
#   Throughout this tutorial, we rely on the same core function MDFA_mse used
#   in previous tutorials. The goal here is to deepen understanding of its
#   parameters within a univariate signal extraction (nowcasting) context.
# =============================================================================



rm(list=ls())

library(xts)
#install.packages("devtools")
library(devtools)
# Load MDFA package from github
devtools::install_github("wiaidp/MDFA")
# MDFA package: EURUSD is now part of the data in the package
library(MDFA)


# Briev overview of wrappers and main function
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
source("Common functions/ideal_filter.r")
source("Common functions/mdfa_trade_func.r")


# =============================================================================
# Example 1: Best Possible MSE Estimate for an ARMA Process
#             (Assumes Knowledge of the True Data-Generating Process)
# -----------------------------------------------------------------------------
# Design:   Univariate, unconstrained MSE
# Goal:     Compute the theoretically optimal one-sided (causal) filter for
#           nowcasting the output of an ideal lowpass filter, using the true
#           ARMA spectrum as the weighting function.
# =============================================================================

# -----------------------------------------------------------------------------
# Step 1: Define the Frequency Grid
# -----------------------------------------------------------------------------
# K controls the resolution of the frequency grid.
# Frequencies are evenly spaced: omega_j = j * pi / K, for j = 0, 1, ..., K.
# A larger K yields a finer grid and more accurate numerical integration,
# but increases computational cost.
K <- 600

# -----------------------------------------------------------------------------
# Step 2: Specify the ARMA(1,1) Process and Compute Its Spectrum
# -----------------------------------------------------------------------------
# Parameters of the ARMA(1,1) process:
#   a1: AR coefficient  (set close to 0 to approximate white noise,
#       mimicking log-returns of typical economic/financial time series)
#   b1: MA coefficient  (set to NULL here, i.e., pure AR(1) process)
a1 <- 0.1
b1 <- NULL

# Compute and plot the ARMA spectral density over the frequency grid
plot_T <- T
par(mfrow = c(1, 1))
spec <- abs(arma_spectrum_func(a1, b1, K, plot_T)$arma_spec)

# -----------------------------------------------------------------------------
# Step 3: Construct the Weighting Function (weight_func)
# -----------------------------------------------------------------------------
# weight_func is a matrix with one column per series:
#   Column 1: spectrum of the TARGET series
#   Column 2: spectrum of the EXPLANATORY (input) series
# For univariate problems, both columns are identical (same series).
weight_func <- cbind(spec, spec)
colnames(weight_func) <- c("spectrum target", "spectrum explanatory")

# -----------------------------------------------------------------------------
# Step 4: Define the Ideal Lowpass Target (Gamma)
# -----------------------------------------------------------------------------
# The target Gamma is an ideal symmetric (two-sided) lowpass filter:
#   - Passes all frequency components with period >= 'periodicity' (i.e., omega <= cutoff)
#   - Blocks all higher-frequency components
# cutoff = pi / periodicity defines the passband boundary in radians.
periodicity <- 10                          # Retain cycles of length >= 10 time periods
cutoff      <- pi / periodicity            # Cutoff frequency in radians

# Gamma is a binary indicator: 1 in the passband, 0 in the stopband
# The small tolerance (1e-9) avoids floating-point boundary issues
Gamma <- (0:(K)) <= K * cutoff / pi + 1.e-9

# -----------------------------------------------------------------------------
# Step 5: Specify the Filter Lag (Timing)
# -----------------------------------------------------------------------------
# Lag controls the timing of the filter output relative to the current observation:
#   Lag  = 0  -> Nowcast  (estimate signal at current time t)
#   Lag  > 0  -> Backcast (estimate signal at a past time t - Lag)
#   Lag  < 0  -> Forecast (estimate signal at a future time t + |Lag|)
Lag <- 0   # Nowcasting

# -----------------------------------------------------------------------------
# Step 6: Specify the Filter Length
# -----------------------------------------------------------------------------
# L is the number of filter coefficients (filter order = L - 1).
# Rule of thumb: the ratio L / K should be kept small to prevent overfitting.
# Here L = 100 and K = 600, giving L/K ≈ 0.17.
L <- 100

# -----------------------------------------------------------------------------
# Step 7: Estimate the Optimal Filter via MDFA (MSE Criterion)
# -----------------------------------------------------------------------------
# MDFA_mse minimizes the integrated mean squared error (MSE) between the
# filter output and the ideal target Gamma, weighted by the spectral density.
# This yields the theoretically optimal causal filter coefficients b.
mdfa_obj_mse <- MDFA_mse(L, weight_func, Lag, Gamma)$mdfa_obj

# Plot coefficients, amplitude and phase response of the estimated filter vs. the ideal target
plot_estimate_func(mdfa_obj_mse, weight_func, Gamma)

# =============================================================================
# Step 8: Simulate Data and Apply Filters
# =============================================================================

# Simulate a realization of the specified ARMA process
set.seed(1)
len <- 1000
x   <- as.vector(arima.sim(n = len, list(ar = a1, ma = b1)))

# -----------------------------------------------------------------------------
# Compute the Ideal (Two-Sided) Lowpass Filter Output
# -----------------------------------------------------------------------------
# The ideal lowpass filter is bi-infinite (uses future AND past observations).
# In practice it is truncated to 2M+1 coefficients (symmetric around t).
# This serves as the benchmark "truth" against which we evaluate the DFA filter.
# NOTE: The ideal filter output is NOT available in real time near the sample end,
#       because it requires future observations.
M          <- 100
id_obj     <- ideal_filter_func(periodicity, M, x)
output_ideal <- id_obj$y

# -----------------------------------------------------------------------------
# Compute the DFA (One-Sided, Causal) Filter Output
# -----------------------------------------------------------------------------
# The DFA filter uses only past and current observations — it is causal and
# therefore available in real time, including at the end of the sample.
b              <- mdfa_obj_mse$b
filt_obj       <- filt_func(x, b)
output_dfa     <- filt_obj$yhat
output_dfa[1:(L - 1)] <- NA   # First L-1 observations unavailable (filter warm-up)

# =============================================================================
# Step 9: Visualize and Compare Filter Outputs
# =============================================================================

# --- Full-Sample Plot --------------------------------------------------------
# The ideal lowpass output (blue) is smooth and symmetric but unavailable at
# the sample end. The DFA output (red) is causal and extends to the last
# observation, at the cost of slight noise leakage and a small positive time shift.
par(mfrow = c(1, 1))
ts.plot(output_ideal, col = "blue",
        main = "Ideal Lowpass Output (blue) vs. DFA Nowcast (red)")
lines(output_dfa, col = "red")

# --- Zoomed-In Plot ----------------------------------------------------------
# Zooming in highlights two characteristic properties of the real-time DFA filter
# relative to the ideal two-sided benchmark:
#   1. NOISE LEAKAGE:  The DFA output is slightly noisier because the one-sided
#                      filter cannot perfectly suppress high-frequency components
#                      without access to future data.
#   2. PHASE DELAY:    The DFA output is slightly retarded (positive time shift),
#                      reflecting the inherent trade-off between timeliness and
#                      smoothness in causal filtering.
anf <- 400
enf <- 500
ts.plot(output_ideal[anf:enf], col = "blue",
        main = "Ideal Lowpass Output (blue) vs. DFA Nowcast (red) — Zoomed")
lines(output_dfa[anf:enf], col = "red")
abline(h = 0)


# =============================================================================
# Example 2: DFA for Lowpass Nowcasting Using DFT-Based Spectral Estimation
# -----------------------------------------------------------------------------
# This example mirrors Example 1 but replaces the known true ARMA spectrum
# with an empirical spectral estimate derived from the Discrete Fourier
# Transform (DFT) of the in-sample data.
#
# KEY QUESTION: How much efficiency is lost when the true spectrum is unknown
#               and must be estimated from data?
#
# EXPERIMENT: Re-run this example with different filter lengths L to observe
#             the trade-off between filter resolution and overfitting:
#
#   1. L <- periodicity * 2   Minimum length needed to attenuate a cycle of the
#                             given periodicity. Amplitude and out-of-sample
#                             performance are acceptable across all AR strengths.
#
#   2. L <- periodicity * 4   Adequate resolution with no signs of overfitting.
#                             Amplitude and out-of-sample MSE remain well-behaved
#                             for both a1 = 0 (white noise) and a1 = 0.9 (strong AR).
#
#   3. L <- periodicity * 8   Early signs of overfitting may appear when a1 = 0.9.
#                             Watch for ripples in the amplitude response.
#
#   4. L <- periodicity * 16  Clear overfitting. Degradation is more severe for
#                             a1 = 0.9 than for a1 = 0. This is encouraging:
#                             log-returns of typical economic/financial data are
#                             close to white noise (a1 ~ 0), so they are less
#                             susceptible to overfitting in practice.
# =============================================================================

# -----------------------------------------------------------------------------
# Step 1: Define the In-Sample Period and Estimate the Spectrum via DFT
# -----------------------------------------------------------------------------
# Only the in-sample portion of the data is used to estimate the spectrum.
# This simulates a realistic setting where the analyst has no access to
# out-of-sample observations at the time of filter design.
in_sample    <- 300
x_insample   <- x[1:in_sample]

# Compute the periodogram (DFT-based spectral estimate) from the in-sample data.
# The periodogram is noisier than the true ARMA spectrum but requires no
# parametric assumptions about the data-generating process.
# Both columns of weight_func_dft are identical (univariate problem).
weight_func_dft <- cbind(per(x_insample, F)$DFT, per(x_insample, F)$DFT)
colnames(weight_func_dft) <- c("spectrum target", "spectrum explanatory")

# K_dft: number of frequency grid points implied by the DFT (= in_sample / 2)
K_dft <- nrow(weight_func_dft) - 1

# -----------------------------------------------------------------------------
# Step 2: Define the Lowpass Target (Gamma) on the DFT Frequency Grid
# -----------------------------------------------------------------------------
# Gamma is the same ideal lowpass target as in Example 1, but now defined on
# the DFT frequency grid (which has resolution pi / K_dft instead of pi / K).
# The small tolerance (1e-9) prevents floating-point boundary issues.
Gamma_dft <- (0:(K_dft)) <= K_dft * cutoff / pi + 1.e-9

# -----------------------------------------------------------------------------
# Step 3: Set Lag and Filter Length
# -----------------------------------------------------------------------------
# Lag = 0: Nowcasting (estimate the signal at the current time point t).
Lag <- 0

# Filter length L:
#   - L = periodicity * 2 is the practical minimum for attenuating a cycle of
#     the specified periodicity; increase L for sharper frequency selectivity,
#     but be aware of increasing overfitting risk (see experiment guide above).
L <- periodicity * 2

# -----------------------------------------------------------------------------
# Step 4: Estimate the DFT-Based DFA Filter (MSE Criterion)
# -----------------------------------------------------------------------------
# MDFA_mse minimizes the MSE between the filter output and the ideal target
# Gamma_dft, weighted by the DFT-based spectral estimate.
mdfa_obj_dft_mse <- MDFA_mse(L, weight_func_dft, Lag, Gamma_dft)$mdfa_obj

# Plot and inspect amplitude and phase response.
# Compared to Example 1 (true spectrum), the DFT-based spectrum is noisier,
# which typically results in more irregular filter coefficients. The degree
# of irregularity increases with L (longer filter = more overfitting risk).
plot_estimate_func(mdfa_obj_dft_mse, weight_func_dft, Gamma_dft)

# =============================================================================
# Step 5: Apply All Three Filters to the Full Data Series
# =============================================================================
# We compare three filter outputs:
#   Blue  — Ideal (two-sided) lowpass:  the benchmark "truth"; unavailable in
#            real time near the sample end (requires future observations).
#   Red   — Best possible one-sided DFA (true ARMA spectrum, from Example 1):
#            the theoretical optimum for causal filtering.
#   Green — DFA based on DFT estimate (this example):
#            the practically feasible causal filter; no knowledge of true DGP.

b_dft          <- mdfa_obj_dft_mse$b
filt_dft_obj   <- filt_func(x, b_dft)
output_dft_dfa <- filt_dft_obj$yhat
output_dft_dfa[1:(L - 1)] <- NA   # Discard filter warm-up period

# --- Full-Sample Comparison Plot ---------------------------------------------
par(mfrow = c(1, 1))
ts.plot(output_ideal, col = "blue",
        main = paste("Ideal Lowpass (blue) | Best Possible DFA (red)",
                     "| DFA via DFT (green)", sep = "\n"))
lines(output_dfa,     col = "red")
lines(output_dft_dfa, col = "green")

# Note: The ideal filter output (blue) is not available near the sample end
# because it requires future observations. Both causal DFA filters (red, green)
# extend to the last observation, trading off some accuracy for timeliness.

# --- Zoomed-In Comparison Plot -----------------------------------------------
# A closer look at a 100-observation window reveals:
#   - How closely the DFT-based filter (green) tracks the oracle filter (red)
#   - Any additional noise leakage or phase delay introduced by using the
#     estimated rather than the true spectrum
anf <- 400
enf <- 500
ts.plot(output_ideal[anf:enf], col = "blue",
        main = paste("Ideal Lowpass (blue) | Best Possible DFA (red)",
                     "| DFA via DFT (green) — Zoomed", sep = "\n"))
lines(output_dfa[anf:enf],     col = "red")
lines(output_dft_dfa[anf:enf], col = "green")
abline(h = 0)

# =============================================================================
# Step 6: Compute Root Mean Squared Error (RMSE) — In-Sample vs. Out-of-Sample
# =============================================================================
# RMSE measures how closely each causal filter approximates the ideal lowpass
# output. Lower RMSE = better approximation of the signal extraction target.
#
# All RMSE values are computed relative to the ideal filter output (benchmark).
# Time points where the ideal filter is undefined (start/end of sample due to
# the two-sided truncation) are excluded via na.exclude().

dat_mat           <- cbind(output_ideal, output_dfa, output_dft_dfa)
dat_mat_in_sample <- na.exclude(dat_mat[1:in_sample, ])
dat_mat_out_sample<- na.exclude(dat_mat[(in_sample + 1):nrow(dat_mat), ])

colnames(dat_mat_in_sample) <-
  colnames(dat_mat_out_sample) <- c("ideal", "best MSE", "dft")

# Compute RMSE for each filter relative to the ideal, in both evaluation periods
mat_mse_result <- rbind(
  # In-sample RMSE (filters were designed on this portion of data)
  c(sqrt(mean((dat_mat_in_sample[, "ideal"] - dat_mat_in_sample[, "best MSE"])^2)),
    sqrt(mean((dat_mat_in_sample[, "ideal"] - dat_mat_in_sample[, "dft"])^2))),
  # Out-of-sample RMSE (genuine hold-out; neither filter has seen this data)
  c(sqrt(mean((dat_mat_out_sample[, "ideal"] - dat_mat_out_sample[, "best MSE"])^2)),
    sqrt(mean((dat_mat_out_sample[, "ideal"] - dat_mat_out_sample[, "dft"])^2)))
)

rownames(mat_mse_result) <- c("In-Sample", "Out-of-Sample")
colnames(mat_mse_result) <- c("Theoretically Optimal (True Model)",
                              "DFA Based on DFT Estimate")

mat_mse_result

# -----------------------------------------------------------------------------
# INTERPRETATION OF RESULTS:
#
#   In-Sample:
#     The DFT-based filter may appear to match or even slightly outperform the
#     true-model filter in-sample. This is an artifact of overfitting — the DFT
#     spectrum adapts to sample-specific noise, inflating apparent in-sample fit.
#
#   Out-of-Sample:
#     The true-model filter is the theoretical best and dominates out-of-sample.
#     However, the efficiency loss of the DFT-based filter is typically modest
#     (often around 10%), confirming that the DFT spectrum is a practical and
#     reliable substitute when the true DGP is unknown.
#
#   Effect of Filter Length L:
#     Increasing L sharpens the frequency response but raises overfitting risk.
#     The out-of-sample RMSE is the most reliable diagnostic: a large gap
#     between in-sample and out-of-sample RMSE signals overfitting.
# -----------------------------------------------------------------------------



# =============================================================================
# Example 3: Simulation Study (Multiple Realizations)
#           ARMA-processes with a Lowpass Gamma Target (Nowcasting, Lag = 0)
# -----------------------------------------------------------------------------
# Concept:
# This example extends the previous setup to a Monte Carlo study. For each
# realization, we:
#  - Generate an ARMA process (one of several sub-cases below)
#  - Estimate the spectrum in-sample using three approaches:
#      a) True model spectrum (MDFA with the exact ARMA parameters)
#      b) DFT-based empirical spectrum (DFT periodogram)
#  - Compute MSE-based nowcasting performance for both the true-model DFA and the
#    DFT-based DFA, relative to the ideal two-sided lowpass benchmark
#  - Repeat across many realizations to assess in-sample vs. out-of-sample
#    performance and the impact of the filter length L
#
# Notes:
#  - Gamma (the target) is a lowpass filter (not allpass); we focus on nowcasting
#    (Lag = 0)
#  - This file uses a single set of settings for all realizations, but you can
#    extend it by looping over different a1, b1, and true_model_order values.
# =============================================================================

# Example sub-cases (uncomment to run individually)
# 3.1) MA(1)
# a1 <- 0.0
# b1 <- 0.7
# true_model_order <- c(0, 0, 1)

# 3.2) ARMA with positive ACF
# a1 <- 0.6
# b1 <- 0.7
# true_model_order <- c(1, 0, 1)

# 3.3) AR with negative ACF
# a1 <- -0.9
# b1 <- 0.0
# true_model_order <- c(1, 0, 0)

# 3.4) Close to noise (typical for FX-log-returns)
# a1 <- -0.08
# b1 <- 0.0
# true_model_order <- c(1, 0, 0)

# Add any other ARMA configurations as needed...

# Common settings for the simulation study
set.seed(1)
len <- 1000                 # Time series length per realization
anzsim <- 500               # Number of simulations
in_sample <- 300              # In-sample window for estimation
K_true <- 600                 # Frequency grid size for the true spectrum
periodicity <- 10               # Target period for the lowpass Gamma
L <- 2 * periodicity            # Filter length (nowcasting; minimal damping length)
Lag <- 0                        # Nowcasting (causal, no lag)
M <- 100                          # Length for the ideal (two-sided) filter

# Storage for results
mse_true <- mse_dft <- NULL

# Progress bar for user feedback
pb <- txtProgressBar(min = 1, max = anzsim, style = 3)

# Simulation loop
for (i in 1:anzsim) {
  # Generate a realization from the chosen ARMA specification
  if (abs(a1) + abs(b1) > 0) {
    x <- as.vector(arima.sim(n = len, list(ar = a1, ma = b1)))
  } else {
    x <- rnorm(len)
  }
  
  # In-sample window used for both model-based and DFT-based spectrum estimation
  x_insample <- x[1:in_sample]
  
  # 1) True model spectrum (MDFA with known ARMA parameters)
  # Estimate the true ARMA spectrum on the in-sample segment
  arima_true_obj <- arima(x_insample, order = true_model_order, include.mean = FALSE)
  spec <- arma_spectrum_func(
    ifelse(!is.na(arima_true_obj$coef["ar1"]), arima_true_obj$coef["ar1"], 0),
    ifelse(!is.na(arima_true_obj$coef["ma1"]), arima_true_obj$coef["ma1"], 0),
    K_true, FALSE
  )$arma_spec
  weight_func_true <- cbind(spec, spec)
  colnames(weight_func_true) <- c("spectrum target", "spectrum explanatory")
  
  Gamma_true <- (0:(K_true)) <= K_true * (cutoff / pi) + 1e-9
  
  mdfa_true_obj <- MDFA_mse(L, weight_func_true, Lag, Gamma_true)$mdfa_obj
  b_true <- mdfa_true_obj$b
  
  # 2) DFT-based spectrum (in-sample estimation)
  weight_func_dft <- cbind(per(x_insample, F)$DFT, per(x_insample, F)$DFT)
  colnames(weight_func_dft) <- c("spectrum target", "spectrum explanatory")
  K_dft <- nrow(weight_func_dft) - 1
  Gamma_dft <- (0:(K_dft)) <= K_dft * (cutoff / pi) + 1e-9
  
  mdfa_dft_obj <- MDFA_mse(L, weight_func_dft, Lag, Gamma_dft)$mdfa_obj
  b_dft <- mdfa_dft_obj$b
  
  # 3) Apply filters to the full series
  # 3a) Ideal (two-sided) filter output (benchmark)
  id_obj <- ideal_filter_func(periodicity, M, x)
  output_ideal <- id_obj$y
  
  # 3b) DFA with true model spectrum
  filt_true_obj <- filt_func(x, b_true)
  output_dfa_true <- filt_true_obj$yhat
  
  # 3c) DFA with DFT-based spectrum
  filt_dft_obj <- filt_func(x, b_dft)
  output_dfa_dft <- filt_dft_obj$yhat
  
  # 4) Accumulate out-of-sample MSE relative to the ideal filter
  # Use non-overlapping tail (from in_sample+1 to len-M) to compare forecasts
  mse_true <- c(mse_true, mean((output_ideal - output_dfa_true)[(in_sample + 1):len - M]^2, na.rm = TRUE))
  mse_dft  <- c(mse_dft,  mean((output_ideal - output_dfa_dft)[(in_sample + 1):len - M]^2, na.rm = TRUE))
  
  setTxtProgressBar(pb, i)
}

# 5) Summary: RMSE ratio (DFA based on DFT vs. true-model DFA)
# The ratio is interpreted as: how close is the DFT-based DFA to the true-model performance
ratio <- sqrt(mean(mse_true) / mean(mse_dft))

ratio
































#----------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
# Example 3: Same as previous exercise but we conduct a simulation study based on multiple 
# realizations of an arma-process. This example corresponds to the simulation 
# studies in the previous tutorial (forecasting) but Gamma is now a lowpass (not an allpass) 
# and we emphasize nowcasting (Lag=0) 

# Example 3.1: MA(1)
a1<-0.
b1<-0.7
true_model_order<-c(0,0,1)
# Example 3.2: ARMA with positive acf
a1<-0.6
b1<-0.7
true_model_order<-c(1,0,1)
# Example 3.3: AR with negative acf
a1<--0.9
b1<-0
true_model_order<-c(1,0,0)
# Example 3.4: close to noise (typical for log-returns of FX-data)
a1<--0.08
b1<-0.0
true_model_order<-c(1,0,0)
# Add any other processes...
# We generate anzsim realizations of length len of the arma-process
set.seed(1)
len<-1000
mse_true_arma<-mse_dfa<-NULL
# Number of simulations
anzsim<-500
# Length of in-sample span
in_sample<-300
# Frequency grid for DFA based on true model
K_true<-600
# Lowpass target  
periodicity<-10
# Default (reasonable) filter length for nowcasting
L<-2*periodicity
# Nowcast
Lag<-0
# Length of ideal filter
M<-100
mse_true<-mse_dft<-NULL
pb <- txtProgressBar(min = 1, max = anzsim, style = 3)
# Loop through all simulations and collect out-of-sample forecast performances
for (i in 1:anzsim)
{
  # Distinguish white noise  
  if (abs(a1)+abs(b1)>0)
  {
    # Generate series  
    x<-as.vector(arima.sim(n=len,list(ar=a1,ma=b1)))
  } else
  {
    x<-rnorm(len)
  }
  # Use in-sample span for model-estimation and for dft  
  x_insample<-x[1:in_sample]
  # True model: estimate model-parameters by relying on classic arima-function
  arima_true_obj<-arima(x_insample,order=true_model_order,include.mean=F)
  # Spectrum based on true model  
  spec<-arma_spectrum_func(ifelse(!is.na(arima_true_obj$coef["ar1"]),arima_true_obj$coef["ar1"],0),ifelse(!is.na(arima_true_obj$coef["ma1"]),arima_true_obj$coef["ma1"],0),K_true,F)$arma_spec
  weight_func<-cbind(spec,spec)
  colnames(weight_func)<-c("spectrum target","spectrum explanatory")
  weight_func_true<-weight_func
  cutoff<-pi/periodicity
  # target true model: frequnecy grid is not the same as for dft below i.e. K is different
  Gamma_true<-(0:(K_true))<=K_true*cutoff/pi+1.e-9
  mdfa_true_obj<-MDFA_mse(L,weight_func_true,Lag,Gamma_true)$mdfa_obj 
  b_true<-mdfa_true_obj$b
  # Use in-sample span for dft  
  weight_func_dft<-cbind(per(x_insample,F)$DFT,per(x_insample,F)$DFT)
  colnames(weight_func_dft)<-c("spectrum target","spectrum explanatory")
  K_dft<-nrow(weight_func_dft)-1
  Gamma_dft<-(0:(K_dft))<=K_dft*cutoff/pi+1.e-9
  # Compute MSE-filter
  mdfa_dft_obj<-MDFA_mse(L,weight_func_dft,Lag,Gamma_dft)$mdfa_obj 
  b_dft<-mdfa_dft_obj$b
  # Filter data
  # 1. ideal filter  
  id_obj<-ideal_filter_func(periodicity,M,x)
  output_ideal<-id_obj$y
  # 2. DFA true
  filt_true_obj<-filt_func(x,b_true)
  output_dfa_true<-filt_true_obj$yhat
  # 3. DFA dft
  filt_dft_obj<-filt_func(x,b_dft)
  output_dfa_dft<-filt_dft_obj$yhat
  # Mean-square out-of-sample filter error  
  mse_true<-c(mse_true,mean((output_ideal-output_dfa_true)[(in_sample+1):len-M]^2,na.rm=T))
  mse_dft<-c(mse_dft,mean((output_ideal-output_dfa_dft)[(in_sample+1):len-M]^2,na.rm=T))
  setTxtProgressBar(pb, i)
}

# Compute the ratio of root mean-square forecast errors:
#   The ratio cannot be larger than 1 asymptotically because our particular design distinguishes arma as the universally best possible design
sqrt(mean(mse_true)/mean(mse_dft))

# Results: 
#   -for L=2*periodicity and in_sample=300, the ratio is typically around 97%: in the mean the non-parametric DFA performs as well (by all practical means) as the best possible forecast approach
#   -for L=2*periodicity and in_sample=100, the ratio is typically around 89%: in the mean the non-parametric DFA performs close to the best possible forecast approach (despite massive overfitting)
#     -Note that L=2*periodicity is fine for damping all components with durations shorter/equal periodicity
#     -But fitting L=2*periodicity=20 parameters for a time series of length in_sample=100 is 
#       not extremely smart (overfitting). See tutorial on regularization...



#---------------------------------------------------------------------------------------
# Example 4: same as above but we compare DFA-dft (as above) and DFA based on Burg's maximum entropy spectral estimate

# Example 4.1: MA(1)
a1<-0.
b1<-0.7
# Example 4.2: ARMA with positive acf
a1<-0.6
b1<-0.7
# Example 4.3: AR with negative acf
a1<--0.9
b1<-0
# Example 4.4: nearly noise
a1<--0.08
b1<-0
# Add any other processes...
# We generate anzsim realizations of length len of the arma-process
set.seed(1)
len<-1000
mse_true_arma<-mse_dfa<-NULL
# Number of simulations
anzsim<-500
# Length of in-sample span
in_sample<-300
# Frequency grid for DFA based on Burg's estimate
K_burg<-600
# Lowpass target  
periodicity<-10
# Default (reasonable) filter length for nowcasting: this is also used for the estimation of Burg's max-entropy spectrum
L<-2*periodicity
# Nowcast
Lag<-0
# Length of ideal filter
M<-100
mse_burg<-mse_dft<-NULL
pb <- txtProgressBar(min = 1, max = anzsim, style = 3)
# Loop through all simulations and collect out-of-sample forecast performances
for (i in 1:anzsim)
{
  # Distinguish white noise  
  if (abs(a1)+abs(b1)>0)
  {
    # Generate series  
    x<-as.vector(arima.sim(n=len,list(ar=a1,ma=b1)))
  } else
  {
    x<-rnorm(len)
    spec<-rep(1,K_burg+1)
  }
  # Use in-sample span for model-estimation and for dft  
  x_insample<-x[1:in_sample]
  # Burg spectral estimate: use AR(L)
  # Restrict length of AR (otherwise numerical optimization fails)
  arima_burg_obj<-arima(x_insample,order=c(min(L,10),0,0),include.mean=F)
  # Spectrum based on burg model  
  spec<-arma_spectrum_func(arima_burg_obj$coef,NULL,K_burg,F)$arma_spec
  weight_func<-cbind(spec,spec)
  colnames(weight_func)<-c("spectrum target","spectrum explanatory")
  weight_func_burg<-weight_func
  cutoff<-pi/periodicity
  # target burg model: frequency grid is not the same as for dft below i.e. K is different
  Gamma_burg<-(0:(K_burg))<=K_burg*cutoff/pi+1.e-9
  mdfa_burg_obj<-MDFA_mse(L,weight_func_burg,Lag,Gamma_burg)$mdfa_obj 
  b_burg<-mdfa_burg_obj$b
  # Use in-sample span for dft  
  weight_func_dft<-cbind(per(x_insample,F)$DFT,per(x_insample,F)$DFT)
  colnames(weight_func_dft)<-c("spectrum target","spectrum explanatory")
  K_dft<-nrow(weight_func_dft)-1
  Gamma_dft<-(0:(K_dft))<=K_dft*cutoff/pi+1.e-9
  # Compute MSE-filter
  mdfa_dft_obj<-MDFA_mse(L,weight_func_dft,Lag,Gamma_dft)$mdfa_obj 
  b_dft<-mdfa_dft_obj$b
  # Filter data
  # 1. ideal filter  
  id_obj<-ideal_filter_func(periodicity,M,x)
  output_ideal<-id_obj$y
  # 2. DFA burg
  filt_burg_obj<-filt_func(x,b_burg)
  output_dfa_burg<-filt_burg_obj$yhat
  # 3. DFA dft
  filt_dft_obj<-filt_func(x,b_dft)
  output_dfa_dft<-filt_dft_obj$yhat
  # Mean-square out-of-sample filter error  
  mse_burg<-c(mse_burg,mean((output_ideal-output_dfa_burg)[(in_sample+1):len-M]^2,na.rm=T))
  mse_dft<-c(mse_dft,mean((output_ideal-output_dfa_dft)[(in_sample+1):len-M]^2,na.rm=T))
  setTxtProgressBar(pb, i)
}

# Compute the ratio of root mean-square forecast errors:

sqrt(mean(mse_burg)/mean(mse_dft))

# Results: 
#   -for L=2*periodicity and in_sample=300, the ratio is 98% i.e. both approaches are indistinguishable by all practical means

