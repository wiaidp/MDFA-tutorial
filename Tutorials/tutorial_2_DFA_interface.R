
# ════════════════════════════════════════════════════════════════════
# TUTORIAL 2: (M)DFA USER INTERFACE, MSE SOLUTIONS AND OVERFITTING
# ════════════════════════════════════════════════════════════════════

# ── PURPOSE ───────────────────────────────────────────────────────
# This tutorial pursues three main objectives:
#   • Familiarize the user with the (M)DFA interface
#   • Develop intuition for and interpretation of MSE solutions
#   • Build a thorough understanding of overfitting in the DFA framework

# ── SCOPE AND EXTENSIONS RELATIVE TO PREVIOUS TUTORIAL ───────────
# The previous forecasting tutorial 1 focused on a specific target:
# an anticipative (Lag < 0) allpass filter.
# This tutorial broadens the scope in several directions:
#
#   • Generic filter targets
#       → Lowpass, bandpass, highpass, Hodrick-Prescott
#         (see Tutorial 9), and arbitrary user-defined targets
#
#   • Multiple spectral estimation approaches
#       → Different spectra are proposed and compared
#
#   • Interpretation of MSE solutions
#       → Key characteristics examined:
#         amplitude function and time-shift function
#
#   • Overfitting
#       → The mechanism of overfitting is examined in depth
#         within the DFA framework
#
#   • Flexible user control
#       → The DFA user interface is explored to understand how
#         the optimization outcome (the one-sided DFA filter)
#         can be shaped and steered
#
#   • Link to trading
#       → Identification of turning points (local maxima and minima)
#         provides a natural bridge to trading applications:
#         → Tactical (short-term) and strategic (medium- to long-term)
#           positioning can be implemented via a single parameter
#           (periodicity)

# ── TUTORIAL DESIGN ───────────────────────────────────────────────
# To keep the focus on core concepts, this tutorial is deliberately
# restricted to:
#   • Univariate examples          (no multivariate extensions)
#   • MSE criterion                (no customization)
#   • Unconstrained filters        (no regularization)
#
# Note: customization and regularization are addressed in
# dedicated subsequent tutorials.

# ── FUNCTIONS AND PARAMETERS ──────────────────────────────────────
# As in the previous tutorial, all examples rely on MDFA_mse.
# The focus here shifts to a deeper understanding of its parameters
# in a univariate signal extraction (nowcasting) framework.
#
# Parameters of particular emphasis in this tutorial:
#   • Gamma        → arbitrary filter target specification
#   • weight_func  → spectral estimate
# ─────────────────────────────────────────────────────────────────


# Start from scratch
rm(list=ls())



library(xts)
#install.packages("devtools")
library(devtools)
# Load MDFA package from github
devtools::install_github("wiaidp/MDFA")
# MDFA package: EURUSD is now part of the data in the package
library(MDFA)

# Gamma and weight_func in the head of MDFA_mse allow for flexible interaction of the user with the estimation algorithm
head(MDFA_mse)

#-----------------------------------------------------------------------------------------------
# Source common functions

source("Common functions/plot_func.r")
source("Common functions/arma_spectrum.r")
source("Common functions/ideal_filter.r")
source("Common functions/mdfa_trade_func.r")

# =============================================================================
# Playing with DFA Targets: Frequency-Domain Filter Design
# =============================================================================
# Overview:
#   The DFA (Direct Filter Approach) requires two user-supplied inputs:
#     (1) A TARGET: the desired frequency response of the optimal filter
#                   (i.e., what the filter should extract or predict).
#     (2) A SPECTRUM: the spectral density of the input data,
#                     used to weight the importance of each frequency
#                     during optimization.
#
#   Given these inputs, DFA returns the filter that minimizes a chosen
#   loss criterion — either:
#     - Classical MSE (Mean Squared Error), or
#     - A generalized criterion addressing the ATS-trilemma
#       (Accuracy, Timeliness, Smoothness trade-off).
#
# Motivation:
#   Unlike classical time series forecasting methods (ARMA, state-space, etc.),
#   DFA allows arbitrary user-defined targets in the frequency domain.
#   This makes it  more flexible than standard SOTA forecast approaches,
#   which implicitly assume particular target specifications.
#
# Design choices for the following examples:
#   - Loss criterion : MSE (classical mean squared error)
#   - Filter design  : Univariate (single input series)
#   - Assumed spectrum: White noise (flat spectral density)
# =============================================================================

# -----------------------------------------------------------------------------
# Frequency Grid Setup
# -----------------------------------------------------------------------------
# DFA operates in the frequency domain over a discrete frequency grid:
#   omega_j = j * pi / K,  for j = 0, 1, 2, ..., K
#
# This partitions the interval [0, pi] into K equally spaced frequency supports.
#
# Practical trade-off when choosing K:
#   - Larger K → denser frequency grid → better MSE approximation
#                (exact when the true process is white noise)
#   - Larger K → longer computation time
#
# We set K = 600, which provides a fine, accurate frequency resolution
# while remaining computationally efficient.

K <- 600

# -----------------------------------------------------------------------------
# Spectral Weighting Function: White Noise Assumption
# -----------------------------------------------------------------------------
# The weighting function (weight_func) is a (K+1) x 2 matrix where:
#   - Column 1: spectrum of the TARGET series
#   - Column 2: spectrum of the EXPLANATORY (input) series
#
# In a univariate design, the target and explanatory series are identical,
# so both columns contain the same spectral values.
#
# White noise assumption: the spectral density is flat (constant = 1, or any positive real number)
# across all frequencies, meaning all frequencies contribute equally
# to the variance of the process.
#
# This is the simplest possible spectral assumption and serves as a
# natural starting point for DFA filter design. It is particularly
# well-suited to noisy, spectrally unstructured data — for example:
#   - Many differenced economic time series (e.g., GDP growth, inflation changes),
#     which often exhibit little residual autocorrelation after differencing.
#   - High-frequency financial returns, which are largely unpredictable
#     and well-approximated by white noise.
#
# When the true spectrum deviates substantially from flat (e.g., strong
# low-frequency dominance in trending or seasonal data), a more
# informative spectral estimate — such as mode-based or the DFT periodogram or an
# AR-based spectrum — should be preferred over the white noise assumption.

weight_func <- matrix(rep(1, 2 * (K + 1)), ncol = 2)
colnames(weight_func) <- c("target", "explanatory")

# -----------------------------------------------------------------------------
# Plot: White Noise Spectrum
# -----------------------------------------------------------------------------
# Visualize the flat white noise spectrum across the frequency interval [0, pi].
# The horizontal line at amplitude = 1 confirms equal weighting of all frequencies.

par(mfrow = c(1, 1))

plot(
  weight_func[, 1],
  type  = "l",
  main  = paste("White noise spectrum, denseness = ", K, sep = ""),
  axes  = FALSE,
  xlab  = "Frequency",
  ylab  = "Spectral Amplitude",
  col   = "black"
)
# Annotate the plot with the spectrum label (taken from column name of weight_func)
mtext(colnames(weight_func)[1], line = -1, col = "black")
# Add frequency axis with human-readable labels in multiples of pi/6
# Each tick corresponds to omega = k*pi/6, for k = 0, 1, ..., 6
axis(
  1,
  at     = c(0, 1:6 * K / 6 + 1),
  labels = c("0", "pi/6", "2pi/6", "3pi/6", "4pi/6", "5pi/6", "pi")
)
axis(2)
box()

# =============================================================================
# Example 1: Classic One-Step-Ahead Forecasting (Replicating SOTA ARMA Models)
# =============================================================================
# Target specification:
#   - We are equally interested in all frequency components of the signal.
#   - The target Gamma is an allpass filter: it passes all frequencies with
#     unit gain and no phase distortion.
#   - In the time domain, this corresponds to a forward shift by one period
#     (i.e., predicting the next observation), which replicates the one-step-
#     ahead forecast of a correctly specified ARMA model.

Gamma <- rep(1, K + 1)

plot(
  Gamma,
  type  = "l",
  main  = paste("Allpass forecast target, denseness = ", K, sep = ""),
  axes  = FALSE,
  xlab  = "Frequency",
  ylab  = "Amplitude",
  col   = "black"
)

# Label the plotted target using the column name from weight_func
mtext("Target", line = -1, col = "black")

# Frequency axis: tick marks at multiples of pi/6 across [0, pi]
axis(1, at = c(0, 1:6 * K / 6 + 1),
     labels = c("0", "pi/6", "2pi/6", "3pi/6", "4pi/6", "5pi/6", "pi"))
axis(2)
box()

# Forecast horizon: Lag = -1 specifies a one-step-ahead forecast
Lag <- -1

# Filter length: number of coefficients in the causal (real-time) forecast filter.
# L = 10 is typically sufficient for most seasonally adjusted economic series.
L <- 10

# -----------------------------------------------------------------------------
# Estimate the MSE-Optimal DFA Filter
# -----------------------------------------------------------------------------
# MDFA_mse computes the filter coefficients that minimize the mean squared error
# between the filter output and the allpass target Gamma, given the assumed
# white noise spectrum (weight_func) and forecast horizon (Lag).

mdfa_obj <- MDFA_mse(L, weight_func, Lag, Gamma)$mdfa_obj

# -----------------------------------------------------------------------------
# Diagnostic Plot: Filter Coefficients, Time-Shift, and Amplitude Response
# -----------------------------------------------------------------------------
# plot_estimate_func visualizes three key properties of the estimated filter:
#   (1) Filter coefficients  : the time-domain weights applied to past observations.
#   (2) Amplitude function   : how much each frequency is amplified or attenuated.
#   (3) Time-shift function  : the phase delay introduced at each frequency.

plot_estimate_func(mdfa_obj, weight_func, Gamma)

# Interpretation of diagnostic output:
#   - Coefficients and amplitude are nearly vanishing.
#   - Minor deviations from the ideal response (i.e., zero) arise because K is finite
#     (the frequency grid is a discrete approximation of the continuous interval).
#   - For K/L > 10, these approximation errors are negligible in practice.


# =============================================================================
# Example 2: Lowpass Filtering — Extracting Trend and Cycle Components
# =============================================================================
# Motivation:
#   Unlike forecasting (Example 1), lowpass filtering is selective in frequency:
#   it retains low-frequency components (slow-moving trends and cycles) while
#   suppressing high-frequency noise.
#
# Use cases:
#   - Noise reduction: fade out high-frequency fluctuations irrelevant to the
#     underlying signal of interest.
#   - Signal extraction: isolate trend or business cycle components from a
#     noisy economic time series.
#   - Trading signals: define holding periods based on a chosen periodicity
#     threshold (see mean holding time below).
#
# Target specification:
#   - The ideal lowpass target Gamma passes all components with periodicity
#     greater than 2 * periodicity (i.e., frequencies below the cutoff).
#   - Components with periodicity below 2 * periodicity are fully suppressed.
#   - The cutoff frequency is defined as: cutoff = pi / periodicity.

# -----------------------------------------------------------------------------
# Lowpass Target: Ideal Frequency Response
# -----------------------------------------------------------------------------

periodicity <- 10                          # Minimum half-period of interest (in time units)
cutoff      <- pi / periodicity            # Cutoff frequency in radians

# Gamma = 1 for frequencies at or below the cutoff; 0 above (step function)
Gamma <- (0:(K)) <= K * cutoff / pi + 1.e-9

par(mfrow = c(1, 1))

plot(
  Gamma,
  type  = "l",
  main  = paste("Ideal lowpass, periodicity = ", periodicity, ", denseness = ", K, sep = ""),
  axes  = FALSE,
  xlab  = "Frequency",
  ylab  = "Amplitude",
  col   = "black"
)

mtext("Target", line = -1, col = "black")

axis(1, at = c(0, 1:6 * K / 6 + 1),
     labels = c("0", "pi/6", "2pi/6", "3pi/6", "4pi/6", "5pi/6", "pi"))
axis(2)
box()

# Interpretation of the ideal lowpass target:
#   - PASSBAND  (frequencies < cutoff): components with cycle duration > 2*periodicity
#     are passed through unchanged (unit gain).
#   - STOPBAND  (frequencies > cutoff): components with cycle duration < 2*periodicity
#     are fully eliminated (zero gain).
#   - The filter is called 'ideal' because the transition between passband and
#     stopband is instantaneous (a perfect step function in frequency).
#   - This target is intuitive and trivial to specify in the frequency domain,
#     but it cannot be implemented exactly with a finite causal filter.

# Since Gamma is real-valued and symmetric about zero frequency, the corresponding
# time-domain filter is also symmetric (two-sided). Symmetric filters introduce
# no phase distortion — all frequencies are shifted by the same amount (zero).
# However, symmetry implies non-causality: the filter requires future observations.

# -----------------------------------------------------------------------------
# Apply the Ideal Lowpass Filter to a White Noise Series
# -----------------------------------------------------------------------------

len    <- 1000    # Length of the simulated input series
set.seed(1)
x      <- rnorm(len)   # Simulate white noise as the test input

# Filter half-length: the ideal lowpass filter is bi-infinite in theory;
# M = 100 provides a close finite approximation (truncated symmetric filter)
M <- 100

# Compute ideal filter coefficients and apply the filter to x
id_obj <- ideal_filter_func(periodicity, M, x)

# -----------------------------------------------------------------------------
# Diagnostic Plots
# -----------------------------------------------------------------------------

# The filter coefficients are symmetric around the center: this confirms the
# filter is two-sided (non-causal) and introduces no phase distortion
ts.plot(id_obj$gamma)

# Overlay raw input (black) and filtered output (red):
# The filtered series is visibly smoother — high-frequency noise has been
# attenuated, leaving only components with periodicity > 2*periodicity
ts.plot(x)
lines(id_obj$y, col = "red",lwd=2)

# Mean holding time: average number of periods between consecutive zero-crossings
# of the filtered series. In a trading context, this represents the expected
# duration of a long or short position when using this filter as a signal.
#   - Higher periodicity → stronger smoothing → fewer zero-crossings → longer holding times
#   - Lower periodicity  → weaker smoothing  → more zero-crossings  → shorter holding times
id_obj$mean_holding_time

# =============================================================================
# Motivation for DFA: Bridging the Ideal Filter and Real-Time Application
# =============================================================================
# Advantage:
#   The user can freely specify any periodicity threshold, making this framework
#   strictly more flexible than classical time series approaches (e.g., HP filter,
#   Baxter-King, or ARMA-based signal extraction), which impose fixed target shapes.
#
# Problem:
#   The ideal lowpass filter is non-causal — it requires future data and therefore
#   cannot be applied in real time. As visible in the plot above, the filtered
#   series does not extend to the end of the sample.
#
# Solution:
#   DFA approximates the ideal target Gamma with a causal (one-sided) finite filter
#   that can be applied in real time using only current and past observations.
#   The following examples demonstrate how DFA achieves this approximation.
# =============================================================================

# =============================================================================
# Example 2 (Continued): Turning-Point Detection and Trading Signal Generation
# =============================================================================
# Setup:
#   - Assume x represents log-returns of a financial asset.
#   - Then cumsum(x) corresponds to the cumulative (log-) price series.
#
# Trading logic based on local drift (ideal lowpass filter output):
#   - The ideal lowpass filter applied to log-RETURNS extracts the LOCAL DRIFT
#     of the (log-) price series — i.e., the slow-moving trend component.
#   - A MAXIMUM in (log-) prices occurs when the local drift crosses zero
#     from ABOVE (positive drift turning negative → price about to decline).
#     → SELL signal.
#   - A MINIMUM in (log-) prices occurs when the local drift crosses zero
#     from BELOW (negative drift turning positive → price about to rise).
#     → BUY signal.
#
# Visual conventions in the plots below:
#   - Blue line        : local drift (ideal lowpass filter output, id_obj$y)
#   - Green vertical lines : BUY  signals (drift crosses zero from below)
#   - Red vertical lines   : SELL signals (drift crosses zero from above)
# =============================================================================

# -----------------------------------------------------------------------------
# Full-Sample Overview Plot
# -----------------------------------------------------------------------------
# Combine (log-) price and local drift into a single matrix for plotting.
# na.exclude removes boundary NAs introduced by the non-causal filter.

mplot <- na.exclude(cbind(cumsum(x), id_obj$y))
colnames(mplot) <- c("(log) price", "local drift")

plot_data <- mplot
head(plot_data)

# Plot (log-) price series with overlaid local drift and trading signals
ts.plot(plot_data[, 1])
mtext(colnames(plot_data)[1], line = -1)

lines(plot_data[, 2], col = "blue")
mtext(colnames(plot_data)[2], line = -2, col = "blue")

# Horizontal reference line at zero (drift sign changes indicate turning points)
abline(h = 0)

# BUY signals: local drift crosses zero from below (negative → positive)
abline(
  v   = 1 + which(plot_data[1:(nrow(plot_data) - 1), 2] < 0 &
                    plot_data[2:nrow(plot_data), 2] > 0),
  col = "green"
)

# SELL signals: local drift crosses zero from above (positive → negative)
abline(
  v   = 1 + which(plot_data[1:(nrow(plot_data) - 1), 2] > 0 &
                    plot_data[2:nrow(plot_data), 2] < 0),
  col = "red"
)

# Note: the full-sample plot may appear cluttered due to the density of signals.
# The zoomed plot below provides a clearer view of the turning-point logic.

# -----------------------------------------------------------------------------
# Zoomed Plot: Observations 400–500
# -----------------------------------------------------------------------------
# Restrict to a 100-observation window to clearly illustrate:
#   - The relationship between (log-) price turning points and drift zero-crossings.
#   - The timing of buy and sell signals relative to price peaks and troughs.

anf <- 400
enf <- 500
plot_data <- mplot[anf:enf, ]

ts.plot(plot_data[, 1], ylim = c(min(plot_data), max(plot_data)))
mtext(colnames(plot_data)[1], line = -1)

lines(plot_data[, 2], col = "blue")
mtext(colnames(plot_data)[2],                                          line = -2, col = "blue")
mtext("Buy  (green vertical lines): signal (blue) crosses zero from below", line = -3, col = "green")
mtext("Sell (red vertical lines):   signal (blue) crosses zero from above", line = -4, col = "red")

abline(h = 0)

# BUY signals within the zoomed window
abline(
  v   = 1 + which(plot_data[1:(nrow(plot_data) - 1), 2] < 0 &
                    plot_data[2:nrow(plot_data), 2] > 0),
  col = "green"
)

# SELL signals within the zoomed window
abline(
  v   = 1 + which(plot_data[1:(nrow(plot_data) - 1), 2] > 0 &
                    plot_data[2:nrow(plot_data), 2] < 0),
  col = "red"
)

# =============================================================================
# Trading Horizons: Tactical vs. Strategic Positioning
# =============================================================================
# The periodicity parameter controls the frequency selectivity of the lowpass
# filter, directly determining the trading horizon and signal frequency:
#
#   Short-term  (tactical)  : periodicity <- 10
#                               Weekly-scale signals; frequent trades.
#
#   Medium-term (strategic) : periodicity <- 20
#                               Monthly-scale signals; moderate trade frequency.
#
#   Long-term   (strategic) : periodicity <- 60   (quarterly)
#                             periodicity <- 250  (yearly; increase M accordingly)
#                               Infrequent, high-conviction trades.
#
# General principle:
#   Higher periodicity → stronger smoothing → fewer, longer-duration signals
#                      → fewer trades → lower transaction costs
#   Lower  periodicity → weaker smoothing  → more, shorter-duration signals
#                      → more trades → higher transaction costs
#
# Important: when increasing periodicity substantially (e.g., to 250),
# the filter half-length M should also be increased to avoid truncation error
# in the ideal filter approximation.
#
# =============================================================================
# Key Limitation and DFA Solution
# =============================================================================
# Problem:
#   The ideal lowpass filter used above is NON-CAUSAL — it requires future
#   observations and therefore cannot be applied in real time. As a result,
#   buy/sell signals cannot be generated at the current time point.
#
# Solution (covered in examples below):
#   Use DFA to approximate the non-causal ideal target Gamma with a CAUSAL
#   (one-sided) finite filter that operates on current and past data only.
#   Trading signals are then derived from zero-crossings of the causal DFA
#   filter output — replacing the non-causal ideal filter in a real-time setting.
# =============================================================================



# =============================================================================
# Example 3: Bandpass Filter Design
# =============================================================================
#   Unlike forecasting, bandpass filtering focuses on isolating specific frequency
#   components rather than capturing all of them.
#
#   When specifying a bandpass filter, the goal is typically one of two things:
#     1. Suppress undesirable components (i.e., high- and low-frequency noise), or
#     2. Emphasize components corresponding to cycles of interest.
#
#   This type of filter is widely used in business-cycle analysis, where analysts
#   seek to isolate medium-term cyclical fluctuations from a time series.

# --- Target Specification ---
# Define the bandpass target Gamma in the frequency domain.
# Each periodicity corresponds to a cutoff frequency via: cutoff = pi / periodicity
# The bandpass retains components with periodicities between periodicity_high and
# periodicity_low (i.e., the frequency band [cutoff_high, cutoff_low]).

periodicity_low  <- 5                        # Lower bound on periodicity (shorter cycles)
cutoff_low       <- pi / periodicity_low     # Corresponding upper cutoff frequency
periodicity_high <- 10                       # Upper bound on periodicity (longer cycles)
cutoff_high      <- pi / periodicity_high    # Corresponding lower cutoff frequency

# Construct the ideal bandpass frequency response:
# = (lowpass at cutoff_low) minus (lowpass at cutoff_high)
# A small constant (1e-9) is added for numerical stability
Gamma <- ((0:(K)) <= K * cutoff_low / pi) - ((0:(K)) <= K * cutoff_high / pi) + 1.e-9

# --- Plot the Ideal Bandpass Target ---
plot(Gamma, type = "l",
     main = paste("Ideal bandpass, periodicities =", periodicity_low, ",",
                  periodicity_high, ", denseness =", K, sep = ""),
     axes = F, xlab = "Frequency", ylab = "Amplitude", col = "black")

# Annotate the plot to identify the curve as the target filter
mtext("Target", line = -1, col = "black")

# Label the x-axis with standardized frequency markers from 0 to pi
axis(1, at = c(0, 1:6 * K / 6 + 1),
     labels = c("0", "pi/6", "2pi/6", "3pi/6", "4pi/6", "5pi/6", "pi"))
axis(2)
box()

# --- Interpretation of the Ideal Bandpass Filter ---
#
#   - Gamma = 1 (passband):  Components with cycle durations between
#     2*periodicity_high and 2*periodicity_low are passed through unchanged.
#   - Gamma = 0 (stopband):  All other frequency components are eliminated.
#
#   This construction is referred to as an 'ideal' bandpass filter because:
#     - It has a perfectly rectangular shape in the frequency domain.
#     - It is straightforward to specify and intuitively easy to interpret.

# --- Note on Filter Coefficients and Causality ---
#
#   The time-domain coefficients of the ideal bandpass can be derived as the
#   difference between two ideal lowpass filters (one at cutoff_low, one at
#   cutoff_high). However, this calculation is omitted here as the primary
#   interest lies in the frequency-domain specification.
#
#   Importantly, like the ideal lowpass filter, the ideal bandpass filter is
#   non-causal — it requires future observations, making it infeasible for
#   real-time applications. A causal approximation can be obtained using the
#   Direct Filter Approach (DFA).



# =============================================================================
# Example 4: Arbitrary (User-Defined) Filter Targets
# =============================================================================
#   This example demonstrates that DFA is not restricted to standard filter shapes
#   such as lowpass or bandpass. The user can define any frequency response that
#   suits their analytical objectives — a level of flexibility not available in
#   classical time series filtering approaches.

# --- Target Specification ---
# Construct an arbitrary frequency response Gamma_arbitrary over [0, pi].
# The default value is 1 (full passthrough), with selective modifications
# applied to specific frequency bands to illustrate a non-standard target shape.

Gamma_arbitrary <- rep(1, K + 1)          # Initialize: pass all frequencies

Gamma_arbitrary[3:20]   <- 0.4            # Partial attenuation in a low-frequency band
Gamma_arbitrary[37:98]  <- 0              # Complete suppression in a mid-frequency band
Gamma_arbitrary[167:208] <- pi            # Amplification beyond unity in another band
Gamma_arbitrary[398:476] <- 0.1          # Near-suppression in a high-frequency band

# --- Plot the Arbitrary Target ---
plot(Gamma_arbitrary, type = "l", main = "Arbitrary target",
     axes = F, xlab = "Frequency", ylab = "Amplitude", col = "black")

# Label the curve as the filter target
mtext("Target", line = -1, col = "black")

# Label x-axis with evenly spaced frequency markers from 0 to pi
axis(1, at = c(0, 1:6 * K / 6 + 1),
     labels = c("0", "pi/6", "2pi/6", "3pi/6", "4pi/6", "5pi/6", "pi"))
axis(2)
box()

# --- Key Takeaways ---
#
#   - Like the ideal lowpass and bandpass filters in previous examples, this
#     arbitrary filter is non-causal and cannot be applied directly in real time.
#   - The DFA framework can approximate any such user-defined target with a
#     causal (one-sided) filter, making real-time estimation feasible.
#   - This generality is a key advantage of DFA over classical time series methods,
#     which are typically constrained to fixed, predefined filter shapes.


# =============================================================================
# Example 5: Approximating the Ideal Lowpass with a Causal Filter (White Noise Spectrum)
# =============================================================================
#   The ideal lowpass filter is non-causal and therefore cannot be applied in real
#   time. This example uses the Direct Filter Approach (DFA) to approximate the
#   ideal lowpass with a causal (one-sided) filter under the assumption of a white
#   noise input spectrum.
#
#   The resulting amplitude and time-shift functions are examined to understand
#   the trade-off between noise suppression and filter delay.

# --- Setup ---
K <- 600    # Frequency grid denseness (number of frequency ordinates = K+1)

# Spectrum: white noise assumption (flat spectrum = 1 at all frequencies)
# The weight matrix has two columns:
#   - Column 1 ("target"):      the desired filter output (Gamma)
#   - Column 2 ("explanatory"): the input series (same as target in univariate design)
weight_func <- matrix(rep(1, 2 * (K + 1)), ncol = 2)
colnames(weight_func) <- c("target", "explanatory")
weight_func_noise <- weight_func

# Define the ideal lowpass target: pass frequencies below cutoff, suppress the rest
periodicity <- 10                        # Retain cycles of length >= 10 time units
cutoff      <- pi / periodicity          # Cutoff frequency corresponding to periodicity
Gamma       <- (0:(K)) <= K * cutoff / pi + 1.e-9   # Binary frequency response (with small numerical offset)

# --- Plot the Ideal Lowpass Target ---
plot(Gamma, type = "l",
     main = paste("Ideal lowpass, periodicity =", periodicity,
                  ", denseness =", K, sep = ""),
     axes = F, xlab = "Frequency", ylab = "Amplitude", col = "black")

# Label the curve as the filter target
mtext("Target", line = -1, col = "black")

# Label x-axis with evenly spaced frequency markers from 0 to pi
axis(1, at = c(0, 1:6 * K / 6 + 1),
     labels = c("0", "pi/6", "2pi/6", "3pi/6", "4pi/6", "5pi/6", "pi"))
axis(2)
box()

# --- Filter Design Parameters ---
Lag <- 0    # Timing mode: 0 = nowcast, >0 = backcast, <0 = forecast
L   <- 200  # Filter length (number of causal coefficients)

# --- DFA Optimization (MSE Criterion) ---
# Estimate the optimal causal filter using the DFA-MSE criterion.
# Under the MSE criterion, the filter minimizes the combined fitting error
# across both the amplitude and time-shift functions simultaneously.
mdfa_obj_noise_mse <- MDFA_mse(L, weight_func_noise, Lag, Gamma)$mdfa_obj

b <- mdfa_obj_noise_mse$b             # Extract estimated filter coefficients

# Plot the filter coefficients, amplitude function, and time-shift function
plot_estimate_func(mdfa_obj_noise_mse, weight_func_noise, Gamma)

# --- Interpretation of DFA Output Plots ---
#
#   1. Filter Coefficients (first plot):
#        The coefficients are one-sided (causal), meaning the filter relies only
#        on current and past observations — no future data is used.
#
#   2. Time-Shift Function (second plot):
#        Because the filter cannot look ahead, its output is delayed relative to
#        the non-causal ideal target. The time-shift (lag) in the passband is
#        approximately 2 to 4 time units.
#
#   3. Amplitude Function (third plot):
#        The DFA amplitude (colored line) approximates the ideal target (violet line).
#        Deviations reflect the fundamental trade-off inherent in causal filtering.
#
#   4. Amplitude vs. Time-Shift Trade-off:
#        - The MSE criterion balances amplitude fit and time-shift fit simultaneously.
#        - Improving amplitude accuracy (better noise suppression) increases the lag.
#        - Reducing the lag (faster response) degrades amplitude accuracy (more noise leakage).
#        - MSE represents one specific, balanced solution to this trade-off.
#          Other criteria (covered in later tutorials) allow customization of this balance.


# --- Comparing Ideal vs. DFA Causal Filter Outputs ---
#
#   Apply both the ideal non-causal filter and the DFA causal filter to a
#   simulated white noise series and compare their outputs visually.

len <- 1000         # Length of the simulated time series
set.seed(1)
x <- rnorm(len)     # Simulate a white noise input series

# Ideal (non-causal) filter: requires observations on both sides of each time point.
# A finite approximation of length M is used (the true ideal filter is bi-infinite).
M      <- 100
id_obj <- ideal_filter_func(periodicity, M, x)
output_ideal <- id_obj$y

# DFA (causal) filter: applied using the estimated one-sided coefficients b
filt_obj   <- filt_func(x, b)
output_dfa <- filt_obj$yhat
output_dfa[1:(L - 1)] <- NA    # Set initial observations to NA (filter warm-up period)

# --- Full-Sample Comparison Plot ---
par(mfrow = c(1, 1))
ts.plot(output_ideal, col = "blue",
        main = "Output of ideal lowpass (blue) vs. DFA causal filter (red)")
lines(output_dfa, col = "red")

# Note: The ideal filter output (blue) is unavailable near the sample boundaries
# due to its non-causal nature. The DFA output (red) extends to the sample end,
# providing a real-time estimate at the cost of some noise leakage and delay.

# --- Zoomed Comparison Plot ---
# Examine a shorter segment for a closer look at differences between the two outputs.
anf <- 400
enf <- 500
ts.plot(output_ideal[anf:enf], col = "blue",
        main = "Output of ideal lowpass (blue) vs. DFA one-sided filter (red)")
lines(output_dfa[anf:enf], col = "red")
abline(h = 0)

# Observations from the zoomed plot:
#   - The DFA output (red) is noisier than the ideal output (blue):
#       this reflects imperfect amplitude suppression in the stopband (noise leakage).
#   - The DFA output is slightly shifted to the right:
#       this reflects the positive time-shift (lag) introduced by causal filtering.
#   - Both effects — noise leakage and delay — are directly visible in the
#     amplitude and time-shift diagnostic plots generated earlier.
#   - Later tutorials demonstrate how to reduce both effects through customization.


# --- Parameter Experimentation ---
#
#   Users are encouraged to modify the following parameters to explore their effects:
#
#   1. periodicity:
#        Changing this adjusts the cutoff frequency of the lowpass target,
#        controlling which cycle lengths are retained or suppressed.
#
#   2. Lag:
#        Lag = 0  → Nowcast: estimate the current filtered value in real time.
#        Lag < 0  → Forecast: estimate a future value of the filtered output
#                   (note: this forecasts the filtered target, not the raw series).
#        Lag > 0  → Backcast: estimate a past value, used when revising historical
#                   data (e.g., macroeconomic data subject to revision).
#                   Backcasting is non-trivial, even for simple lowpass targets.


# --- Summary ---
#
#   - The user specifies a target filter Gamma that reflects their analytical interest.
#   - DFA derives an optimal causal approximation (here: MSE) to that target.
#   - The resulting real-time estimate is available through to the end of the sample.
#   - The estimate is 'as close as possible' to the ideal target output in the
#     mean-square sense, subject to the constraints of causal filtering.
#   - This framework is more flexible and general than classical time series approaches.

# =============================================================================
# Example 6: DFA Approximation of Ideal Lowpass — AR(1) Input Spectrum
# =============================================================================
#   This example mirrors Example 5 but replaces the white noise spectrum with
#   an AR(1) spectrum. The goal is to illustrate how the input spectrum influences
#   the DFA filter design: the spectrum acts as a frequency-dependent weighting
#   in the MSE criterion, causing the filter to fit the target more closely at
#   frequencies that carry more power in the input signal.

# --- Spectrum Specification ---
K <- 600        # Frequency grid denseness (number of frequency ordinates = K + 1)

# Define an AR(1) process with autoregressive coefficient a1.
# Setting a1 > 0 produces a spectrum concentrated at low frequencies (persistent process).
# b1 = NULL indicates no moving average component (pure AR(1)).
a1     <- 0.9
b1     <- NULL
plot_T <- TRUE   # Plot the spectrum during computation

# Compute the AR(1) spectrum over the frequency grid [0, pi]
spec <- arma_spectrum_func(a1, b1, K, plot_T)$arma_spec

# Construct the weight matrix:
#   - Column 1 ("spectrum target"):      spectrum of the target series
#   - Column 2 ("spectrum explanatory"): spectrum of the input series
# In a univariate setting, both columns are identical.
weight_func <- cbind(spec, spec)
colnames(weight_func) <- c("spectrum target", "spectrum explanatory")
weight_func_ar1 <- weight_func

# --- Ideal Lowpass Target ---
# Define the same ideal lowpass target as in Example 5 for direct comparison.
periodicity <- 10                          # Retain cycles of length >= 2*10 time units
cutoff      <- pi / periodicity            # Corresponding cutoff frequency
Gamma       <- (0:(K)) <= K * cutoff / pi + 1.e-9   # Binary frequency response

# --- Plot the Ideal Lowpass Target ---
plot(Gamma, type = "l",
     main = paste("Ideal lowpass, periodicity =", periodicity,
                  ", denseness =", K, sep = ""),
     axes = F, xlab = "Frequency", ylab = "Amplitude", col = "black")

# Label the curve as the filter target
mtext("Target", line = -1, col = "black")

# Label x-axis with evenly spaced frequency markers from 0 to pi
axis(1, at = c(0, 1:6 * K / 6 + 1),
     labels = c("0", "pi/6", "2pi/6", "3pi/6", "4pi/6", "5pi/6", "pi"))
axis(2)
box()

# --- Filter Design Parameters ---
Lag <- 0    # Timing mode: 0 = nowcast, >0 = backcast, <0 = forecast
L   <- 200  # Filter length (number of causal coefficients)

# --- DFA Optimization (MSE Criterion, AR(1) Spectrum) ---
# Estimate the optimal causal filter using the DFA-MSE criterion with the AR(1) spectrum.
# The spectrum enters the MSE criterion as a frequency-dependent weight:
# frequencies with higher spectral power receive greater emphasis in the optimization.
mdfa_obj_ar1_mse <- MDFA_mse(L, weight_func_ar1, Lag, Gamma)$mdfa_obj

b <- mdfa_obj_ar1_mse$b    # Extract estimated filter coefficients

# Plot filter coefficients, amplitude function, and time-shift function
plot_estimate_func(mdfa_obj_ar1_mse, weight_func, Gamma)
# The filter assigns more weight to lag 0 (top left panel)
# The time-shift is smaller (than white noise case)
# The amplitude is closer to target in passband of ideal filter. 
# But the amplitude is farther away from zero in the stopband (noise leakage)


# --- Comparing White Noise vs. AR(1) Filter Designs ---
#
#   Only the spectrum has changed relative to Example 5 (white noise → AR(1)).
#   The comparison below highlights how this change affects the DFA solution.
plot_compare_two_DFA_designs(mdfa_obj_noise_mse, mdfa_obj_ar1_mse,
                             weight_func_noise, weight_func_ar1, Gamma)

# --- Interpretation of Comparison Plots ---
#
#   Top plot (white noise, Example 5):
#     - The spectrum (red) is flat: all frequencies contribute equally to the
#       MSE criterion, so the amplitude function fits the target uniformly
#       across the frequency band.
#
#   Bottom plot (AR(1), Example 6):
#     - The spectrum (red) is concentrated at low frequencies (since a1 = 0.9 > 0),
#       meaning low-frequency components dominate the AR(1) process.
#     - As a result, the DFA amplitude (black) matches the target (violet) more
#       closely at low frequencies, at the expense of a poorer fit at higher
#       frequencies (i.e., greater noise leakage in the stopband).
#
#   Key Principle:
#     The input spectrum modulates the frequency-specific quality of the DFA fit.
#     The DFA amplitude (and time-shift) match the target most closely at
#     frequencies where the spectrum is largest — those frequencies contribute
#     most to the MSE objective and are therefore prioritized by the optimizer.


# =============================================================================
# Example 7: Effect of a Deliberately Modified (Tweaked) Spectrum on DFA Design
# =============================================================================
#   Building on the spectral modulation insight from Example 6, this example
#   artificially amplifies the AR(1) spectrum at a single frequency (pi/2)
#   to verify that the DFA filter adapts its fit accordingly.
#
#   Since the ideal lowpass target is zero at pi/2 (that frequency lies in the
#   stopband), we expect the DFA amplitude function to be driven closer to zero
#   at pi/2 when that frequency is heavily weighted in the spectrum.

# --- Construct the Tweaked Spectrum ---
# Start from the AR(1) spectrum and introduce a large spike at frequency pi/2.
# The midpoint of the frequency grid corresponds to pi/2.
weight_func_tweaked <- weight_func_ar1
weight_func_tweaked[nrow(weight_func_tweaked) / 2, ] <- 100   # Artificial spike at pi/2

# --- Plot the Tweaked Spectrum ---
plot(abs(weight_func_tweaked[, 1]), type = "l",
     main = paste("Tweaked AR(1) spectrum, denseness =", K, sep = ""),
     axes = F, xlab = "Frequency", ylab = "Amplitude", col = "black")

# Label x-axis with evenly spaced frequency markers from 0 to pi
axis(1, at = c(0, 1:6 * K / 6 + 1),
     labels = c("0", "pi/6", "2pi/6", "3pi/6", "4pi/6", "5pi/6", "pi"))
axis(2)
box()

# --- Expected Effect of the Spectral Spike ---
#
#   1. The spectrum at pi/2 has been artificially inflated to a very large value (100).
#   2. Since the MSE criterion weights fitting errors by spectral power, the DFA
#      optimizer will prioritize reducing the error at pi/2 above all other frequencies.
#   3. Because the ideal lowpass target is zero at pi/2 (stopband), the DFA
#      amplitude function is expected to approach zero near pi/2 — better stopband
#      suppression at that frequency compared to the unmodified AR(1) design.

# --- Ideal Lowpass Target (unchanged) ---
periodicity <- 10
cutoff      <- pi / periodicity
Gamma       <- (0:(K)) <= K * cutoff / pi + 1.e-9

# --- Plot the Ideal Lowpass Target ---
plot(Gamma, type = "l",
     main = paste("Ideal lowpass, periodicity =", periodicity,
                  ", denseness =", K, sep = ""),
     axes = F, xlab = "Frequency", ylab = "Amplitude", col = "black")

# Label the curve as the filter target
mtext("Target", line = -1, col = "black")

# Label x-axis with evenly spaced frequency markers from 0 to pi
axis(1, at = c(0, 1:6 * K / 6 + 1),
     labels = c("0", "pi/6", "2pi/6", "3pi/6", "4pi/6", "5pi/6", "pi"))
axis(2)
box()

# --- Filter Design Parameters ---
Lag <- 0    # Nowcast
L   <- 200  # Filter length

# --- DFA Optimization (MSE Criterion, Tweaked Spectrum) ---
# Estimate the optimal causal filter using the tweaked spectrum.
mdfa_obj_tweaked_mse <- MDFA_mse(L, weight_func_tweaked, Lag, Gamma)$mdfa_obj

# Plot filter coefficients, amplitude function, and time-shift function
plot_estimate_func(mdfa_obj_tweaked_mse, weight_func_tweaked, Gamma)

# --- Comparing Tweaked vs. Original AR(1) Filter Designs ---
#
#   The comparison below confirms whether the expected spectral modulation
#   effect materializes in the DFA amplitude function.
plot_compare_two_DFA_designs(mdfa_obj_tweaked_mse, mdfa_obj_ar1_mse,
                             weight_func_tweaked, weight_func_ar1, Gamma)

# --- Interpretation of Results ---
#
#   Comparing the tweaked design (top) with the original AR(1) design (bottom):
#
#   1. The two amplitude functions are broadly similar across most frequencies,
#      confirming that the tweak is localized in its effect.
#
#   2. Near frequency pi/2, the tweaked amplitude function approaches zero —
#      much closer to the ideal lowpass target value of zero at that frequency.
#      This confirms that the artificially large spectral weight at pi/2 forces
#      the DFA optimizer to prioritize suppression of that frequency component.
#
#   3. The underlying mechanism is the same as in Example 6:
#        - Target value at pi/2 = 0  (ideal lowpass suppresses that component)
#        - Spectral weight at pi/2 = 100  (DFA heavily penalizes any deviation
#          from the target at that frequency)
#        - Combined effect: the amplitude is driven close to zero at pi/2,
#          improving stopband attenuation at that specific frequency.
#
#   4. A similar improvement is observed in the time-shift function near pi/2,
#      which will be examined further in the next example.



# =============================================================================
# Example 8: DFA Approximation with a Spectral Spike at a Low Frequency
# =============================================================================
#   This example extends Example 6 (AR(1) spectrum) by introducing an artificial
#   spike at a low frequency — near zero — rather than at pi/2 as in Example 7.
#
#   The purpose is to further verify the spectral modulation principle:
#   the DFA filter matches the target most closely at frequencies where the
#   input spectrum (and hence the MSE weighting) is largest.
#
#   Since the chosen frequency lies in the passband of the ideal lowpass
#   (target = 1 near zero), we expect the amplitude function to be driven
#   closer to 1 at that frequency, and the time-shift to approach zero —
#   both consistent with a better local fit to the target.

# --- Construct the Tweaked Spectrum ---
# Start from the AR(1) spectrum and introduce a large spike at frequency index 10,
# corresponding to frequency pi * (10 - 1) / 600 ≈ 0.047 (near zero).
weight_func_tweaked <- weight_func_ar1
weight_func_tweaked[10, ] <- 100    # Artificial spike near frequency zero

# --- Plot the Tweaked Spectrum ---
plot(abs(weight_func_tweaked[, 1]), type = "l",
     main = paste("Tweaked AR(1) spectrum, denseness =", K, sep = ""),
     axes = F, xlab = "Frequency", ylab = "Amplitude", col = "black")

# Label x-axis with evenly spaced frequency markers from 0 to pi
axis(1, at = c(0, 1:6 * K / 6 + 1),
     labels = c("0", "pi/6", "2pi/6", "3pi/6", "4pi/6", "5pi/6", "pi"))
axis(2)
box()

# --- Expected Effect of the Spectral Spike ---
#
#   1. The spectrum at index 10 (frequency ≈ 0.047, near zero) has been
#      artificially inflated to a very large value (100).
#   2. Since the MSE criterion weights fitting errors by spectral power, the DFA
#      optimizer will strongly prioritize reducing the approximation error near
#      that frequency.
#   3. Since the ideal lowpass target equals 1 in the passband (near zero),
#      the DFA amplitude is expected to approach 1 more closely at that frequency,
#      and the time-shift is expected to approach zero — confirming an improved
#      local fit to the target in both dimensions.

# --- Ideal Lowpass Target (unchanged) ---
periodicity <- 10                          # Retain cycles of length >= 10 time units
cutoff      <- pi / periodicity            # Corresponding cutoff frequency
Gamma       <- (0:(K)) <= K * cutoff / pi + 1.e-9   # Binary frequency response

# --- Plot the Ideal Lowpass Target ---
plot(Gamma, type = "l",
     main = paste("Ideal lowpass, periodicity =", periodicity,
                  ", denseness =", K, sep = ""),
     axes = F, xlab = "Frequency", ylab = "Amplitude", col = "black")

# Label the curve as the filter target
mtext("Target", line = -1, col = "black")

# Label x-axis with evenly spaced frequency markers from 0 to pi
axis(1, at = c(0, 1:6 * K / 6 + 1),
     labels = c("0", "pi/6", "2pi/6", "3pi/6", "4pi/6", "5pi/6", "pi"))
axis(2)
box()

# --- Filter Design Parameters ---
Lag <- 0    # Timing mode: 0 = nowcast, >0 = backcast, <0 = forecast
L   <- 200  # Filter length (number of causal coefficients)

# --- DFA Optimization (MSE Criterion, Tweaked Spectrum) ---
# Estimate the optimal causal filter using the tweaked AR(1) spectrum.
mdfa_obj_tweaked_mse <- MDFA_mse(L, weight_func_tweaked, Lag, Gamma)$mdfa_obj

# Plot filter coefficients, amplitude function, and time-shift function
plot_estimate_func(mdfa_obj_tweaked_mse, weight_func_tweaked, Gamma)

# --- Interpretation of Results ---
#
#   The output confirms the expected behavior:
#   - The amplitude function approaches the target value (1) more closely near
#     the tweaked frequency, compared to the unmodified AR(1) design.
#   - The time-shift function similarly approaches zero at that frequency,
#     indicating a reduction in local filter delay.
#
#   This result reinforces the core principle of spectral modulation in DFA:
#
#     "The DFA amplitude and time-shift functions match the target most closely
#      at frequencies where the spectrum (weighting function) is largest."
#
#   In other words, the spectrum acts as a frequency-specific priority signal
#   for the MSE optimizer: heavily weighted frequencies receive a more accurate
#   approximation of the target, while less weighted frequencies tolerate larger
#   deviations.


# --- Summary: Generality of the DFA Framework ---
#
#   These examples collectively illustrate the broad flexibility of DFA
#   along two key dimensions:
#
#   1. Target Specification — the desired filter output Gamma can represent:
#        - One- or multi-step-ahead forecasting targets
#        - Lowpass filters (nowcast, forecast, or backcast)
#        - Bandpass, highpass, or fully arbitrary frequency response shapes
#        - Any user-defined target that reflects specific analytical objectives
#
#   2. Spectrum Specification — the input weighting function can be derived from:
#        - Parametric models (e.g., ARMA processes)
#        - Non-parametric estimates (e.g., the discrete Fourier transform)
#        - Deliberately modified or tweaked spectra (as demonstrated here)
#        - Any other weighting scheme relevant to the application


# --- Outlook: Customization (Covered in Later Tutorials) ---
#
#   Beyond the MSE criterion demonstrated here, DFA supports explicit
#   customization of the amplitude-shift trade-off:
#
#   - Noise Leakage Control:
#       Suppress unwanted frequency components more aggressively to improve
#       the reliability and interpretability of filter output signals.
#
#   - Time-Shift (Lag) Control:
#       Reduce filter delay to enable faster signal detection, which is
#       particularly relevant in real-time applications such as trading,
#       nowcasting, or early warning systems.
#
#   These customization tools allow the user to move beyond the balanced
#   MSE compromise and tailor the filter to their specific performance priorities.



# =============================================================================
# Example 9: Introducing Overfitting — Effect of Reducing Filter Length L
# =============================================================================
#   This example uses the same tweaked AR(1) spectrum as Example 8 (a large spike
#   at a low frequency near zero), but reduces the filter length L substantially.
#
#   The purpose is to illustrate the concept of overfitting in the DFA context:
#   a filter with many coefficients (large L) has high flexibility and can match
#   the target very closely at specific frequencies — including artificially
#   inflated ones — at the cost of fitting noise rather than signal structure.
#   Reducing L limits this flexibility and produces a more constrained, robust design.

# --- Tweaked AR(1) Spectrum (same as Example 8) ---
# Start from the AR(1) spectrum and introduce a large spike near frequency zero.
weight_func_tweaked <- weight_func_ar1
weight_func_tweaked[10, ] <- 100    # Artificial spike at frequency index 10 (≈ pi * 9/600)

# --- Plot the Tweaked Spectrum ---
plot(abs(weight_func_tweaked[, 1]), type = "l",
     main = paste("Tweaked AR(1) spectrum, denseness =", K, sep = ""),
     axes = F, xlab = "Frequency", ylab = "Amplitude", col = "black")

# Label x-axis with evenly spaced frequency markers from 0 to pi
axis(1, at = c(0, 1:6 * K / 6 + 1),
     labels = c("0", "pi/6", "2pi/6", "3pi/6", "4pi/6", "5pi/6", "pi"))
axis(2)
box()

# --- Motivation: From Overfitting to Constrained Estimation ---
#
#   Observations from Example 8 (L = 200):
#   1. The amplitude and time-shift functions matched the target near-perfectly
#      at the spike frequency — an unusually good local fit.
#   2. In most real-world settings, such a perfect fit is a warning sign of
#      overfitting: the filter has adapted to an artificial feature of the
#      spectrum rather than genuine signal structure.
#   3. The root cause is excessive flexibility: with L = 200 coefficients,
#      the filter has enough degrees of freedom to accommodate very narrow
#      spectral features, including artificial spikes.
#   4. This example investigates what happens when L is reduced significantly,
#      limiting the filter's flexibility and its ability to chase spike features.

# --- Ideal Lowpass Target (unchanged) ---
periodicity <- 10                          # Retain cycles of length >= 10 time units
cutoff      <- pi / periodicity            # Corresponding cutoff frequency
Gamma       <- (0:(K)) <= K * cutoff / pi + 1.e-9   # Binary frequency response

# --- Plot the Ideal Lowpass Target ---
plot(Gamma, type = "l",
     main = paste("Ideal lowpass, periodicity =", periodicity,
                  ", denseness =", K, sep = ""),
     axes = F, xlab = "Frequency", ylab = "Amplitude", col = "black")

# Label the curve as the filter target
mtext("Target", line = -1, col = "black")

# Label x-axis with evenly spaced frequency markers from 0 to pi
axis(1, at = c(0, 1:6 * K / 6 + 1),
     labels = c("0", "pi/6", "2pi/6", "3pi/6", "4pi/6", "5pi/6","pi"))
axis(2)
box()

# --- Filter Design Parameters ---
Lag <- 0    # Nowcast
L   <- 10   # Filter length: deliberately reduced to limit degrees of freedom

# --- DFA Optimization (MSE Criterion, Tweaked Spectrum, Small L) ---
# Estimate the optimal causal filter using the tweaked spectrum and restricted filter length.
mdfa_obj_tweaked_mse <- MDFA_mse(L, weight_func_tweaked, Lag, Gamma)$mdfa_obj

# Plot filter coefficients, amplitude function, and time-shift function
plot_estimate_func(mdfa_obj_tweaked_mse, weight_func_tweaked, Gamma)

# --- Interpretation of Results (L = 10) ---
#
#   Compared to Example 8 (L = 200), the fit at the spike frequency is noticeably
#   less precise — the amplitude and time-shift no longer snap to the target value
#   at the tweaked frequency.
#
#   This behavior is explained by the following:
#
#   1. Filter flexibility and L:
#        The transfer function of a filter of length L is a polynomial of degree
#        L - 1 in exp(i*omega). The number of peaks and troughs in the amplitude
#        and time-shift functions is therefore directly controlled by L.
#        - Small L → few, broad ripples → smooth but inflexible amplitude function
#        - Large L → many, narrow ripples → flexible but potentially overfit
#
#   2. Overfitting signature:
#        Many narrow peaks and dips in the amplitude or time-shift function
#        (as seen with L = 200 and the tweaked spectrum) are a direct indicator
#        of overfitting. The filter is exploiting its degrees of freedom to chase
#        artificial spectral features rather than capturing genuine signal structure.
#
#   3. Constraining L as a regularization tool:
#        Reducing L is the simplest way to prevent overfitting. With L = 10,
#        the filter lacks the resolution to match a narrow spike in the spectrum,
#        producing a smoother and more robust amplitude function overall.

# --- Additional Experiments ---
#
#   The following experiments illustrate the boundary cases of filter flexibility:
#
#   Experiment 1: Very large L (e.g., L = 600 or L = 2*K)
#     - At L = 2*K, the DFA-MSE solution achieves a perfect fit at all discrete
#       frequency ordinates: the amplitude exactly matches the target and the
#       time-shift is zero throughout the passband.
#     - This is the extreme case of overfitting: the filter has as many parameters
#       as there are frequency equations, leaving no degrees of freedom for
#       out-of-sample generalization.
#     - For L > 2*K, the system becomes singular (more parameters than equations)
#       and the numerical optimization breaks down entirely.
#
#   Experiment 2: L = 10 with an increasingly large spike (e.g., weight_func_tweaked[10,] <- 10000)
#     - With small L, the filter cannot overfit regardless of spike magnitude.
#     - As the spike value increases, the amplitude and time-shift are gradually
#       driven toward the target value at the spike frequency — but this improvement
#       comes at the cost of a progressively worse fit at all other frequencies.
#     - The optimizer must make a trade-off: improving the fit at the spike frequency
#       inevitably degrades it elsewhere (the MSE budget is redistributed).
#     - In contrast, with L = 200 the filter accommodated the spike without
#       severely degrading the fit at other frequencies, precisely because it had
#       sufficient degrees of freedom to do so — at the cost of overfitting.


#--------------------------------------------------------------------------------------
# Wrap-Up: Key Concepts and Takeaways
#--------------------------------------------------------------------------------------

# 1. DFA-MSE User Interface: L, weight_func, Lag, Gamma
#
#    The four inputs to the MDFA_mse() wrapper each play a distinct role:
#
#    Gamma — the filter target:
#      Specifies the desired frequency response of the output signal.
#      Supported target types include:
#        - Allpass:   classical one- or multi-step-ahead forecasting
#        - Lowpass:   trend extraction (nowcast, forecast, or backcast)
#        - Bandpass:  business-cycle or other frequency-band isolation
#        - Arbitrary: any user-defined shape matching specific research needs
#      The user is responsible for specifying a target that reflects their
#      analytical objectives.
#
#    weight_func — the spectral weighting function:
#      Modulates the frequency-specific quality of the DFA fit.
#      Interpretations and sources include:
#        - Parametric spectrum: derived from a fitted ARMA model
#        - Non-parametric spectrum: estimated via the discrete Fourier transform
#        - Tweaked spectrum: deliberately modified to enforce better fit at
#          specific frequencies of particular interest
#      The spectrum determines where the MSE budget is allocated: frequencies
#      with higher spectral weight receive a more accurate approximation of the target.
#
#    L — the filter length (degrees of freedom):
#      Controls the flexibility of the causal approximation.
#        - Small L/K: limited flexibility, reduced risk of overfitting, smoother output
#        - L = 2*K:   extreme overfitting — the target is matched perfectly at all
#                     discrete frequency ordinates, but the filter is unreliable
#                     for any practical application
#        - L > 2*K:   system is singular; numerical optimization fails
#      Choosing L appropriately is the simplest regularization tool available.
#
#    Lag — the timing offset of the target:
#      Determines whether the filter estimates a past, current, or future value
#      of the target output:
#        - Lag = 0:  Nowcast — estimate the current filtered value in real time
#        - Lag < 0:  Forecast — estimate a future value of the filtered output
#                    (note: this forecasts the target's output, not the raw series)
#        - Lag > 0:  Backcast — estimate a past value; used for historical revision
#                    (in state-space literature this is called 'smoothing',
#                     as opposed to 'filtering' which corresponds to nowcasting;
#                     the Kalman filter is the nowcast, the Kalman smoother is the backcast)


# 2. Turning-Point Detection
#
#    Turning points in a price or economic series can be defined in two ways:
#
#    Definition 1 — Local extrema:
#      Peaks and troughs of a smooth trend component, extracted via an ideal lowpass.
#
#    Definition 2 — Zero-crossings (preferred):
#      Zero-crossings of an ideal lowpass applied to the first differences of the data.
#      This definition tends to be more robust and analytically tractable.
#
#    The parameter 'periodicity' allows the user to define the trend according to
#    their investment horizon or research objective:
#      - Large periodicity: captures long-term cycles (suitable for pension funds,
#        macro analysis, or long-horizon positioning)
#      - Small periodicity: captures short-term cycles (suitable for hedge funds,
#        tactical allocation, or high-frequency signal generation)


# 3. The DFA-MSE Criterion: Amplitude vs. Time-Shift Trade-off
#
#    Given filter length L, the MSE criterion seeks the best simultaneous fit of:
#      - The amplitude function: governs noise leakage (false or unreliable signals)
#      - The time-shift function: governs filter delay (lateness of signal detection)
#
#    Ideally, both noise leakage and delay would be zero. In practice:
#      - Perfect noise suppression (amplitude = target) generally implies a larger delay
#      - Minimal delay generally implies greater noise leakage
#      - These requirements are in fundamental tension and cannot be met simultaneously
#     
#
#    The MSE criterion resolves this tension by finding a balanced compromise.
#    Unlike classical time series approaches, DFA allows the user to customize
#    this balance explicitly — emphasizing noise suppression or timeliness depending
#    on the application. This customization is covered in detail in later tutorials
#    (see: McElroy & Wildi, trilemma paper).


# 4. Overfitting in DFA: When to Be Concerned
#
#    Overfitting is most likely to occur when:
#      1. The spectrum (weight_func) is noisy or contains sharp spikes
#      2. The target (Gamma) is irregular or non-smooth
#      3. The filter length L is large relative to the frequency grid size K
#
#    Symptoms of overfitting:
#      - Many narrow peaks and dips in the amplitude or time-shift function
#      - Near-perfect fit at specific frequencies driven by artificial spectral features
#      - Poor generalization to actual time series data
#
#    Mitigation strategies:
#      - Reduce L to constrain degrees of freedom (simplest approach)
#      - Apply regularization penalties to the DFA criterion (covered in later tutorials)


#--------------------------------------------------------------------------------------
# Possible Extensions to thi tutorial
#
#   - Specify the filter target directly in the time domain
#   - Incorporate explicit filter coefficient constraints (e.g., sum-to-one,
#     zero at specific frequencies, or monotonicity constraints)
#--------------------------------------------------------------------------------------



