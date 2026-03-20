# =============================================================================
# Tutorial 4
# =============================================================================
# Purpose:
#   1. Illustrate multivariate filter designs using an artificial bivariate
#      leading-indicator setup.
#   2. Compare MSE performances (in-sample, out-of-sample, frequency-domain)
#      between univariate DFA and bivariate MDFA.
#   3. Explore overfitting risks inherent to multivariate frameworks.
#   4. Assess the value of the bivariate approach across multiple scenarios:
#      data-generating process, indicator noise level, filter length, sample size.
#   5. Establish the equivalence between the frequency-domain optimization
#      criterion (MSE) and the classic time-domain in-sample MSE.
#
# Key functions and parameters:
#   - MDFA_mse: main wrapper used throughout (same as previous tutorials).
#   - Filter length L: critical parameter — a bivariate filter requires 2*L
#     coefficients, making it significantly more prone to overfitting than
#     its univariate counterpart for the same L.
# =============================================================================

rm(list = ls())

library(xts)
# install.packages("devtools")  # Uncomment if devtools is not yet installed
library(devtools)

# Install and load the MDFA package from GitHub
devtools::install_github("wiaidp/MDFA")
library(MDFA)


# -----------------------------------------------------------------------------
# Brief overview of available wrapper functions
# -----------------------------------------------------------------------------
head(MDFA_mse)              # MSE-optimal filter (unconstrained)
head(MDFA_mse_constraint)   # MSE-optimal filter with constraints
head(MDFA_cust)             # Customized filter criterion (unconstrained)
head(MDFA_cust_constraint)  # Customized filter criterion with constraints
head(MDFA_reg)              # Regularized filter (unconstrained)
head(MDFA_reg_constraint)   # Regularized filter with constraints
head(mdfa_analytic)         # Core analytic estimation function


# -----------------------------------------------------------------------------
# Source shared utility functions
# -----------------------------------------------------------------------------
source("Common functions/plot_func.r")          # Plotting utilities
source("Common functions/arma_spectrum.r")       # ARMA spectrum computation
source("Common functions/ideal_filter.r")        # Ideal lowpass filter
source("Common functions/mdfa_trade_func.r")     # Trading signal utilities
source("Common functions/play_with_bivariate.r") # Bivariate experiment helpers


# =============================================================================
# Example 1.
# Experiment Design: Bivariate Leading-Indicator Setup
# =============================================================================
# Motivation:
#   Multivariate designs are particularly powerful when a leading indicator
#   is available alongside the target variable.
#
# Setup:
#   - The target series is a simulated AR(1) process.
#   - The leading indicator is a noisy, one-step-ahead version of the target.
#   - We compare bivariate MDFA vs. univariate DFA in terms of:
#       * In-sample time-domain MSE
#       * Out-of-sample time-domain MSE
#       * In-sample frequency-domain MSE (the MDFA criterion value)
#
# Analysis structure:
#   - Three 'toggle breakpoints' mark key diagnostic outputs in the script.
#   - Sensitivity is examined over: sample size, filter length L,
#     data-generating process, and indicator noise level.
# =============================================================================


# -----------------------------------------------------------------------------
# Sample size configuration
# -----------------------------------------------------------------------------
# Long total sample ensures reliable out-of-sample performance estimates.
# Extra length also needed for the non-causal ideal lowpass filter,
# which requires future observations for initialization.
lenh <- 10000

# In-sample length used for filter estimation.
# As the ratio L/len increases, overfitting worsens.
# The bivariate filter (with 2*L coefficients) is more susceptible
# to overfitting than the univariate filter (with L coefficients).
len <- 300


# -----------------------------------------------------------------------------
# Data-generating process: AR(1) with coefficient a1
# -----------------------------------------------------------------------------
# a1 > 0: positive autocorrelation (typical in economic data)
# a1 = 0: white noise
# a1 < 0: negative autocorrelation (can arise from overdifferencing)
a1 <- 0.9

# Simulate the target AR(1) series (lenh+1 points; extra point for lead construction)
set.seed(1)
x_long <- arima.sim(list(ar = a1), n = lenh + 1)


# -----------------------------------------------------------------------------
# Leading indicator construction
# -----------------------------------------------------------------------------
# The leading indicator is the target series plus scaled idiosyncratic noise,
# shifted one time step ahead.
# A larger scale_idiosyncratic value makes the indicator noisier (less informative).
scale_idiosyncratic <- 0.4

if (abs(scale_idiosyncratic) == 0)
  print("Design is not uniquely specified: both explanatory series are identical (up to a one-step shift).")

set.seed(2)
eps <- rnorm(lenh + 1)
indicator <- x_long + sqrt(var(x_long)) * scale_idiosyncratic * eps

# Assemble data matrix:
#   Column 1 (target):           clean AR(1) series x_long
#   Column 2 (x):                same AR(1) series (used as the main explanatory variable)
#   Column 3 (leading indicator): noisy one-step-ahead version of x_long
data_matrix <- na.exclude(cbind(
  x_long[1:lenh],
  x_long[1:lenh],
  c(indicator[2:(lenh + 1)])
))
dimnames(data_matrix)[[2]]<- c("target", "x", "leading indicator")
head(data_matrix)


# -----------------------------------------------------------------------------
# Ideal lowpass filter configuration
# -----------------------------------------------------------------------------
# M is the half-length of the symmetric ideal lowpass filter.
# Effective filter length = 2*M - 1.
# The first M observations are consumed during filter initialization
# and must be excluded from in-sample comparisons.
M <- 100

# Extract the in-sample window (length = len + 1, offset by M to align
# time-domain and frequency-domain MSE computations)
data_matrix_in_sample <- data_matrix[M:(M + len), ]

# Verify that column 3 leads column 1/2 (visually apparent in the tail)
tail(round(data_matrix_in_sample, 4))


# -----------------------------------------------------------------------------
# Target filter specification
# -----------------------------------------------------------------------------
periodicity <- 6                # Target cycle length in time steps
cutoff <- pi / periodicity      # Corresponding frequency cutoff

# Apply the ideal lowpass filter to the full long series
y <- ideal_filter_func(periodicity, M, x_long)$y[1:lenh]


# -----------------------------------------------------------------------------
# Spectral weight matrix (DFT of in-sample data)
# -----------------------------------------------------------------------------
# spec_comp computes the multivariate DFT for all columns of data_matrix_in_sample.
# Important: the DFT is complex-valued and must NOT be reduced to its modulus,
# as the complex argument (phase) encodes the relative lead/lag between
# the explanatory series and the target — this phase information is essential
# for the bivariate MDFA to exploit the leading indicator.
weight_func_bivariate <- spec_comp(
  nrow(data_matrix_in_sample),
  data_matrix_in_sample,
  0
)$weight_func

# Number of frequency grid points (K+1 total, including 0)
K <- nrow(weight_func_bivariate) - 1

# Target frequency response: ideal lowpass (1 below cutoff, 0 above)
Gamma <- (0:K) <= K * cutoff / pi + 1.e-9


# -----------------------------------------------------------------------------
# Filter estimation parameters
# -----------------------------------------------------------------------------
Lag <- 0        # 0 = nowcast; Lag > 0 = backcast; Lag < 0 = forecast
L   <- 2 * periodicity  # Filter length (bivariate will use 2*L coefficients total)


# -----------------------------------------------------------------------------
# Bivariate MDFA filter estimation
# -----------------------------------------------------------------------------
mdfa_obj_bivariate <- MDFA_mse(L, weight_func_bivariate, Lag, Gamma)$mdfa_obj

# Extract and label filter coefficients (L rows x 2 columns: x and leading indicator)
b__bivariate <- mdfa_obj_bivariate$b
dimnames(b__bivariate)[[2]] <- c("x", "leading indicator")
dimnames(b__bivariate)[[1]] <- paste("Lag ", 0:(L - 1), sep = "")


# =============================================================================
# Toggle Breakpoint 1: Inspect bivariate filter coefficients
# =============================================================================
# Examine the estimated weights for each explanatory series across lags.
# The leading indicator coefficients should reflect its one-step anticipation
# of the target, as well as attenuation due to its noise component.
head(b__bivariate)


# Apply bivariate filter to generate filtered output (full long sample)
yhat_bivariate_leading_indicator <- filt_func(
  data_matrix[, 2:ncol(data_matrix)],
  b__bivariate
)$yhat


# -----------------------------------------------------------------------------
# Univariate DFA filter estimation (benchmark)
# -----------------------------------------------------------------------------
# Uses only the first two columns of the spectral weight matrix (target + x).
# This serves as the univariate benchmark from previous tutorials.
weight_func_univariate <- weight_func_bivariate[, 1:2]
mdfa_obj_univariate    <- MDFA_mse(L, weight_func_univariate, Lag, Gamma)$mdfa_obj
b_univariate           <- mdfa_obj_univariate$b

# Apply univariate filter to generate filtered output (full long sample)
yhat_univariate <- filt_func(as.matrix(data_matrix[, 2]), b_univariate)$yhat


# -----------------------------------------------------------------------------
# MSE performance comparison
# -----------------------------------------------------------------------------
y_target_leading_indicator <- y

# In-sample window: observations M to M+len (aligned with filter estimation data)
anf <- M
enf <- M + len
mse_in <- round(c(
  mean(na.exclude((yhat_bivariate_leading_indicator - y_target_leading_indicator)[anf:enf])^2),
  mean(na.exclude((yhat_univariate - y_target_leading_indicator)[anf:enf])^2)
), 3)

# Out-of-sample window: all observations after the in-sample period
anf <- M + 1 + len
enf <- lenh
mse_out <- round(c(
  mean(na.exclude((yhat_bivariate_leading_indicator - y_target_leading_indicator)[anf:enf])^2),
  mean(na.exclude((yhat_univariate - y_target_leading_indicator)[anf:enf])^2)
), 3)

# Assemble performance summary table
perf_mse <- rbind(
  mse_out,
  mse_in,
  c(round(mdfa_obj_bivariate$MS_error, 3), round(mdfa_obj_univariate$MS_error, 3))
)
colnames(perf_mse) <- c("bivariate MDFA", "DFA")
rownames(perf_mse) <- c(
  "Out-of-sample MSE (time domain)",
  "In-sample MSE (time domain)",
  "In-sample criterion value (frequency domain)"
)


# =============================================================================
# Toggle Breakpoint 2: Analyze MSE results
# =============================================================================
# Expected findings:
#   1. Bivariate MDFA achieves lower out-of-sample MSE than univariate DFA,
#      confirming the value of the leading indicator.
#   2. Bivariate MDFA shows a larger in-sample vs. out-of-sample MSE gap,
#      reflecting greater overfitting risk (2*L coefficients vs. L).
#   3. The frequency-domain criterion (last row) closely replicates the in-sample
#      time-domain MSE (2nd-row) up to negligible finite-sample discrepancy,
#      confirming theoretical equivalence between the two perspectives.
round(perf_mse, 3)


# -----------------------------------------------------------------------------
# Plot: full-sample comparison of filtered series
# -----------------------------------------------------------------------------
par(mfrow = c(1, 1))
mplot <- mplot_all <- cbind(yhat_univariate, y, yhat_bivariate_leading_indicator)
ymin <- min(mplot, na.rm = TRUE)
ymax <- max(mplot, na.rm = TRUE)

ts.plot(
  mplot[, 1],
  main = paste("Out-of-sample MSE  |  MDFA:", round(perf_mse[1], 3),
               "  DFA:", round(perf_mse[2], 3)),
  ylab = "",
  col = "blue",
  ylim = c(ymin, ymax)
)
lines(mplot[, 2], col = "red")
lines(mplot[, 3], col = "green")
mtext("DFA",    line = -2, at = len / 2, col = "blue")
mtext("Target", line = -1, at = len / 2, col = "red")
mtext("MDFA",   line = -3, at = len / 2, col = "green")


# =============================================================================
# Toggle Breakpoint 3: Zoom in to inspect filtered series visually
# =============================================================================
# Zooming in reveals that the MDFA output is marginally smoother and slightly left-shifted (faster)
# relative to DFA — this temporal lead, due to the leading indicator, is the primary driver of the
# bivariate design's efficiency gains.
anf <- 300
enf <- 400
mplot <- mplot_all[anf:enf, ]
ymin  <- min(mplot, na.rm = TRUE)
ymax  <- max(mplot, na.rm = TRUE)

ts.plot(
  mplot[, 1],
  main = paste("Out-of-sample MSE  |  MDFA:", round(perf_mse[1], 3),
               "  DFA:", round(perf_mse[2], 3)),
  ylab = "",
  col = "blue",
  ylim = c(ymin, ymax)
)
lines(mplot[, 2], col = "red")
lines(mplot[, 3], col = "green")
mtext("DFA",    line = -2, at = (enf - anf) / 2, col = "blue")
mtext("Target", line = -1, at = (enf - anf) / 2, col = "red")
mtext("MDFA",   line = -3, at = (enf - anf) / 2, col = "green")



# =============================================================================
# Example 2: Baseline bivariate design — strong autocorrelation, moderate noise
# =============================================================================
# This example replicates the lengthy setup above using the wrapper function
# play_bivariate_func(), which encapsulates all steps for convenience.
# We explore the effect of varying:
#   - a1:                   AR(1) coefficient (controls autocorrelation strength)
#   - scale_idiosyncratic:  noise level of the leading indicator
#   - len:                  in-sample span used for filter estimation
#   - L:                    filter length (the bivariate filter requires 2*L
#                           coefficients, doubling the degrees of freedom
#                           relative to the univariate benchmark)

a1                   <- 0.9   # Strong positive autocorrelation
scale_idiosyncratic  <- 0.4   # Moderate indicator noise
len                  <- 300   # Reasonably long in-sample span
L                    <- 2 * periodicity  # L=12: sufficient to attenuate stopband
# components (periodicity = 6)

play_obj <- play_bivariate_func(a1, scale_idiosyncratic, len, L)

play_obj$b__bivariate  # Inspect estimated filter coefficients (Toggle Point 1)
play_obj$perf_mse      # Inspect MSE performance table   (Toggle Points 2 & 3)

# -----------------------------------------------------------------------------
# Toggle Point 1 — Filter coefficients:
#   - Coefficients on the first series (x) resemble a standard one-sided
#     lowpass filter with weights decaying smoothly across lags.
#   - Coefficients on the second series (leading indicator) concentrate most
#     weight on the most recent observation (lag 0), which is the data point
#     that looks one step into the future. Older lags of the indicator are
#     essentially redundant, since they are noisier than x itself.
#   - Because the indicator is noisy, the weight at lag 0 is notable but
#     not overwhelmingly dominant.
#
# Toggle Point 2 — MSE performances:
#   - Row 2 (in-sample time-domain MSE) and Row 3 (in-sample frequency-domain
#     criterion value) are close, confirming internal consistency. A large
#     discrepancy could signal overfitting driven by an insufficiently dense
#     frequency grid (K/L too small, leading to ill-conditioning).
#   - Out-of-sample MSEs (Row 1) are higher than in-sample values for both
#     filters, reflecting the expected degree of overfitting.
#   - The bivariate filter (Column 1) outperforms the univariate DFA
#     (Column 2) both in-sample and out-of-sample. The in-sample advantage
#     is expected by construction; the out-of-sample advantage is less
#     obvious, given that the bivariate design uses twice as many
#     coefficients and is therefore more prone to overfitting.
#
# Toggle Point 3 — Visual inspection of filter outputs:
#   - The bivariate filter (green) output generally anticipates turning points
#     when compared to DFA (blue),
#     consistent with its exploitation of the leading indicator.
#   - Both one-sided filter outputs show a slight delay relative to the
#     ideal lowpass target — though this delay is modest due to the strong
#     autocorrelation (a1 = 0.9), which makes the smoothing task easier.
# -----------------------------------------------------------------------------


# =============================================================================
# Example 3: Near-white-noise process — harder filtering task
# =============================================================================
# Same setup as Example 2, but with a1 close to zero (near white noise),
# approximating the autocorrelation structure of log-returns of EURUSD.
# This makes the filtering task substantially more difficult.

a1                   <- 0.08  # Near white noise (minimal autocorrelation)
scale_idiosyncratic  <- 0.4   # Same moderate indicator noise as Example 1
len                  <- 300   # Same in-sample span as Example 1
L                    <- 2 * periodicity  # Same filter length as Example 1

play_obj <- play_bivariate_func(a1, scale_idiosyncratic, len, L)

play_obj$b__bivariate
play_obj$perf_mse

# -----------------------------------------------------------------------------
# Toggle Point 1 — Filter coefficients:
#   - Coefficients on the first series decay more slowly than in Example 1,
#     reflecting heavier smoothing required to suppress noise in a
#     near-white-noise process.
#   - Coefficients on the leading indicator again concentrate most weight
#     at lag 0 (the forward-looking observation), consistent with Example 1.
#     The remaining weights are small due to the indicator's noise level.
#
# Toggle Point 2 — MSE performances:
#   - In-sample time-domain MSE and frequency-domain criterion value remain
#     tightly matched, confirming no ill-conditioning issues.
#   - Relative overfitting (gap between in-sample and out-of-sample MSE) is
#     somewhat smaller than in Example 1: heavier smoothing partially offsets
#     overfitting by reducing sensitivity to in-sample noise.
#   - Efficiency gains from the bivariate design are comparable to Example 1.
#
# Toggle Point 3 — Visual inspection of filter outputs:
#   - Filtered outputs are less smooth than in Example 1, as expected given
#     the noisier input; both outputs show slightly more delay due to heavier
#     smoothing requirements.
#   - The bivariate filter output (green) leads the univariate output (blue) by approximately
#     one time step, consistent with the one-period anticipation built into
#     the leading indicator.
# -----------------------------------------------------------------------------


# =============================================================================
# Example 4: Uninformative leading indicator — very high noise level
# =============================================================================
# Same as Example 3, but with an extremely noisy leading indicator
# (scale_idiosyncratic = 10). At this noise level the indicator carries
# virtually no useful information, so we expect the bivariate MDFA to
# offer no advantage over — or even underperform — the univariate DFA,
# due to the additional overfitting risk from the extra L coefficients.

a1                   <- 0.08  # Near white noise
scale_idiosyncratic  <- 10    # Extremely noisy leading indicator
len                  <- 300   # Same in-sample span
L                    <- 2 * periodicity  # Same filter length

play_obj <- play_bivariate_func(a1, scale_idiosyncratic, len, L)

play_obj$b__bivariate
play_obj$perf_mse

# -----------------------------------------------------------------------------
# Toggle Point 1 — Filter coefficients:
#   - Coefficients on the first series are similar to Example 2, as the
#     target data-generating process is unchanged.
#   - Coefficients on the second series (leading indicator) are close to
#     zero across all lags: the optimizer correctly identifies that the
#     indicator is pure noise and assigns it negligible weight.
#
# Toggle Point 2 — MSE performances:
#   - In-sample time-domain and frequency-domain MSEs remain closely matched,
#     confirming no numerical issues despite the uninformative indicator.
#   - Out-of-sample overfitting is similar to Example 2 in magnitude.
#   - The bivariate filter marginally outperforms the univariate filter
#     in-sample (it has more degrees of freedom), but is marginally
#     outperformed out-of-sample due to overfitting. The differences are negligible.
#   - Importantly, the out-of-sample loss from including a completely
#     uninformative indicator is small, demonstrating that the bivariate
#     framework is reasonably robust to irrelevant explanatory variables.
#
# Toggle Point 3 — Visual inspection of filter outputs:
#   - The two filter outputs nearly overlap, as expected when the indicator
#     contributes no useful information and its coefficients are near zero.
# -----------------------------------------------------------------------------


# =============================================================================
# Example 5: Short in-sample span — examining overfitting under data scarcity
# =============================================================================
# Same as Example 4 (near-white-noise, moderate indicator noise), but with a
# drastically reduced in-sample length (len = 100) and moderate indicator noise. This stresses the
# bivariate design: fitting 2*L = 24 coefficients on only 100 observations
# corresponds to solving a system with 49 complex-valued frequency equations
# plus 2 real-valued boundary equations (at frequencies 0 and pi),
# yielding exactly 100 frequency-domain observations — a very tight fit.

a1                   <- 0.08  # Near white noise
scale_idiosyncratic  <- 0.4   # Moderate indicator noise
len                  <- 100   # Short in-sample span (data scarcity scenario)
L                    <- 2 * periodicity  # Same filter length (but now 24
# coefficients fitted to 100 obs.)

play_obj <- play_bivariate_func(a1, scale_idiosyncratic, len, L)

play_obj$b__bivariate
play_obj$perf_mse

# -----------------------------------------------------------------------------
# Toggle Point 1 — Filter coefficients:
#   - Remarkably similar to Example 2, suggesting that the estimation is
#     stable even under the reduced sample size.
#
# Toggle Point 2 — MSE performances:
#   - In-sample time-domain MSE (Row 2) and frequency-domain criterion
#     value (Row 3) now diverge noticeably:
#       * The criterion values (last row) are artificially small because both filters
#         overfit the coarse frequency grid (only ~50 frequency points
#         available for 12 or 24 unknowns).
#       * The bivariate filter overfits more severely, as expected, given
#         its larger parameter count.
#   - Despite the short sample, out-of-sample MSEs are surprisingly similar
#     to those in Example 2, indicating that the filter generalizes well
#     even when estimated on limited data.
#   - Out-of-sample efficiency gains of the bivariate design are also
#     comparable to Example 2 — a notably encouraging result given the
#     tight estimation conditions.
#
# Toggle Point 3 — Visual inspection of filter outputs:
#   - The bivariate filter output remains faster (left-shifted) than the
#     univariate output out-of-sample — a reassuring result, given that
#     24 coefficients were estimated on only 100 observations.
# -----------------------------------------------------------------------------

# =============================================================================
# Example 6: Extreme overfitting — bivariate filter with 2*L/len = 0.5
# =============================================================================
# Same setup as Example 5 (near-white-noise, moderate indicator noise), but
# with an aggressively large filter length L = 25. The bivariate design now
# estimates 2*L = 50 coefficients on only 100 in-sample observations —
# a 1:2 parameter-to-observation ratio that constitutes a severe overfitting
# stress test. The key question is whether out-of-sample performance degrades
# catastrophically, or whether the frequency-domain estimation framework
# provides some implicit regularization.

a1                  <- 0.08  # Near white noise
scale_idiosyncratic <- 0.4   # Moderate indicator noise
len                 <- 100   # Short in-sample span (~one quarter of daily data);
# relevant for non-stationary settings where the
# data-generating process changes rapidly
L                   <- 25    # Aggressively long filter: 2*L = 50 coefficients
# estimated on 100 observations

play_obj <- play_bivariate_func(a1, scale_idiosyncratic, len, L)

play_obj$b__bivariate
play_obj$perf_mse
yhat_mat_overfitting <- play_obj$mplot_all  # Store outputs for later comparison

# -----------------------------------------------------------------------------
# Toggle Point 1 — Filter coefficients:
#   - Despite the extreme parameter-to-observation ratio, the coefficients
#     remain interpretable: the leading indicator's largest weight is still
#     correctly assigned to lag 0 (the forward-looking observation),
#     consistent with prior examples.
#
# Toggle Point 2 — MSE performances:
#   - Remarkably, the bivariate filter still marginally outperforms the
#     univariate DFA out-of-sample, suggesting that the frequency-domain
#     estimation framework provides a degree of implicit regularization
#     even under extreme overfitting conditions.
#
# Toggle Point 3 — Visual inspection of filter outputs:
#   - The bivariate filter output remains faster (left-shifted) relative
#     to the univariate output, even when 50 coefficients are estimated
#     on only 100 observations — a surprisingly robust result.
# -----------------------------------------------------------------------------


# =============================================================================
# Example 7: Asymptotic benchmark — large sample, moderate filter length
# =============================================================================
# Contrasting with the short-sample examples above, we now use a very long
# in-sample span (len = 3000, equivalent to more than 10 years of daily data)
# with a moderate filter length. This approximates the asymptotic regime,
# where overfitting is negligible and all MSE measures — in-sample and
# out-of-sample, time-domain and frequency-domain — should converge to
# the same value for each filter.

a1                  <- 0.08         # Near white noise
scale_idiosyncratic <- 0.4          # Moderate indicator noise
len                 <- 3000         # Very long in-sample span (asymptotic regime)
L                   <- 2*periodicity # Moderate filter length (L = 12)

play_obj <- play_bivariate_func(a1, scale_idiosyncratic, len, L)

play_obj$b__bivariate
play_obj$perf_mse

# -----------------------------------------------------------------------------
# Toggle Point 1 — Filter coefficients:
#   - Coefficients are consistent with earlier examples and align with
#     theoretical expectations: well-shaped lowpass for x, concentrated
#     weight at lag 0 for the leading indicator.
#
# Toggle Point 2 — MSE performances:
#   - As expected in the asymptotic regime: all four MSE values
#     (in-sample time-domain, in-sample frequency-domain, out-of-sample
#     time-domain for both filters) are tightly matched, confirming that
#     overfitting is negligible at this sample size.
#
# Toggle Point 3 — Visual inspection of filter outputs:
#   - Outputs are smooth and well-behaved, consistent with a well-identified
#     filter estimated on abundant data.
# -----------------------------------------------------------------------------


# =============================================================================
# Example 8: Asymptotic regime with large L — diminishing returns to filter length
# =============================================================================
# Same as Example 7 (large sample, near-white-noise) but with a much longer
# filter length (L = 50). The purpose is to demonstrate that, asymptotically,
# increasing L beyond 2*periodicity yields negligible improvement in
# out-of-sample performance. This validates the practical rule of thumb
# L = 2*periodicity as a sensible default, particularly when data is scarce
# or the data-generating process is non-stationary.

a1                  <- 0.08  # Near white noise
scale_idiosyncratic <- 0.4   # Moderate indicator noise
len                 <- 3000  # Very long in-sample span (asymptotic regime)
L                   <- 50    # Large filter length to probe asymptotic saturation

play_obj <- play_bivariate_func(a1, scale_idiosyncratic, len, L)

play_obj$b__bivariate
play_obj$perf_mse
yhat_mat_asymptotic <- play_obj$mplot_all  # Store outputs for final comparison


# =============================================================================
# Final comparison: massively overfitted MDFA vs. asymptotic MDFA and DFA
# =============================================================================
# We overlay three filter outputs over the last 100 (out-of-sample) observations of the
# long sample to visually compare:
#   - Asymptotic MDFA (blue):  bivariate filter, len=3000, L=50 — best-case benchmark
#   - Asymptotic DFA (green):  univariate filter, len=3000, L=50 — best-case benchmark
#   - Overfitted MDFA (red):   bivariate filter, len=100,  L=25 — stress-test case
#
# Key observation:
#   The massively overfitted bivariate MDFA (red) preserves the desirable
#   temporal lead (left-shift) relative to DFA, but exhibits inflated
#   amplitude swings (exaggerated peaks and troughs) — a characteristic
#   signature of overfitting in the time domain.

anf  <- nrow(yhat_mat_asymptotic) - 100
enf  <- nrow(yhat_mat_asymptotic)
mplot <- cbind(
  yhat_mat_asymptotic[, 3],   # Asymptotic MDFA output
  yhat_mat_overfitting[, 3],  # Massively overfitted MDFA output
  yhat_mat_asymptotic[, 1]    # Asymptotic DFA output
)[anf:enf, ]

ts.plot(
  mplot[, 1],
  ylim = c(min(mplot), max(mplot)),
  main = paste("Asymptotic MDFA (blue) and DFA (green) with large L",
               "vs. massively overfitted bivariate MDFA (red)"),
  col  = "blue"
)
lines(mplot[, 2], col = "red")
lines(mplot[, 3], col = "green")
mtext("Asymptotic MDFA (bivariate)",     line = -1, col = "blue")
mtext("Asymptotic DFA (univariate)",     line = -2, col = "green")
mtext("Massively overfitted MDFA",       line = -3, col = "red")
abline(h = 0)


# =============================================================================
# Key Takeaways
# =============================================================================
# - Leading Indicators:
#     Leading indicators can be exploited in multivariate designs to achieve
#     superior out-of-sample performance.
#
# - Noise Level:
#     The degree of outperformance is inversely related to the noise level
#     in the leading indicator: less noise yields greater gains.
#
# - Interpretability:
#     The bivariate design produces straightforwardly interpretable outcomes;
#     filter coefficients assigned to the leading series are intuitively
#     appealing and easy to diagnose.
#
# - Overfitting:
#     Multivariate designs are inherently more prone to overfitting than
#     simpler univariate alternatives. Severe overfitting manifests in
#     two key ways:
#     a. A discrepancy between in-sample frequency-domain MSE (the proper
#          optimization objective of MDFA) and in-sample time-domain MSE.
#     b. A pronounced discrepancy between in-sample frequency-domain MSE
#          (the optimization criterion) and out-of-sample time-domain MSE.
#
# - Internal DFA Regularization:
#     Despite strong overfitting, MDFA can still perform well due to its
#     internal regularization mechanisms. Explicit regularization features,
#     offering finer control over this behavior, will be introduced in
#     Tutorial 7.
#
# - Lead:
#     Across all analyzed examples — ranging from very long samples
#     (asymptotic true model) to very short samples with strong overfitting —
#     the bivariate design consistently retains a lead over the univariate
#     benchmark, unless the leading indicator is excessively noisy
#     (i.e., uninformative).
# =============================================================================
