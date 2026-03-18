# =============================================================================
# Tutorial 5
# =============================================================================
# Purposes:
#   I) Replicate one-sided and symmetric classic filter designs by (M)DFA, specifically:
#     1. Hodrick-Prescott (HP) lowpass filter
#     2. Christiano-Fitzgerald (CF) bandpass filter
#     3. Hamilton filter 
#   II) Address non-stationarity
#   III) Demonstrate the MDFA wrapper MDFA_mse_constraint()
#     a. This MDFA wrapper extends application of the MDFA to non-stationary time series with
#       unit roots at omega=0 (I(1) or I(2) processes)
#     b. Unit roots of the process are matched by imposing constraints to the causal (MDFA-) filter 
#     c. The constraints ensure a `perfect' tracking of the target at frequency zero 
#       -the unit roots are cancelled by imposing the constraints
#       -the filter error is stationary (finite variance): in- and out-of-sample
#         -An infinite variance error would contradict the MSE principle
# =============================================================================
# Approach
# =============================================================================
# a. Filter Class:
#     All (real-time, causal) filters are constrained MSE designs — no customization or explicit
#     regularization applied — but possibly with unit-root constraints imposed at frequency zero.
#     The constraints reflect unit root assumptions of the underlying (implicit) models.
#     Imposing the constraints ensures that the filter error variance remains finite when applied to 
#       non-stationary integrated processes
#
# b. Inputs Required for Replication:
#     To replicate these filters via MDFA, three ingredients must be supplied:
#       - weight_func (pseudo-spectrum)
#       - Gamma (target gain function)
#       - Constraints at frequency zero
#     These are handled by the MDFA_mse_constraint() wrapper.
#
#     Deriving weight_func:
#       The pseudo-spectrum is derived from the implicit model-based
#       representation of each filter:
#         - HP: ARIMA(0,2,2) — the MA(2) coefficients are uniquely determined
#               by lambda (the free HP tuning parameter); see McElroy's paper
#               (available in the literature folder on GitHub) for details.
#               Importantly, these MA(2) parameters do not depend on the data.
#         - CF: ARIMA(0,1,0) (random walk)
#         - Hamilton: AR(p) fitted at forecast horizon h (stationary model)
#
#     Deriving Gamma:
#       - HP: target is derived from the implicit model; see McElroy's paper.
#       - CF: target is the ideal bandpass as specified by the original authors.
#       -Hamilton: one-sided Hamilton filter (is already causal, but can be customized)
# c. Unit-Root Constraints at Frequency Zero:
#     Since implicit models of HP and CF are non-stationary, classic constraints must
#     be imposed at frequency zero:
#
#       - HP (lowpass):
#           i.   Amplitude at frequency zero must equal one
#                  (i <- T; weight_constraint <- 1)
#           ii.  Time-shift at frequency zero must vanish
#                  (i2 <- T; shift_constraint <- 0)
#           These two constraints are necessary to cancel the double unit root
#           of the implicit ARIMA(0,2,2) model.
#
#       - CF (bandpass):
#           i.   Amplitude at frequency zero must vanish
#                  (i <- T; weight_constraint <- 0)
#           ii.  No second-order constraint is required, since the implicit
#                model is a simple random walk (i2 <- F).
#       -Hamilton does not impose unit root constraints
#
# d. Replication Benchmark — mFilter Package:
#     MDFA output is compared against the R-package mFilter:
#
#       - HP: Replication can be made arbitrarily tight by increasing the
#               frequency-grid resolution in MDFA.
#
#       - CF: The mFilter package delivers incorrect one-sided CF filters.
#               In this case, replicability is verified by comparing MDFA
#               against the classical model-based (time-domain) solution.
#               As with HP, replication accuracy can be tightened arbitrarily
#               by selecting a larger K.
#       - Hamilton: trivial since the target filter is already one-sided (could be customized) 
# =============================================================================


rm(list = ls())

# =============================================================================
# Load Libraries
# =============================================================================
library(mFilter)       # Classic filter designs (HP, CF, etc.)
library(quantmod)      # Download macroeconomic data from FRED
library(tis)           # Time index and NBER recession shading utilities
# Install and load the MDFA package from GitHub
devtools::install_github("wiaidp/MDFA", force = T)
library(MDFA)

# =============================================================================
# Source Helper Functions
# =============================================================================
source(paste(getwd(), "/Common functions/hpFilt.r",   sep = ""))
source(paste(getwd(), "/Common functions/plot_func.r", sep = ""))

# =============================================================================
# Load or Download GDP Data (FRED: GDPC1 — Real US GDP, Quarterly)
# =============================================================================
# Set to TRUE to re-download the latest data from FRED;
# set to FALSE to use the locally saved dataset.
download_data <- F

if (download_data) {
  getSymbols('GDPC1', src = 'FRED')
  mydata <- GDPC1
} else {
  load("mydata")
}

# =============================================================================
# Prepare GDP Data
# =============================================================================
start_year <- 1960
end_date   <- format(Sys.time(), "%Y-%m-%d")
end_year   <- as.double(substr(end_date, 1, 4))
start_date <- paste(start_year, "-01-01", sep = "")

# Subset data to the selected date range
data_sample <- mydata[paste("/",    end_date,   sep = "")]
data_sample <- mydata[paste(start_date, "/",    sep = "")]

# Express GDP in log-levels (scaled by 100)
lgdp <- ts(100 * log(data_sample), start = start_year, frequency = 4)
nobs <- length(lgdp)

# HP smoothing parameter: lambda = 1600 is the conventional choice
# for quarterly data (Hodrick & Prescott, 1997)
lambda_hp <- 1600

# =============================================================================
# Exercise 1: Replication of the Hodrick-Prescott (HP) Filter
# =============================================================================

# -----------------------------------------------------------------------------
# 1.1 Derive MA Coefficients of the Implicit ARIMA(0,2,2) Model via hpFilt
#   Background on HP is provided in tutorial 2.0 of  the M-SSA tutorial (on a separate github repository) 
# -----------------------------------------------------------------------------
# The HP filter implicitly assumes an ARIMA(0,2,2) data-generating process.
# The MA(2) coefficients of this model are uniquely determined by lambda and
# are required to construct the pseudo-spectrum (weight_func) for MDFA.
# See McElroy (2008) in the literature folder on GitHub (specifically: `HP filter' paper in subfolder `Related topics') 
#   for full derivation.
# Note: hpFilt is a standalone utility — it is not part of the MDFA package.
head(hpFilt)

x   <- lgdp
len <- L_hp <- length(x)
q   <- 1 / lambda_hp

# Compute MA coefficients of the implicit ARIMA(0,2,2) model
hp_filt_obj <- hpFilt(q, L_hp)

tail(hpFilt, 2)
hp_filt_obj$ma_model

# Extract the two MA coefficients (indices 2 and 3; index 1 is the intercept)
# These depend solely on lambda, not on the data
ma_coeff <- hp_filt_obj$ma_model[2:3]
ma_coeff
# -The implicit model is ARIMA(0,2,2): after second order differences, 
#   the process is MA(2) with the above parameters uniquely determinded by lambda
# -This model is used for deriving the spectrum in MDFA, see below.
# -----------------------------------------------------------------------------
# 1.2 Apply HP Filter to GDP Using mFilter
# -----------------------------------------------------------------------------
# The mFilter package is used here as the replication benchmark.
# Below, the HP filter will be independently replicated via MDFA.

# Apply HP filter
x_hp <- hpfilter(x, type = "lambda", freq = lambda_hp)

# mFilter returns the gap filter matrix (fmatrix); transform to trend filter
# by subtracting from the identity matrix
parm <- diag(rep(1, len)) - x_hp$fmatrix

# --- Plot: Filter Coefficients and GDP Series ---
par(mfrow = c(2, 1))
title_more <- NA
insamp     <- 1.e+99

# Symmetric (midpoint) vs. real-time (endpoint) HP filter coefficients
mplot      <- cbind(parm[, len / 2], parm[, 1])
plot_title <- "HP lambda=1600: Symmetric Filter (red) vs. Real-Time Filter (blue)"
axis_d     <- 1:len - 1
colo       <- c("red", "blue")
mplot_func(mplot, axis_d, plot_title, title_more, insamp, colo)

# Log GDP vs. HP trend
mplot      <- cbind(rep(NA, len), rep(NA, len), rep(NA, len), x, x_hp$trend)
plot_title <- "Log US GDP (blue) vs. HP Trend (red)"
plot(mplot[, 4], col = "blue", xlab = "", ylab = "", main = plot_title)
nberShade()
lines(mplot[, 5], col = "red")

# -----------------------------------------------------------------------------
# 1.3 Replicate HP Filter via MDFA
# -----------------------------------------------------------------------------
# Two inputs are required:
#   - weight_func: pseudo-spectrum of the implicit ARIMA(0,2,2) model
#   - Gamma:       target gain function of the symmetric HP trend filter

# Frequency-grid resolution (higher K yields tighter replication)
K <- 1200

# --- 1.3.1 Compute Pseudo-Spectral Density ---
# Based on the ARIMA(0,2,2) model underlying the HP filter
# (see McElroy (2008) or Maravall & Kaiser, p. 179).
# For lambda = 1600: MA coefficients are approximately -1.77709 and 0.79944.
# Note: MDFA takes the square-root of the spectrum (i.e., the DFT modulus).
weight_func_h <- abs(
  (1 + ma_coeff[1] * exp(-1.i * (0:K) * pi / K) +
     ma_coeff[2] * exp(-1.i * 2 * (0:K) * pi / K)) /
    (1 - exp(-1.i * (0:K) * pi / K))^2
)

# Both the target and the explanatory series use the same spectrum
# (univariate filter: target = explanatory variable)
weight_func <- cbind(weight_func_h, weight_func_h)

# --- 1.3.2 Compute Target Gamma: Symmetric HP Trend Filter ---
#  See McElroy (2008) for derivation.
Gamma <- 0:K
for (k in 0:K) {
  omegak   <- k * pi / K
  Gamma[k + 1] <- (1 / lambda_hp) /
    (1 / lambda_hp + abs(1 - exp(1.i * omegak))^4)
}

# --- Plot: Target Gamma and Log Pseudo-Spectrum ---
par(mfrow = c(2, 1))
colo   <- c("blue", "red")
insamp <- 1.e+99

# Frequency axis labels (0, pi/6, 2pi/6, ..., pi)
freq_axe <- rep(NA, K + 1)
freq_axe[1] <- 0
freq_axe[1 + (1:6) * K / 6] <- c(paste(c("", 2:5), "pi/6", sep = ""), "pi")

# Target gain function
mplot      <- as.matrix(Gamma)
plot_title <- "Target Gamma: HP Lowpass (lambda = 1600)"
mplot_func(mplot, freq_axe, plot_title, title_more, insamp, colo)

# Log pseudo-spectrum (weight_func is the square-root, so square it for the spectrum)
mplot      <- as.matrix(c(NA, log(weight_func[2:(K + 1), 1]^2)))
plot_title <- "Log Pseudo-Spectrum: Implicit ARIMA(0,2,2) Model (lambda = 1600)"
mplot_func(mplot, freq_axe, plot_title, title_more, insamp, colo)


# -----------------------------------------------------------------------------
# 1.3.3 Set Up MDFA: Replicating the HP Filter via MDFA_mse_constraint
# -----------------------------------------------------------------------------
# The MSE wrapper MDFA_mse_constraint is used here with the following setup:
#
# Constraints at frequency zero (required by the implicit ARIMA(0,2,2) model):
#   i1 <- T, weight_constraint <- 1 : filter amplitude equals one at freq. zero
#   i2 <- T, shift_constraint  <- 0 : filter time-shift vanishes at freq. zero
#
# Handling the spectral singularity at frequency zero:
#   The pseudo-spectrum of the ARIMA(0,2,2) model diverges at frequency zero
#   (due to the double unit root), which would cause numerical issues.
#   This singularity is removed by setting weight_func[1, ] <- 0 (an arbitrary
#   value). The choice is inconsequential because the constraints i1 and i2
#   fully determine filter behavior at frequency zero — any other finite value
#   would yield identical results.
#
# Additional settings:
#   Lag = 0  : nowcast (Lag < 0 for forecasts; Lag > 0 for backcasts)
#   cutoff   : frequency at which Gamma drops below 0.5; not relevant in MSE
#              designs (only matters for customization), but must be supplied
# -----------------------------------------------------------------------------

weight_func_hp <- weight_func

# Set spectral value at frequency zero to 0 to avoid numerical singularity
# (the double unit root makes the true spectrum infinite at this frequency)
weight_func_hp[1, ] <- 0

# Filter length equals sample length
L <- len

# Constraint flags: impose both amplitude and shift constraints at freq. zero
i1 <- T    # Amplitude constraint: filter passes the trend in original scale
i2 <- T    # Phase constraint:     no time-shift at frequency zero

# Constraint values at frequency zero
weight_constraint <- 1    # Amplitude = 1: trend passed at original scale
shift_constraint  <- 0    # Time-shift = 0: no phase distortion at freq. zero
# A non-zero shift_constraint would introduce a
# right-shift (lag) or left-shift (lead)


# Cutoff: frequency where Gamma drops below 0.5
# (not used in the MSE criterion, but required as a function argument)
cutoff <- pi * which(Gamma < 0.5)[1] / length(Gamma)

# Nowcast (real-time estimate with no lag)
Lag <- 0


# Estimate the one-sided HP filter via MDFA, imposing the above constraints
#   Use MDFA wrapper MDFA_mse_constraint
imdfa_hp <- MDFA_mse_constraint(
  L, weight_func_hp, Lag, Gamma,
  i1, i2, weight_constraint, shift_constraint
)$mdfa_obj

# -----------------------------------------------------------------------------
# Compare One-Sided HP Filter Coefficients: mFilter vs. MDFA
# -----------------------------------------------------------------------------
# The two coefficient vectors are virtually indistinguishable.
# Residual differences can be reduced arbitrarily by increasing K (the
# frequency-grid resolution), at the cost of additional computation time.
# This confirms that MDFA successfully replicates the HP filter.
# Once replicated, customization (e.g., timeliness optimization) can be
# applied — see the MDFA book for illustrations.
# -----------------------------------------------------------------------------

par(mfrow = c(2, 1))
colo   <- c("blue", "red")
insamp <- 1.e+99

# Full coefficient comparison (all lags)
mplot      <- cbind(imdfa_hp$b, parm[1:L, max(0, Lag) + 1])
rownames(mplot) <- paste("Lag ", 0:(nrow(mplot) - 1))
colnames(mplot) <- c("MDFA Replication", "HP Real-Time (mFilter)")
plot_title <- "HP Real-Time Filter: MDFA Replication vs. mFilter — All Lags"
freq_axe   <- rownames(mplot)
title_more <- c("MDFA", "HP")
mplot_func(mplot, freq_axe, plot_title, title_more, insamp, colo)

# Zoomed-in comparison (first 21 lags)
mplot      <- mplot[1:21, ]
rownames(mplot) <- paste("Lag ", 0:(nrow(mplot) - 1))
colnames(mplot) <- c("MDFA Replication", "HP Real-Time (mFilter)")
plot_title <- "HP Real-Time Filter: MDFA Replication vs. mFilter — Lags 0–20"
freq_axe   <- rownames(mplot)
title_more <- c("MDFA", "HP")
mplot_func(mplot, freq_axe, plot_title, title_more, insamp, colo)

# -----------------------------------------------------------------------------
# 1.3.4 Verify Constraints at Frequency Zero
# -----------------------------------------------------------------------------
# First-order constraint (i1): sum of coefficients should equal 1
print(paste("Transfer function at frequency zero (should be 1): ",
            round(sum(imdfa_hp$b), 3), sep = ""))

# Second-order constraint (i2): time-shift should vanish
print(paste("Time-shift at frequency zero (should be 0): ",
            round((1:(L - 1)) %*% imdfa_hp$b[2:L], 10), sep = ""))


# =============================================================================
# Exercise 2: Replication of the Christiano-Fitzgerald (CF) Bandpass Filter
# =============================================================================
# The CF filter differs from HP in two key respects:
#   - Target:  an ideal bandpass (rather than a lowpass)
#   - Implicit model: a random walk / ARIMA(0,1,0) (rather than ARIMA(0,2,2))
# Consequently, only a first-order constraint (i1) is required at freq. zero.
# =============================================================================

# -----------------------------------------------------------------------------
# 2.1 Set Up Parameters for CF Replication
# -----------------------------------------------------------------------------
x   <- lgdp
len <- length(x)

# Frequency-grid resolution
# (increasing K tightens the approximation at the cost of computation time)
K <- 1200

# Bandpass cutoff frequencies (in radians), derived from cycle lengths in quarters
len1    <- 8     # Upper cutoff: 8-quarter (2-year) cycles
len2    <- 40    # Lower cutoff: 40-quarter (10-year) cycles
cutoff1 <- 2 * pi / len1
cutoff2 <- 2 * pi / len2

# Target: ideal bandpass gain (1 within the band, 0 outside)
Gamma_cf <- ((0:K) > K * cutoff2 / pi) & ((0:K) < K * cutoff1 / pi)

# Pseudo-spectrum: square-root of the random-walk spectral density
# (the implicit model underlying the CF filter is an ARIMA(0,1,0))
weight_func_cf <- matrix(rep(1 / abs(1 - exp(1.i * (0:K) * pi / K)), 2), ncol = 2)
K <- nrow(weight_func_cf) - 1

# Remove spectral singularity at frequency zero (unit root in random walk)
weight_func_cf[1, ] <- 0

# Filter length equals sample length
L <- len

# Constraint flags: CF assumes an I(1) model — only first-order constraint needed
i1 <- T    # Amplitude at freq. zero must equal 0 (bandpass suppresses trend)
i2 <- F    # No second-order constraint needed (single unit root suffices)

# Constraint values at frequency zero
weight_constraint <- 0    # Bandpass must have zero gain at freq. zero
shift_constraint  <- 0    # Irrelevant when i2 <- F; any value can be assigned

# Nowcast
Lag <- 0

# -----------------------------------------------------------------------------
# 2.2 Estimate One-Sided CF Filter via MDFA
# -----------------------------------------------------------------------------
imdfa_cf <- MDFA_mse_constraint(
  L, weight_func_cf, Lag, Gamma_cf,
  i1, i2, weight_constraint, shift_constraint
)$mdfa_obj

par(mfrow=c(1,1))
# Plot amplitude of the estimated one-sided CF filter
omega_k  <- pi * (0:K) / K
amp_mse  <- abs(imdfa_cf$trffkt)
mplot    <- as.matrix(amp_mse)
mplot[1, ] <- NA
colnames(mplot) <- NA

ax <- rep(NA, nrow(mplot))
ax[1 + (0:6) * ((nrow(mplot) - 1) / 6)] <- c(0, "pi/6", "2pi/6", "3pi/6",
                                             "4pi/6", "5pi/6", "pi")
plot_title <- paste("Amplitude: One-Sided CF Bandpass Filter",
                    "(Cutoffs pi/", len2 / 2, " and pi/", len1 / 2, ")",
                    sep = "")
insamp     <- 1.e+90
title_more <- dimnames(mplot)[[2]]
colo       <- c("blue", "red")
mplot_func(mplot, ax, plot_title, title_more, insamp, colo)

# -----------------------------------------------------------------------------
# 2.3 Verify Constraints at Frequency Zero
# -----------------------------------------------------------------------------
# Only the amplitude constraint was imposed (i1 <- T, i2 <- F)

# First-order constraint: sum of coefficients should equal 0 (bandpass)
print(paste("Transfer function at frequency zero (should be 0): ",
            round(sum(imdfa_cf$b), 3), sep = ""))

# Second-order constraint: time-shift is unconstrained — reported for reference
print(paste("Time-shift at frequency zero (unconstrained): ",
            round((1:(L - 1)) %*% imdfa_cf$b[2:L], 3), sep = ""))


# -----------------------------------------------------------------------------
# 2.4 Verifying the MDFA One-Sided CF Filter
# -----------------------------------------------------------------------------
# To verify the MDFA solution, we ideally would compare it against the
# mFilter package. However, mFilter produces incorrect one-sided CF filters,
# as demonstrated below.
# Instead, we verify the MDFA solution against the classical model-based
# (time-domain) one-sided CF filter, which serves as the correct benchmark.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 2.4.1 First Attempt: mFilter Benchmark (Illustrating Incorrect Results)
# -----------------------------------------------------------------------------
# We apply cffilter() with both root = F and root = T, and verify whether
# the first-order constraint (amplitude = 0 at frequency zero) is satisfied.

# CF filter with root = F
x_cf     <- cffilter(x, pu = len2, pl = len1, root = F,
                     drift = F, nfix = NULL, theta = 1)
parm_cf  <- x_cf$fmatrix

# First-order constraint check: sum of coefficients should equal 0
# Does not work!
print(paste("Transfer function at frequency zero (root=F, should be 0): ",
            round(sum(parm_cf[, 1]), 3), sep = ""))

# CF filter with root = T
x_cf_T    <- cffilter(x, pu = len2, pl = len1, root = T,
                      drift = F, nfix = NULL, theta = 1)
parm_cf_T <- x_cf_T$fmatrix

# First-order constraint check
# Still wrong: sum does not vanish
print(paste("Transfer function at frequency zero (root=T, should be 0): ",
            round(sum(parm_cf_T[, 1]), 3), sep = ""))

# mFilter with root = T
x_mF_T    <- mFilter(x, filter = "CF", pu = len2, pl = len1, root = T,
                     drift = F, nfix = NULL, theta = 1)
parm_mF_T <- x_mF_T$fmatrix

# First-order constraint check
# Still a miss!
print(paste("Transfer function at frequency zero (mFilter root=T, should be 0): ",
            round(sum(parm_mF_T[, 1]), 3), sep = ""))

# Neither cffilter() nor mFilter() satisfies the first-order constraint.
# Setting root = T results in a severely misspecified filter.
# The plot below makes these discrepancies visible.

colo       <- c("blue", "red", "green")
mplot      <- cbind(imdfa_cf$b, parm_cf[, 1], parm_mF_T[, 1])
colnames(mplot) <- c("MDFA", "mFilter: root=F", "mFilter: root=T")
plot_title <- "Real-Time CF Filter: MDFA (blue) vs. mFilter root=F (red) vs. mFilter root=T (green)"
freq_axe   <- paste("Lag ", 0:(len - 1), sep = "")
title_more <- colnames(mplot)
mplot_func(mplot, freq_axe, plot_title, title_more, insamp, colo)

# Observations:
#   - MDFA (blue) and mFilter with root=F (red) match closely in the interior,
#     but show non-negligible discrepancies near the boundaries (lags 0 and len).
#   - mFilter with root=T (green) is severely misspecified throughout.

# -----------------------------------------------------------------------------
# 2.4.2 Correct Benchmark: Classical Model-Based (Time-Domain) CF Filter
# -----------------------------------------------------------------------------
# The classical one-sided CF filter is derived analytically as a truncated
# ideal bandpass, augmented by fore- and backcasts at the sample boundaries.
# This provides the correct benchmark for verifying the MDFA solution.
# A tighter match can be obtained by increasing K (frequency-grid resolution).

ord <- 100000
b   <- 0:ord

# Ideal bandpass coefficients (sinc function representation)
b[1 + 1:ord] <- (sin((1:ord) * 2 * pi / len1) -
                   sin((1:ord) * 2 * pi / len2)) / (pi * (1:ord))
b[1] <- 2 / len1 - 2 / len2

# Truncate to sample length
b_finite <- b[1:len]

# Augment boundary coefficients with contributions from forecasts and backcasts
b_finite[1] <- b_finite[1]   + sum(b[2:ord])           # Lag-0: add forecast tail
b_finite[len] <- b_finite[len] + sum(b[(len + 1):ord])   # Lag-len: add backcast tail

# Compare MDFA and classical model-based coefficients
# The two series overlap almost perfectly, confirming successful replication.
colo       <- c("red", "blue")
mplot      <- cbind(b_finite, imdfa_cf$b)
colnames(mplot) <- c("Classical Model-Based", "MDFA")
plot_title <- "Real-Time CF Filter: Classical Model-Based (red) vs. MDFA (blue)"
freq_axe   <- rep(NA, len)
freq_axe[1] <- 0
freq_axe[(1:6) * len / 6] <- paste("Lag ", as.integer(1 + (1:6) * len / 6), sep = "")
title_more <- colnames(mplot)
mplot_func(mplot, freq_axe, plot_title, title_more, insamp, colo)


# =============================================================================
# Exercise 3: Replication of the Hamilton Filter
# =============================================================================
# The Hamilton filter is a causal (one-sided) linear filter, which means
# its target Gamma is complex-valued (it carries a phase component).
#
# Solution strategy:
#   Rather than working with a complex-valued target, we transfer the phase
#   from Gamma into the pseudo-spectrum (weight_func). This yields a
#   real-valued target and a real-valued filter, while preserving the
#   correct phase structure of the original Hamilton filter.

# Replication is perfect because target is already causal. In a further step 
#   the filter could be customized (for example addressing the time-shift or lag)
# =============================================================================

y   <- lgdp
len <- length(y)

# -----------------------------------------------------------------------------
# 3.1 Estimate Hamilton Filter Coefficients via OLS
# -----------------------------------------------------------------------------
# Hamilton (2018) proposes regressing y_{t+h} on y_t, y_{t-1}, ..., y_{t-p+1}
# Settings for quarterly data: h = 8 quarters (2 years), p = 4 lags

h <- 2 * 4    # Forecast horizon: 8 quarters
p <- 4        # Number of autoregressive lags

# Construct lagged regressor matrix and target vector
explanatory <- y[(p):(len - h)]
for (i in 1:(p - 1))
  explanatory <- cbind(explanatory, y[(p - i):(len - h - i)])
target <- y[(h + p):len]

# Estimate OLS regression: y_{t+h} = alpha + beta * [y_t, ..., y_{t-p+1}] + e_t
lm_obj <- lm(y[(h + p):len] ~ explanatory)

# Note: in practice, only the first lag coefficient tends to be significant
# for non-stationary economic series — the model approximates a random walk.
summary(lm_obj)
ar_vec <- lm_obj$coefficients[1 + 1:p]

hamilton_filter<-c(1,rep(0,h-1),-ar_vec)

# Hamilton filter coefficients: [1, 0, ..., 0, -ar_vec] (length h + p)
plot(hamilton_filter, col = "black", xlab = "", ylab = "", main = "Hamilton Filter",type="l")


# Intercept (can optionally be included in the filter output)
intercept <- lm_obj$coefficients[1]
  
# -----------------------------------------------------------------------------
# 3.2 Replication of the Hamilton Filter via MDFA
# -----------------------------------------------------------------------------
K <- 1200

# --- 3.2.1 Compute Pseudo-Spectral Density ---
# Fit an AR(p) model to the data to derive the spectral density of the
# data-generating process. This serves as weight_func for MDFA.

explanatory_spect <- y[(p):(len - 1)]
for (i in 1:(p - 1))
  explanatory_spect <- cbind(explanatory_spect, y[(p - i):(len - 1 - i)])
target_spect  <- y[(1 + p):len]
lm_obj_spect  <- lm(target_spect ~ explanatory_spect)
ar_spect_vec  <- lm_obj_spect$coefficients[1 + 1:p]

# Evaluate the AR(p) transfer function on the frequency grid
ar_inv_spect <- rep(0, K + 1)
for (j in 1:p)
  ar_inv_spect <- ar_inv_spect + ar_spect_vec[j] * exp(-1.i * j * (0:K) * pi / K)

# Spectral density: |1 / (1 - AR(z))|
ar_spect  <- abs(1 / (1 - ar_inv_spect))

par(mfrow=c(1,1))
omega_k<-pi*(0:K)/K
mplot<-as.matrix(ar_spect)
mplot[1,]<-NA
colnames(mplot)<-NA
ax<-rep(NA,nrow(mplot))
ax[1+(0:6)*((nrow(mplot)-1)/6)]<-c(0,"pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi")
plot_title<-paste("Spectrum Hamilton Filter: AR(",p,") Spectrum" ,sep="")
insamp<-1.e+90
colo<-c("black","red")
title_more<-NULL
mplot_func(mplot, ax, plot_title, title_more, insamp, colo)


weight_func <- abs(cbind(ar_spect, ar_spect))

# --- 3.2.2 Compute Target Gamma ---
# The Hamilton filter is treated as an ordinary (MA-) filter applied directly to x_t
# Note: the spectrum above is the inversion of the AR, as applied to epsilon_t.
# The MA transfer function encodes the Hamilton filter structure.

ma_inv_spect <- rep(0, K + 1)
for (j in 1:p)
  ma_inv_spect <- ma_inv_spect +
  ar_vec[j] * exp(-1.i * (h - 1 + j) * (0:K) * pi / K)

ma_gamma <- (1 - ma_inv_spect)

# Gamma must be real-valued: take the modulus
Gamma_ham <- abs(ma_gamma)

# Transfer the phase of ma_gamma from Gamma into the pseudo-spectrum
# This ensures a real-valued target and a real-valued estimated filter
weight_func_ham <- cbind(weight_func[, 1] * exp(-1.i * Arg(ma_gamma)),
                         weight_func[, 2])

# --- Plot: Target Gamma and Spectrum ---
par(mfrow = c(2, 1))
colo   <- c("blue", "red")
insamp <- 1.e+99

freq_axe <- rep(NA, K + 1)
freq_axe[1] <- 0
freq_axe[1 + (1:6) * K / 6] <- c(paste(c("", 2:5), "pi/6", sep = ""), "pi")

mplot      <- abs(as.matrix(Gamma_ham))
plot_title <- "Target Gamma: Hamilton Filter"
mplot_func(mplot, freq_axe, plot_title, title_more, insamp, colo)

mplot      <- as.matrix(abs(weight_func_ham))
plot_title <- "Spectrum AR(p) Model"
mplot_func(mplot, freq_axe, plot_title, title_more, insamp, colo)

# --- 3.2.3 Replicate Hamilton Filter via MDFA ---
# No unit-root constraints are needed (Hamilton filter is based on a stationary AR(p))
L                 <- length(hamilton_filter)
i1                <- F
i2                <- F
weight_constraint <- 1
shift_constraint  <- 1
Lag               <- 0

imdfa_ham <- MDFA_mse_constraint(
  L, weight_func_ham, Lag, Gamma_ham,
  i1, i2, weight_constraint, shift_constraint
)$mdfa_obj

# --- Plot: Amplitude of MDFA-Estimated Hamilton Filter ---
par(mfrow = c(2, 1))
omega_k    <- pi * (0:K) / K
amp_mse    <- abs(imdfa_ham$trffkt)
mplot      <- cbind(as.matrix(amp_mse),abs(as.matrix(Gamma_ham)))
mplot[1, ] <- NA
colnames(mplot) <- rep(NA,2)

ax <- rep(NA, nrow(mplot))
ax[1 + (0:6) * ((nrow(mplot) - 1) / 6)] <- c(0, "pi/6", "2pi/6", "3pi/6",
                                             "4pi/6", "5pi/6", "pi")
plot_title <- "Amplitude: Hamilton (red) replicated by DFA (blue): both overlap"
insamp     <- 1.e+90
title_more <- dimnames(mplot)[[2]]
colo       <- c("blue", "red")
mplot_func(mplot, ax, plot_title, title_more, insamp, colo)

# Overlay MDFA and Hamilton filter coefficients: perfect replication
ts.plot(cbind(imdfa_ham$b, hamilton_filter), col = c("blue", "red"),main="Filter: Hamilton (red) replicated by DFA (blue): both overlap")


# =============================================================================
# Summary and Final Considerations
# =============================================================================
# This tutorial demonstrated how classic linear filters can be replicated
# within the MDFA framework by supplying the appropriate inputs:
#
# Replication Recipe:
#   1. Derive weight_func (pseudo-spectrum) from the filter's implicit model
#        or from the data directly (e.g., via the DFT or a fitted AR model).
#   2. Derive Gamma (target gain) from the filter's analytical definition
#        or from the literature.
#   3. Apply unit-root constraints if the implicit model is non-stationary.
#   4. Call MDFA_mse_constraint() to obtain the replicated filter.
#     In the absence of unit roots one might use the MSE-wrapper MDFA_mse() 
#     instead of MDFA_mse_constraint() 
#
# Key Results:
#   - HP filter:       replicated via ARIMA(0,2,2) spectrum and two constraints
#   - CF filter:       replicated via random-walk spectrum and one constraint;
#                      mFilter was shown to produce incorrect one-sided filters
#   - Hamilton filter: replicated via AR(p) spectrum with phase transfer trick
#
# Extensions (once a filter has been replicated):
#   - Customization:   e.g., timeliness-optimized HP (see MDFA book)
#   - Regularization:  control overfitting in short samples
#   - Hybrid designs:  combine spectra from data-fitted models with targets
#                      from HP, CF, or other classic filters
#
# Any linear filter can, in principle, be replicated and subsequently
# extended using the MDFA framework.
# =============================================================================


























#------------------------------------
# 2.4 We now verify that the above one-sided filter, obtained by MDFA, replicates the one-sided CF-filter
#   -For that purpose we would have relied on the mFilter-package: 
#     -Unfortunately mFilter is wrong, as can be seen below 
#   -Therefore we rely on the classic model-based solution for deriving the one-sided filter and we check that this corresponds to MDFA


# 2.4.1 First attempt: based on mFilter (mFilter does not work properly)
# We here rely on R-package mFilter for the CF-filter: comparisons with the MDFA-package are provided below
x_cf<-cffilter(x,pu=len2,pl=len1,root=F,drift=F, nfix=NULL,theta=1)
parm_cf<-x_cf$fmatrix

# Check first-order constraint: should give 0 i.e. amplitude in frequency zero should vanish
print(paste("Transfer function in frequency zero: ",
            round(sum(parm_cf[,1]),3),sep=""))
  x_cf_T<-cffilter(x,pu=len2,pl=len1,root=T,drift=F, nfix=NULL,theta=1)
parm_cf_T<-x_cf_T$fmatrix

# Check first-order: should give 0
print(paste("Transfer function in frequency zero: ",
            round(sum(parm_cf_T[,1]),3),sep=""))

#  Obviously, the code does not seem to work properly. 
x_mF_T<-mFilter(x,filter="CF",pu=len2,pl=len1,root=T,drift=F, 
                  nfix=NULL,theta=1)
parm_mF_T<-x_mF_T$fmatrix

# Check first-order: should give 0
print(paste("Transfer function in frequency zero: ",
            round(sum(parm_mF_T[,1]),3),sep=""))

#  Neither $mFilter$ nor $cffilter$  comply with the first-order constraint in frequency zero. Selecting $root=T$ results in a severly misspecified filter.
colo<-c("blue","red","green")
mplot<-cbind(imdfa_cf$b,parm_cf[,1],parm_mF_T[,1])
colnames(mplot)<-c("DFA","mFilter: plot=F","mFilter: plot=T")
plot_title<-"Real-time CF-filters: DFA (blue) vs. mFilter (red and green)"
freq_axe<-paste("Lag ",0:(len-1),sep="")
title_more<-colnames(mplot)
mplot_func(mplot,freq_axe,plot_title,title_more,insamp,colo)

# As can be seen, DFA (blue) and $mFilter$ or $cffilter$ based on $root=F$ (red) match closely up to the boundaries, lags 0 and $\Sexpr{len}$, where non-negligible discrepancies can be observed. 
# In contrast, the coefficients of $mFilter$ for $root=T$ (green line) are `off the mark'.

#----------------------
# 2.4.2 Cross-check real-time DFA coefficients with the (classic MSE time-domain) model-based solution 
#   Finally we can verify replication of CF-filter by MDFA
ord<-100000
b<-0:ord
b[1+1:ord]<-(sin((1:ord)*2*pi/len1)-sin((1:ord)*2*pi/len2))/(pi*(1:ord))
b[1]<-2/len1-2/len2
# Real-time filter based on for- and backcasts
b_finite<-b[1:len]
# The lag-0 coefficient is augmented by forecasts
b_finite[1]<-b_finite[1]+sum(b[2:ord])
# The lag-len coefficient is augmenetd by backcasts
b_finite[len]<-b_finite[len]+sum(b[(len+1):ord])
# Compare DFA and model-based coefficients
#   Both coefficients overlap almost perfectly: tighter approximations could be obtained by increasing K (resolution of frequency-grid)
colo<-c("red","blue")
mplot<-cbind(b_finite,imdfa_cf$b)
plot_title<-"Real-time CF-filters: Forecast/backcast (red) vs. DFA (blue)"
freq_axe<-rep(NA,len)
freq_axe[1]<-0
freq_axe[(1:6)*len/6]<-paste("Lag ",as.integer(1+(1:6)*len/6),sep="")
mplot_func(mplot,freq_axe,plot_title,title_more,insamp,colo)




#--------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
# Exercise 3: Replication Hamilton filter
# The slight difficulty here is that the (target) Hamilton filter is causal. Hence the target Gamma is complex
# -Solution: we transfer the phase from the target to the spectrum 
#   This way the target is real-valued and the filter is real-valued too

y<-lgdp
len<-length(y)
#------------------
# 3.1 Hamilton filter
# Settings proposed by Hamilton for quarterly data
h<-2*4
p<-4

# Regression of y_{t+h} on y_t,y_{t-1},...,y_{t-p}
explanatory<-y[(p):(len-h)]
for (i in 1:(p-1))
  explanatory<-cbind(explanatory,y[(p-i):(len-h-i)])
target<-y[(h+p):len]
# The sample begins in 1960
lm_obj<-lm(y[(h+p):len]~explanatory)

# Only the first coefficient is significant, as is often the case in applications to non-stationary economic series.
#   Model is close to a random-walk
summary(lm_obj)
ar_vec<-lm_obj$coefficients[1+1:p]

# Specify Hamilton filter
hamilton_filter<-c(1,rep(0,h-1),-ar_vec)
# We can include an intercept 
intercept<-lm_obj$coefficients[1]

#----------------------------
# 3.2 Replication by DFA
K<-1200

# 3.2.1.Compute spectral density
# First need AR model
explanatory_spect<-y[(p):(len-1)]
for (i in 1:(p-1))
  explanatory_spect<-cbind(explanatory_spect,y[(p-i):(len-1-i)])
target_spect<-y[(1+p):len]
lm_obj_spect<-lm(target_spect~explanatory_spect)
ar_spect_vec<-lm_obj_spect$coefficients[1+1:p]

# Spectrum: AR(p) filter
ar_inv_spect<-rep(0,K+1)
for (j in 1:p)#j<-1
{
  ar_inv_spect<-ar_inv_spect+ar_spect_vec[j]*exp(-1.i*j*(0:(K))*pi/(K))
}  
ar_spect<-abs(1/(1-ar_inv_spect))
ts.plot(ar_spect)

weight_func<-abs(cbind(ar_spect,ar_spect))

# 3.2.2.Compute target Gamma:
# Note that we treat this as a MA filter, applied directly to x_t 
#   (in contrast, the spectrum is a AR filter applied to epsilon_t)
ma_inv_spect<-rep(0,K+1)
for (j in 1:p)#j<-1
{
  ma_inv_spect<-ma_inv_spect+ar_vec[j]*exp(-1.i*(h-1+j)*(0:(K))*pi/(K))
}  
ma_gamma<-(1-ma_inv_spect)
# Target: Gamma must be real (absolute value)
Gamma_ham<-abs(ma_gamma)
# We shift spectrum of target by argument of ma_gamma to account for phase of target (ma_gamma)
weight_func_ham<-cbind(weight_func[,1]*exp(-1.i*Arg(ma_gamma)),weight_func[,2])

# Target is identity
#Gamma_ham<-rep(1,K+1)

par(mfrow=c(2,1))
colo<-c("blue","red")
insamp<-1.e+99
mplot<-abs(as.matrix(Gamma_ham))
plot_title<-"Target Gamma"
freq_axe<-rep(NA,K+1)
freq_axe[1]<-0
freq_axe[1+(1:6)*K/6]<-c(paste(c("",2:5),"pi/6",sep=""),"pi")
mplot_func(mplot,freq_axe,plot_title,title_more,insamp,colo)
# Plot log spectrum: weight_func must be squared
mplot<-as.matrix(abs(weight_func_ham))
plot_title<-"Spectrum"
mplot_func(mplot,freq_axe,plot_title,title_more,insamp,colo)


L<-length(hamilton_filter)
i1<-i2<-F
weight_constraint<-shift_constraint<-1
Lag<-0

imdfa_ham<-MDFA_mse_constraint(L,weight_func_ham,Lag,Gamma_ham,i1,i2,weight_constraint,shift_constraint)$mdfa_obj

par(mfrow=c(1,1))
omega_k<-pi*(0:K)/K
amp_mse<-abs(imdfa_ham$trffkt)
mplot<-as.matrix(amp_mse)
mplot[1,]<-NA
colnames(mplot)<-NA
ax<-rep(NA,nrow(mplot))
ax[1+(0:6)*((nrow(mplot)-1)/6)]<-c(0,"pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi")
plot_title<-paste("Amplitude ")
insamp<-1.e+90
title_more<-dimnames(mplot)[[2]]
colo<-c("blue","red")
mplot_func(mplot, ax, plot_title, title_more, insamp, colo)

# Perfect replication
ts.plot(cbind(imdfa_ham$b,hamilton_filter),col=c("blue","red"))



#-----------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
# Final considerations
#   -We showed how to replicate classic filters in MDFA
#     Derive Gamma and weight_func as specified in literature
#     Apply constraints (if required)
#     Apply MDFA_mse_constraint wrapper for replication
#   -Any other (linear) filter could be replicated analogously by MDFA
#   -Once replicated, additional features could be applied
#     -Customization: an example of a customized HP is presented in the MDFA-book
#     -Regularization
#     -Thus any filter could be 'tweaked' to the purpose of a particular application or the preferences of a particular user
#   -Hybrid designs could be obtained straightforwardly
#     Example hybrid design
#       -Use (pseudo-) spectrum from a model fitted to the data or the from the dft
#       -Use target from HP or CF
