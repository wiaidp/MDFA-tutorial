# ==============================================================================
# Tutorial 6
# ==============================================================================
#
# Purpose:
#   - Introduce the new MDFA wrapper MDFA_cust(), which generalizes MDFA_mse()
#   - Illustrate the decomposition of the classical MSE criterion into its
#     Accuracy, Timeliness, and Smoothness components: the ATS trilemma
#     (see McElroy and Wildi, 2020)
#   - Analyze filter characteristics (amplitude and phase shift) when
#     emphasizing Timeliness (the T component of the ATS trilemma)
#   - Analyze filter characteristics (amplitude and phase shift) when
#     emphasizing Smoothness (the S component of the ATS trilemma)
#   - Analyze filter characteristics (amplitude and phase shift) when
#     emphasizing both T and S simultaneously, illustrating the power of
#     the trilemma relative to the classical MSE-based dilemma
#   - Compare the univariate customized DFA to the bivariate leading-indicator
#     MSE-MDFA design introduced in the previous tutorial
#   - Compare all filter designs to a customized bivariate MDFA
#   - Conduct extensive out-of-sample simulation studies
#
# New hyperparameters:
#   - lambda : controls the emphasis on Timeliness in MDFA_cust() / MDFA_reg()
#   - eta    : controls the emphasis on Smoothness in MDFA_cust() / MDFA_reg()
#   - cutoff : delimites passband and stopband
# ==============================================================================
# Reference:
#   McElroy, T. and Wildi, M. (2020).
#   "The Trilemma between Accuracy, Timeliness and Smoothness in
#    Real-Time Signal Extraction."
#   International Journal of Forecasting, 36(2), 423-438.
#   https://doi.org/10.1016/j.ijforecast.2019.04.010
#   (Working paper available in the Literature folder of this GitHub repository)
#
# ==============================================================================
# Theoretical Background: ATS Decomposition
# ==============================================================================
#
# The MSE criterion admits an additive decomposition: MSE = A + T + S + R, where:
#   - A : contribution of amplitude mismatch between the causal and acausal
#         filters in the passband of the target filter
#   - T : contribution of phase-shift mismatch in the passband
#   - S : contribution of amplitude mismatch in the stopband
#   - R : contribution of phase-shift mismatch in the stopband
#
# Since R ≈ 0 in practice, the decomposition simplifies to: MSE ≈ A + T + S
#
# The customized criterion generalizes the MSE as follows:
#   MSE_cust = MSE + lambda * T + eta * S = A + (1 + lambda) * T + (1 + eta) * S
#
# Key properties:
#   - lambda = eta = 0 : replicates the classical MSE (no customization)
#   - lambda > 0       : emphasizes T, yielding a faster filter with reduced lag
#   - eta > 0          : emphasizes S, yielding stronger noise suppression
#                        and smoother filter output
#   - lambda > 0 and eta > 0 : emphasizes both T and S simultaneously;
#                        judicious choices can yield a filter that is both
#                        faster and smoother
#
# MSE_cust defines an efficient frontier over the ATS trilemma:
#   - In any customized MDFA solution, improving any one of A, T, or S
#     inevitably comes at the cost of at least one of the others
#
# ==============================================================================
# Alternative Approaches Addressing Smoothness and Timeliness in Prediction
# ==============================================================================
#
# - M-SSA: addresses Smoothness in the time domain by controlling the frequency
#   of zero-crossings (sign changes) in the filter output
#   - A dedicated M-SSA tutorial is available on GitHub
#   - Reference:
#     Wildi, M. (2026). The Accuracy-Smoothness Dilemma in Prediction:
#     A Novel Multivariate M-SSA Forecast Approach.
#     Journal of Time Series Analysis. 
#     https://doi.org/10.48550/arXiv.2602.13722
#
# - DFP/PCS (Look-Ahead predictors): address Timeliness by directly minimizing
#   the lead of the filter output; no other linear predictor can improve upon
#   DFP/PCS for a given level of tracking accuracy
#   - A dedicated tutorial is currently in preparation
#   - Reference:
#     Wildi, M. (2026). Forecasting on the Accuracy-Timeliness Frontier:
#     Two Novel Look-Ahead Predictors.
#     https://doi.org/10.48550/arXiv.2602.23087
#
# ==============================================================================
# Background: MDFA_cust() Wrapper
# ==============================================================================
#
# - MDFA_cust() selects the relevant hyperparameters and calls the core
#   mdfa_analytic() function:
#     - mdfa_analytic() is considerably more general, accepting a broader set
#       of hyperparameters
#     - MDFA_cust() simplifies the interface by exposing only the hyperparameters
#       relevant to ATS customization
#
# - Hyperparameters of MDFA_cust():
#     - L           : filter length
#     - weight_func : spectral density (spectrum)
#     - Lag         : target lead/lag (forecast, nowcast, or backcast)
#     - Gamma       : target filter
#     - cutoff      : frequency separating the pass- and stopband
#                     (A, T, and S are defined over either band as determined by cutoff)
#     - lambda      : weight assigned to T is (1 + lambda)
#     - eta         : weight assigned to S is (1 + eta)
#
# - Note: cutoff, lambda and eta are absent from MDFA_mse(), which implicitly sets
#   lambda = eta = 0, corresponding to pure MSE minimization
# - Note: mdfa_analytic() enforces non-negativity of lambda and eta;
#   negative values are automatically converted via abs()
# ==============================================================================
# Note on Leading Indicator Examples 6 and 7
# (see Tutorial 4 for background)
# ==============================================================================
# - The bivariate leading-indicator MSE-MDFA is generally faster than the univariate
#   MSE-DFA, as the additional explanatory variable carries a lead of one time unit
# - Key question: can a customized univariate DFA match or even surpass the
#   speed of the bivariate MSE-MDFA through ATS customization alone,
#   i.e., without resorting to any leading indicator?
# ==============================================================================


rm(list = ls())

# ------------------------------------------------------------------------------
# Load libraries
# ------------------------------------------------------------------------------
library(xts)
# install.packages("devtools")  # Uncomment if devtools not yet installed
library(devtools)

# Load MDFA package from GitHub
devtools::install_github("wiaidp/MDFA")  # Uncomment if MDFA not yet installed
# Note: EURUSD data is bundled within the MDFA package
library(MDFA)

# ------------------------------------------------------------------------------
# Brief overview of available wrapper functions and main estimation function
# ------------------------------------------------------------------------------
# Wrappers:
#   MDFA_mse                 - MSE criterion (no constraints)
#   MDFA_mse_constraint      - MSE criterion (with constraints: non-stationarity)
#   MDFA_cust                - Customized criterion (lambda/eta, no constraints)
#   MDFA_cust_constraint     - Customized criterion (lambda/eta, with constraints)
#   MDFA_reg                 - Regularized criterion (no constraints)
#   MDFA_reg_constraint      - Regularized criterion (with constraints)
# Main estimation function:
#   mdfa_analytic            - Core analytic DFA estimation routine

# Main MSE routine
head(MDFA_mse)
# MDFA_cust() has additional hyperparameters cutoff, lambda and eta
head(MDFA_cust)
# MDFA_mse_constraint: this wrapper was used in tutorial 5 (constraints for addressing integrated processes)
head(MDFA_mse_constraint)
# Additional wrappers and generic mdfa_analytic routine: the regularization feature will be introduced in the next tutorial
#head(MDFA_cust_constraint)
#head(MDFA_reg)
#head(MDFA_reg_constraint)
#head(mdfa_analytic)

# ------------------------------------------------------------------------------
# Source common utility functions
# ------------------------------------------------------------------------------
source("Common functions/plot_func.r")
source("Common functions/arma_spectrum.r")
source("Common functions/ideal_filter.r")
source("Common functions/mdfa_trade_func.r")
source("Common functions/play_with_bivariate.r")
source("Common functions/functions_trilemma.r")
source("Common functions/compute_customized_designs.r")

# ==============================================================================
# Example 1: Emphasizing Timeliness (lambda > 0, eta = 0)
# Univariate setting
# ==============================================================================
#
# The customized DFA criterion decomposes the MSE into three components:
#   Accuracy  (A) : standard MSE passband/stopband fit
#   Timeliness (T): penalizes phase shifts (controlled by lambda)
#   Smoothness (S): penalizes roughness of the filter output (controlled by eta)
#
# Here we fix eta = 0 (no smoothness emphasis) and vary lambda to isolate
# the effect of increasing timeliness emphasis.

# The setting lambda=eta=0 replicates the classical MSE DFA: the new wrapper MDFA_cust()
#   is more general than MDFA_mse() of previous tutorials
# ------------------------------------------------------------------------------

# --- Data simulation ----------------------------------------------------------
len <- 300
a1  <- 0.0          # AR coefficient (white noise when a1 = 0)
b1  <- NULL         # MA coefficients (none)
set.seed(1)
x   <- as.vector(arima.sim(n = len, list(ar = a1, ma = b1)))

# --- Spectral setup -----------------------------------------------------------
K          <- 600    # Number of frequency grid points (grid: 0, pi/K, ..., pi)
plot_T     <- TRUE

# Spectrum weight function (both columns identical for univariate case)
weight_func <- abs(cbind(
  arma_spectrum_func(a1, b1, K, plot_T)$arma_spec,
  arma_spectrum_func(a1, b1, K, plot_T)$arma_spec
))

plot(weight_func[, 1], type = "l",
     main = paste("White noise spectrum, denseness =", K, sep = ""),
     axes = F, xlab = "Frequency", ylab = "Amplitude", col = "black")
mtext(colnames(weight_func)[1], line = -1, col = "black")
axis(1, at = c(0, 1:6 * K / 6 + 1),
     labels = c("0", "pi/6", "2pi/6", "3pi/6", "4pi/6", "5pi/6", "pi"))
axis(2)
box()


# --- Target (ideal) filter ----------------------------------------------------
periodicity <- 6                          # Target: cycles of period <= 2*6
cutoff      <- pi / periodicity           # Corresponding cutoff frequency
# Gamma: ideal low-pass target (1 in passband [0, cutoff], 0 elsewhere)
Gamma       <- (0:K) <= K * cutoff / pi + 1e-9
# Plot the lowpass target filter:
plot(Gamma, type = "l",
     main = paste("Target, denseness =", K, sep = ""),
     axes = F, xlab = "Frequency", ylab = "Amplitude", col = "black")
mtext("Target", line = -1, col = "black")
axis(1, at = c(0, 1:6 * K / 6 + 1),
     labels = c("0", "pi/6", "2pi/6", "3pi/6", "4pi/6", "5pi/6", "pi"))
axis(2)
box()


# --- Filter design parameters -------------------------------------------------
L   <- 20    # Filter length (number of coefficients)
Lag <- 0     # Lag = 0 => nowcast (real-time signal extraction)

# --- Timeliness parameter grid ------------------------------------------------
# lambda = 0   : pure MSE (no timeliness emphasis)
# lambda > 0   : increasing timeliness emphasis
lambda_vec <- c(0, 2^(0:7))              # 0, 1, 2, 4, 8, ..., 128
eta_vec    <- rep(0, length(lambda_vec)) # eta fixed at 0 throughout

# --- Compute filter designs across lambda grid --------------------------------
for (i in seq_along(lambda_vec)) {
  
  lambda   <- lambda_vec[i]
  eta      <- eta_vec[i]
  # ==============================================================================
  # - Increasing lambda > 0 reduces T, thereby decreasing the time-shift of the
  #   filter output to better match the target in the passband
  # - For Lag = 0 (nowcast): the target is a zero-phase (two-sided symmetric)
  #   ideal lowpass filter; increasing lambda drives the causal filter's shift towards zero
  #   (i.e., coincident with the target)
  # - For Lag = -1 (forecast): the target has a phase shift of -1; increasing
  #   lambda drives the causal filter's shift towards -1
  #   (i.e., the causal filter output becomes anticipative)
  # - Note: perfect anticipation is unattainable in practice; customization
  #   seeks the best achievable approximation under the ATS trilemma formulation
  # ==============================================================================
  mdfa_obj <- MDFA_cust(L, weight_func, Lag, Gamma, cutoff, lambda, eta)$mdfa_obj
  
  # Store transfer functions, filter coefficients, and filtered outputs
  if (i == 1) {
    trffkt_mat <- mdfa_obj$trffkt
    b_mat      <- mdfa_obj$b
    yhat_mat   <- filt_func(x, mdfa_obj$b)$yhat
  } else {
    trffkt_mat <- cbind(trffkt_mat, mdfa_obj$trffkt)
    b_mat      <- cbind(b_mat,      mdfa_obj$b)
    yhat_mat   <- cbind(yhat_mat,   filt_func(x, mdfa_obj$b)$yhat)
  }
}

# --- Plot: Amplitude and time-shift functions ---------------------------------
par(mfrow = c(2, 2))

# Frequency axis labels
ax <- rep(NA, K + 1)
ax[1 + (0:6) * (K / 6)] <- c("0", "pi/6", "2pi/6", "3pi/6", "4pi/6", "5pi/6", "pi")

col_labels <- paste0("(lambda=", lambda_vec, ", eta=", eta_vec, ")")
colo       <- rainbow(length(lambda_vec))

# 1. Raw amplitude functions
mplot <- abs(trffkt_mat)
dimnames(mplot)[[2]] <- paste("Amplitude", col_labels)
mplot_func(mplot, ax,
           plot_title  = "Amplitude functions",
           title_more  = dimnames(mplot)[[2]],
           insamp      = 1e+90,
           colo        = colo)

# 2. Scaled amplitude functions (each rescaled to max = 1 for visual comparison)
mplot_scaled <- abs(trffkt_mat)
for (i in seq_len(ncol(mplot_scaled)))
  mplot_scaled[, i] <- mplot_scaled[, i] / max(mplot_scaled[, i])
dimnames(mplot_scaled)[[2]] <- paste("Scaled amplitude", col_labels)
mplot_func(mplot_scaled, ax,
           plot_title  = "Scaled amplitude functions",
           title_more  = dimnames(mplot_scaled)[[2]],
           insamp      = 1e+90,
           colo        = colo)

# 3. Time-shift functions: phase / omega  (units: time periods)
#    A negative shift indicates the filter leads the target.
mplot_shift <- Arg(trffkt_mat) / (pi * (0:K) / K)
dimnames(mplot_shift)[[2]] <- paste("Time-shift", col_labels)
mplot_func(mplot_shift, ax,
           plot_title  = "Time-shift functions",
           title_more  = dimnames(mplot_shift)[[2]],
           insamp      = 1e+90,
           colo        = colo)
#======================================================
# Key Outcomes:
#
# 1. Baseline (lambda = eta = 0):
#      Corresponds to the classical MSE-optimal DFA design.
#
# 2. Effect of increasing lambda (lambda > 0):
#      - PRIMARY BENEFIT: Reduces lag (positive time-shift)
#        within the passband, though not uniformly so (see lower panel).
#      - SIDE EFFECT: Tends to compress the amplitude function
#        (see top-left panel); however, this can be mitigated
#        to a reasonable extent via simple rescaling (top right panel).
#
# 3. Phase shift (lag) reduction:
#      - At lambda = 0 (MSE design): lag ≈ between 1 and 1.5 (see red line left bottom plot)
#      - At lambda > 8:              lag ≈ 0 (see violet line left bottom plot)
#
# 4. Diminishing returns (lambda > 8):
#      Further increases to lambda yield smaller changes
#      to the filter shape (at some point no meaningful improvement).
#======================================================

# --- Plot: Filter outputs -----------------------------------------------------
anf <- 250   # Start index for output display window
enf <- 300   # End index for output display window

par(mfrow = c(1, 1))

# Time axis labels for output window
n_out <- enf - anf + 1
ax_out <- rep(NA, n_out)
ax_out[1 + (0:6) * floor((n_out - 1) / 6)] <-
  as.integer(anf + (0:6) * floor((n_out - 1) / 6))

# 4. Raw filter outputs
mplot_out <- yhat_mat[anf:enf, ]
dimnames(mplot_out)[[2]] <- paste("Output", col_labels)
mplot_func(mplot_out, ax_out,
           plot_title  = "Filter outputs",
           title_more  = dimnames(mplot_out)[[2]],
           insamp      = 1e+90,
           colo        = colo)

# 5. Scale filter outputs (rescaled by max amplitude for fair visual comparison) and Zoom in
anf <- 260   # Start index for output display window
enf <- 280   # End index for output display window
# Time axis labels for output window
n_out <- enf - anf + 1
ax_out <- rep(NA, n_out)
ax_out[1 + (0:6) * floor((n_out - 1) / 6)] <-
  as.integer(anf + (0:6) * floor((n_out - 1) / 6))

mplot_out_scaled <- yhat_mat[anf:enf, ]
for (i in seq_len(ncol(mplot_out_scaled)))
  mplot_out_scaled[, i] <- mplot_out_scaled[, i] / max(abs(trffkt_mat)[, i])
dimnames(mplot_out_scaled)[[2]] <- paste("Scaled output", col_labels)
mplot_func(mplot_out_scaled, ax_out,
           plot_title  = "Scaled filter outputs",
           title_more  = dimnames(mplot_out_scaled)[[2]],
           insamp      = 1e+90,
           colo        = colo)

# ==============================================================================
# Key Outcomes:
#
# 1. Effect of increasing lambda:
#      - Augments the left-shift in the filter output (i.e., reduces lag).
#
# 2. Upper bound on left-shift (nowcasting):
#      - The left-shift is bounded above by 1 time-unit.
#      - The time-shift function cannot turn negative, since the
#        target is a nowcast (Lag = 0).
#
# 3. Achieving larger left-shifts via forecasting:
#      - Setting Lag < 0 (forecasting) in combination with lambda > 0
#        allows for more pronounced left-shifts.
# ==============================================================================



# ==============================================================================
# Example 2: Emphasizing Smoothness (lambda = 0, eta > 0)
# Univariate setting
# ==============================================================================
#
# Here we fix lambda = 0 (no timeliness emphasis) and vary eta to isolate
# the effect of increasing smoothness emphasis.
#
# The smoothness penalty (eta) penalizes high-frequency content in the
# filter output (suppression of stopband noise), resulting in smoother (less noisy) estimates at the cost
# of some accuracy and potential phase distortion.
# ------------------------------------------------------------------------------

# --- Smoothness parameter grid ------------------------------------------------
eta_vec    <- 0.8 * 0:6                  # 0, 1, 2, 3, 4, 5, 6
lambda_vec <- rep(0, length(eta_vec))    # lambda fixed at 0 throughout

# --- Compute and plot all designs (reuses the helper function) ----------------
# This replicates the full loop + plotting pipeline from Example 1 in one call.
compute_customized_designs_func(
  lambda_vec  = lambda_vec,
  eta_vec     = eta_vec,
  L           = L,
  weight_func = weight_func,
  Lag         = Lag,
  Gamma       = Gamma,
  cutoff      = cutoff
)


# ==============================================================================
# Key Outcomes:
#
# 1. PRIMARY BENEFIT of increasing eta:
#      - Pulls the amplitude towards zero in the stopband,
#        resulting in improved noise attenuation (top plots).
#
# 2. SIDE EFFECTS of increasing eta:
#      - INCREASED LAG: Time-shift generally increases due to
#        stronger smoothing (bottom-left plot).
#      - SHRINKAGE: Amplitude in the passband shrinks towards zero
#        (top-left plot); this effect can be partially compensated
#        for via simple rescaling (top-right plot).
#
# 3. Filtered Series (bottom-right plot):
#      - Increasing eta produces a smoother filtered series.
#      - Stronger smoothing induces an additional right-shift (lag).
# ==============================================================================





# ==============================================================================
# Example 3: Emphasizing Both Timeliness and Smoothness
# ==============================================================================
# Comparison:
#   - MSE-optimal design:    lambda = 0,  eta = 0  (baseline)
#   - Customized design:     lambda = 10, eta = 4  (emphasizing Timeliness and Smoothness
#                                                   at the cost of Accuracy and overall MSE)

eta_vec    <- c(0, 4)
lambda_vec <- c(0, 20)

compute_customized_designs_func(lambda_vec, eta_vec, L, weight_func, Lag, Gamma, cutoff)


# ==============================================================================
# Key Outcomes:
#
# 1. Desirable Effects:
#      - TIMELINESS (lambda): Increasing lambda reduces the time-shift
#        in the passband (bottom-left panel, cyan line).
#      - SMOOTHNESS (eta): Increasing eta suppresses the amplitude in
#        the stopband, yielding stronger noise attenuation (top panels).
#
# 2. Side Effects — The Trilemma (A, T, S):
#      - SHRINKAGE: The amplitude shrinks towards zero in the passband
#        (top-left panel), adversely affecting accuracy (amplitude fit
#        in the passband); this can be partially compensated for via
#        rescaling (top-right panel).
#      - DISTORTION: Jointly emphasizing timeliness and smoothness
#        (large lambda and eta) distorts the filter characteristics;
#        specifically, instead of remaining flat across the passband,
#        the amplitude develops a peak near the cutoff frequency (top right).
#      - MSE DETERIORATION: Distortion and shrinkage jointly degrade
#        accuracy and overall MSE performance.
#      - EFFICIENT FRONTIER: Improvements in timeliness (T) and
#        smoothness (S) come at a disproportionate cost to accuracy
#        (A) and MSE, reflecting the inherent trilemma among the
#        three criteria for MDFA optimized predictors.
#
# 3. Filtered Series (bottom-right panel):
#      - The filtered series appears smoother and left-shifted,
#        but MSE performance deteriorates disproportionately.
#
# Note: The double-score design — simultaneously improving Smoothness (S) and
#       Timeliness (T) — is analyzed in depth through an extensive
#       out-of-sample simulation exercise in the next example 4.
# ==============================================================================



# ==============================================================================
# Example 4: Replicate an example in McElroy and Wildi (2019)
# ==============================================================================

# Compute A, T, S and MSE statistics


eta_vec<-c(0,1)
# Specify the fixed lambda
lambda_vec<-c(0,128)

# Specify the processes: ar(1) with coefficients -0.9,0.1 and 0.9
#   One can specify different processes at once and the code will loop through 
#   For simplicity we here use positive and nearly zero autocorrelation
a_vec<-c(0.9,0.08,-0.9)
# Ordinary ATS-components
scaled_ATS<-F
# Generate a single realization of the processes
anzsim<-1
# Use periodogram
mba<-F
estim_MBA<-T
L_sym<-1000
# Length of long data (for computing the target)
len1<-3000
# Length of in-sample span
len<-120
# Frequency grid: length of periodogram i.e. len/2
K<-len/2
# Periodicity
periodicity<-12
cutoff<-pi/periodicity
# Specify filter length
L<-2*periodicity
# Real-time design
Lag<-0
# no constraints
i1<-i2<-F
# Use original (not differenced) data
dif<-F


# Proceed to estimation
for_sim_obj<-for_sim_out(a_vec,len1,len,cutoff,L,mba,estim_MBA,L_sym,
                         Lag,i1,i2,scaled_ATS,lambda_vec,eta_vec,anzsim,K,dif)



# ATS-components
#   -Decomposition of MSE into Accuracy (A), Timeliness (T) and Smoothness (S), see trilemma paper Wildi/Mcelroy
#   -Customization here emphasizes T (lambda>0) and S (eta>0) simultaneously at cost of A
#   -As can be seen in table below: A (of customized design) degrades massiveley in favour of substantial improvements by T and S
#   -MSE (of customized) degrades obviously, since customization is a deliberate departure of MSE-performances
#     A degrades overproportionally when improving S/T
ats_sym_ST<-for_sim_obj$ats_sym
# 2 Curvature, Peak Correlation, ...
amp_shift_mat_sim<-for_sim_obj$amp_shift_mat_sim
# 3. Amplitude and time-shifts
amp_sim_per<-for_sim_obj$amp_sim_per
shift_sim_per<-for_sim_obj$shift_sim_per
# 4. Output series
xff_sim<-for_sim_obj$xff_sim
# 5. Peak correlation and Curvature
amp_shift_mat_sim_ST<-for_sim_obj$amp_shift_mat_sim
dim_names<-for_sim_obj$dim_names
# Process 1 (positive acf) or 2 (zero acf) 
DGP<-1
ats_sym_ST[-1,,DGP,1]

# Amplitude and time shift
par(mfrow=c(1,2))
mplot<-amp_sim_per[,-1,DGP,1]
dimnames(mplot)[[2]]<-paste("Amplitude (",lambda_vec,",",eta_vec,")",sep="")
ax<-rep(NA,ncol(mplot))
ax[1+(0:6)*((nrow(mplot)-1)/6)]<-c(0,"pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi")
plot_title<-paste("Amplitude functions: a1=",a_vec[DGP],sep="")
insamp<-1.e+90
title_more<-dimnames(mplot)[[2]]
colo<-rainbow(ncol(mplot))
mplot_func(mplot, ax, plot_title, title_more, insamp, colo)
mplot<-shift_sim_per[,-1,DGP,1]
dimnames(mplot)[[2]]<-paste("Time-shifts (",lambda_vec,",",eta_vec,")",sep="")
ax<-rep(NA,ncol(mplot))
ax[1+(0:6)*((nrow(mplot)-1)/6)]<-c(0,"pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi")
plot_title<-paste("Time-shifts: a1=",a_vec[DGP],sep="")
insamp<-1.e+90
title_more<-dimnames(mplot)[[2]]
colo<-rainbow(ncol(mplot))
mplot_func(mplot, ax, plot_title, title_more, insamp, colo)

par(mfrow=c(1,1))
series_vec<-c(2,3)
xf_per<-xff_sim[940:(940+len),,DGP,1]#dim(xff_sim)
dimnames(xf_per)[[2]]<-dim_names[[1]]
anf<-1
enf<-len
mplot<-scale(cbind(xf_per[,series_vec[1]],xf_per[,series_vec[2]])[anf:enf,])  #head(xf_per)
plot(as.ts(mplot[,1]),type="l",axes=F,col="red",ylim=c(min(na.exclude(mplot)),
                                                       max(na.exclude(mplot))),ylab="",xlab="",
     main=paste("MSE (red) vs. Customized (cyan): a1=",a_vec[DGP],sep=""),lwd=1)
mtext("MSE", side = 3, line = -1,at=(enf-anf)/2,col=colo[1])
i<-2
lines(as.ts(mplot[,i]),col=colo[2],lwd=2)
mtext(paste("Customized: ",dimnames(xf_per)[[2]][series_vec[2]],sep=""), side = 3, line = -i,at=(enf-anf)/2,col=colo[2])
axis(1,at=c(1,rep(0,6))+as.integer((0:6)*(enf-anf)/6),
     labels=as.integer(anf+(0:6)*(enf-anf)/6))
axis(2)
box()








# ================================================================================
# Example 5: Simulation Exercise — Interpreting Timeliness (T) and Smoothness (S)
# ================================================================================
# Motivation:
#   - The ATS criteria (Accuracy, Timeliness, Smoothness) are relatively new concepts,
#     and their practical interpretation may not be immediately intuitive.
#   - Rather than relying solely on T and S directly, we map them to two well-established,
#     scale-invariant alternative statistics that are easier to interpret:
#
#       1. PEAK CORRELATION (proxy for Timeliness, T):
#            - Shift the output of a one-sided filter relative to the target (output of the
#              symmetric filter) until the correlation between the two series is maximized.
#            - A smaller shift implies a faster filter (smaller lag).
#            - Scale invariant: rescaling the filter output does not affect this measure.
#            - Expectation: emphasizing T (lambda > 0) yields faster filters
#              (smaller shift at peak correlation).
#
#       2. RELATIVE CURVATURE (proxy for Smoothness, S):
#            - Smoothness of a series is measured via its mean-squared second-order differences,
#              normalized by the series variance (i.e., relative curvature).
#            - Strong noise leakage (poor stopband attenuation) produces noisy filter outputs
#              with large squared second-order differences (high curvature).
#            - Weak noise leakage (strong stopband attenuation) produces smooth outputs
#              with small squared second-order differences (low curvature).
#            - Scale invariant: rescaling the filter output does not affect this measure.
#            - Expectation: emphasizing S (eta > 0) yields smaller (better) curvature.
#
# ---------------------------------------------------------------------------------
# Experimental Design:
#
#   Data:
#     - Multiple realizations of three different AR(1) processes, covering positive,
#       near-zero, and negative autocorrelation structures (a_vec = 0.9, 0.08, -0.9).
#
#   Filters Compared:
#     1. SYMMETRIC TARGET FILTER:
#          - Two-sided filter (looks into the future); serves as the ideal reference.
#          - One-sided filters are expected to lag behind this target
#            (positive shift at peak correlation).
#
#     2. ONE-SIDED BENCHMARK (best possible):
#          - Best possible one-sided MSE filter, assuming full knowledge of the
#            true data-generating process (no estimation error).
#          - Serves as the gold standard for MSE performance.
#
#     3. EMPIRICAL DFA FILTERS (estimated from 120 in-sample observations via periodogram):
#          a. MSE design:       lambda = 0,   eta = 0    (baseline)
#          b. Smooth design:    lambda = 0,   eta = 1.5  (emphasizes S at cost of A and T)
#          c. Best-mix design:  lambda = 30,  eta = 1    (emphasizes T and S at cost of A)
#          d. Fast design:      lambda = 500, eta = 0.3  (emphasizes T at cost of A and S)
#
#   Performance Metrics (computed in-sample and out-of-sample for each realization):
#     1. Peak-correlation shift  (proxy for Timeliness)
#     2. Relative curvature      (proxy for Smoothness)
#     3. Time-domain MSE         (proxy for Accuracy)
#
#   Output:
#     - Empirical distributions (box-plots) of all three metrics,
#       shown separately for in-sample and out-of-sample evaluations.
#
# ---------------------------------------------------------------------------------
# Expectations:
#
#   1. MSE Designs:
#        - BEST POSSIBLE BENCHMARK will achieve the lowest out-of-sample MSE
#          among all designs.
#        - DFA-MSE (periodogram-based) will outperform all customized designs
#          in terms of out-of-sample MSE.
#        - DFA-MSE will outperform the best possible benchmark in-sample
#          (due to overfitting to the periodogram), but will underperform out-of-sample.
#
#   2. Customized Designs:
#        - FAST design (high lambda):
#            Best peak-correlation (smallest lag, ideally zero shift relative to target),
#            but noisier output (weaker smoothness) compared to the MSE design.
#
#        - SMOOTH design (high eta):
#            Best curvature (strongest noise suppression, smoothest output),
#            but larger lag compared to the MSE design.
#
#        - BEST-MIX design (balanced lambda and eta):
#            Achieves a DOUBLE SCORE — simultaneously outperforming the best possible
#            MSE benchmark in terms of both peak correlation (faster) and curvature
#            (smoother). This double score is unattainable from a pure MSE perspective.
#            The ATS trilemma enables joint improvements in T and S at the
#            expense of A (and MSE).
# ================================================================================

# --- Simulation Settings ---

# Number of realizations for computing empirical distributions
anzsim <- 100

# AR(1) coefficients: positive, near-zero, and negative autocorrelation
# These cover a wide range of applications
a_vec <- c(0.9, 0.08, -0.9)

# Filter designs (columns correspond to MSE, Smooth, Best-mix, Fast):
#   lambda controls timeliness emphasis
#   eta    controls smoothness emphasis
lambda_vec <- c(0,   0,   30,  500)
eta_vec    <- c(0,   1.5, 1,   0.3)

# Display design parameter table for reference
table_mat          <- rbind(lambda_vec, eta_vec)
colnames(table_mat) <- c("MSE", "Emphasize S only", "Emphasize S and T", "Emphasize T only")
rownames(table_mat) <- c("lambda", "eta")
table_mat

# Use raw (unscaled) ATS components (irrelevant for the outcome)
scaled_ATS <- FALSE

# Spectral estimation: use periodogram (non-parametric DFA, see tutorial 3 for background)
#   Advantage:
#     - No reliance on a parametric model of the data.
#   Disadvantages:
#     - More susceptible to overfitting than model-based alternatives.
#     - Outperformed by DFA based on the true model (in terms of MSE performances).
#   Note: the non-parametric DFA therefore faces an additional handicap
#         relative to the best possible MSE benchmark, which assumes full
#         knowledge of the true data-generating process (no estimation error).
mba       <- FALSE
estim_MBA <- TRUE

# Length of the symmetric (two-sided) reference filter
L_sym <- 1000

# Total series length (for computing target and out-of-sample performance)
len1 <- 3000

# In-sample span for DFA optimization (120 = 10 years of monthly data)
len <- 120

# Frequency grid: K must not exceed half the in-sample length (periodogram size)
K <- len / 2

# Target: ideal low-pass removing cycles shorter than 2 * periodicity
periodicity <- 12
cutoff      <- pi / periodicity

# Filter length: standard DFA setting (see earlier tutorials)
L <- 2 * periodicity

# Nowcast (no forecast horizon)
Lag <- 0

# No stationarity constraints (see Tutorial 5)
i1 <- i2 <- FALSE

# Use levels (not differenced data)
# Note: differencing would be appropriate for trending (non-stationary) series
dif <- FALSE

# --- Run Simulation ---
for_sim_obj <- for_sim_out(
  a_vec, len1, len, cutoff, L, mba, estim_MBA, L_sym, Lag,
  i1, i2, scaled_ATS, lambda_vec, eta_vec, anzsim, K, dif
)

# --- Extract Results ---
amp_shift_mat_sim <- for_sim_obj$amp_shift_mat_sim
amp_sim_per       <- for_sim_obj$amp_sim_per
shift_sim_per     <- for_sim_obj$shift_sim_per
xff_sim           <- for_sim_obj$xff_sim
xff_sim_sym       <- for_sim_obj$xff_sim_sym
ats_sym           <- for_sim_obj$ats_sym
dim_names         <- for_sim_obj$dim_names

# --- Plot Results ---
# Box-plots of MSE, peak-correlation, and curvature (in- and out-of-sample)
# for each data-generating process specified in a_vec
colo <- c("red", "orange", "yellow", "green", "blue")

Perf_meas_sel <- c(3, 7, 4, 8, 5, 9, 6, 10)

# First process: a1=0.9
DGP<-1

# Plot MSE and peak-correlation metrics (2x2 grid)
par(mfrow = c(2, 2))
for (Perf_meas in Perf_meas_sel[1:4])
{
  boxplot(
    list(
      amp_shift_mat_sim[1, Perf_meas, DGP, ],
      amp_shift_mat_sim[2, Perf_meas, DGP, ],
      amp_shift_mat_sim[3, Perf_meas, DGP, ],
      amp_shift_mat_sim[4, Perf_meas, DGP, ],
      amp_shift_mat_sim[5, Perf_meas, DGP, ]
    ),
    outline  = TRUE,
    names    = c("Best MSE", paste("(", lambda_vec, ",", eta_vec, ")", sep = "")),
    main     = paste(dim_names[[2]][Perf_meas], ", a1 =", a_vec[DGP]),
    cex.axis = 0.8,
    col      = colo
  )
}

# ==============================================================================
# Key Outcomes — Curvature (top row) and Peak-Correlation (bottom row),
#                in-sample (left) and out-of-sample (right):
#
# Focus: out-of-sample performance (right panels), which is more practically
#        relevant than in-sample performance.
#
# 1. BEST MSE / red (true model):
#      - Marginally outperforms non-parametric MSE-DFA (orange) in terms of
#        curvature (slightly lower); peak-correlation is virtually identical.
#
# 2. SMOOTH design / yellow (high eta, emphasizes S):
#      - Best curvature among all designs (top-right panel).
#      - Largest loss in timeliness: median lag ≈ 4 (bottom-right panel).
#
# 3. FAST design / blue (high lambda, emphasizes T):
#      - Best peak-correlation among all designs (bottom-right panel).
#      - Largest loss in smoothness: median curvature ≈ 0.3 (top-right panel).
#
# 4. BEST-MIX Design (Green) — Balanced lambda and eta, emphasizing T and S:
#
#    - DOUBLE OUTPERFORMANCE: Simultaneously surpasses the best (true model) MSE benchmark
#      (red) in both curvature (top-right panel) and peak correlation
#      (bottom-right panel).
#
#    - This outcome is particularly noteworthy: achieving superiority over the
#      best MSE benchmark on two fronts at once is impossible from a pure MSE
#      standpoint, underscoring the added value of the ATS trilemma framework.
#      The result is further strengthened by the fact that the underlying
#      predictor is non-parametric — a setting which favours flexibility
#      at the cost of estimation efficiency.
# ==============================================================================

  
# Plot MSE
par(mfrow = c(1, 2))
for (Perf_meas in Perf_meas_sel[5:6])
{
  boxplot(
    list(
      amp_shift_mat_sim[1, Perf_meas, DGP, ],
      amp_shift_mat_sim[2, Perf_meas, DGP, ],
      amp_shift_mat_sim[3, Perf_meas, DGP, ],
      amp_shift_mat_sim[4, Perf_meas, DGP, ],
      amp_shift_mat_sim[5, Perf_meas, DGP, ]
    ),
    outline  = TRUE,
    names    = c("Best MSE", paste("(", lambda_vec, ",", eta_vec, ")", sep = "")),
    main     = paste(dim_names[[2]][Perf_meas], ", a1 =", a_vec[DGP]),
    cex.axis = 0.8,
    col      = colo,
    notch    = FALSE
  )
}

# ==============================================================================
# Key Outcomes (out-of-sample right panels):
#
# - The best MSE predictor (based on the true model) achieves the lowest
#   out-of-sample MSE, outperforming all competing designs.
#
# - It is followed, in order, by the non-parametric MSE-DFA (orange),
#   the S-only design (yellow), and the T-only design (dark blue).
#
# - Placing weight on S and T (green) incurs disproportionately
#   large losses in A (and thus MSE).
# ==============================================================================


# Second process: a1~0
DGP<-2

# Plot MSE and peak-correlation metrics (2x2 grid)
par(mfrow = c(2, 2))
for (Perf_meas in Perf_meas_sel[1:4])
{
  boxplot(
    list(
      amp_shift_mat_sim[1, Perf_meas, DGP, ],
      amp_shift_mat_sim[2, Perf_meas, DGP, ],
      amp_shift_mat_sim[3, Perf_meas, DGP, ],
      amp_shift_mat_sim[4, Perf_meas, DGP, ],
      amp_shift_mat_sim[5, Perf_meas, DGP, ]
    ),
    outline  = TRUE,
    names    = c("Best MSE", paste("(", lambda_vec, ",", eta_vec, ")", sep = "")),
    main     = paste(dim_names[[2]][Perf_meas], ", a1 =", a_vec[DGP]),
    cex.axis = 0.8,
    col      = colo
  )
}

# Plot curvature metrics (1x2 grid)
par(mfrow = c(1, 2))
for (Perf_meas in Perf_meas_sel[5:6])
{
  boxplot(
    list(
      amp_shift_mat_sim[1, Perf_meas, DGP, ],
      amp_shift_mat_sim[2, Perf_meas, DGP, ],
      amp_shift_mat_sim[3, Perf_meas, DGP, ],
      amp_shift_mat_sim[4, Perf_meas, DGP, ],
      amp_shift_mat_sim[5, Perf_meas, DGP, ]
    ),
    outline  = TRUE,
    names    = c("Best MSE", paste("(", lambda_vec, ",", eta_vec, ")", sep = "")),
    main     = paste(dim_names[[2]][Perf_meas], ", a1 =", a_vec[DGP]),
    cex.axis = 0.8,
    col      = colo,
    notch    = FALSE
  )
}

# ==============================================================================
# Key Outcomes (white noise process)
#
# Similar to a1=0.9 (strong autocorrelation) above
# ==============================================================================


# Third process: a1=-0.9
DGP<-3

# Plot MSE and peak-correlation metrics (2x2 grid)
par(mfrow = c(2, 2))
for (Perf_meas in Perf_meas_sel[1:4])
{
  boxplot(
    list(
      amp_shift_mat_sim[1, Perf_meas, DGP, ],
      amp_shift_mat_sim[2, Perf_meas, DGP, ],
      amp_shift_mat_sim[3, Perf_meas, DGP, ],
      amp_shift_mat_sim[4, Perf_meas, DGP, ],
      amp_shift_mat_sim[5, Perf_meas, DGP, ]
    ),
    outline  = TRUE,
    names    = c("Best MSE", paste("(", lambda_vec, ",", eta_vec, ")", sep = "")),
    main     = paste(dim_names[[2]][Perf_meas], ", a1 =", a_vec[DGP]),
    cex.axis = 0.8,
    col      = colo
  )
}

# Plot curvature metrics (1x2 grid)
par(mfrow = c(1, 2))
for (Perf_meas in Perf_meas_sel[5:6])
{
  boxplot(
    list(
      amp_shift_mat_sim[1, Perf_meas, DGP, ],
      amp_shift_mat_sim[2, Perf_meas, DGP, ],
      amp_shift_mat_sim[3, Perf_meas, DGP, ],
      amp_shift_mat_sim[4, Perf_meas, DGP, ],
      amp_shift_mat_sim[5, Perf_meas, DGP, ]
    ),
    outline  = TRUE,
    names    = c("Best MSE", paste("(", lambda_vec, ",", eta_vec, ")", sep = "")),
    main     = paste(dim_names[[2]][Perf_meas], ", a1 =", a_vec[DGP]),
    cex.axis = 0.8,
    col      = colo,
    notch    = FALSE
  )
}

# ==============================================================================
# Key Outcomes (negative autocorrelation)
#
# Similar to a1=0,0.9 above
# ==============================================================================



# Comparison filter outputs
# Best MSE (red) vs. customized mixed (green)
# Customized is smoother and faster
DGP<-2
par(mfrow=c(1,1))
amp_shift_mat_sim<-for_sim_obj$amp_shift_mat_sim
amp_sim_per<-for_sim_obj$amp_sim_per
shift_sim_per<-for_sim_obj$shift_sim_per
xff_sim<-for_sim_obj$xff_sim
xff_sim_sym<-for_sim_obj$xff_sim_sym
ats_sym<-for_sim_obj$ats_sym
dim_names<-for_sim_obj$dim_names
xf_per<-xff_sim[940:(940+2*len),,DGP,10]
dimnames(xf_per)[[2]]<-dim_names[[1]]
anf<-1
enf<-2*len
mplot<-scale(cbind(xf_per[,1],xf_per[,4])[anf:enf,])  #head(xf_per)
plot(as.ts(mplot[,1]),type="l",axes=F,col="red",ylim=c(min(na.exclude(mplot)),
                                                       max(na.exclude(mplot))),ylab="",xlab="",
     main=paste("Benchmark MSE (red) vs. Customized balanced (green)",sep=""),lwd=2)
mtext("in sample",side = 3, line = -1,at=60,col="black")
mtext("out-of-sample",side = 3, line = -1,at=180,col="black")
mtext("Benchmark MSE", side = 3, line = -1,at=(enf-anf)/2,col="red")
i<-2
lines(as.ts(mplot[,i]),col=colo[4],lwd=2)
mtext(paste("Customized: ",dimnames(xf_per)[[2]][4],sep=""), side = 3, line = -i,at=(enf-anf)/2,col=colo[4])
abline(v=120)
axis(1,at=c(1,rep(0,6))+as.integer((0:6)*(enf-anf)/6),
     labels=as.integer(anf+(0:6)*(enf-anf)/6))
axis(2)
box()







# ==============================================================================
# Example 6: Bivariate Leading Indicator vs. Univariate Customized DFA
# ==============================================================================
#
# Background:
#   - See tutorial 4 (example 2) for details on the play_bivariate_func function.
#   - The bivariate design incorporates a leading indicator and was shown to
#     outperform the univariate DFA in terms of MSE and left-shift (advance), both in-sample (by
#     construction) and out-of-sample (as expected).
#   - It follows that the bivariate design will also outperform all univariate
#     customized designs in terms of MSE (in- and out-of-sample).
#   - But what about left-shift or advancement? 
#
# Research Question:
#   - Can the 'best-mix' customized univariate DFA outperform the bivariate MSE design
#     (with leading indicator) in terms of timeliness (lead) and smoothness
#     (curvature)?
#
# Experimental Design:
#   - The function mdfa_mse_leading_indicator_vs_dfa_customized sets up the
#     competition by comparing three designs:
#       1. Univariate DFA-MSE:          lambda = 0,  eta = 0  (baseline)
#       2. Univariate 'best-mix' DFA:   lambda = 30, eta = 1  (customized)
#       3. Bivariate MSE design:        incorporates a noisy leading indicator
# ==============================================================================


# --- Simulation Settings ---

# AR(1) coefficient (data generating process)
# We assume the data to be close to white noise 
# Typical assumption for differenced economic time series
a1 <- 0.08

# Number of simulation replications
anzsim <- 500

# Customization settings: MSE baseline and `mix': 
# Emphasize T (lambda=20) and S (eta=0.5)
lambda_vec <- c(0, 20)
eta_vec    <- c(0, 0.5)


# Target filter: typical target in business-cycle analysis with monthly data
# Ideal low-pass with specified periodicity and cutoff: filter removes all 
# components with duration < 2*12 months (2 years)
periodicity <- 12
cutoff      <- pi / periodicity

# Full sample length (for applying the symmetric ideal low-pass)
len1 <- 3000

# In-sample span (for spectrum estimation via DFT): 20 years of monthly observations
len <- 240
# Typical length setting (see previous tutorials)
L   <- 2 * periodicity

# Nowcast (no forecast horizon)
Lag <- 0

# No coefficient restrictions: series is stationary 
i1 <- i2 <- FALSE

# MDFA baseline: MSE design
#   These hyperparameters are applied to bivariate design
#   The above lambda_vec and eta_vec are applied to univariate DFA
lambda_mdfa <- eta_mdfa <- 0

# Speed-up flag: omits non-essential MDFA statistics (does not affect results)
troikaner <- FALSE

# --- Run the Competition ---
# Univariate MSE and customized DFA vs. bivariate leading indicator
cust_leading_obj <- mdfa_mse_leading_indicator_vs_dfa_customized(
  anzsim, a1, cutoff, L, lambda_vec, eta_vec,
  len1, len, i1, i2, Lag, lambda_mdfa, eta_mdfa, troikaner
)


# ==============================================================================
# Key Outcomes (assuming a1 = 0.08: near white noise, typical of log-returns
#               of positive economic time series):
#
# 1. Smoothness / Curvature:
#      - IN-SAMPLE:     Bivariate MSE (brown) beats DFA-MSE (orange), as expected. 
#                       But customized DFA (green) is even smoother and outperforms.
#      - OUT-OF-SAMPLE: same hierarchy is maintained
#
# 2. Timeliness / Lag at Peak Correlation:
#      - OUT-OF-SAMPLE: Best-mix customized (green) leads the bivariate design
#                       (brown) by approximately one time-point — a noteworthy
#                       and unexpected result.
#                       Bivariate (brown) leads univariate MSE (orange) by
#                       approximately one time-point, as expected given the
#                       leading indicator embedded in the bivariate design.
#
# 3. MSE:
#      - Results are fully as expected:
#          Bivariate MSE (brown) < Univariate MSE (orange) < Best-mix (green)
#        both in- and out-of-sample, confirming the inherent trilemma between
#        Accuracy (A), Timeliness (T), and Smoothness (S).
# ==============================================================================

# --- Plot 1: Curvature ---
par(mfrow = c(1, 2))
boxplot(
  list(
    cust_leading_obj$perf_in_sample[, 1, 1],
    cust_leading_obj$perf_in_sample[, 1, 2],
    cust_leading_obj$perf_in_sample[, 1, 3]
  ),
  outline = TRUE,
  names   = c(paste("DFA(", lambda_vec, ",", eta_vec, ")", sep = ""),
              "MDFA-MSE Leading Indicator"),
  main    = paste("Curvature in-sample, a1 =", a1),
  cex.axis = 0.8,
  col     = c("orange", "green", "brown")
)
boxplot(
  list(
    cust_leading_obj$perf_out_sample[, 1, 1],
    cust_leading_obj$perf_out_sample[, 1, 2],
    cust_leading_obj$perf_out_sample[, 1, 3]
  ),
  outline = TRUE,
  names   = c(paste("DFA(", lambda_vec, ",", eta_vec, ")", sep = ""),
              "MDFA-MSE Leading Indicator"),
  main    = paste("Curvature out-of-sample, a1 =", a1),
  cex.axis = 0.8,
  col     = c("orange", "green", "brown")
)

# ==============================================================================
# Outcomes (assuming a1 = 0.08: near white noise, typical of log-returns
#           of positive economic time series):
#
# Smoothness / Curvature (out-of-sample, right panels):
#
#    - The customized non-parametric DFA design (green) outperforms both the
#      bivariate MSE predictor (brown) and the univariate MSE-DFA (orange)
#      in terms of curvature.  
# ==============================================================================


# --- Plot 2: Lag at Peak Correlation ---
par(mfrow = c(1, 2))
boxplot(
  list(
    cust_leading_obj$perf_in_sample[, 2, 1],
    cust_leading_obj$perf_in_sample[, 2, 2],
    cust_leading_obj$perf_in_sample[, 2, 3]
  ),
  outline = TRUE,
  names   = c(paste("DFA(", lambda_vec, ",", eta_vec, ")", sep = ""),
              "MDFA-MSE Leading Indicator"),
  main    = paste("Peak-Correlation in-sample, a1 =", a1),
  cex.axis = 0.8,
  col     = c("orange", "green", "brown")
)
boxplot(
  list(
    cust_leading_obj$perf_out_sample[, 2, 1],
    cust_leading_obj$perf_out_sample[, 2, 2],
    cust_leading_obj$perf_out_sample[, 2, 3]
  ),
  outline = TRUE,
  names   = c(paste("DFA(", lambda_vec, ",", eta_vec, ")", sep = ""),
              "MDFA-MSE Leading Indicator"),
  main    = paste("Peak-Correlation out-of-sample, a1 =", a1),
  cex.axis = 0.8,
  col     = c("orange", "green", "brown")
)


# ==============================================================================
# Outcomes
#
# Timeliness / Lag at Peak Correlation:
#
#    - OUT-OF-SAMPLE: The median lag of the customized design (green) leads
#                    that of the bivariate design (brown) by one time-point —
#                    a noteworthy and unexpected result, given that the former
#                    does not exploit any leading indicator information.
#
#                    In turn, the bivariate design (brown) leads the univariate
#                    MSE predictor (orange) by one time-point, as expected,
#                    since the bivariate design explicitly incorporates a
#                    leading indicator.
#
# ==============================================================================


# --- Plot 3: MSE ---
boxplot(
  list(
    cust_leading_obj$perf_in_sample[, 3, 1],
    cust_leading_obj$perf_in_sample[, 3, 2],
    cust_leading_obj$perf_in_sample[, 3, 3]
  ),
  outline = TRUE,
  names   = c(paste("DFA(", lambda_vec, ",", eta_vec, ")", sep = ""),
              "MDFA-MSE Leading Indicator"),
  main    = paste("MSE in-sample, a1 =", a1),
  cex.axis = 0.8,
  col     = c("orange", "green", "brown")
)
boxplot(
  list(
    cust_leading_obj$perf_out_sample[, 3, 1],
    cust_leading_obj$perf_out_sample[, 3, 2],
    cust_leading_obj$perf_out_sample[, 3, 3]
  ),
  outline = TRUE,
  names   = c(paste("DFA(", lambda_vec, ",", eta_vec, ")", sep = ""),
              "MDFA-MSE Leading Indicator"),
  main    = paste("MSE out-of-sample, a1 =", a1),
  cex.axis = 0.8,
  col     = c("orange", "green", "brown")
)

# ==============================================================================
# Outcomes
#
# - The bivariate MSE predictor (brown) achieves the lowest MSE, as expected.
#
# - The customized DFA design (green) yields the worst overall performance,
#   incurring disproportionate losses in both A and MSE — a consequence of
#   placing excessive weight on S and T at the expense of accuracy.
#
# ==============================================================================



# --- Plot 4: Filter Outputs (Last Sample of Simulation) ---
par(mfrow = c(1, 1))
mplot <- scale(cust_leading_obj$filter_output_in_sample)
dimnames(mplot)[[2]] <- dimnames(cust_leading_obj$filter_output_in_sample)[[2]]

colo_cust <- c("orange", "green", "brown")

plot(
  as.ts(mplot[, 1]),
  type = "l",
  axes = FALSE,
  col  = colo_cust[1],
  ylim = c(min(na.exclude(mplot)), max(na.exclude(mplot))),
  ylab = "",
  xlab = "",
  main = "Filter Outputs: Last Realization",
  lwd  = 1
)
mtext(dimnames(mplot)[[2]][1], side = 3, line = -1,
      at = nrow(mplot) / 2, col = colo_cust[1])

for (i in 2:(ncol(mplot) - 1)) {
  lines(mplot[, i], col = colo_cust[i], lwd = 1)
  mtext(dimnames(mplot)[[2]][i], side = 3, line = -i,
        at = nrow(mplot) / 2, col = colo_cust[i])
}

axis(1, at     = c(1, rep(0, 6)) + as.integer((0:6) * nrow(mplot) / 6),
     labels = c(1, rep(0, 6)) + as.integer((0:6) * nrow(mplot) / 6))
axis(2)
box()














#---------------------------------------------------------------------------------------------------------
# Example 7: same as example 4 but we now allow for customization of the bivariate design
# Contenders in this competition: uni and bivariate MSE as well as uni and bivariate customized (the latter is new)


# Use the same settings as above but add a 'best-mix' customization for the bivariate filter
#   You might have to load the settings in the previous example in order to run this piece (if not done yet)
lambda_mdfa<-c(0,30)
eta_mdfa<-c(0,1.)


cust_leading_obj<-mdfa_mse_leading_indicator_vs_dfa_customized(anzsim,a1,
                                                               cutoff,L,lambda_vec,eta_vec,len1,len,i1,i2,Lag,lambda_mdfa,eta_mdfa,troikaner)  


colo<-rainbow(length(lambda_mdfa)+length(lambda_vec))
par(mfrow=c(1,2))
# 1. Curvature
#   in-sample
boxplot(list(cust_leading_obj$perf_in_sample[,1,1],cust_leading_obj$perf_in_sample[,1,2],cust_leading_obj$perf_in_sample[,1,3],cust_leading_obj$perf_in_sample[,1,4]),outline=T,names=c(paste("DFA(",lambda_vec,",",eta_vec,")",sep=""),paste("MDFA(",lambda_mdfa,",",eta_mdfa,")",sep="")),main=paste("Curvature in-sample, a1=",a1,sep=""),cex.axis=0.8,col=colo)
#   out-of-sample
boxplot(list(cust_leading_obj$perf_out_sample[,1,1],cust_leading_obj$perf_out_sample[,1,2],cust_leading_obj$perf_out_sample[,1,3],cust_leading_obj$perf_out_sample[,1,4]),outline=T,names=c(paste("DFA(",lambda_vec,",",eta_vec,")",sep=""),paste("MDFA(",lambda_mdfa,",",eta_mdfa,")",sep="")),main=paste("Curvature out-of-sample, a1=",a1,sep=""),cex.axis=0.8,col=colo)



# 2. Peak correlation 
#   in-sample
boxplot(list(cust_leading_obj$perf_in_sample[,2,1],cust_leading_obj$perf_in_sample[,2,2],cust_leading_obj$perf_in_sample[,2,3],cust_leading_obj$perf_in_sample[,2,4]),outline=T,names=c(paste("DFA(",lambda_vec,",",eta_vec,")",sep=""),paste("MDFA(",lambda_mdfa,",",eta_mdfa,")",sep="")),main=paste("Peak Correlation in-sample, a1=",a1,sep=""),cex.axis=0.8,col=colo)
#   out-of-sample 
boxplot(list(cust_leading_obj$perf_out_sample[,2,1],cust_leading_obj$perf_out_sample[,2,2],cust_leading_obj$perf_out_sample[,2,3],cust_leading_obj$perf_out_sample[,2,4]),outline=T,names=c(paste("DFA(",lambda_vec,",",eta_vec,")",sep=""),paste("MDFA(",lambda_mdfa,",",eta_mdfa,")",sep="")),main=paste("Peak Correlation out-of-sample, a1=",a1,sep=""),cex.axis=0.8,col=colo)


# 3. MSE 
#   in sample
boxplot(list(cust_leading_obj$perf_in_sample[,3,1],cust_leading_obj$perf_in_sample[,3,2],cust_leading_obj$perf_in_sample[,3,3],cust_leading_obj$perf_in_sample[,3,4]),outline=T,names=c(paste("DFA(",lambda_vec,",",eta_vec,")",sep=""),paste("MDFA(",lambda_mdfa,",",eta_mdfa,")",sep="")),main=paste("MSE in-sample, a1=",a1,sep=""),cex.axis=0.8,col=colo)
#   out-of-sample 
boxplot(list(cust_leading_obj$perf_out_sample[,3,1],cust_leading_obj$perf_out_sample[,3,2],cust_leading_obj$perf_out_sample[,3,3],cust_leading_obj$perf_out_sample[,3,4]),outline=T,names=c(paste("DFA(",lambda_vec,",",eta_vec,")",sep=""),paste("MDFA(",lambda_mdfa,",",eta_mdfa,")",sep="")),main=paste("MSE out-of-sample, a1=",a1,sep=""),cex.axis=0.8,col=colo)

# Multivariate customized 
#   outperforms all other contenders with respect to lead and curvature out-of-sample (though outperformance with respect to customized DFA is modest)
#   outperforms customized univariate in terms of MSE out-of-sample
#   is outperformed in terms of MSE by both MSE-designs out-of-sample
# Conclusion: 
#   1. Performances in terms of Smoothess (smaller curvature) and Timeliness (smaller lag at peak correlation) are
#     obtained mainly by design, i.e. by the art of optimizing the relevant features of the filter, rather than by adding 
#     a (cheating...) leading time series.
#   2. MSE-performances may degrade substantially as a side-effect of addressing S and T at the expense of A
#     but some users (me included) really don't care about that collateral damage to MSE





#---------------------------------------------------------------------------------------------------------------------------------
# Wrap-up
# -The MSE-norm can be split into four error components, Accuracy, Timeliness, Smoothness and Residual, which are weighted equally in the original MSE norm
#   -S measures the noise suppression by the one-sided filter in the stopband
#   -T measures the shift (delay) of the one-sided filter in the passband
#   -A measures the level-tracking of the one-sided filter in the passband
#   -R measures the contribution of the shift in the stopband: in our applications this component vanishes invariably (because the target vanishes in the stopband)
#   -The remaining A, T and S account for the dilemma of optimizing amplitude (A and S) as well as phase (T) functions, recall tutorial 2 example 5
#     -One cannot improve both amplitude and phase functions fits simultaneously and arbitrarily well across the full frequency-band
# -Playing with the trilemma
#   -Emphasizing any (A-, T- or S-) error component (assigning unilaterally a larger weight to that component uniquely) inflates the other ones as well as the sum (i.e. MSE)
#   -Emphasizing either combination of two components (assigning larger weights to these two components) inflates overprortionally the remaining one as well as MSE
#   -Emphasizing S and T (at costs of A) improves simultaneously the amplitude (noise suppression in stop-band) as well as the phase (shift in the passband)
#      at cost of amplitude in passband (A or Accuracy-component: accounts for level-tracking ability of the one-sided filter) 
#   -Therefore amplitude and phase function fits can be improved simultaneously on parts of the frequency band: the stop-band (amplitude) and the passband (phase)   
#   -Classic econometric approaches are immanently incapable of tackling that problem because the classic maximum likelihood approach emphasizes a 
#     (one-step ahead) forecast problem; the forecasting target is an allpass filter i.e. there is no split of the frequency-band into pass- and stopbands
#     or, stated otherwise, S (smoothness) does not exist.
# -A comparison with classic econometric/time series approaches
#   -Classic one-step ahead mean-square forecasting decomposes the MSE-norm into A and T only (there is no S): classic econometric approaches emphasize a 
#     dilemma; they are by design incapable of addressing the proposed trilemma
#   -By tracking arbitrary targets (see tutorial 2) the DFA is more general than classic econometric approaches
#     1. DFA can replicate classic (one-steap ahead allpass) approaches, see tutorial 1
#     2. By allowing more generic targets (lowpass, bandpass) a trilemma can be spanned upon pass- and stop-bands of the target
#     3. The trilemma enables to address amplitude AND phase fits simultaneously in relevant frequency-bands
# -Interpretation of S and T
#   -Smoothness is intimately related to the classic curvature statistic (mean-square second order differences) which measures ... well... the curvature (i.e. smoothness) of the filter output
#   -Timeliness is intimately related to the classic peak-correlation concept (shift outputs of one-sided and of target filters until correlation is maximized)
#   -Since S and T can be improved simulatenously in the ATS-trilemma, at costs of A and overall MSE, we conclude that curvature and lag (at peak-correlation) 
#     can be improved simultenously, too.
#   -Our simulation studies above confirm this claim, in-sample as well as out-of-sample
#     -Curvature as well as lag at peak-correlation can be improved both substantially i.e. improvements are not marginal
#     -But A(ccuracy) and MSE-performances degrade: 
#       -This loss is mainly due to shrinkage of the amplitude function in the passband 
#       -The shrinkage could be remedied easily, at least to some extent, by re-scaling of the filter output. 
#       -Stated otherwise: part of the loss in A- and MSE-performances could be overcome by a very simple transformation which does not affect 
#         the scale-invariant (relative) curvature and lag at peak-correlation measures
# -Selecting lambda,eta
#   -In contrast to the previous regularization feature, for which 'best' weights could be derived in terms of smallest tic (see tutorial 5), 
#     customization of a filter by means of lambda (Timeliness) and eta (Smoothness) cannot claim uncontroversial 'optimality'
#   -As suggested by its naming, customization (the selection of lambda/eta) depends mainly (solely) on the user or, more precisely, on the purpose of the analysis
#   -The 'utility' of a particular lambda/eta setting belies in the mindset of its user
# -Summary of empirical studies
#   -MSE designs outperform customized designs in- and out-of-sample, as expected
#   -The bivariate leading indicator MSE-design outperforms all other contenders in-sample and out-of-sample in terms of MSE (assuming overfitting is not excessively heavy)
#   -A univariate suitably customized DFA filter can outperform all MSE-designs in terms of curvature and (lag at) peak-correlation simultaneously in-sample and out-of-sample
#     1. the best possible univariate MSE-approach assuming knowledge of the true data-generating process is outperformed
#     2. more surprisingly, perhaps, the bivariate MSE leading indicator design is outperformed: this last result confirms that gains can be substantial in both dimensions at once



# -Cautionary words: avoid confusions
#   -All reported measures are aggregates of stochastic events
#   -Improved S and T or, equivalently, improved curvature and peak correlation imply that the corresponding filter output improves 'in the mean'
#     -At some turning-points the lead can be larger or smaller (than indicated by the aggregate peak-correlation number): there is variation (no determinism)
#     -Sometimes the output is less (or more) smooth than assumed by the aggregate curvature number
#     -Note that usage of a leading indicator in the bivariate design leads to 'more determinism' i.e. the anticipation (by exactly one time point) is less random or more regular than for the customized univariate design
#   -Applying a customized filter to white noise can (and does) not improve our inferential ability about the (completely random) future
#     -Neither forecasting nor any derived statistic (for example trading performances based on the sign of the filter output) can be improved
#     -In the mean, the customized filter will cross the zero line earlier than the MSE-filter: this will indeed be observed
#     -But this feature would be completely 'useless' in the context of an iid process (independent identically distributed): no utility could be derived for the user
#     -In contrast, usage of the leading indicator in the bivariate design would have lead to substantial trading performance (due to effective non-causality of the design) 
#   -But all is not lost... in real-world markets liquidity is finite (adjustments are not immediate) and a substantial share of traders are relying on classic ('slow') MA-filters. 
#     -In such a context, improved timing by customized designs could deliver 
#     -However, experience suggests that faster is not always better
#     -In any case, the ATS-trilemma enables to trigger whatever is felt judicious: now it's up to the user...




