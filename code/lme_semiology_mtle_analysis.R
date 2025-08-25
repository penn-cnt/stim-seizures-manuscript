# Install packages if needed:
# install.packages("lme4")
# install.packages("lmerTest")
# install.packages("pbkrtest")

library(lme4)
library(lmerTest)  # for Satterthwaite DOF
library(pbkrtest) # for KR DOF
library(jsonlite)

# =============================================================================
# CONFIGURATION - LOAD FROM JSON CONFIG FILE
# =============================================================================
# Function to load configuration from JSON file created by Python config
load_config <- function(config_path = NULL) {
  # Default path - look for r_config.json in common locations
  if (is.null(config_path)) {
    possible_paths <- c(
      # Look in the parent directory's data folder
      file.path(dirname(getwd()), "stim_seizures_data", "METADATA", "r_config.json"),
      # Look in the CNT directory
      "/Users/wojemann/Documents/CNT/stim_seizures_data/METADATA/r_config.json",
      # Look in current directory
      "r_config.json"
    )
    
    for (path in possible_paths) {
      if (file.exists(path)) {
        config_path <- path
        break
      }
    }
  }
  
  if (is.null(config_path) || !file.exists(config_path)) {
    stop(paste("Configuration file not found. Please run the Python config script first:",
               "\npython config.py",
               "\nThis will create the r_config.json file that R needs to read."))
  }
  
  config <- jsonlite::fromJSON(config_path)
  cat("Configuration loaded from:", config_path, "\n")
  return(config)
}

# Load configuration
config <- load_config()
prodatapath <- config$prodatapath
metapath <- config$metapath
figpath <- config$figpath

cat("Configuration:\n")
cat("Processed data path:", prodatapath, "\n")
cat("Metadata path:", metapath, "\n")
cat("Figure path:", figpath, "\n\n")

# Helper function to read CSV files from processed data
read_data <- function(filename) {
  filepath <- file.path(prodatapath, filename)
  if (!file.exists(filepath)) {
    warning(paste("File not found:", filepath))
    return(NULL)
  }
  return(read.csv(filepath, stringsAsFactors = FALSE))
}

# Read the data
df <- read_data("modeling_df.csv")

# Binarize age, duration, and age_at_onset at their medians
cat("\n--- Binarizing age, duration, and age_at_onset at their medians ---\n")
df$age_bin <- as.integer(df$age > median(df$age, na.rm=TRUE))
df$duration_bin <- as.integer(df$duration > median(df$duration, na.rm=TRUE))
df$age_at_onset_bin <- as.integer(df$age_at_onset > median(df$age_at_onset, na.rm=TRUE))

# 1. Typical vs. Atypical Stim Seizures (KR-corrected p-value)
cat("\n--- Typical vs. Atypical Stim Seizures (Satterthwaite-corrected p-value) ---\n")
cat("Fixed effects: Regression coefficients (Estimate), standard errors, Kenward-Roger degrees of freedom, t-values, and p-values for each predictor.\n")
model_typical <- lmer(MCC ~ typical + (1|patient), data = df, REML=FALSE)
print(summary(model_typical))
cat("Random effect variance: Variance attributed to the random intercept for patient, and residual variance.\n")
# print(VarCorr(model_typical),comp="Variance")

# 2. Typical vs. Typical x Center (KR LRT)
cat("\n--- Typical vs. Typical x Center (KR LRT) ---\n")
cat("KR LRT: F-test statistic, numerator and denominator degrees of freedom (Kenward-Roger), scaling factor, and p-value for the interaction between typical and center.\n")
model_typical_center <- lmer(MCC ~ typical * adult + (1|patient), data = df, REML=FALSE)
print(summary(model_typical_center))
kr_test_interaction <- KRmodcomp(model_typical_center, model_typical)
print(kr_test_interaction)

# 3. LME Coefficients for various models (Standard t-tests)
# a. MTLE
cat("\n--- MTLE Model ---\n")
cat("Fixed effects: Regression coefficients, standard errors, Satterthwaite degrees of freedom, t-values, and p-values for each predictor.\n") # nolint
model_mtle <- lmer(MCC ~ mtle + (1|patient), data = df, REML=FALSE)
# print(summary(model_mtle)$coefficients)
cat("Random effect variance: Variance attributed to the random intercept for patient, and residual variance.\n") # nolint
print(VarCorr(model_mtle),comp="Variance")


# b. Center x MTLE
cat("\n--- Adult x MTLE Model ---\n")
cat("Fixed effects: Regression coefficients, standard errors, Satterthwaite degrees of freedom, t-values, and p-values for each predictor.\n") # nolint
model_center_mtle <- lmer(MCC ~ adult * mtle + (1|patient), data = df, REML=FALSE)
# print(summary(model_center_mtle)$coefficients)
cat("Random effect variance: Variance attributed to the random intercept for patient, and residual variance.\n") # nolint
print(VarCorr(model_center_mtle),comp="Variance")

# c. Duration x MTLE (binarized)
cat("\n--- Duration_bin x MTLE Model ---\n")
cat("Fixed effects: Regression coefficients, standard errors, Satterthwaite degrees of freedom, t-values, and p-values for each predictor.\n") # nolint
model_durationbin_mtle <- lmer(MCC ~ duration_bin * mtle + (1|patient), data = df, REML=FALSE)
# print(summary(model_durationbin_mtle)$coefficients)
cat("Random effect variance: Variance attributed to the random intercept for patient, and residual variance.\n") # nolint
print(VarCorr(model_durationbin_mtle),comp="Variance")

# d. Age x MTLE (binarized)
cat("\n--- Age_bin x MTLE Model ---\n")
cat("Fixed effects: Regression coefficients, standard errors, Satterthwaite degrees of freedom, t-values, and p-values for each predictor.\n") # nolint
model_agebin_mtle <- lmer(MCC ~ age_bin * mtle + (1|patient), data = df, REML=FALSE)
# print(summary(model_agebin_mtle)$coefficients)
cat("Random effect variance: Variance attributed to the random intercept for patient, and residual variance.\n") # nolint
print(VarCorr(model_agebin_mtle),comp="Variance")

# e. Age at Onset x MTLE (binarized and unbinarized)
cat("\n--- Age at Onset_bin x MTLE Model ---\n")
cat("Fixed effects: Regression coefficients, standard errors, Satterthwaite degrees of freedom, t-values, and p-values for each predictor.\n") # nolint
model_ageonsetbin_mtle <- lmer(MCC ~ age_at_onset_bin * mtle + (1|patient), data = df, REML=FALSE)
# print(summary(model_ageonsetbin_mtle)$coefficients)
cat("Random effect variance: Variance attributed to the random intercept for patient, and residual variance.\n") # nolint
print(VarCorr(model_ageonsetbin_mtle),comp="Variance")

# ===== OLS ANALYSES (NO RANDOM EFFECTS) =====
cat("\n\n========== OLS ANALYSES (NO RANDOM EFFECTS) ==========\n")

# OLS Coefficients for various models (Standard t-tests)
# a. MTLE
cat("\n--- MTLE Model (OLS) ---\n")
cat("OLS coefficients: Regression coefficients, standard errors, t-values, and p-values for each predictor (no random effects).\n") # nolint
model_mtle_ols <- lm(MCC ~ mtle, data = df)
print(summary(model_mtle_ols))

# b. Adult x MTLE
cat("\n--- Adult x MTLE Model (OLS) ---\n")
cat("OLS coefficients: Regression coefficients, standard errors, t-values, and p-values for each predictor (no random effects).\n") # nolint
model_adult_mtle_ols <- lm(MCC ~ adult * mtle, data = df)
print(summary(model_adult_mtle_ols))

# c. Duration x MTLE (binarized)
cat("\n--- Duration_bin x MTLE Model (OLS) ---\n")
cat("OLS coefficients: Regression coefficients, standard errors, t-values, and p-values for each predictor (no random effects).\n") # nolint
model_durationbin_mtle_ols <- lm(MCC ~ duration_bin * mtle, data = df)
print(summary(model_durationbin_mtle_ols))

# d. Age x MTLE (binarized)
cat("\n--- Age_bin x MTLE Model (OLS) ---\n")
cat("OLS coefficients: Regression coefficients, standard errors, t-values, and p-values for each predictor (no random effects).\n") # nolint
model_agebin_mtle_ols <- lm(MCC ~ age_bin * mtle, data = df)
print(summary(model_agebin_mtle_ols))

# e. Age at Onset x MTLE (binarized)
cat("\n--- Age at Onset_bin x MTLE Model (OLS) ---\n")
cat("OLS coefficients: Regression coefficients, standard errors, t-values, and p-values for each predictor (no random effects).\n") # nolint
vars_needed <- c("MCC", "mtle", "age_at_onset_bin", "patient")
df_complete <- df[complete.cases(df[, vars_needed]), ]

model_mtle_cc <- lm(MCC ~ mtle, data = df_complete)
model_ageonsetbin_mtle_ols <- lm(MCC ~ age_at_onset_bin * mtle, data = df_complete)
print(summary(model_ageonsetbin_mtle_ols))


# --- Alternative contrast for Adult x MTLE (OLS) ---
cat("\n--- Alternative contrast: adultTrue + adultTrue:mtleTrue (OLS) ---\n")
# Extract coefficients and covariance matrix
coefs_adult_mtle <- coef(summary(model_adult_mtle_ols))
cov_adult_mtle <- vcov(model_adult_mtle_ols)
# Estimate: sum of main effect and interaction
estimate_adult_mtle <- coef(model_adult_mtle_ols)["mtleTrue"] + coef(model_adult_mtle_ols)["adultTrue:mtleTrue"]
# Variance: var1 + var3 + 2*cov13
var1_adult_mtle <- cov_adult_mtle["mtleTrue", "mtleTrue"]
var3_adult_mtle <- cov_adult_mtle["adultTrue:mtleTrue", "adultTrue:mtleTrue"]
cov13_adult_mtle <- cov_adult_mtle["mtleTrue", "adultTrue:mtleTrue"]
se_adult_mtle <- sqrt(var1_adult_mtle + var3_adult_mtle + 2 * cov13_adult_mtle)
t_stat_adult_mtle <- estimate_adult_mtle / se_adult_mtle
df_adult_mtle <- model_adult_mtle_ols$df.residual
p_value_adult_mtle <- 2 * pt(-abs(t_stat_adult_mtle), df = df_adult_mtle)
cat(sprintf("beta: %.4f, se: %.4f, t-stat: %.4f, p-value: %.4g\n", estimate_adult_mtle, se_adult_mtle, t_stat_adult_mtle, p_value_adult_mtle))

center_ps = c(coefs_adult_mtle["mtleTrue","Pr(>|t|)"],coefs_adult_mtle["adultTrue:mtleTrue","Pr(>|t|)"],p_value_adult_mtle)
cat("Adult x MTLE Model (OLS) Bonferroni-adjusted p-values:")
cat("\n","mtleTrue","adultTrue:mtleTrue","p_value_adult_mtle","\n")
print(p.adjust(center_ps,method="bonferroni"))

# --- Alternative contrast for Age at Onset_bin x MTLE (OLS) ---
cat("\n--- Alternative contrast: age_at_onset_bin + age_at_onset_bin:mtleTrue (OLS) ---\n")
coefs_onset_mtle <- coef(summary(model_ageonsetbin_mtle_ols))
cov_onset_mtle <- vcov(model_ageonsetbin_mtle_ols)
estimate_onset_mtle <- coef(model_ageonsetbin_mtle_ols)["mtleTrue"] + coef(model_ageonsetbin_mtle_ols)["age_at_onset_bin:mtleTrue"]
var1_onset_mtle <- cov_onset_mtle["mtleTrue", "mtleTrue"]
var3_onset_mtle <- cov_onset_mtle["age_at_onset_bin:mtleTrue", "age_at_onset_bin:mtleTrue"]
cov13_onset_mtle <- cov_onset_mtle["mtleTrue", "age_at_onset_bin:mtleTrue"]
se_onset_mtle <- sqrt(var1_onset_mtle + var3_onset_mtle + 2 * cov13_onset_mtle)
t_stat_onset_mtle <- estimate_onset_mtle / se_onset_mtle
df_onset_mtle <- model_ageonsetbin_mtle_ols$df.residual
p_value_onset_mtle <- 2 * pt(-abs(t_stat_onset_mtle), df = df_onset_mtle)
cat(sprintf("beta: %.4f, se: %.4f, t-stat: %.4f, p-value: %.4g\n", estimate_onset_mtle, se_onset_mtle, t_stat_onset_mtle, p_value_onset_mtle))
age_ps = c(coefs_onset_mtle["mtleTrue","Pr(>|t|)"],coefs_onset_mtle["age_at_onset_bin:mtleTrue","Pr(>|t|)"],p_value_onset_mtle)
cat("Age at Onset_bin x MTLE Model (OLS) Bonferroni-adjusted p-values:")
cat("\n","mtleTrue","age_at_onset_bin:mtleTrue","p_value_onset_mtle","\n")
print(p.adjust(age_ps,method="bonferroni"))

# ===== F-TESTS FOR OLS MODEL COMPARISONS =====

# Compare adult*mtle interaction model to mtle-only model
cat("\n--- F-test: Adult x MTLE vs. MTLE-only Model (OLS) ---\n")
cat("F-test: Compares nested OLS models. Shows residual degrees of freedom, residual sum of squares, difference in degrees of freedom, difference in sum of squares, F statistic, and p-value.\n") # nolint
f_test_adult_mtle <- anova(model_mtle_ols, model_adult_mtle_ols)
print(f_test_adult_mtle)

# Compare age_at_onset_bin*mtle interaction model to mtle-only model
cat("\n--- F-test: Age at Onset_bin x MTLE vs. MTLE-only Model (OLS) ---\n")
cat("F-test: Compares nested OLS models. Shows residual degrees of freedom, residual sum of squares, difference in degrees of freedom, difference in sum of squares, F statistic, and p-value.\n") # nolint
f_test_onset_mtle <- anova(model_mtle_cc, model_ageonsetbin_mtle_ols)
print(f_test_onset_mtle)

# ===== OLS MODEL FIT STATISTICS =====

# Model fit statistics for age-based and duration-based models
cat("\n--- Model Fit Statistics for OLS Models ---\n")
cat("For each OLS model: AIC (Akaike Information Criterion), BIC (Bayesian Information Criterion), R-squared (proportion of variance explained), and adjusted R-squared.\n") # nolint 
age_duration_models_ols <- list(
  mtle_only = model_mtle_ols,
  age_bin_x_mtle = model_agebin_mtle_ols,
  age_at_onset_bin_x_mtle = model_ageonsetbin_mtle_ols,
  duration_bin_x_mtle = model_durationbin_mtle_ols,
  adult_x_mtle = model_adult_mtle_ols
)

for (name in names(age_duration_models_ols)) {
  m <- age_duration_models_ols[[name]]
  cat(sprintf("\nModel: %s\n", name))
  cat(sprintf("  AIC: %.2f\n", AIC(m)))
  cat(sprintf("  BIC: %.2f\n", BIC(m)))
  cat(sprintf("  R-squared: %.3f\n", summary(m)$r.squared))
  cat(sprintf("  Adj R-squared: %.3f\n", summary(m)$adj.r.squared))
}
