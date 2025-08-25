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

# ===== RECRUITMENT ANALYSIS (NEW) =====

cat("\n\n========== RECRUITMENT ANALYSIS ==========\n")

# ===== TOTAL RECRUITMENT ANALYSIS =====
cat("\n--- TOTAL RECRUITMENT ANALYSIS ---\n")

# Load total recruitment data
recruitment_total <- read_data("recruitment_df_total.csv")
recruitment_total$stim <- as.logical(recruitment_total$stim)
cat(sprintf("Loaded total recruitment data: %d observations\n", nrow(recruitment_total)))
cat(sprintf("Patients: %d\n", length(unique(recruitment_total$patient))))
cat(sprintf("Stim seizures: %d, Spontaneous seizures: %d\n", 
            sum(recruitment_total$stim), sum(!recruitment_total$stim)))

# LME model for total recruitment: Recruitment ~ stim + (1|patient)
cat("\n--- Total Recruitment: LME Model with Satterthwaite correction ---\n")
model_total_recruitment <- lmer(Recruitment ~ stim + (1|patient), data = recruitment_total, REML=FALSE)
print(summary(model_total_recruitment)$coefficients)
cat("Random effect variance:\n")
print(VarCorr(model_total_recruitment), comp="Variance")

# ===== ONSET RECRUITMENT ANALYSIS =====
cat("\n--- ONSET RECRUITMENT ANALYSIS ---\n")

# Load onset recruitment data
recruitment_onset <- read_data("recruitment_df_onset.csv")
recruitment_onset$stim <- as.logical(recruitment_onset$stim)
cat(sprintf("Loaded onset recruitment data: %d observations\n", nrow(recruitment_onset)))
cat(sprintf("Patients: %d\n", length(unique(recruitment_onset$patient))))
cat(sprintf("Stim seizures: %d, Spontaneous seizures: %d\n", 
            sum(recruitment_onset$stim), sum(!recruitment_onset$stim)))

# LME model for onset recruitment: Recruitment ~ stim + (1|patient)
cat("\n--- Onset Recruitment: LME Model with Satterthwaite correction ---\n")
model_onset_recruitment <- lmer(Recruitment ~ stim + (1|patient), data = recruitment_onset, REML=FALSE)
print(summary(model_onset_recruitment)$coefficients)
cat("Random effect variance:\n")
print(VarCorr(model_onset_recruitment), comp="Variance")

# ===== SPREAD RECRUITMENT ANALYSIS =====
cat("\n--- SPREAD RECRUITMENT ANALYSIS ---\n")

# Load spread recruitment data
recruitment_spread <- read_data("recruitment_df_spread.csv")
recruitment_spread$stim <- as.logical(recruitment_spread$stim)
cat(sprintf("Loaded spread recruitment data: %d observations\n", nrow(recruitment_spread)))
cat(sprintf("Patients: %d\n", length(unique(recruitment_spread$patient))))
cat(sprintf("Stim seizures: %d, Spontaneous seizures: %d\n", 
            sum(recruitment_spread$stim), sum(!recruitment_spread$stim)))

# LME model for spread recruitment: Recruitment ~ stim + (1|patient)
cat("\n--- Spread Recruitment: LME Model with Satterthwaite correction ---\n")
model_spread_recruitment <- lmer(Recruitment ~ stim + (1|patient), data = recruitment_spread, REML=FALSE)
print(summary(model_spread_recruitment)$coefficients)
cat("Random effect variance:\n")
print(VarCorr(model_spread_recruitment), comp="Variance")

# ===== SUMMARY OF RECRUITMENT EFFECTS =====
cat("\n--- SUMMARY OF RECRUITMENT EFFECTS ---\n")

# Extract coefficients and p-values for stim effect in each model
total_stim_coef <- summary(model_total_recruitment)$coefficients["stimTRUE", "Estimate"]
total_stim_p <- summary(model_total_recruitment)$coefficients["stimTRUE", "Pr(>|t|)"]

onset_stim_coef <- summary(model_onset_recruitment)$coefficients["stimTRUE", "Estimate"]
onset_stim_p <- summary(model_onset_recruitment)$coefficients["stimTRUE", "Pr(>|t|)"]

spread_stim_coef <- summary(model_spread_recruitment)$coefficients["stimTRUE", "Estimate"]
spread_stim_p <- summary(model_spread_recruitment)$coefficients["stimTRUE", "Pr(>|t|)"]

cat(sprintf("Total Recruitment: Stim effect = %.3f, p = %.6f\n", total_stim_coef, total_stim_p))
cat(sprintf("Onset Recruitment: Stim effect = %.3f, p = %.6f\n", onset_stim_coef, onset_stim_p))
cat(sprintf("Spread Recruitment: Stim effect = %.3f, p = %.6f\n", spread_stim_coef, spread_stim_p))

# ===== OLS ANALYSES (NO RANDOM EFFECTS) ===== 
