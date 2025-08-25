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

# Set working directory (adjust path as needed)
# setwd("/mnt/sauce/littlab/users/wojemann/stim-seizures")

# Load the data
time_df_all <- read_data("time_df_all.csv")
spread_df_all <- read_data("spread_df_all.csv") 

# Check data structure
cat("=== DATA STRUCTURE ===\n")
cat("Time DF dimensions:", dim(time_df_all), "\n")
cat("Spread DF dimensions:", dim(spread_df_all), "\n")

cat("\nTime DF columns:", colnames(time_df_all), "\n")
cat("Spread DF columns:", colnames(spread_df_all), "\n")

# Convert typical to factor
time_df_all$typical <- as.factor(time_df_all$typical)
spread_df_all$typical <- as.factor(spread_df_all$typical)

# Check levels
cat("\nTypical levels in time_df_all:", levels(time_df_all$typical), "\n")
cat("Typical levels in spread_df_all:", levels(spread_df_all$typical), "\n")

# Summary statistics
cat("\n=== SUMMARY STATISTICS ===\n")
cat("Time DF - Typical distribution:\n")
print(table(time_df_all$typical))

cat("\nSpread DF - Typical distribution:\n") 
print(table(spread_df_all$typical))

cat("\n=== LINEAR MIXED EFFECTS MODELS ===\n")

# Analysis 1: Seizure timing (onset_med_25) comparing Atypical vs Typical
cat("\n--- ANALYSIS 1: SEIZURE TIMING (onset_med_25) ---\n")

# Filter to only Atypical and Typical (exclude "All" if present)
time_df_filtered <- time_df_all[time_df_all$typical %in% c("Atypical", "Typical"), ]
time_df_filtered$typical <- droplevels(time_df_filtered$typical)

# Set Typical as reference level (so coefficient represents Atypical effect)
time_df_filtered$typical <- relevel(time_df_filtered$typical, ref = "Typical")

cat("Sample sizes after filtering:\n")
print(table(time_df_filtered$typical))

# Fit linear mixed effects model: onset_med_25 ~ typical + (1|patient)
timing_model <- lmer(onset_med_25 ~ typical + (1|patient), data = time_df_filtered, REML=FALSE)

cat("\nTiming Model Summary:\n")
print(summary(timing_model))

# Descriptive statistics
cat("\nDescriptive statistics for timing by group:\n")
timing_desc <- aggregate(onset_med_25 ~ typical, data = time_df_filtered, 
                        FUN = function(x) c(mean = mean(x, na.rm = TRUE), 
                                          median = median(x, na.rm = TRUE),
                                          sd = sd(x, na.rm = TRUE),
                                          n = length(x)))
print(timing_desc)

# Analysis 2: Seizure spread (Fraction) comparing Atypical vs Typical  
cat("\n--- ANALYSIS 2: SEIZURE SPREAD (Fraction) ---\n")

# Filter to only Atypical and Typical (exclude "All" if present)
spread_df_filtered <- spread_df_all[spread_df_all$typical %in% c("Atypical", "Typical"), ]
spread_df_filtered$typical <- droplevels(spread_df_filtered$typical)

# Set Typical as reference level
spread_df_filtered$typical <- relevel(spread_df_filtered$typical, ref = "Typical")

cat("Sample sizes after filtering:\n")
print(table(spread_df_filtered$typical))

# Fit linear mixed effects model: Fraction ~ typical + (1|patient)
spread_model <- lmer(Fraction ~ typical + (1|patient), data = spread_df_filtered, REML=FALSE)

cat("\nSpread Model Summary:\n")
print(summary(spread_model))
spread_ols = lm(Fraction ~ typical, data = spread_df_filtered)
print(summary(spread_ols))

# Descriptive statistics
cat("\nDescriptive statistics for spread by group:\n")
spread_desc <- aggregate(Fraction ~ typical, data = spread_df_filtered,
                        FUN = function(x) c(mean = mean(x, na.rm = TRUE),
                                          median = median(x, na.rm = TRUE), 
                                          sd = sd(x, na.rm = TRUE),
                                          n = length(x)))
print(spread_desc)

# Summary of all results
cat("\n=== SUMMARY OF RESULTS ===\n")
cat("All models use random intercepts for patients\n")
cat("All p-values corrected using Satterthwaite degrees of freedom\n\n")

cat("\nAnalysis completed!\n")