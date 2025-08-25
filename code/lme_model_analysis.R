# Mixed Effects Model Analysis for Seizure Detection Data
# This script performs LME analyses with random intercepts for patients
# on threshold comparisons, model agreements, and stim vs spontaneous seizure comparisons

# Load required libraries
library(lme4)
library(lmerTest)
library(dplyr)
library(tidyr)
library(emmeans)
library(multcomp)
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

# Helper function to read CSV files
read_data <- function(filename) {
  filepath <- file.path(prodatapath, filename)
  if (!file.exists(filepath)) {
    warning(paste("File not found:", filepath))
    return(NULL)
  }
  return(read.csv(filepath, stringsAsFactors = FALSE))
}

# Helper function to read metadata files
read_metadata <- function(filename) {
  filepath <- file.path(metapath, filename)
  if (!file.exists(filepath)) {
    warning(paste("Metadata file not found:", filepath))
    return(NULL)
  }
  return(read.csv(filepath, stringsAsFactors = FALSE))
}

cat("=== Mixed Effects Model Analysis for Seizure Detection ===\n\n")

# ===================================================================
# 1. THRESHOLD COMPARISON ANALYSIS (Stim vs Spontaneous)
# ===================================================================

cat("1. THRESHOLD COMPARISON ANALYSIS\n")
cat("================================\n")

# Load threshold data
plot_thresholds <- read_data("plot_thresholds_for_lme.csv")

if (!is.null(plot_thresholds)) {
  cat("Data loaded successfully. Shape:", nrow(plot_thresholds), "x", ncol(plot_thresholds), "\n")
  cat("Columns:", paste(colnames(plot_thresholds), collapse = ", "), "\n\n")
  
  # Convert factors
  plot_thresholds$patient <- as.factor(plot_thresholds$patient)
  plot_thresholds$stim <- as.factor(plot_thresholds$stim)
  levels(plot_thresholds$stim) <- c("Spontaneous", "Stimulated")
  
  # Descriptive statistics
  cat("Descriptive Statistics:\n")
  desc_stats <- plot_thresholds %>%
    group_by(stim) %>% 
    summarise(
      n = n(),
      mean_threshold = mean(threshold, na.rm = TRUE),
      median_threshold = median(threshold, na.rm = TRUE),
      sd_threshold = sd(threshold, na.rm = TRUE),
      .groups = 'drop'
    )
  print(desc_stats)
  cat("\n")
  
  # Mixed effects model for threshold comparison
  cat("Mixed Effects Model for Threshold Comparison:\n")
  threshold_model <- lmer(threshold ~ stim + (1|patient), data = plot_thresholds)
  
  # Model summary
  cat("Model Summary:\n")
  print(summary(threshold_model))
  cat("\n")
  
  # ANOVA for fixed effects
  cat("ANOVA for Fixed Effects:\n")
  threshold_anova <- anova(threshold_model)
  print(threshold_anova)
  cat("\n")
  
  # Post-hoc comparisons using emmeans
  cat("Post-hoc Comparisons:\n")
  threshold_emm <- emmeans(threshold_model, ~ stim)
  threshold_contrasts <- contrast(threshold_emm, method = "pairwise")
  print(threshold_contrasts)
  cat("\n")
  
  # Effect size (Cohen's d equivalent for LME)
  cat("Effect Size Estimation:\n")
  threshold_means <- emmeans(threshold_model, ~ stim)
  threshold_pairs <- pairs(threshold_means)
  print(confint(threshold_pairs))
  cat("\n")
  
  # Variance homogeneity test accounting for patient clustering
  cat("VARIANCE HOMOGENEITY TESTS:\n")
  cat("===========================\n")
  
  # Method 1: Levene test on patient-averaged data
  cat("Method 1 - Levene Test on Patient-Averaged Data:\n")
  threshold_by_patient <- plot_thresholds %>%
    group_by(patient, stim) %>%
    summarise(mean_threshold = mean(threshold, na.rm = TRUE), .groups = 'drop') %>%
    filter(!is.na(mean_threshold))
  
  if (nrow(threshold_by_patient) > 0) {
    if (!require(car, quietly = TRUE)) {
      install.packages("car")
      library(car)
    }
    
    levene_patient_avg <- car::leveneTest(mean_threshold ~ stim, data = threshold_by_patient)
    print(levene_patient_avg)
    cat("\n")
  }
  

  
} else {
  cat("Threshold data not found. Skipping analysis.\n\n")
}

# ===================================================================
# 2. MODEL AGREEMENT DIFFERENCE ANALYSIS (NDD vs Benchmark Models)
# ===================================================================

cat("2. MODEL AGREEMENT DIFFERENCE ANALYSIS\n")
cat("======================================\n")

# Load onset agreement data
onset_agreements <- read_data("onset_all_plot_agreements_for_lme.csv")

if (!is.null(onset_agreements)) {
  cat("Data loaded successfully. Shape:", nrow(onset_agreements), "x", ncol(onset_agreements), "\n")
  cat("Columns:", paste(colnames(onset_agreements), collapse = ", "), "\n\n")
  
  # Data preparation for difference calculations
  cat("Preparing data for pairwise difference analysis...\n")
  
  # Check the structure of the data
  cat("Unique models:", paste(unique(onset_agreements$model), collapse = ", "), "\n")
  cat("Unique patients:", length(unique(onset_agreements$patient)), "\n")
  
  # Filter to only include seizure detection models (not Interrater)
  onset_models <- onset_agreements %>%
    filter(model != "Interrater") %>%
    mutate(
      model = as.factor(model),
      patient = as.factor(patient)
    )
  
  # Check for complete cases (seizures with all three models)
  structure_check <- onset_models %>%
    group_by(patient, approximate_onset) %>%
    summarise(
      n_models = n(),
      models_present = paste(sort(unique(model)), collapse = ", "),
      .groups = 'drop'
    )
  
  complete_seizures <- structure_check %>%
    filter(n_models == length(unique(onset_models$model)))
  
  cat("Seizures with all models present:", nrow(complete_seizures), "out of", nrow(structure_check), "\n")
  
  # Filter to only include complete cases for proper pairing
  if (nrow(complete_seizures) > 0) {
    cat("Using only seizures with complete model data for difference analysis\n")
    onset_models_complete <- onset_models %>%
      semi_join(complete_seizures, by = c("patient", "approximate_onset"))
    
    cat("Filtered data shape:", nrow(onset_models_complete), "x", ncol(onset_models_complete), "\n\n")
    
  } else {
    cat("Warning: No seizures found with all models present. Cannot perform difference analysis.\n\n")
    onset_models_complete <- NULL
  }
  
  if (!is.null(onset_models_complete) && nrow(onset_models_complete) > 0) {
    
    # Descriptive statistics by model
    cat("Descriptive Statistics by Model:\n")
    desc_stats_onset <- onset_models_complete %>%
      group_by(model) %>%
      summarise(
        n = n(),
        mean_dice = mean(MCC, na.rm = TRUE),
        median_dice = median(MCC, na.rm = TRUE),
        sd_dice = sd(MCC, na.rm = TRUE),
        .groups = 'drop'
      )
    print(desc_stats_onset)
    cat("\n")
    
    # Reshape data to wide format for difference calculations
    onset_wide <- onset_models_complete %>%
      dplyr::select(patient, approximate_onset, model, MCC) %>%
      pivot_wider(names_from = model, values_from = MCC)
    
    cat("Wide format data shape:", nrow(onset_wide), "x", ncol(onset_wide), "\n")
    cat("Available models in wide format:", paste(setdiff(colnames(onset_wide), c("patient", "approximate_onset")), collapse = ", "), "\n\n")
    
    # Calculate differences: NDD - Benchmark models
    differences_data <- onset_wide %>%
      mutate(
        patient = as.factor(patient)
      )
    
    # Initialize vectors to store p-values for Bonferroni correction
    p_values <- c()
    comparison_names <- c()
    
    # Analysis 1: NDD vs AbsSlp
    if ("NDD" %in% colnames(onset_wide) && "AbsSlp" %in% colnames(onset_wide)) {
      cat("=== ANALYSIS 1: NDD vs AbsSlp ===\n")
      
      # Calculate difference (NDD - AbsSlp)
      differences_data$diff_NDD_AbsSlp <- differences_data$NDD - differences_data$AbsSlp
      
      # Remove rows with missing differences
      diff_data_1 <- differences_data %>%
        filter(!is.na(diff_NDD_AbsSlp))
      
      cat("Number of seizures with complete NDD-AbsSlp pairs:", nrow(diff_data_1), "\n")
      
      # Descriptive statistics for differences
      cat("Difference Statistics (NDD - AbsSlp):\n")
      cat("Mean difference:", round(mean(diff_data_1$diff_NDD_AbsSlp), 4), "\n")
      cat("SD of differences:", round(sd(diff_data_1$diff_NDD_AbsSlp), 4), "\n")
      cat("95% CI of mean difference:", 
          round(mean(diff_data_1$diff_NDD_AbsSlp) - 1.96 * sd(diff_data_1$diff_NDD_AbsSlp) / sqrt(nrow(diff_data_1)), 4), "to",
          round(mean(diff_data_1$diff_NDD_AbsSlp) + 1.96 * sd(diff_data_1$diff_NDD_AbsSlp) / sqrt(nrow(diff_data_1)), 4), "\n\n")
      
      # Mixed effects model on differences with patient random effect
      cat("Mixed Effects Model (Difference ~ 1 + (1|patient)):\n")
      model_diff_1 <- lmer(diff_NDD_AbsSlp ~ 1 + (1|patient), data = diff_data_1)
      
      # Model summary
      print(summary(model_diff_1))
      cat("\n")
      
      # Extract p-value for the intercept (tests if mean difference != 0)
      model_summary_1 <- summary(model_diff_1)
      p_val_1 <- model_summary_1$coefficients[1, "Pr(>|t|)"]
      p_values <- c(p_values, p_val_1)
      comparison_names <- c(comparison_names, "NDD_vs_AbsSlp")
      
      cat("Intercept p-value (test of mean difference = 0):", p_val_1, "\n\n")
    }
    
    # Analysis 2: NDD vs DL
    if ("NDD" %in% colnames(onset_wide) && "DL" %in% colnames(onset_wide)) {
      cat("=== ANALYSIS 2: NDD vs DL ===\n")
      
      # Calculate difference (NDD - DL)
      differences_data$diff_NDD_DL <- differences_data$NDD - differences_data$DL
      
      # Remove rows with missing differences
      diff_data_2 <- differences_data %>%
        filter(!is.na(diff_NDD_DL))
      
      cat("Number of seizures with complete NDD-DL pairs:", nrow(diff_data_2), "\n")
      
      # Descriptive statistics for differences
      cat("Difference Statistics (NDD - DL):\n")
      cat("Mean difference:", round(mean(diff_data_2$diff_NDD_DL), 4), "\n")
      cat("SD of differences:", round(sd(diff_data_2$diff_NDD_DL), 4), "\n")
      cat("95% CI of mean difference:", 
          round(mean(diff_data_2$diff_NDD_DL) - 1.96 * sd(diff_data_2$diff_NDD_DL) / sqrt(nrow(diff_data_2)), 4), "to",
          round(mean(diff_data_2$diff_NDD_DL) + 1.96 * sd(diff_data_2$diff_NDD_DL) / sqrt(nrow(diff_data_2)), 4), "\n\n")
      
      # Mixed effects model on differences with patient random effect
      cat("Mixed Effects Model (Difference ~ 1 + (1|patient)):\n")
      model_diff_2 <- lmer(diff_NDD_DL ~ 1 + (1|patient), data = diff_data_2)
      
      # Model summary
      print(summary(model_diff_2))
      cat("\n")
      
      # Extract p-value for the intercept
      model_summary_2 <- summary(model_diff_2)
      p_val_2 <- model_summary_2$coefficients[1, "Pr(>|t|)"]
      p_values <- c(p_values, p_val_2)
      comparison_names <- c(comparison_names, "NDD_vs_DL")
      
      cat("Intercept p-value (test of mean difference = 0):", p_val_2, "\n\n")
    }
    
    # Bonferroni correction
    if (length(p_values) > 0) {
      cat("=== BONFERRONI CORRECTION ===\n")
      cat("Number of comparisons:", length(p_values), "\n")
      cat("Alpha level: 0.05\n")
      cat("Bonferroni-corrected alpha:", round(0.05 / length(p_values), 4), "\n\n")
      
      # Apply Bonferroni correction
      p_adjusted <- p.adjust(p_values, method = "bonferroni")
      
      # Results summary
      results_summary <- data.frame(
        Comparison = comparison_names,
        Raw_p_value = round(p_values, 6),
        Bonferroni_p_value = round(p_adjusted, 6),
        Significant_raw = p_values < 0.05,
        Significant_bonferroni = p_adjusted < 0.05
      )
      
      cat("SUMMARY OF RESULTS:\n")
      print(results_summary)
      cat("\n")
      
      # Interpretation
      cat("INTERPRETATION:\n")
      for (i in 1:length(comparison_names)) {
        cat("- ", comparison_names[i], ": ", 
            ifelse(p_adjusted[i] < 0.05, "Significant difference", "No significant difference"),
            " (Bonferroni-adjusted p = ", round(p_adjusted[i], 4), ")\n", sep = "")
      }
      cat("\n")
    }
    
  } else {
    cat("Cannot perform difference analysis due to missing data.\n\n")
  }
  
} else {
  cat("Onset agreement data not found. Skipping analysis.\n\n")
}

# ===================================================================
# 2a. LSTM MODEL VS HUMAN INTERRATER DIFFERENCE ANALYSIS
# ===================================================================

cat("2a. LSTM MODEL VS HUMAN INTERRATER DIFFERENCE ANALYSIS\n")
cat("=======================================================\n")

# Load the long-format data (LSTM vs human, long-form)
model_interrater_long <- read_data("model-interrater_agreement.csv")

if (!is.null(model_interrater_long)) {
  cat("Data loaded successfully. Shape:", nrow(model_interrater_long), "x", ncol(model_interrater_long), "\n")
  cat("Columns:", paste(colnames(model_interrater_long), collapse = ", "), "\n\n")
  
  # Data preparation for difference calculations
  cat("Preparing data for LSTM vs Human difference analysis...\n")
  
  # Convert to factors
  model_interrater_long$patient <- as.factor(model_interrater_long$patient)
  model_interrater_long$annotator <- as.factor(model_interrater_long$annotator)
  
  # Check available annotators
  cat("Unique annotators:", paste(unique(model_interrater_long$annotator), collapse = ", "), "\n")
  cat("Unique patients:", length(unique(model_interrater_long$patient)), "\n")
  
  # Check for complete cases (seizures with both annotators)
  structure_check <- model_interrater_long %>%
    group_by(patient, approximate_onset) %>%
    summarise(
      n_annotators = n(),
      annotators_present = paste(sort(unique(annotator)), collapse = ", "),
      .groups = 'drop'
    )
  
  complete_seizures <- structure_check %>%
    filter(n_annotators == length(unique(model_interrater_long$annotator)))
  
  cat("Seizures with both annotators present:", nrow(complete_seizures), "out of", nrow(structure_check), "\n")
  
  # Filter to only include complete cases for proper pairing
  if (nrow(complete_seizures) > 0) {
    cat("Using only seizures with complete annotator data for difference analysis\n")
    interrater_complete <- model_interrater_long %>%
      semi_join(complete_seizures, by = c("patient", "approximate_onset"))
    
    cat("Filtered data shape:", nrow(interrater_complete), "x", ncol(interrater_complete), "\n\n")
    
    # Descriptive statistics by annotator
    cat("Descriptive Statistics by Annotator:\n")
    desc_stats_2a <- interrater_complete %>%
      group_by(annotator) %>%
      summarise(
        n = n(),
        mean_phi = mean(Phi, na.rm = TRUE),
        median_phi = median(Phi, na.rm = TRUE),
        sd_phi = sd(Phi, na.rm = TRUE),
        .groups = 'drop'
      )
    print(desc_stats_2a)
    cat("\n")
    
    # Reshape data to wide format for difference calculations
    interrater_wide <- interrater_complete %>%
      dplyr::select(patient, approximate_onset, annotator, Phi) %>%
      pivot_wider(names_from = annotator, values_from = Phi)
    
    cat("Wide format data shape:", nrow(interrater_wide), "x", ncol(interrater_wide), "\n")
    cat("Available annotators in wide format:", paste(setdiff(colnames(interrater_wide), c("patient", "approximate_onset")), collapse = ", "), "\n\n")
    
    # Calculate differences: Model - Human (assuming one is LSTM/Model and other is Human)
    annotator_names <- setdiff(colnames(interrater_wide), c("patient", "approximate_onset"))
    
    if (length(annotator_names) == 2) {
      # Determine which is model and which is human based on names
      model_col <- annotator_names[grepl("LSTM|Model|NDD", annotator_names, ignore.case = TRUE)]
      human_col <- annotator_names[!annotator_names %in% model_col]
      
      # If can't determine automatically, use first vs second
      if (length(model_col) == 0) {
        model_col <- annotator_names[1]
        human_col <- annotator_names[2]
        cat("Note: Could not automatically identify model vs human annotator.\n")
        cat("Using", model_col, "as model and", human_col, "as human.\n\n")
      }
      
      cat("=== ANALYSIS: ", model_col, " vs ", human_col, " ===\n", sep = "")
      
      # Calculate difference (Model - Human)
      diff_data <- interrater_wide %>%
        mutate(
          patient = as.factor(patient),
          diff_model_human = .data[[model_col]] - .data[[human_col]]
        ) %>%
        filter(!is.na(diff_model_human))
      
      cat("Number of seizures with complete Model-Human pairs:", nrow(diff_data), "\n")
      
      # Descriptive statistics for differences
      cat("Difference Statistics (", model_col, " - ", human_col, "):\n", sep = "")
      cat("Mean difference:", round(mean(diff_data$diff_model_human), 4), "\n")
      cat("SD of differences:", round(sd(diff_data$diff_model_human), 4), "\n")
      cat("95% CI of mean difference:", 
          round(mean(diff_data$diff_model_human) - 1.96 * sd(diff_data$diff_model_human) / sqrt(nrow(diff_data)), 4), "to",
          round(mean(diff_data$diff_model_human) + 1.96 * sd(diff_data$diff_model_human) / sqrt(nrow(diff_data)), 4), "\n\n")
      
      # Mixed effects model on differences with patient random effect
      cat("Mixed Effects Model (Difference ~ 1 + (1|patient)):\n")
      model_diff_interrater <- lmer(diff_model_human ~ 1 + (1|patient), data = diff_data)
      
      # Model summary
      print(summary(model_diff_interrater))
      cat("\n")
      
      # Extract p-value for the intercept (tests if mean difference != 0)
      model_summary_interrater <- summary(model_diff_interrater)
      p_val_interrater <- model_summary_interrater$coefficients[1, "Pr(>|t|)"]
      
      cat("Intercept p-value (test of mean difference = 0):", p_val_interrater, "\n")
             cat("Interpretation: ", 
           ifelse(p_val_interrater < 0.05, "Significant difference", "No significant difference"),
           " between ", model_col, " and ", human_col, " (p = ", round(p_val_interrater, 4), ")\n\n", sep = "")
       
       # Store p-value for combined correction (will be used later)
       interrater_p_value <- p_val_interrater
       interrater_comparison_name <- paste(model_col, "vs", human_col, sep = "_")
       
     } else {
       cat("Error: Expected exactly 2 annotators, found", length(annotator_names), "\n")
       cat("Available annotators:", paste(annotator_names, collapse = ", "), "\n\n")
       interrater_p_value <- NULL
       interrater_comparison_name <- NULL
     }
     
   } else {
     cat("Warning: No seizures found with both annotators present. Cannot perform difference analysis.\n\n")
     interrater_p_value <- NULL
     interrater_comparison_name <- NULL
   }
   
} else {
  cat("Model/interrater long-format data not found. Skipping analysis.\n\n")
  interrater_p_value <- NULL
  interrater_comparison_name <- NULL
}

# ===================================================================
# 2b. COMBINED MODEL PERFORMANCE BONFERRONI CORRECTION
# ===================================================================

cat("2b. COMBINED MODEL PERFORMANCE BONFERRONI CORRECTION\n")
cat("====================================================\n")

# Combine p-values from model comparisons (section 2) and interrater comparison (section 2a)
combined_p_values <- c()
combined_comparison_names <- c()

# Add model comparison p-values if they exist
if (exists("p_values") && length(p_values) > 0) {
  combined_p_values <- c(combined_p_values, p_values)
  combined_comparison_names <- c(combined_comparison_names, comparison_names)
}

# Add interrater p-value if it exists
if (!is.null(interrater_p_value)) {
  combined_p_values <- c(combined_p_values, interrater_p_value)
  combined_comparison_names <- c(combined_comparison_names, interrater_comparison_name)
}

if (length(combined_p_values) > 0) {
  cat("=== COMBINED BONFERRONI CORRECTION FOR ALL MODEL PERFORMANCE COMPARISONS ===\n")
  cat("Total number of comparisons:", length(combined_p_values), "\n")
  cat("Comparisons included:\n")
  for (i in 1:length(combined_comparison_names)) {
    cat("  ", i, ". ", combined_comparison_names[i], " (raw p = ", round(combined_p_values[i], 4), ")\n", sep = "")
  }
  cat("\nAlpha level: 0.05\n")
  cat("Bonferroni-corrected alpha:", round(0.05 / length(combined_p_values), 4), "\n\n")
  
  # Apply Bonferroni correction
  combined_p_adjusted <- p.adjust(combined_p_values, method = "bonferroni")
  
  # Combined results summary
  combined_results_summary <- data.frame(
    Comparison = combined_comparison_names,
    Raw_p_value = round(combined_p_values, 6),
    Bonferroni_p_value = round(combined_p_adjusted, 6),
    Significant_raw = combined_p_values < 0.05,
    Significant_bonferroni = combined_p_adjusted < 0.05
  )
  
  cat("COMBINED SUMMARY OF ALL MODEL PERFORMANCE RESULTS:\n")
  print(combined_results_summary)
  cat("\n")
  
  # Combined interpretation
  cat("COMBINED INTERPRETATION (after Bonferroni correction for all model comparisons):\n")
  for (i in 1:length(combined_comparison_names)) {
    cat("- ", combined_comparison_names[i], ": ", 
        ifelse(combined_p_adjusted[i] < 0.05, "Significant difference", "No significant difference"),
        " (Bonferroni-adjusted p = ", round(combined_p_adjusted[i], 4), ")\n", sep = "")
  }
  cat("\n")
  
} else {
  cat("No p-values available for combined correction.\n\n")
}

# ===================================================================
# 3. LSTM/NDD STIM VS SPONTANEOUS SEIZURE ANALYSIS
# ===================================================================

cat("3. LSTM/NDD STIM VS SPONTANEOUS SEIZURE ANALYSIS\n")
cat("================================================\n")

# Load stim vs spontaneous data
stim_spont_data <- read_data("stim_vs_spont_agreements_for_lme.csv")

if (!is.null(stim_spont_data)) {
  cat("Data loaded successfully. Shape:", nrow(stim_spont_data), "x", ncol(stim_spont_data), "\n")
  
  # Prepare data - this should only contain NDD model data
  stim_spont_data <- stim_spont_data %>%
    mutate(
      stim = as.factor(stim),
      patient = as.factor(patient)
    )
  
  # Rename stim levels for clarity
  levels(stim_spont_data$stim) <- c("Spontaneous", "Stimulated")
  
  # Descriptive statistics
  cat("Descriptive Statistics by Seizure Type:\n")
  desc_stats_stim <- stim_spont_data %>%
    group_by(stim) %>%
    summarise(
      n = n(),
      mean_MCC = mean(MCC, na.rm = TRUE),
      median_MCC = median(MCC, na.rm = TRUE),
      sd_MCC = sd(MCC, na.rm = TRUE),
      .groups = 'drop'
    )
  print(desc_stats_stim)
  cat("\n")
  
  # Mixed effects model for stim vs spontaneous
  cat("Mixed Effects Model for Stim vs Spontaneous Agreement:\n")
  if ("patient" %in% colnames(stim_spont_data) && length(unique(stim_spont_data$patient)) > 1) {
    stim_model <- lmer(MCC ~ stim + (1|patient), data = stim_spont_data)
  } else {
    stim_model <- lm(MCC ~ stim, data = stim_spont_data)
  }
  
  # Model summary
  print(summary(stim_model))
  cat("\n")
  
  # ANOVA for fixed effects
  cat("ANOVA for Fixed Effects:\n")
  stim_anova <- anova(stim_model)
  print(stim_anova)
  cat("\n")
  
  # Post-hoc comparisons
  cat("Post-hoc Comparisons:\n")
  stim_emm <- emmeans(stim_model, ~ stim)
  stim_contrasts <- contrast(stim_emm, method = "pairwise")
  print(stim_contrasts)
  cat("\n")
  
  # Effect size estimation
  cat("Effect Size Estimation:\n")
  print(confint(stim_contrasts))
  cat("\n")
  
} else {
  cat("Stim vs spontaneous data not found. Skipping analysis.\n\n")
}

# ===================================================================
# 4. COMBINED ANALYSIS AND VISUALIZATION SUGGESTIONS
# ===================================================================

cat("4. SUMMARY AND RECOMMENDATIONS\n")
cat("===============================\n")

cat("Analysis completed successfully!\n\n")

cat("Key findings summary:\n")
cat("- All analyses used mixed effects models with random intercepts for patients\n")
cat("- This accounts for the hierarchical structure of the data (seizures nested within patients)\n")
cat("- Post-hoc comparisons used appropriate multiple comparison corrections\n")
cat("- Section 2B provides fixed effects comparison including Interrater without multiple comparison correction\n\n")

cat("Analysis complete!\n") 