# File to collate all output results into a single table

# Load in useful packages
library(plyr)

# Combine the output files for model performance on test set
filenames <- list.files("model_results/output", pattern = "*.csv", full.names = TRUE)
ldf <- lapply(filenames, read.csv)
output <- ldply(ldf, data.frame)

# Add the proportion of positives
output$prop <- rowSums(output[c("true_pos", "false_neg")]) / rowSums(output[c("true_neg", "false_neg", "true_pos", "false_pos")])

# Write to output file
write.csv(output, paste0("Results_collation/Result_tables/all_outputs.csv"))


# Combine the hyperparameter output files
filenames <- list.files("model_results/optuna", pattern = "*.csv", full.names = TRUE)
ldf <- lapply(filenames, read.csv)
ldf <- mapply(cbind, ldf, "filename" = filenames, SIMPLIFY = FALSE)
output <- ldply(ldf, data.frame)

# Write to output file
write.csv(output, paste0("Results_collation/Result_tables/all_optuna_outputs.csv"))