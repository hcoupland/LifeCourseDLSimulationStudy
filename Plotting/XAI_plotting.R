### File to plot explainability results

# Load required packages
library(list)
library(dplyr)
library(plyr)
library(reshape2)
library(ggplot2)
library(stringr)
library(egg)
library(abind)
library(ggpubr)
library(ggbeeswarm)
library(pheatmap)
library(scales)
library(viridis)
library(lubridate)
library(ggExtra)
library(tidyr)
library(tidyverse)
library(ggnewscale)
library(colorspace)
library(cowplot)
library(ggsci)
library(patchwork)
library(scales)

# Set the directories
setwd("model_results/explr/shap/values")
output_dir <- "Plotting/SHAP_plots/"

load_data <- function(model, filenames_all) {
  # Function to load in the SHAP values
  filenames <- grep(model, filenames_all, fixed = TRUE, value = TRUE)

  num_files <- ifelse(model == "XGBoost", 4, 5)

  # Select only the most recent files
  if (length(filenames)>num_files) {

    datetime_str <- str_extract(filenames, "\\d{8}-\\d{6}")
    datetime <- as.POSIXct(datetime_str, format = "%Y%m%d-%H%M%S")

    # Bind datetime with filenames and sort by datetime
    sorted_files <- filenames[order(datetime, decreasing = TRUE)]

    # Select only the 5 most recent files
    most_recent_files <- head(sorted_files, 5)

    # Display the most recent files
    filenames <- most_recent_files
  }

  # Load and reshape the saved SHAP values
  if (model != "XGBoost") {
    # DL models
    shap0_idx <- grep("shap0", filenames, fixed = TRUE, value = TRUE)
    shap1_idx <- grep("shap1", filenames, fixed = TRUE, value = TRUE)
    xtest_idx <- grep("xtest", filenames, fixed = TRUE, value = TRUE)
    ytest_idx <- grep("ytest", filenames, fixed = TRUE, value = TRUE)
    indices_idx <- grep("indices", filenames, fixed = TRUE, value = TRUE)

    ldf <- lapply(shap1_idx, read.csv)
    shap1 <- ldf[[1]]
    shap1 <- aperm(array(unlist(shap1), dim = c(nrow(shap1), n_years, n_vars)), c(1, 3, 2))

    ldf <- lapply(shap0_idx, read.csv)
    shap0 <- ldf[[1]]
    shap0 <- aperm(array(unlist(shap0), dim = c(nrow(shap0), n_years, n_vars)), c(1, 3, 2))

    ldf <- lapply(ytest_idx, read.csv)
    ytest <- ldf[[1]]

    ldf <- lapply(xtest_idx, read.csv)
    xtest <- ldf[[1]]
    xtest <- aperm(array(unlist(xtest), dim = c(nrow(xtest), n_years, n_vars)), c(1, 3, 2))

    ldf <- lapply(indices_idx, read.csv)
    indices <- ldf[[1]]

  } else {
    # XGBoost
    shap0_idx <- grep("shap", filenames, fixed = TRUE, value = TRUE)
    xtest_idx <- grep("xtest", filenames, fixed = TRUE, value = TRUE)
    ytest_idx <- grep("ytest", filenames, fixed = TRUE, value = TRUE)
    indices_idx <- grep("indices", filenames, fixed = TRUE, value = TRUE)

    ldf <- lapply(shap0_idx, read.csv)
    shap0 <- ldf[[1]]
    shap0 <- aperm(array(unlist(shap0), dim = c(nrow(shap0), n_years, n_vars)), c(1, 3, 2))

    ldf <- lapply(ytest_idx, read.csv)
    ytest <- ldf[[1]]

    ldf <- lapply(xtest_idx, read.csv)
    xtest <- ldf[[1]]
    xtest <- aperm(array(unlist(xtest), dim = c(nrow(xtest), n_years, n_vars)), c(1, 3, 2))

    ldf <- lapply(indices_idx, read.csv)
    indices <- ldf[[1]]

    shap1 <- "Not available for XGB"
  }

  return(list("shap1" = shap1, "shap0" = shap0, "ytest" = ytest,"xtest" =  xtest,"indices" = indices))
}

# Names for data to create the plots for
function_names <- c( "critical_1", "order1diff_var6_var4",  "order2_64",
                     "order4_645",  "repeat2_2_var6", "repeat2_var4_var5",  "repeat_var6_r4",
                     "sensitive_var4_n2_p5", "timing_1event_0year_45", "timing_2event_1year_45",
                     "timing_2event_4year_45", "weighted_var4_n1")
paper_names <- c(paste0("Period", 1:3), paste0("Repeats", 1:3), paste0("Order", 1:3), paste0("Timing", 1:3))

# Model parameters
stoc <- "stoc10"
rand <- "randtr7"
n_years <- 16
var_names <- c("Parental origin", "Mother's age", "Parental diabetes", "Loss", "SES", "Dynamic")
n_vars <- length(var_names)

# Filenames to load from
filenames_base <- list.files(".", pattern = "*.csv", full.names = TRUE)
filenames_base <- grep(stoc, filenames_base, fixed = TRUE, value = TRUE)
filenames_base <- grep(rand, filenames_base, fixed = TRUE, value = TRUE)

for (LCP_rule in paper_names) {
  print(LCP_rule)

  # Select outputs for that LCP rules
  filenames_all <- grep(LCP_rule, filenames_base, fixed = TRUE, value = TRUE)

  # Extract the SHAP values
  resnet_out <- load_data("ResNet", filenames_all)
  shap1_resnet <- resnet_out$shap1
  shap0_resnet <- resnet_out$shap0
  ytest_resnet <- resnet_out$ytest
  xtest_resnet <- resnet_out$xtest
  indices_resnet <- resnet_out$indices

  lstma_out <- load_data("LSTMAttention", filenames_all)
  shap1_lstma <- lstma_out$shap1
  shap0_lstma <- lstma_out$shap0
  ytest_lstma <- lstma_out$ytest
  xtest_lstma <- lstma_out$xtest
  indices_lstma <- lstma_out$indices

  mlstm_out <- load_data("MLSTMFCN", filenames_all)
  shap1_mlstm <- mlstm_out$shap1
  shap0_mlstm <- mlstm_out$shap0
  ytest_mlstm <- mlstm_out$ytest
  xtest_mlstm <- mlstm_out$xtest
  indices_mlstm <- mlstm_out$indices

  it_out <- load_data("InceptionTime", filenames_all)
  shap1_it <- it_out$shap1
  shap0_it <- it_out$shap0
  ytest_it <- it_out$ytest
  xtest_it <- it_out$xtest
  indices_it <- it_out$indices

  xgb_out <- load_data("XGBoost", filenames_all)
  shap0_xgb <- xgb_out$shap0
  ytest_xgb <- xgb_out$ytest
  xtest_xgb <- xgb_out$xtest
  indices_xgb <- xgb_out$indices

  # Rescale xtest
  xtest_all <- xtest_it
  xtest_all[, 2, ] <- 2 * xtest_all[, 2,]
  xtest_all[, 4, ] <- floor(xtest_all[, 4,] * 5)
  xtest_all[, 5, ] <- floor(xtest_all[, 5,] * 3)
  xtest_all[, 6, ] <- floor(xtest_all[, 6,] * 6)

  # Identify the true labels
  true_zero_idx <- which(ytest_it[, 1] == 0)
  true_ones_idx <- which(ytest_it[, 1] == 1)

  # Prepare the data for plotting
  xtest_data <- melt(xtest_all)
  shap0_it <- melt(shap0_it)
  shap0_resnet <- melt(shap0_resnet)
  shap0_lstma <- melt(shap0_lstma)
  shap0_mlstm <- melt(shap0_mlstm)
  shap1_it <- melt(shap1_it)
  shap1_resnet <- melt(shap1_resnet)
  shap1_lstma <- melt(shap1_lstma)
  shap1_mlstm <- melt(shap1_mlstm)
  shap0_xgb <- melt(shap0_xgb)

  all_data <- xtest_data
  names(all_data) <- c("Individual", "Feature", "Age", "Data")
  all_data$IT <- shap1_it$value
  all_data$LSTMA <- shap1_lstma$value
  all_data$MLF <- shap1_mlstm$value
  all_data$ResNet <- shap1_resnet$value
  all_data$XGB <- shap0_xgb$value

  # Make sure all SHAP values are for the positive class
  all_data$IT[all_data$Individual %in% true_ones_idx] <-  shap0_it$value[shap0_it$Var1 %in% true_ones_idx]
  all_data$LSTMA[all_data$Individual %in% true_ones_idx] <-  shap0_lstma$value[shap0_lstma$Var1 %in% true_ones_idx]
  all_data$MLF[all_data$Individual %in% true_ones_idx] <-  shap0_mlstm$value[shap0_mlstm$Var1 %in% true_ones_idx]
  all_data$ResNet[all_data$Individual %in% true_ones_idx] <-  shap0_resnet$value[shap0_resnet$Var1 %in% true_ones_idx]

  shap_data <- pivot_longer(all_data, Data:XGB)
  names(shap_data) <- c("Individual", "Feature", "Age", "Model", "SHAP_Value")
  feature_data <- shap_data[shap_data$Model == "Data", ]
  shap_data <- shap_data[shap_data$Model != "Data", ]

  # Find where all models correctly classified an individual
  dlmodels_agree <- (indices_it[, 2] == indices_lstma[, 2]) & (indices_mlstm[, 2] == indices_resnet[, 2]) & (indices_it[, 2] == indices_resnet[, 2])
  allmodels_agree <- dlmodels_agree & (indices_xgb[, 1] == indices_resnet[, 1])

  shap_data$Sign <- "Neg"
  shap_data$Sign[shap_data$Individual %in% true_ones_idx] <- "Pos"

  feature_data$Sign <- "Neg"
  feature_data$Sign[feature_data$Individual %in% true_ones_idx] <- "Pos"

  true_zero_idx_agree <- which(ytest_it[, 1] == 0 & allmodels_agree)
  true_ones_idx_agree <- which(ytest_it[, 1] == 1 & allmodels_agree)

  # For cetain data sets select the individuals to plot
  if (LCP_rule == "Timing1") {
    individual_1 <- true_zero_idx_agree[15]
    individual_2 <- true_ones_idx_agree[10]
  } else if (LCP_rule == "Order1") {
    individual_1 <- true_zero_idx_agree[4]
    individual_2 <- true_ones_idx_agree[1]
  } else if (LCP_rule == "Repeats3") {
    individual_1 <- true_zero_idx_agree[5]
    individual_2 <- true_ones_idx_agree[4]
  } else if (LCP_rule == "Period3") {
    individual_1 <- true_zero_idx_agree[30]
    individual_2 <- true_ones_idx_agree[13]
  } else if (LCP_rule == "Period1") {
    individual_1 <- true_zero_idx_agree[5]
    individual_2 <- true_ones_idx_agree[1]
  } else if (LCP_rule == "Order3") {
    individual_1 <- true_zero_idx_agree[2]
    individual_2 <- true_ones_idx_agree[2]
  } else {
    individual_1 <- true_zero_idx_agree[1]
    individual_2 <- true_ones_idx_agree[1]
  }
  print(c(individual_1, individual_2))

  shap_inds <- shap_data %>% filter(Individual %in% c(individual_1, individual_2))

  # Calculate marginal values
  shap_marginal_feature <- shap_data %>%
    group_by(Model, Age, Individual, Sign) %>%
    summarise_at(vars(SHAP_Value), list(Mean_SHAP = mean))

  shap_marginal_time <- shap_data %>%
    group_by(Model, Feature, Individual, Sign) %>%
    summarise_at(vars(SHAP_Value), list(Mean_SHAP = mean))

  feature_marginal_feature <- feature_data %>%
    group_by(Age, Individual, Sign) %>%
    summarise_at(vars(SHAP_Value), list(Mean_feature = mean))

  feature_marginal_time <- feature_data %>%
    group_by(Feature, Individual, Sign) %>%
    summarise_at(vars(SHAP_Value), list(Mean_feature = mean))

  # Plotting limits for SHAP values
  shap_limits <- range(c(shap_inds$SHAP_Value, shap_marginal_feature$Mean_SHAP, shap_marginal_time$Mean_SHAP), na.rm = TRUE)

  plot_data_time <- shap_marginal_time %>% left_join(feature_marginal_time, by = c("Individual", "Sign", "Feature"))
  plot_data_feature <- shap_marginal_feature %>% left_join(feature_marginal_feature, by = c("Individual", "Sign", "Age"))

  # Individual-level data plot
  plot_ind <- ggplot(feature_data %>% filter(Individual %in% c(individual_1, individual_2)),
                     aes(x = Age, y = as.factor(Feature), fill = SHAP_Value)) +
    geom_tile(color = "white", size = 0.1) +
    scale_fill_gradientn(colors = c("white", "blue", "navy"),
                         name = "Feature Value",
                         guide = guide_colorbar(barwidth = 0.5,
                                                barheight = 8,
                                                title.position = "left")) +
    theme_minimal() +
    scale_y_discrete(
      breaks = 1:n_vars,
      labels = var_names,
      expand = c(0, 0)
    )  +
    scale_x_continuous(breaks = c(0, 15)) +
    labs(title = paste("Individual Features"), x = "Age", y = "Feature") +
    facet_grid(. ~ Sign) +
    removeGrid() +
    theme(
      strip.background = element_blank(),
      strip.text = element_text(size = 10, face = "bold"),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
      legend.title = element_text(angle = 90, hjust = 0.5, vjust = 0,
                                  size = rel(0.9)),
      legend.position = "right",
      axis.title.y = element_blank(),
      axis.title.x = element_blank(),
      axis.text.x = element_text(hjust = -0.3),
      axis.text = element_text(size = 10)
    )

  # Marginalised individual-level plot
  plot_marginalised_ind <- ggplot(
    shap_marginal_time  %>% filter(Individual %in% c(individual_1, individual_2)),
    aes(y = as.factor(Feature), x = Model, fill=Mean_SHAP)
  ) +
    geom_tile(color = "white", size = 0.1) +
    scale_fill_continuous_diverging(
      palette = "Blue-Red 2", rev = TRUE, l2 = 98, l1 = 40,
      name = "SHAP Value",
      limits = shap_limits,
      guide = guide_colorbar(
        barwidth = 0.5,
        barheight = 8,
        title.position = "left")
    ) +
    theme_minimal() +
    labs(title = "Marginalised SHAP Values", x = "Model", y = "Feature") +
    scale_y_discrete(
      breaks = 1:n_vars,
      labels = var_names,
      expand = c(0, 0)
    ) +
    theme(
      strip.background = element_blank(),
      strip.text = element_text(size = 10, face = "bold"),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
      axis.text.x = element_text(angle = 45, hjust = 1),
      axis.title.x = element_blank(),
      axis.title.y = element_blank(),
      legend.title = element_text(angle = 90, hjust = 0.5, vjust = 0,
                                  size = rel(0.9)),
      legend.position = "none",
      axis.text = element_text(size = 10)
    ) +
    facet_grid(. ~ Sign) +
    removeGrid()

  # SHAP heatmaps for individual 1
  heatmap_ind1 <- ggplot(shap_data %>% filter(Individual %in% c(individual_1)),
                         aes(x = Age, y = as.factor(Feature), fill = SHAP_Value)) +
    geom_tile(color = "white", size = 0.1) +
    scale_fill_continuous_diverging(
      palette = "Blue-Red 2", rev = TRUE, l2 = 98, l1 = 40,
      name = "SHAP Value",
      limits = shap_limits,
      guide = guide_colorbar(
        barwidth = 0.8,
        barheight = 10,
        title.position = "left")
    ) +
    theme_minimal() +
    scale_y_discrete(
      breaks = 1:n_vars,
      labels = var_names,
      expand = c(0, 0)
    ) +
    scale_x_continuous(breaks = c(0, 15)) +
    theme(
      strip.background = element_blank(),
      strip.text = element_text(size = 10, face = "bold"),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
      legend.title = element_text(angle = 90, hjust = 0.5, vjust = 0,
                                  size = rel(0.9)),
      legend.position = "none",
      axis.text.x = element_text(hjust = -0.3),
      axis.title.y = element_blank(),
      axis.title.x = element_blank(),
      axis.text = element_text(size = 10)
    ) +
    labs(title = paste("SHAP Values for Negative Individual"), x = "Age", y = "Feature") +
    facet_grid(. ~ Model) +
    removeGrid()

  # SHAP heatmaps for individual 2
  heatmap_ind2 <- ggplot(shap_data %>% filter(Individual %in% c(individual_2)),
                         aes(x = Age, y = as.factor(Feature), fill = SHAP_Value)) +
    geom_tile(color = "white", size = 0.1) +
    scale_fill_continuous_diverging(
      palette = "Blue-Red 2", rev = TRUE, l2 = 98, l1 = 40,
      name = "SHAP Value",
      limits = shap_limits,
      guide = guide_colorbar(
        barwidth = 0.5,
        barheight = 8,
        title.position = "left")
    ) +
    theme_minimal() +
    scale_y_discrete(
      breaks = 1:n_vars,
      labels = var_names,
      expand = c(0, 0)
    ) +
    scale_x_continuous(breaks = c(0, 15)) +
    theme(
      strip.background = element_blank(),
      strip.text = element_text(size = 10, face = "bold"),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
      axis.title.x = element_blank(),
      legend.title = element_text(angle = 90, hjust = 0.5, vjust = 0,
                                  size = rel(0.9)),
      legend.position = "none",
      axis.title.y = element_blank(),
      axis.text.x = element_text(hjust=-0.3),
      axis.text = element_text(size = 10)
    ) +
    labs(title = paste("SHAP Values for Positive Individual"), x = "Age", y = "Feature") +
    facet_grid(. ~ Model) +
    removeGrid()

  legend <- get_legend(
    heatmap_ind1 +
      theme(legend.position = "right")  # Extract legend from beeswarm_time
  )

  combined_plot <- plot_grid(
    heatmap_ind1,
    heatmap_ind2,
    ncol = 1,  # Stack plots vertically
    align = "v",  # Align vertically
    rel_heights = c(1, 1)  # Equal height for both plots
  )

  # Add the legend to the combined plot
  final_plot_heatmap <- plot_grid(
    combined_plot, legend,
    ncol = 2,  # Place legend to the right
    rel_widths = c(4.4, 0.6)  # Adjust legend width
  )

  combined_plot2 <- plot_grid(
    plot_ind, plot_marginalised_ind,
    ncol = 2,  # Stack plots vertically
    align = "h",  # Align vertically
    rel_heights = c(1, 1)  # Equal height for both plots
  )

  final_plot1 <- combined_plot2  /
    final_plot_heatmap  +
    plot_layout(heights = c(1, 2 ))

  # Print the plot
  print(final_plot1)

  # Save the final plot
  name <- paste0("Big_SHAP1_", LCP_rule, "_", stoc, "_", rand, "_ind", individual_1, "_ind",  individual_2)
  ggsave(paste0(output_dir, name , ".pdf"), final_plot1, dpi = 300, width = 170, height = 200, units = "mm")

  # Scaling the features ot put on the same scale
  plot_data_time_scaled <- plot_data_time %>%
    group_by(Feature) %>%
    mutate(Mean_feature_scaled := scales::rescale(Mean_feature)) %>%
    ungroup()

  # Beeswarm plot marginalised by features
  beeswarm_time_scaled <- ggplot(plot_data_time_scaled,
                                 aes(x = Mean_SHAP, y = as.factor(Feature), color = Mean_feature_scaled)) +
    geom_quasirandom(size = 1.5, alpha = 0.4, stroke=NA) +
    geom_vline(xintercept = 0) +
    scale_color_gradient2(low = "#008afb", mid = "#9927af", high = "#ff1565", midpoint = 0.5,
                          name = "Feature value",
                          breaks = c(0, 1),  # Specify breaks at the ends of the scale
                          labels = c("Low", "High"),
                          guide = guide_colorbar(
                            barwidth = 0.8,
                            barheight = 10,
                            title.position = "left") ) +
    theme_minimal() +
    scale_y_discrete(breaks = 1:n_vars, labels = var_names, expand = c(0, 0)) +  # Adjust y-axis labels and spacing
    scale_x_continuous(n.breaks = 3) +
    facet_grid(. ~ Model) +
    labs(title = "Beeswarm Plot (Marginalised Over Time)", y = "Feature", x = "Mean SHAP Value", color = "Feature Value") +
    theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
          axis.text = element_text(size = 10),
          strip.text = element_text(size = 10, face = "bold"),
          panel.spacing = unit(1.4, "lines"),
          axis.title.y = element_blank(),
          legend.title = element_text(angle = 90, hjust = 0.5, vjust = 0,
                                      size = rel(0.9)),
          legend.position = "none"
    )

  # Scaling the features ot put on the same scale
  plot_data_feature_scaled <- plot_data_feature %>%
    group_by(Age) %>%
    mutate(Mean_feature_scaled := scales::rescale(Mean_feature)) %>%
    ungroup()

  # Beeswarm plot marginalised by features
  beeswarm_feature_scaled <- ggplot(plot_data_feature_scaled,
                                    aes(x = Mean_SHAP, y = Age, color = Mean_feature)) +
    geom_quasirandom(alpha = 0.4, size = 1.5, groupOnX = FALSE, stroke = NA) +
    geom_vline(xintercept = 0) +
    scale_color_gradient2(ow = "#008afb", mid = "#9927af", high = "#ff1565",
                          midpoint = 0.5,
                          name = "Feature value",
                          breaks = c(0, 1),  # Specify breaks at the ends of the scale
                          labels = c("Low", "High"),
                          guide = guide_colorbar(
                            barwidth = 0.8,
                            barheight = 10,
                            title.position = "left")) +
    theme_minimal() +
    facet_grid(. ~ Model) +
    scale_x_continuous(n.breaks = 3) +
    labs(title = "Beeswarm Plot (Marginalised Over Features)", y = "Age", x = "Mean SHAP Value", color = "Feature Value") +
    theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
          axis.text = element_text(size = 10),
          strip.text = element_text(size = 10, face = "bold"),
          panel.spacing = unit(1.4, "lines"),
          legend.title = element_text(angle = 90, hjust = 0.5, vjust = 0,
                                      size = rel(0.9)),
          legend.position = "none"
    )

  # Extract the shared legend
  shared_legend <- get_legend(
    beeswarm_time_scaled +
      theme(legend.position = "right")  # Ensure the legend is positioned correctly
  )

  # Combine the plots vertically, aligning axes
  combined_plot <- plot_grid(
    beeswarm_time_scaled + theme(legend.position = "none"),  # Remove individual legend
    beeswarm_feature_scaled + theme(legend.position = "none"),  # Remove individual legend
    ncol = 1,  # Stack plots vertically
    align = "v",  # Align axes vertically
    axis = "l"  # Align y-axes on the left
  )

  # Add the shared legend to the combined plot
  final_plot2 <- plot_grid(
    combined_plot,
    shared_legend,
    ncol = 2,  # Place legend to the right of plots
    rel_widths = c(4.4, 0.6)  # Adjust widths for plots and legend
  )

  # Display the final plot
  print(final_plot2)

  # Save the final plot
  name <- paste0("Big_SHAP2_", LCP_rule,"_", stoc,"_", rand,"_ind", individual_1,"_ind",  individual_2)
  ggsave(paste0(output_dir, name ,"_big.pdf"), final_plot2, dpi = 300, width = 170, height = 200, units = "mm")

}
