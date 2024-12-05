#### Examining the Model outputs and plotting

# Load required packages
library(dplyr)
library(tidyr)
library(xtable)
library(ggplot2)
library(viridis)
library(ggthemes)
library(ggsci)

# Load collated model results
all_output <- read.csv("Results_collation/Result_tables/all_outputs.csv")
all_output$X <- NULL

function_names <- c(
  "critical_1", "sensitive_var4_n2_p5", "weighted_var4_n1",
  "repeat_var6_r4", "repeat2_2_var6", "repeat2_var4_var5",
  "order1diff_var6_var4", "order2_64", "order4_645",
  "timing_1event_0year_45", "timing_2event_1year_45", "timing_2event_4year_45"
)

paper_names <- c(paste0("Period", 1:3), paste0("Repeats", 1:3), paste0("Order", 1:3), paste0("Timing", 1:3))

all_output <- all_output[all_output$data_name %in% paper_names, ]


date_y <- test_y %>%
  group_by(data_name, model_name, randnum_train) %>%
  top_n(1, date)

# Outputting the best fits table
write.csv(date_y, "Plotting/plot_data.csv")

###### Making the table for publication
data_agg <- aggregate(date_y, by = list(date_y$data_name, date_y$model_name, date_y$stoc, date_y$randnum_stoc, date_y$randnum_split, date_y$epochs, date_y$folds, date_y$trials, date_y$hype, date_y$imp), FUN = mean)
names_cols <- c("data_name", "model_name", "stoc", "randnum_stoc", "randnum_split", "epochs", "folds", "trials", "hype", "imp")
data_agg <- data_agg[, !(names(data_agg) %in% names_cols)]
names(data_agg)[seq_along(names_cols)] <- names_cols

table_y <- data_agg

# Make percentages
table_y[, c(
  "accuracy",
  "precision",
  "recall",
  "f1",
  "auc",
  "av_prec",
  "brier",
  "pr_auc")] <- table_y[, c("accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "av_prec",
        "brier",
        "pr_auc")] * 100

table_y <- table_y[, c("data_name", "model_name", "f1", "auc", "pr_auc", "brier")]
colnames(table_y) <- c("LCP",  "Model", "AAF1", "AUROC", "AUPRC", "Brier")

table_y <- table_y %>%  pivot_wider(names_from = Model, values_from = AAF1:Brier)
colnames(table_y) <- sub('(\\w+)_(\\w+)', '\\2_\\1', colnames(table_y))

table_y <- table_y[order(table_y$LCP), ]
table_y <- table_y[, order(colnames(table_y))]

print(xtable(table_y, type = "latex"), include.rownames = FALSE)

#### Make plot for publication

name_dict <- cbind(sort(paper_names), c(rep("Period", 3), rep("Repeats", 3), rep("Order", 3), rep("Timing", 3)))
colnames(name_dict) <- c("data_name", "lcp_name")

model_dict <- cbind(sort(unique(date_y$model_name)), c("IT", "LR", "LSTMA", "MLF", "ResNet", "XGB"))
colnames(model_dict) <- c("model_name", "Model")

# Prepare data to plot
data <- date_y
data <- merge(data, name_dict)
data <- merge(data, model_dict)
data$Model <- factor(data$Model, levels=c("LR", "XGB", "MLF", "IT", "ResNet", "LSTMA"))
data$lcp_name <- factor(data$lcp_name, levels=c("Period", "Repeat", "Order", "Timing"))

# Create the plot
plot <- ggplot(data, aes(x = Model, y = pr_auc_out, fill = Model, color = Model)) +
  geom_boxplot(alpha = 0.5, width = 0.4) +
  geom_point(stroke = NA, alpha = 0.4, color="black", position = position_jitter(width = 0.1)) +
  facet_wrap(~ lcp_name, nrow = 1, strip.position = "bottom") +
  scale_fill_viridis(discrete = TRUE, option = "H") +
  scale_color_viridis(discrete = TRUE, option = "H") +
  labs(
    x = NULL,
    y = "AUPRC (%)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1),
    panel.spacing = unit(1.4, "lines"),
    legend.position = "none",
    strip.placement = "outside"
  )


# Display the plot
print(plot)

# Save the plot
ggsave("Plotting/Performance plots/AUPRC_LCP_plot.pdf", plot, dpi = 300, width = 120, height = 200, units = "mm")