# Load simcausal library
library(simcausal)
library(reticulate)
# Import NumPy for data handling
np <- import("numpy")

# Set seed for reproducibility
randnum <- 7
set.seed(randnum)

# Define model variables
num_timeindep <- 3 # number of variables in model
num_timedep <- 3 # number of variables in model
n_people <- 100000 # number of people in model
n_time_out <- 16 # number of time points in model
n_time <- n_time_out - 1
num_var <- num_timedep + num_timeindep

# Load DAG coefficients
load_dag_coeffs <- function(file_path) {
    dag_coeffs <- read.csv(file_path)
    dag_coeffs <- data.frame(strsplit(unlist(dag_coeffs), ","))
    names(dag_coeffs) <- dag_coeffs[1, ]
    return(dag_coeffs[-1, ])
}

# Load coefficients from CSVs
dag_coeffs <- load_dag_coeffs("Data_simulation/DAG_params.csv")
dag_coeffs2 <- load_dag_coeffs("Data_simulation/DAG_params2.csv")

# Extract the coefficients
zero_coeffs_loss_intercept <- as.numeric(dag_coeffs$zero_coeffs_loss1)
pois_coeffs_loss_intercept <- as.numeric(dag_coeffs$pois_coeffs_loss1)
zero_coeffs_loss_paret2 <- as.numeric(dag_coeffs$zero_coeffs_loss2)
pois_coeffs_loss_paret2 <- as.numeric(dag_coeffs$pois_coeffs_loss2)
zero_coeffs_loss_parorigin <- as.numeric(dag_coeffs$zero_coeffs_loss3)
pois_coeffs_loss_parorigin <- as.numeric(dag_coeffs$pois_coeffs_loss3)
zero_coeffs_loss_magecat1 <- as.numeric(dag_coeffs$zero_coeffs_loss4)
pois_coeffs_loss_magecat1 <- as.numeric(dag_coeffs$pois_coeffs_loss4)
zero_coeffs_loss_magecat2 <- as.numeric(dag_coeffs$zero_coeffs_loss5)
pois_coeffs_loss_magecat2 <- as.numeric(dag_coeffs$pois_coeffs_loss5)
zero_coeffs_loss_lag <- as.numeric(dag_coeffs$zero_coeffs_loss6)
pois_coeffs_loss_lag <- as.numeric(dag_coeffs$pois_coeffs_loss6)

zero_coeffs_ses_intercept <- as.numeric(dag_coeffs$zero_coeffs_ses1)
pois_coeffs_ses_intercept <- as.numeric(dag_coeffs$pois_coeffs_ses1)
zero_coeffs_ses_paret2 <- as.numeric(dag_coeffs$zero_coeffs_ses2)
pois_coeffs_ses_paret2 <- as.numeric(dag_coeffs$pois_coeffs_ses2)
zero_coeffs_ses_parorigin <- as.numeric(dag_coeffs$zero_coeffs_ses3)
pois_coeffs_ses_parorigin <- as.numeric(dag_coeffs$pois_coeffs_ses3)
zero_coeffs_ses_magecat1 <- as.numeric(dag_coeffs$zero_coeffs_ses4)
pois_coeffs_ses_magecat1 <- as.numeric(dag_coeffs$pois_coeffs_ses4)
zero_coeffs_ses_magecat2 <- as.numeric(dag_coeffs$zero_coeffs_ses5)
pois_coeffs_ses_magecat2 <- as.numeric(dag_coeffs$pois_coeffs_ses5)
zero_coeffs_ses_lag <- as.numeric(dag_coeffs$zero_coeffs_ses6)
pois_coeffs_ses_lag <- as.numeric(dag_coeffs$pois_coeffs_ses6)

zero_coeffs_dyn_intercept <- as.numeric(dag_coeffs$zero_coeffs_dyn1)
pois_coeffs_dyn_intercept <- as.numeric(dag_coeffs$pois_coeffs_dyn1)
zero_coeffs_dyn_paret2 <- as.numeric(dag_coeffs$zero_coeffs_dyn2)
pois_coeffs_dyn_paret2 <- as.numeric(dag_coeffs$pois_coeffs_dyn2)
zero_coeffs_dyn_parorigin <- as.numeric(dag_coeffs$zero_coeffs_dyn3)
pois_coeffs_dyn_parorigin <- as.numeric(dag_coeffs$pois_coeffs_dyn3)
zero_coeffs_dyn_magecat1 <- as.numeric(dag_coeffs$zero_coeffs_dyn4)
pois_coeffs_dyn_magecat1 <- as.numeric(dag_coeffs$pois_coeffs_dyn4)
zero_coeffs_dyn_magecat2 <- as.numeric(dag_coeffs$zero_coeffs_dyn5)
pois_coeffs_dyn_magecat2 <- as.numeric(dag_coeffs$pois_coeffs_dyn5)
zero_coeffs_dyn_loss <- as.numeric(dag_coeffs$zero_coeffs_dyn6)
pois_coeffs_dyn_loss <- as.numeric(dag_coeffs$pois_coeffs_dyn6)
zero_coeffs_dyn_ses <- as.numeric(dag_coeffs$zero_coeffs_dyn7)
pois_coeffs_dyn_ses <- as.numeric(dag_coeffs$pois_coeffs_dyn7)
zero_coeffs_dyn_lag <- as.numeric(dag_coeffs$zero_coeffs_dyn8)
pois_coeffs_dyn_lag <- as.numeric(dag_coeffs$pois_coeffs_dyn8)

max_loss <- as.numeric(dag_coeffs$`max_loss.99.5%`)
max_ses <- as.numeric(dag_coeffs$`max_ses.99.5%`)
max_dyn <- as.numeric(dag_coeffs$`max_dyn.99.5%`)
lim_ses <- as.numeric(dag_coeffs$lim_ses)
gdm_pos_prop <- as.numeric(dag_coeffs$gdm_table.yes)

parorigin_zero <- as.numeric(dag_coeffs2$parorigin_zero.0)
parorigin_one <- as.numeric(dag_coeffs2$parorigin_one.1)
magecat_zero <- as.numeric(dag_coeffs2$magecat_zero.0)
magecat_one <- as.numeric(dag_coeffs2$magecat_one.1)
magecat_two <- as.numeric(dag_coeffs2$magecat_two.2)
paret2_parorigin0_zero <- as.numeric(dag_coeffs2$paret2_parorigin0_zero)
paret2_parorigin0_one <- as.numeric(dag_coeffs2$paret2_parorigin0_one)
paret2_parorigin1_zero <- as.numeric(dag_coeffs2$paret2_parorigin1_zero)
paret2_parorigin1_one <- as.numeric(dag_coeffs2$paret2_parorigin1_one)
paret2_coeff_zero <- paret2_parorigin1_zero - paret2_parorigin0_zero
paret2_coeff_one <- paret2_parorigin1_one - paret2_parorigin0_one

# Define the DAG structure
dag <- DAG.empty() +
    node("par.origin", distr = "rcat.b0", prob = c(parorigin_zero, parorigin_one), replaceNAw0 = TRUE) +
    node("mage.cat", distr = "rcat.b0", prob = c(magecat_zero, magecat_one, magecat_two), replaceNAw0 = TRUE) +
    node("pare.t2", distr = "rcat.b0", prob = c(paret2_parorigin0_zero + paret2_coeff_zero * par.origin, paret2_parorigin0_one + paret2_coeff_one * par.origin), replaceNAw0 = TRUE) +
    node("zero.inflation.loss",
        t = 0, distr = "rbern",
        prob = plogis(zero_coeffs_loss_intercept + zero_coeffs_loss_paret2 * pare.t2 + zero_coeffs_loss_parorigin * par.origin + zero_coeffs_loss_magecat1 * (mage.cat == 1) + zero_coeffs_loss_magecat2 * (mage.cat == 2)), EFU = FALSE, replaceNAw0 = TRUE
    ) +
    node("poisson.loss",
        t = 0, distr = "rpois",
        lambda = exp(pois_coeffs_loss_intercept + pois_coeffs_loss_paret2 * pare.t2 + pois_coeffs_loss_parorigin * par.origin + pois_coeffs_loss_magecat1 * (mage.cat == 1) + pois_coeffs_loss_magecat2 * (mage.cat == 2)), EFU = FALSE, replaceNAw0 = TRUE
    ) +
    node("Loss", t = 0, distr = "rconst", const = (zero.inflation.loss[t] * 0 + (1 - zero.inflation.loss[t]) * poisson.loss[t]), EFU = FALSE, replaceNAw0 = TRUE) +
    node("zero.inflation.loss",
        t = 1:n_time, distr = "rbern",
        prob = plogis(zero_coeffs_loss_intercept + zero_coeffs_loss_paret2 * pare.t2 + zero_coeffs_loss_parorigin * par.origin + zero_coeffs_loss_magecat1 * (mage.cat == 1) + zero_coeffs_loss_magecat2 * (mage.cat == 2) + zero_coeffs_loss_lag * min(Loss[t - 1], max_loss)), EFU = FALSE, replaceNAw0 = TRUE
    ) +
    node("poisson.loss",
        t = 1:n_time, distr = "rpois",
        lambda = exp(pois_coeffs_loss_intercept + pois_coeffs_loss_paret2 * pare.t2 + pois_coeffs_loss_parorigin * par.origin + pois_coeffs_loss_magecat1 * (mage.cat == 1) + pois_coeffs_loss_magecat2 * (mage.cat == 2) + pois_coeffs_loss_lag * min(Loss[t - 1], max_loss)), EFU = FALSE, replaceNAw0 = TRUE
    ) +
    node("Loss", t = 1:n_time, distr = "rconst", const = (zero.inflation.loss[t] * 0 + (1 - zero.inflation.loss[t]) * poisson.loss[t]), EFU = FALSE, replaceNAw0 = TRUE) +
    node("zero.inflation.ses",
        t = 0, distr = "rbern",
        prob = plogis(zero_coeffs_ses_intercept + zero_coeffs_ses_paret2 * pare.t2 + zero_coeffs_ses_parorigin * par.origin + zero_coeffs_ses_magecat1 * (mage.cat == 1) + zero_coeffs_ses_magecat2 * (mage.cat == 2)), EFU = FALSE, replaceNAw0 = TRUE
    ) +
    node("poisson.ses",
        t = 0, distr = "rpois",
        lambda = exp(pois_coeffs_ses_intercept + pois_coeffs_ses_paret2 * pare.t2 + pois_coeffs_ses_parorigin * par.origin + pois_coeffs_ses_magecat1 * (mage.cat == 1) + pois_coeffs_ses_magecat2 * (mage.cat == 2)), EFU = FALSE, replaceNAw0 = TRUE
    ) +
    node("SES", t = 0, distr = "rconst", const = (zero.inflation.ses[t] * 0 + (1 - zero.inflation.ses[t]) * min(poisson.ses[t], lim_ses)), EFU = FALSE, replaceNAw0 = TRUE) +
    node("zero.inflation.ses",
        t = 1:n_time, distr = "rbern",
        prob = plogis(zero_coeffs_ses_intercept + zero_coeffs_ses_paret2 * pare.t2 + zero_coeffs_ses_parorigin * par.origin + zero_coeffs_ses_magecat1 * (mage.cat == 1) + zero_coeffs_ses_magecat2 * (mage.cat == 2) + zero_coeffs_ses_lag * min(SES[t - 1], max_ses)), EFU = FALSE, replaceNAw0 = TRUE
    ) +
    node("poisson.ses",
        t = 1:n_time, distr = "rpois",
        lambda = exp(pois_coeffs_ses_intercept + pois_coeffs_ses_paret2 * pare.t2 + pois_coeffs_ses_parorigin * par.origin + pois_coeffs_ses_magecat1 * (mage.cat == 1) + pois_coeffs_ses_magecat2 * (mage.cat == 2) + pois_coeffs_ses_lag * min(SES[t - 1], max_ses)), EFU = FALSE, replaceNAw0 = TRUE
    ) +
    node("SES", t = 1:n_time, distr = "rconst", const = (zero.inflation.ses[t] * 0 + (1 - zero.inflation.ses[t]) * min(poisson.ses[t], lim_ses)), EFU = FALSE, replaceNAw0 = TRUE) +
    node("zero.inflation.dyn",
        t = 0, distr = "rbern",
        prob = plogis(zero_coeffs_dyn_intercept + zero_coeffs_dyn_paret2 * pare.t2 + zero_coeffs_dyn_parorigin * par.origin + zero_coeffs_dyn_magecat1 * (mage.cat == 1) + zero_coeffs_dyn_magecat2 * (mage.cat == 2) + zero_coeffs_dyn_loss * min(Loss[t], max_loss) + zero_coeffs_dyn_ses * min(SES[t], max_ses)), EFU = FALSE, replaceNAw0 = TRUE
    ) +
    node("poisson.dyn",
        t = 0, distr = "rpois",
        lambda = exp(pois_coeffs_dyn_intercept + pois_coeffs_dyn_paret2 * pare.t2 + pois_coeffs_dyn_parorigin * par.origin + pois_coeffs_dyn_magecat1 * (mage.cat == 1) + pois_coeffs_dyn_magecat2 * (mage.cat == 2) + pois_coeffs_dyn_loss * min(Loss[t], max_loss) + pois_coeffs_dyn_ses * min(SES[t], max_ses)), EFU = FALSE, replaceNAw0 = TRUE
    ) +
    node("Dynamic", t = 0, distr = "rconst", const = (zero.inflation.dyn[t] * 0 + (1 - zero.inflation.dyn[t]) * poisson.dyn[t]), EFU = FALSE, replaceNAw0 = TRUE) +
    node("zero.inflation.dyn",
        t = 1:n_time, distr = "rbern",
        prob = plogis(zero_coeffs_dyn_intercept + zero_coeffs_dyn_paret2 * pare.t2 + zero_coeffs_dyn_parorigin * par.origin + zero_coeffs_dyn_magecat1 * (mage.cat == 1) + zero_coeffs_dyn_magecat2 * (mage.cat == 2) + zero_coeffs_dyn_loss * min(Loss[t], max_loss) + zero_coeffs_dyn_ses * min(SES[t], max_ses) + zero_coeffs_dyn_lag * min(Dynamic[t - 1], max_dyn)), EFU = FALSE, replaceNAw0 = TRUE
    ) +
    node("poisson.dyn",
        t = 1:n_time, distr = "rpois",
        lambda = exp(pois_coeffs_dyn_intercept + pois_coeffs_dyn_paret2 * pare.t2 + pois_coeffs_dyn_parorigin * par.origin + pois_coeffs_dyn_magecat1 * (mage.cat == 1) + pois_coeffs_dyn_magecat2 * (mage.cat == 2) + pois_coeffs_dyn_loss * min(Loss[t], max_loss) + pois_coeffs_dyn_ses * min(SES[t], max_ses) + pois_coeffs_dyn_lag * min(Dynamic[t - 1], max_dyn)), EFU = FALSE, replaceNAw0 = TRUE
    ) +
    node("Dynamic", t = 1:n_time, distr = "rconst", const = (zero.inflation.dyn[t] * 0 + (1 - zero.inflation.dyn[t]) * poisson.dyn[t]), EFU = FALSE, replaceNAw0 = TRUE)


dag <- set.DAG(dag)

# Simulate from the DAG
dat_long <- sim(dag, n = n_people)

# Prepare the data for python
data <- dat_long[-1]
x_out <- array(dim = c(n_people, num_timeindep + num_timedep, n_time_out))

for (i in 1:num_timeindep) {
    x_out[, i, ] <- array(replicate(n_time_out, unlist(data[i])), dim = c(n_people, n_time_out))
}
for (i in (num_timeindep + 1):(num_timeindep + num_timedep)) {
    seqi <- seq(from = 3 * i - num_timeindep - 3, by = num_timedep * 3, to = (num_timeindep + n_time_out * num_timedep * 3))
    x_out[, i, ] <- array(unlist(data[seqi]), dim = c(n_people, n_time_out))
}

# Load custom functions
source("Data_simulation/LCP_rule_functions_gdm.R")

# Define function names and corresponding paper names
function_names <- c(
    "critical_1", "sensitive_var4_n2_p5", "weighted_var4_n1",
    "repeat_var6_r4", "repeat2_2_var6", "repeat2_var4_var5",
    "order1diff_var6_var4", "order2_64", "order4_645",
    "timing_1event_0year_45", "timing_2event_1year_45", "timing_2event_4year_45"
)

paper_names <- c(paste0("Period", 1:3), paste0("Repeats", 1:3), paste0("Order", 1:3), paste0("Timing", 1:3))

# Initialize table to store results
prop_table <- c()

# Simulate and store results
for (i in seq_along(function_names)) {
    print(function_names[i])

    # Evaluate the function
    y_out <- eval(parse(text = (paste0(function_names[i], "(x_out)"))))

    # Calculate proportion and store results
    name <- paper_names[i]
    proportion <- sum(y_out) / length(y_out)
    prop_table <- rbind(prop_table, c(name, proportion))

    # Save data to Python
    np$save(paste0("Data_simulation/Simulated_Data/data_gdm_rand", randnum, "_X.npy"), r_to_py(x_out))
    np$save(paste0("Data_simulation/Simulated_Data/data_gdm_", name, "_rand", randnum, "_Y.npy"), r_to_py(y_out))
}

# Write proportions table to CSV
names(prop_table) <- c("Data", "Proportion 1s")
write.csv(prop_table, paste0("Data_simulation/Simulated_Data/gdm_data_rand", randnum, ".csv"))