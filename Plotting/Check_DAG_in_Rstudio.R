
# Load simcausal library
library(simcausal)
library(reticulate)
np <- import("numpy")


# #load required packages
# library(haven)
# library(dplyr)
# library(lme4)
# library(MarginalMediation)
# library(tidyverse)
# library(ggplot2)
# library(forecast)
# library(lme4)
# library(pscl)
# library(MASS)
# library(fitdistrplus)
# library(coin)
# library(tidyr)
# library(nlme)
# require(gamlss.dist)
# Sys.setenv(LANG="en")

num_timeindep <- 3 # number of variables in model
num_timedep <- 3 # number of variables in model
n_people <- 200000 # number of people in model
n_time_out <- 16 #number of time points in model
n_time <- n_time_out - 1
num_var <- num_timedep + num_timeindep


randnum <- 7

set.seed(randnum)

# # Creates DAG (DAG designed on DANLIFE data for GDM)
# dag <- DAG.empty() +
#     node("mumage", distr = "rcat.b0", prob = c(0.040128, 0.751853, 0.208019), replaceNAw0 = TRUE) +  # <20, 20-30, >30  ## rcat.factor
#     node("parentaldiabetes", distr = "rcat.b0", prob = c(0.893539, 0.106461), replaceNAw0 = TRUE) +
#     node("parentalorigin", distr = "rcat.b0", prob = c(0.985538, 0.014462), replaceNAw0 = TRUE) +
#     node("ACEs", distr = "rbern", prob = (depends on mumage, parental origin and parental diabetes - 5 groups), replaceNAw0 = TRUE) +  ## runif? (min and max), rnorm, mean and sd?
#     node("mumagefirst", distr = "rbern", prob = c(mumage +/- **some disribution from data depending on parental origin and aces**), replaceNAw0 = TRUE) +
#     node("GDM", t = 0:n_time, distr = "rbern", prob = (depends on aces, mumagefirst, parental origin and parental diabetes), EFU = FALSE, replaceNAw0 = TRUE)


dag_coeffs <- read.csv("C:/Users/hlc17/Documents/PostDoc/Simulation paper/Downloads from DANLIFE/DAG_params.csv")
dag_coeffs <- data.frame(strsplit(unlist(dag_coeffs),","))
names(dag_coeffs) <- dag_coeffs[1,]
dag_coeffs <- dag_coeffs[-1,]



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


dag_coeffs2 <- read.csv("C:/Users/hlc17/Documents/PostDoc/Simulation paper/Downloads from DANLIFE/DAG_params2.csv")
dag_coeffs2 <- data.frame(strsplit(unlist(dag_coeffs2),","))
names(dag_coeffs2) <- dag_coeffs2[1,]
dag_coeffs2 <- dag_coeffs2[-1,]

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

#     node("mumage", distr = "rcat.b0", prob = c(0.040128, 0.751853, 0.208019), replaceNAw0 = TRUE) +  # <20, 20-30, >30  ## rcat.factor
#     node("parentaldiabetes", distr = "rcat.b0", prob = c(0.893539, 0.106461), replaceNAw0 = TRUE) +
#     node("parentalorigin", distr = "rcat.b0", prob = c(0.985538, 0.014462), replaceNAw0 = TRUE) +


dag <- DAG.empty() +
  node("par.origin", distr = "rcat.b0", prob = c(parorigin_zero, parorigin_one), replaceNAw0 = TRUE) +
  node("mage.cat", distr = "rcat.b0", prob = c(magecat_zero, magecat_one, magecat_two), replaceNAw0 = TRUE) +
  node("pare.t2", distr = "rcat.b0", prob = c(paret2_parorigin0_zero + paret2_coeff_zero * par.origin, paret2_parorigin0_one + paret2_coeff_one * par.origin), replaceNAw0 = TRUE) +
  node("zero.inflation.loss", t=0, distr = "rbern",
       prob = plogis(zero_coeffs_loss_intercept + zero_coeffs_loss_paret2 * pare.t2 + zero_coeffs_loss_parorigin * par.origin +zero_coeffs_loss_magecat1 * (mage.cat == 1) + zero_coeffs_loss_magecat2 * (mage.cat == 2)), EFU=FALSE, replaceNAw0 = TRUE) +
  node("poisson.loss", t=0, distr = "rpois",
       lambda = exp(pois_coeffs_loss_intercept + pois_coeffs_loss_paret2 * pare.t2 + pois_coeffs_loss_parorigin * par.origin +pois_coeffs_loss_magecat1 * (mage.cat == 1) + pois_coeffs_loss_magecat2 * (mage.cat == 2)), EFU=FALSE, replaceNAw0 = TRUE) +
  node("Loss", t=0, distr = "rconst", const = (zero.inflation.loss[t] * 0 + (1 - zero.inflation.loss[t]) * poisson.loss[t]), EFU=FALSE, replaceNAw0 = TRUE) +
  node("zero.inflation.loss", t=1:n_time, distr = "rbern",
       prob = plogis(zero_coeffs_loss_intercept + zero_coeffs_loss_paret2 * pare.t2 + zero_coeffs_loss_parorigin * par.origin +zero_coeffs_loss_magecat1 * (mage.cat == 1) + zero_coeffs_loss_magecat2 * (mage.cat == 2)  + zero_coeffs_loss_lag * min(Loss[t-1], max_loss)), EFU=FALSE, replaceNAw0 = TRUE) +
  node("poisson.loss", t=1:n_time, distr = "rpois",
       lambda = exp(pois_coeffs_loss_intercept + pois_coeffs_loss_paret2 * pare.t2 + pois_coeffs_loss_parorigin * par.origin +pois_coeffs_loss_magecat1 * (mage.cat == 1) + pois_coeffs_loss_magecat2 * (mage.cat == 2) + pois_coeffs_loss_lag * min(Loss[t-1], max_loss)), EFU=FALSE, replaceNAw0 = TRUE) +
  node("Loss", t=1:n_time, distr = "rconst", const = (zero.inflation.loss[t] * 0 + (1 - zero.inflation.loss[t]) * poisson.loss[t]), EFU=FALSE, replaceNAw0 = TRUE) +
  node("zero.inflation.ses", t=0, distr = "rbern",
       prob = plogis(zero_coeffs_ses_intercept + zero_coeffs_ses_paret2 * pare.t2 + zero_coeffs_ses_parorigin * par.origin +zero_coeffs_ses_magecat1 * (mage.cat == 1) + zero_coeffs_ses_magecat2 * (mage.cat == 2)), EFU=FALSE, replaceNAw0 = TRUE) +
  node("poisson.ses", t=0, distr = "rpois",
       lambda = exp(pois_coeffs_ses_intercept + pois_coeffs_ses_paret2 * pare.t2 + pois_coeffs_ses_parorigin * par.origin +pois_coeffs_ses_magecat1 * (mage.cat == 1) + pois_coeffs_ses_magecat2 * (mage.cat == 2)), EFU=FALSE, replaceNAw0 = TRUE) +
  node("SES", t=0, distr = "rconst", const = (zero.inflation.ses[t] * 0 + (1 - zero.inflation.ses[t]) * min(poisson.ses[t],lim_ses)), EFU=FALSE, replaceNAw0 = TRUE) +
  node("zero.inflation.ses", t=1:n_time, distr = "rbern",
       prob = plogis(zero_coeffs_ses_intercept + zero_coeffs_ses_paret2 * pare.t2 + zero_coeffs_ses_parorigin * par.origin +zero_coeffs_ses_magecat1 * (mage.cat == 1) + zero_coeffs_ses_magecat2 * (mage.cat == 2)  + zero_coeffs_ses_lag * min(SES[t-1], max_ses)), EFU=FALSE, replaceNAw0 = TRUE) +
  node("poisson.ses", t=1:n_time, distr = "rpois",
       lambda = exp(pois_coeffs_ses_intercept + pois_coeffs_ses_paret2 * pare.t2 + pois_coeffs_ses_parorigin * par.origin +pois_coeffs_ses_magecat1 * (mage.cat == 1) + pois_coeffs_ses_magecat2 * (mage.cat == 2) + pois_coeffs_ses_lag * min(SES[t-1], max_ses)), EFU=FALSE, replaceNAw0 = TRUE) +
  node("SES", t=1:n_time, distr = "rconst", const = (zero.inflation.ses[t] * 0 + (1 - zero.inflation.ses[t]) * min(poisson.ses[t],lim_ses)), EFU=FALSE, replaceNAw0 = TRUE) +
  node("zero.inflation.dyn", t=0, distr = "rbern",
       prob = plogis(zero_coeffs_dyn_intercept + zero_coeffs_dyn_paret2 * pare.t2 + zero_coeffs_dyn_parorigin * par.origin +zero_coeffs_dyn_magecat1 * (mage.cat == 1) + zero_coeffs_dyn_magecat2 * (mage.cat == 2) + zero_coeffs_dyn_loss * min(Loss[t], max_loss) + zero_coeffs_dyn_ses * min(SES[t], max_ses)), EFU=FALSE, replaceNAw0 = TRUE) +
  node("poisson.dyn", t=0, distr = "rpois",
       lambda = exp(pois_coeffs_dyn_intercept + pois_coeffs_dyn_paret2 * pare.t2 + pois_coeffs_dyn_parorigin * par.origin +pois_coeffs_dyn_magecat1 * (mage.cat == 1) + pois_coeffs_dyn_magecat2 * (mage.cat == 2) + pois_coeffs_dyn_loss * min(Loss[t], max_loss) + pois_coeffs_dyn_ses * min(SES[t], max_ses)), EFU=FALSE, replaceNAw0 = TRUE) +
  node("Dynamic", t=0, distr = "rconst", const = (zero.inflation.dyn[t] * 0 + (1 - zero.inflation.dyn[t]) * poisson.dyn[t]), EFU=FALSE, replaceNAw0 = TRUE) +
  node("zero.inflation.dyn", t=1:n_time, distr = "rbern",
       prob = plogis(zero_coeffs_dyn_intercept + zero_coeffs_dyn_paret2 * pare.t2 + zero_coeffs_dyn_parorigin * par.origin +zero_coeffs_dyn_magecat1 * (mage.cat == 1) + zero_coeffs_dyn_magecat2 * (mage.cat == 2) + zero_coeffs_dyn_loss * min(Loss[t], max_loss) + zero_coeffs_dyn_ses * min(SES[t], max_ses) + zero_coeffs_dyn_lag * min(Dynamic[t-1], max_dyn)), EFU=FALSE, replaceNAw0 = TRUE) +
  node("poisson.dyn", t=1:n_time, distr = "rpois",
       lambda = exp(pois_coeffs_dyn_intercept + pois_coeffs_dyn_paret2 * pare.t2 + pois_coeffs_dyn_parorigin * par.origin +pois_coeffs_dyn_magecat1 * (mage.cat == 1) + pois_coeffs_dyn_magecat2 * (mage.cat == 2) + pois_coeffs_dyn_loss * min(Loss[t], max_loss) + pois_coeffs_dyn_ses * min(SES[t], max_ses) + pois_coeffs_dyn_lag * min(Dynamic[t-1], max_dyn)), EFU=FALSE, replaceNAw0 = TRUE) +
  node("Dynamic", t=1:n_time, distr = "rconst", const = (zero.inflation.dyn[t] * 0 + (1 - zero.inflation.dyn[t]) * poisson.dyn[t]), EFU=FALSE, replaceNAw0 = TRUE) 


#  node("mental_health", t = 0, distr = "rbern", prob = (0.1 + 0.1 * sex + 0.05 * poverty), replaceNAw0 = TRUE) +
#  node("mental_health", t = 1:n_time, distr = "rbern", prob = (0.1 + 0.1 * sex + 0.05 * poverty + 0.1 * mental_health[t - 1]), replaceNAw0 = TRUE) +


#a.	Material deprivation (Family poverty and parental long-term unemployment)
#b.	Loss of threat of loss (death of parent, death of sibling, somatic illness of parent, somatic illness of sibling)
#c.	Family dynamics (foster care, sibling psychiatric illness, parent psychiatric illness, parental alcohol abuse, parental drug abuse, maternal separation)


dag <- set.DAG(dag)

# plot DAG
#plotDAG(dag)
dat_long <- sim(dag, n = n_people)

data <- dat_long[-1]
x_out <- array(dim = c(n_people, num_timeindep + num_timedep, n_time_out))
# y_out <- array(dim = c(n_people))

for (i in 1:num_timeindep){
  x_out[, i, ] <- array(replicate(n_time_out, unlist(data[i])), dim = c(n_people, n_time_out))
}
for (i in (num_timeindep + 1):(num_timeindep + num_timedep)){
  seqi <- seq(from = 3 * i - num_timeindep - 3, by = num_timedep * 3, to = (num_timeindep + n_time_out * num_timedep * 3))
  print(seqi)
  print(names(data)[seqi])
  x_out[, i, ] <- array(unlist(data[seqi]), dim = c(n_people, n_time_out))
}

# y_out <- array(unlist(data[seq(from = 10, by = 5, to = dim(data)[2])]), dim = c(dim(data)[1], n_time + 1))


### want gdm similar to 2.5%

#########################################################################################
data_plot <- data
data_plot$ID <- seq(1, nrow(data_plot))

data_plot <- data_plot[,!grepl("poisson",names(data_plot))]
data_plot <- data_plot[,!grepl("zero",names(data_plot))]

data_plot <- data_plot %>% pivot_longer(cols = Loss_0:Dynamic_15,
                           names_to = "ACE",
                           values_to = "Count") %>% 
  separate(col=ACE, into = c("ACE","Time"), sep=-2)

library(stringr)
data_plot$ACE <- str_remove(data_plot$ACE, "_")
data_plot$Time <- str_remove(data_plot$Time, "_")

data_plot <-data_plot %>% pivot_wider(names_from = ACE, values_from = Count)

data_plot$par.origin <- factor(data_plot$par.origin)

ggplot(data_plot, aes(fill=par.origin, y=Loss, x=Time)) + 
  geom_bar(position="dodge", stat="identity")



#########################################################################################

# criticaln <- function(x, time_start, period, var_num) {
#   n_people <- dim(x)[1]
#   y_output <- array(0, n_people)
#   time_end <- time_start + period
# 
#   for (i in seq_len(n_people)) {
#     if (sum(x[i, var_num, time_start:time_end]) == 0) {
#       y_output[i] <- 1
#     }
#   }
#   return(y_output)
# }
# 
# print("criticaln")
# for (var_num in 4:6){
#   for (time_start in c(1, 6)){
#   for (period in c(5, 10, 15)){
#     if (period + time_start > n_time_out){period <- n_time_out - time_start}
#     y_test <- criticaln(x=x_out, time_start=time_start, period=period, var_num = var_num)
#     print(c("var = ",var_num,"; time_start = ",time_start,"; period = ",period,"; prop = ",sum(y_test) / length(y_test)))
#   } }}
# 
# ### all a bit high - cause lots of zeros - maybe can't use

# criticalnopp <- function(x, time_start, period, var_num) {
#   n_people <- dim(x)[1]
#   y_output <- array(0, n_people)
#   time_end <- time_start + period
#   
#   for (i in seq_len(n_people)) {
#     if (sum(x[i, var_num, time_start:time_end]) != 0) {
#       y_output[i] <- 1
#     }
#   }
#   return(y_output)
# }
# 
# print("criticalnopp")
# for (var_num in 4:6){
#   time_start = 1
#     for (period in c(2, 3, 5)){
#       if (period + time_start > n_time_out){period <- n_time_out - time_start}
#       y_test <- criticalnopp(x=x_out, time_start=time_start, period=period, var_num = var_num)
#       print(c("var = ",var_num,"; time_start = ",time_start,"; period = ",period,"; prop = ",sum(y_test) / length(y_test)))
#     } }
# 
# ### all a bit high - cause lots of zeros - maybe can't use - let's also give this a miss (it's just snesitive period now anyway)


sensitiven <- function(x, n, var_num, period, time_start) {
  n_people <- dim(x)[1]
  y_output <- array(0, n_people)
  time_end <- time_start + period
  for (i in seq_len(n_people)) {
    if (sum(x[i, var_num, time_start:time_end]) >= n) {
      y_output[i] <- 1
    }
  }
  return(y_output)
}

## critical can be sensitive with 1

print("criticaln")
for (var_num in 4:6){
  n = 1
  if (var_num==4 ){
    for (time_start in c(1,5)){
      for (period in c(1, 2, 5)){
        if (period + time_start > n_time_out){period <- n_time_out - time_start}
        y_test <- sensitiven(x=x_out, time_start=time_start, period=period, var_num = var_num, n=n)
        print(c("var = ",var_num,"; n = ",n,"; time_start = ",time_start,"; period = ",period,"; prop = ",sum(y_test) / length(y_test)))
      } }} else if ( var_num==5){
        for (time_start in c(1,5)){
          for (period in c(1, 2, 5)){
            if (period + time_start > n_time_out){period <- n_time_out - time_start}
            y_test <- sensitiven(x=x_out, time_start=time_start, period=period, var_num = var_num, n=n)
            print(c("var = ",var_num,"; n = ",n,"; time_start = ",time_start,"; period = ",period,"; prop = ",sum(y_test) / length(y_test)))
          } }} else {
            for (time_start in c(1,5)){
              for (period in c(1, 2, 5)){
                if (period + time_start > n_time_out){period <- n_time_out - time_start}
                y_test <- sensitiven(x=x_out, time_start=time_start, period=period, var_num = var_num, n=n)
                print(c("var = ",var_num,"; n = ",n,"; time_start = ",time_start,"; period = ",period,"; prop = ",sum(y_test) / length(y_test)))
              } }} }

## okay let's take [1] "var = "          "4"               "; n = "          "1"               "; time_start = " "1"               "; period = "     "2"               "; prop = "       "0.04582" 
critical_1 <- function(x){
  return(sensitiven(x=x, time_start=1, period=1, var_num = 4, n=1))
}
critical_2 <- function(x){
  return(sensitiven(x=x, time_start=1, period=2, var_num = 4, n=1))
}

critical_1_var6 <- function(x){
  return(sensitiven(x=x, time_start=1, period=1, var_num = 6, n=1))
}

print("sensitiven")
for (var_num in 4:6){
  time_start=1
  if (var_num==4 ){
  for (n in 1:2){
      for (period in c(2, 5)){
        if (period + time_start > n_time_out){period <- n_time_out - time_start}
        y_test <- sensitiven(x=x_out, time_start=time_start, period=period, var_num = var_num, n=n)
        print(c("var = ",var_num,"; n = ",n,"; time_start = ",time_start,"; period = ",period,"; prop = ",sum(y_test) / length(y_test)))
      } }} else if ( var_num==6){
        for (n in 2:3){
            for (period in c(2, 5)){
              if (period + time_start > n_time_out){period <- n_time_out - time_start}
              y_test <- sensitiven(x=x_out, time_start=time_start, period=period, var_num = var_num, n=n)
              print(c("var = ",var_num,"; n = ",n,"; time_start = ",time_start,"; period = ",period,"; prop = ",sum(y_test) / length(y_test)))
            } }} else {
        for (n in 3){
            for (period in c(2, 5)){
              if (period + time_start > n_time_out){period <- n_time_out - time_start}
              y_test <- sensitiven(x=x_out, time_start=time_start, period=period, var_num = var_num, n=n)
              print(c("var = ",var_num,"; n = ",n,"; time_start = ",time_start,"; period = ",period,"; prop = ",sum(y_test) / length(y_test)))
            } }} }


sensitive_var4_n2_p5 <- function(x){
  return(sensitiven(x=x, time_start=1, period=5, var_num = 4, n=2))
}

sensitive_var5_n3_p2 <- function(x){
  return(sensitiven(x=x, time_start=1, period=2, var_num = 5, n=3))
}

sensitive_var6_n2_p2 <- function(x){
  return(sensitiven(x=x, time_start=1, period=2, var_num = 6, n=2))
}

sensitive_var6_n3_p5 <- function(x){
  return(sensitiven(x=x, time_start=1, period=5, var_num = 6, n=3))
}

# sensitiven_soft <- function(x, n, var_num, period, time_start) {
#   ## like sensitive but with probability and events outside influence
#   n_people <- dim(x)[1]
#   n_time <- dim(x)[3]
#   y_output <- array(0, n_people)
#   time_end <- time_start + period
#   for (i in seq_len(n_people)) {
#     # if (sum(x[i, var_num, time_start:time_end]) >= n) {
#     #   y_output[i] <- 1
#     # }
#     y_output[i] <- rbern(n = 1, p = min(0.9 * sum(x[i, var_num, time_start:time_end]) + 0.1 * (sum(x[i, var_num, 1:n_time])-sum(x[i, var_num, time_start:time_end])),1))
#   }
#   return(y_output)
# }
# 
# print("sensitiven_soft")
# for (var_num in 4:6){
#   time_start=1
#   if (var_num==4 ){
#     for (n in 1:2){
#       for (period in c(2, 5)){
#         if (period + time_start > n_time_out){period <- n_time_out - time_start}
#         y_test <- sensitiven_soft(x=x_out, time_start=time_start, period=period, var_num = var_num, n=n)
#         print(c("var = ",var_num,"; n = ",n,"; time_start = ",time_start,"; period = ",period,"; prop = ",sum(y_test) / length(y_test)))
#       } }} else if ( var_num==6){
#         for (n in 2:3){
#           for (period in c(2, 5)){
#             if (period + time_start > n_time_out){period <- n_time_out - time_start}
#             y_test <- sensitiven_soft(x=x_out, time_start=time_start, period=period, var_num = var_num, n=n)
#             print(c("var = ",var_num,"; n = ",n,"; time_start = ",time_start,"; period = ",period,"; prop = ",sum(y_test) / length(y_test)))
#           } }} else {
#             for (n in 3:4){
#               for (period in c(2, 5)){
#                 if (period + time_start > n_time_out){period <- n_time_out - time_start}
#                 y_test <- sensitiven_soft(x=x_out, time_start=time_start, period=period, var_num = var_num, n=n)
#                 print(c("var = ",var_num,"; n = ",n,"; time_start = ",time_start,"; period = ",period,"; prop = ",sum(y_test) / length(y_test)))
#               } }} }
# 
# 
# sensitiven_soft2 <- function(x, n, var_num, period, time_start) {
#   ## like sensitive but with probability and events outside influence
#   n_people <- dim(x)[1]
#   y_output <- array(0, n_people)
#   time_end <- time_start + period
#   for (i in seq_len(n_people)) {
#     # if (sum(x[i, var_num, time_start:time_end]) >= n) {
#     #   y_output[i] <- 1
#     # }
#     y_output[i] <- rbern(n = 1, p = plogis(sum(x[i, var_num, time_start:time_end]) + 0.1 * (sum(x[i, var_num, ])-sum(x[i, var_num, time_start:time_end]))))
#   }
#   return(y_output)
# }
# 
# print("sensitiven_soft2")
# for (var_num in 4:6){
#   time_start=1
#   if (var_num==4 ){
#     for (n in 1:2){
#       for (period in c(2, 5)){
#         if (period + time_start > n_time_out){period <- n_time_out - time_start}
#         y_test <- sensitiven_soft2(x=x_out, time_start=time_start, period=period, var_num = var_num, n=n)
#         print(c("var = ",var_num,"; n = ",n,"; time_start = ",time_start,"; period = ",period,"; prop = ",sum(y_test) / length(y_test)))
#       } }} else if ( var_num==6){
#         for (n in 2:3){
#           for (period in c(2, 5)){
#             if (period + time_start > n_time_out){period <- n_time_out - time_start}
#             y_test <- sensitiven_soft2(x=x_out, time_start=time_start, period=period, var_num = var_num, n=n)
#             print(c("var = ",var_num,"; n = ",n,"; time_start = ",time_start,"; period = ",period,"; prop = ",sum(y_test) / length(y_test)))
#           } }} else {
#             for (n in 3:4){
#               for (period in c(2, 5)){
#                 if (period + time_start > n_time_out){period <- n_time_out - time_start}
#                 y_test <- sensitiven_soft2(x=x_out, time_start=time_start, period=period, var_num = var_num, n=n)
#                 print(c("var = ",var_num,"; n = ",n,"; time_start = ",time_start,"; period = ",period,"; prop = ",sum(y_test) / length(y_test)))
#               } }} }
# 
# 
# sensitiven_soft3 <- function(x, n, var_num, period, time_start) {
#   ## like sensitive but with probability and events outside influence
#   n_people <- dim(x)[1]
#   y_output <- array(0, n_people)
#   time_end <- time_start + period
#   for (i in seq_len(n_people)) {
#     # if (sum(x[i, var_num, time_start:time_end]) >= n) {
#     #   y_output[i] <- 1
#     # }
#     y_output[i] <- rbern(n = 1, p = min(sum(x[i, var_num, time_start:time_end])>1 + 0.1 * ((sum(x[i, var_num, ])-sum(x[i, var_num, time_start:time_end]))>1),1))
#   }
#   return(y_output)
# }
# 
# print("sensitiven_soft3")
# for (var_num in 4:6){
#   time_start=1
#   if (var_num==4 ){
#     for (n in 1:2){
#       for (period in c(2, 5)){
#         if (period + time_start > n_time_out){period <- n_time_out - time_start}
#         y_test <- sensitiven_soft3(x=x_out, time_start=time_start, period=period, var_num = var_num, n=n)
#         print(c("var = ",var_num,"; n = ",n,"; time_start = ",time_start,"; period = ",period,"; prop = ",sum(y_test) / length(y_test)))
#       } }} else if ( var_num==6){
#         for (n in 2:3){
#           for (period in c(2, 5)){
#             if (period + time_start > n_time_out){period <- n_time_out - time_start}
#             y_test <- sensitiven_soft3(x=x_out, time_start=time_start, period=period, var_num = var_num, n=n)
#             print(c("var = ",var_num,"; n = ",n,"; time_start = ",time_start,"; period = ",period,"; prop = ",sum(y_test) / length(y_test)))
#           } }} else {
#             for (n in 3:4){
#               for (period in c(2, 5)){
#                 if (period + time_start > n_time_out){period <- n_time_out - time_start}
#                 y_test <- sensitiven_soft3(x=x_out, time_start=time_start, period=period, var_num = var_num, n=n)
#                 print(c("var = ",var_num,"; n = ",n,"; time_start = ",time_start,"; period = ",period,"; prop = ",sum(y_test) / length(y_test)))
#               } }} }


weightedn <- function(x, n, var_num) {
  n_people <- dim(x)[1]
  y_output <- array(0, n_people)
  for (i in seq_len(n_people)) {
    count_infant <- sum(x[i, var_num, 1:2])
    count_youngchild <- sum(x[i, var_num, 3:5])
    count_child <- sum(x[i, var_num, 6:12])
    count_teen <- sum(x[i, var_num, 13:15])
    if ((count_infant + count_youngchild / 2 + count_child / 5 + count_teen / 10) > n) {
      y_output[i] <- 1
    }
  }
  return(y_output)
}

print("weightedn")
for (var_num in 4:6){
  for (n in c(0.5, 1, 1.5)){
    y_test <- weightedn(x=x_out, n=n, var_num = var_num)
    print(c("var = ",var_num,"; n = ",n,"; prop = ",sum(y_test) / length(y_test)))
  } }

weighted_var4_n1 <- function(x) {
  return(weightedn(x=x_out, n=1, var_num = 4))
}

weighted_var4_n15 <- function(x) {
  return(weightedn(x=x_out, n=1.5, var_num = 4))
}

weighted_var6_n15 <- function(x) {
  return(weightedn(x=x_out, n=1.5, var_num = 6))
}

# weightedn_soft <- function(x, n, var_num) {
#   n_people <- dim(x)[1]
#   y_output <- array(0, n_people)
#   for (i in seq_len(n_people)) {
#     count_infant <- sum(x[i, var_num, 1:2])
#     count_youngchild <- sum(x[i, var_num, 3:5])
#     count_child <- sum(x[i, var_num, 6:12])
#     count_teen <- sum(x[i, var_num, 13:15])
#     if ((count_infant + count_youngchild / 2 + count_child / 5 + count_teen / 10) > n) {
#       y_output[i] <- rbern(n=1, p =plogis(count_infant + count_youngchild / 2 + count_child / 5 + count_teen / 10))
#     }
#   }
#   return(y_output)
# }

repeatn <- function(x, n, var_num) {
  # Function for event repeated n times
  
  n_people <- dim(x)[1]
  n_time <- dim(x)[3]
  y_output <- array(0, n_people)
  for (i in seq_len(n_people)) {
    # for each individual in cohort
    # set count to zero
    count <- 0
    for (j in seq_len(n_time)) {
      # for each time point
      if (x[i, var_num, j] > 0) {
        # check if they have an xnum_var event
        count <- count + 1
      } else {
        count <- 0
      }
      if (count == n) {
        # count will be equal to the number of events in a row
        # if count >= n then that individual gets a positive outcome
        y_output[i] <- 1
      }
    }
  }
  return(y_output)
}

print("repeatn")
for (var_num in 4:6){
  for (period in 1:4){
    y_test <- repeatn(x=x_out,  n=period, var_num = var_num)
    print(c("var = ",var_num,"; n = ",period,"; prop = ",sum(y_test) / length(y_test)))
  } }


repeat_var4_r2 <- function(x) {
  return(repeatn(x=x,  n=2, var_num = 4))
}

repeat_var4_r3 <- function(x) {
  return(repeatn(x=x,  n=3, var_num = 4))
}

repeat_var6_r3 <- function(x) {
  return(repeatn(x=x,  n=3, var_num = 6))
}

repeat_var6_r4 <- function(x) {
  return(repeatn(x=x,  n=4, var_num = 6))
}

print("repeatn_hard")
for (var_num1 in 4:6){
  for (var_num2 in 4:6){
    if (var_num1 != var_num2 & var_num2>var_num1){y_test <- as.numeric(repeatn(x=x_out,  n=2, var_num = var_num1)==1 & repeatn(x=x_out,  n=2, var_num = var_num2)==1)
    print(c("var1 = ",var_num1, "; var2 = ",var_num2,"; prop = ",sum(y_test) / length(y_test)))}
  } }

repeat2_var4_var6 <- function(x) {
  return(as.numeric(repeatn(x=x,  n=2, var_num = 6)==1 & repeatn(x=x,  n=2, var_num = 4)==1))
}

repeat2_var4_var5 <- function(x) {
  return(as.numeric(repeatn(x=x,  n=2, var_num = 5)==1 & repeatn(x=x,  n=2, var_num = 4)==1))
}

repeat2_var5_var6 <- function(x) {
  return(as.numeric(repeatn(x=x,  n=2, var_num = 6)==1 & repeatn(x=x,  n=2, var_num = 4)==1))
}



# repeatn_soft <- function(x, var_num) {
#   # Function for event repeated n times
#   
#   n_people <- dim(x)[1]
#   n_time <- dim(x)[3]
#   y_output <- array(0, n_people)
#   for (i in seq_len(n_people)) {
#     # for each individual in cohort
#     # set count to zero
#     count <- 0
#     scores <- c()
#     for (j in seq_len(n_time)) {
#       # for each time point
#       if (x[i, var_num, j] > 0) {
#         # check if they have an xnum_var event
#         count <- count + 1
#       } else {
#         count <- 0
#       }
#       scores <- c(scores, count)
#       #if (count == n) {
#         # count will be equal to the number of events in a row
#         # if count >= n then that individual gets a positive outcome
#         
#       #}
#     }
# 
#     y_output[i] <- rbern(n=1,p=plogis(0.3 * max(scores)^2-4))
#   }
#   return(y_output)
# }
# 
# print("repeatn_soft")
# for (var_num in 4:6){
# 
#     y_test <- repeatn_soft(x=x_out,  var_num = var_num)
#     print(c("var = ",var_num,"; prop = ",sum(y_test) / length(y_test)))
#   } 


repeat2_2 <- function(x, var_num) {
  # function for two events repeated twice
  
  # create empty output array
  n_people <- dim(x)[1]
  n_time <- dim(x)[3]
  y_output <- array(0, n_people)
  for (i in seq_len(n_people)) {
    # for each individual in cohort
    # set counts to zero
    count1 <- 0
    count2 <- 0
    mark <- 0
    outj <- 0
    for (j in seq_len(n_time)) {
      # for each time point
      if (x[i, var_num, j] > 0) {
        # check if they have an xnum_var event
        count1 <- count1 + 1
      } else {
        count1 <- 0
      }
      if (count1 == 2) {
        # if have a repeat of 2 then increase mark and note time
        mark <- 1
        outj <- j
      }
    }
    if (mark == 1 && outj < n_time) {
      # if have at least one repeat
      for (k in (outj + 1):n_time) {
        # search rest of time frame
        if (x[i, var_num, k] > 0) {
          count2 <- count2 + 1
        } else {
          count2 <- 0
        }
        if (count2 == 2) {
          # if have two repeats of two get positive outcome
          y_output[i] <- 1
        }
      }
    }
  }
  return(y_output)
}

print("repeat2_2")
for (var_num in 4:6){
    y_test <- repeat2_2(x=x_out, var_num = var_num)
    print(c("var = ",var_num,"; prop = ",sum(y_test) / length(y_test)))
}

repeat2_2_var4 <- function(x) {
  return(repeat2_2(x=x, var_num = 4))
}

repeat2_2_var6 <- function(x) {
  return(repeat2_2(x=x, var_num = 6))
}


# order1 <- function(x, var_num1, var_num2) {
#   n_people <- dim(x)[1]
#   n_time <- dim(x)[3]
#   y_output <- array(0, n_people)
#   for (i in seq_len(n_people)) {
#     time3 <- c()
#     time4 <- c()
#     for (j in seq_len(n_time)) {
#       if (x[i, var_num1, j] > 0) {
#         time3 <- c(time3, j)
#       }
#       if (x[i, var_num2, j] > 0) {
#         time4 <- c(time4, j)
#       }
#     }
#     if (!(is.null(time3) || is.null(time4))) {
#       if (time3[1] < time4[1]) {
#         y_output[i] <- 1
#       }
#     } else { y_output[i] <- NA }
#   }
#   return(y_output)
# }
# 
# print("order1")
# for (var_num1 in 4:6){
#   for (var_num2 in 4:6){
#     y_test <- order1(x=x_out, var_num1 = var_num1, var_num2 = var_num2)
#     print(c("var1 = ",var_num1,"; var2 = ",var_num2,"; prop = ",sum(y_test[!is.na(y_test)]) / length(y_test[!is.na(y_test)])))
#     print(table(y_test, useNA = "always"))
#   } }

order1diff <- function(x, var_num1, var_num2) {
  n_people <- dim(x)[1]
  n_time <- dim(x)[3]
  y_output <- array(0, n_people)
  for (i in seq_len(n_people)) {
    time1 <- c()
    time2 <- c()
    for (j in seq_len(n_time)) {
      if (x[i, var_num1, j] > 0) {
        time1 <- c(time1, j)
      }
      if (x[i, var_num2, j] > 0) {
        time2 <- c(time2, j)
      }
    }
    #print(time1)
    #print(time2)
    if (!(is.null(time1) || is.null(time2))) {
      if (time1[1] < time2[1]) {
        y_output[i] <- 1
      }
    }  else if (!is.null(time1)){
      y_output[i] <- 0#1
    }   else if (!is.null(time2)){
      y_output[i] <- 0
    } #else { y_output[i] <- NA }
  }
  #print(y_output)
  return(y_output)
}

print("order1")
for (var_num1 in 4:6){
  if (var_num1 == 4){
    for (var_num2 in 5:6){
      y_test <- order1diff(x=x_out, var_num1 = var_num1, var_num2 = var_num2)
      print(c("var1 = ",var_num1,"; var2 = ",var_num2,"; prop = ",sum(y_test[!is.na(y_test)]) / length(y_test[!is.na(y_test)])))
      print(table(y_test, useNA = "always"))
    }
  }
  if (var_num1 == 5){
    for (var_num2 in c(4,6)){
      y_test <- order1diff(x=x_out, var_num1 = var_num1, var_num2 = var_num2)
      print(c("var1 = ",var_num1,"; var2 = ",var_num2,"; prop = ",sum(y_test[!is.na(y_test)]) / length(y_test[!is.na(y_test)])))
      print(table(y_test, useNA = "always"))
    }
  }
  if (var_num1 == 6){
    for (var_num2 in 4:5){
      y_test <- order1diff(x=x_out, var_num1 = var_num1, var_num2 = var_num2)
      print(c("var1 = ",var_num1,"; var2 = ",var_num2,"; prop = ",sum(y_test[!is.na(y_test)]) / length(y_test[!is.na(y_test)])))
      print(table(y_test, useNA = "always"))
    }
  }
}

order1diff_var4_var6 <- function(x) {
  return( order1diff(x=x, var_num1 = 4, var_num2 = 6))
}

order1diff_var4_var5 <- function(x) {
  return( order1diff(x=x, var_num1 = 4, var_num2 = 5))
}

order1diff_var6_var4 <- function(x) {
  return( order1diff(x=x, var_num1 = 6, var_num2 = 4))
}

order1diff_var5_var4 <- function(x) {
  return( order1diff(x=x, var_num1 = 5, var_num2 = 4))
}

# order1diff_soft <- function(x, var_num1, var_num2) {
#   n_people <- dim(x)[1]
#   n_time <- dim(x)[3]
#   y_output <- array(0, n_people)
#   for (i in seq_len(n_people)) {
#     time1 <- c()
#     time2 <- c()
#     for (j in seq_len(n_time)) {
#       if (x[i, var_num1, j] > 0) {
#         time1 <- c(time1, j)
#       }
#       if (x[i, var_num2, j] > 0) {
#         time2 <- c(time2, j)
#       }
#     }
#     #print(time1)
#     #print(time2)
#     if (!(is.null(time1) || is.null(time2))) {
#       #if (time1[1] < time2[1]) {
#         y_output[i] <- 1
#       #}
#     }  else if (!is.null(time1)){
#       y_output[i] <- rbern(n=1, p=0.1)
#     }   else if (!is.null(time2)){
#       y_output[i] <- rbern(n=1, p=0.01)
#     } else { y_output[i] <- 0 }
#   }
#   #print(y_output)
#   return(y_output)
# }
# 
# print("order1_soft")
# for (var_num1 in 4:6){
#   if (var_num1 == 4){
#     for (var_num2 in 5:6){
#       y_test <- order1diff_soft(x=x_out, var_num1 = var_num1, var_num2 = var_num2)
#       print(c("var1 = ",var_num1,"; var2 = ",var_num2,"; prop = ",sum(y_test[!is.na(y_test)]) / length(y_test[!is.na(y_test)])))
#       print(table(y_test, useNA = "always"))
#     }
#   }
#   if (var_num1 == 5){
#     for (var_num2 in c(4,6)){
#       y_test <- order1diff_soft(x=x_out, var_num1 = var_num1, var_num2 = var_num2)
#       print(c("var1 = ",var_num1,"; var2 = ",var_num2,"; prop = ",sum(y_test[!is.na(y_test)]) / length(y_test[!is.na(y_test)])))
#       print(table(y_test, useNA = "always"))
#     }
#   }
#   if (var_num1 == 6){
#     for (var_num2 in 4:5){
#       y_test <- order1diff_soft(x=x_out, var_num1 = var_num1, var_num2 = var_num2)
#       print(c("var1 = ",var_num1,"; var2 = ",var_num2,"; prop = ",sum(y_test[!is.na(y_test)]) / length(y_test[!is.na(y_test)])))
#       print(table(y_test, useNA = "always"))
#     }
#   }
# }


order2 <- function(x, var_num1, var_num2) {
  n_people <- dim(x)[1]
  n_time <- dim(x)[3]
  y_output <- array(0, n_people)
  for (i in seq_len(n_people)) {
    time3 <- c()
    time4 <- c()
    for (j in seq_len(n_time)) {
      if (x[i, var_num1, j] > 0) {
        time3 <- c(time3, j)
      }
      if (x[i, var_num2, j] > 0) {
        time4 <- c(time4, j)
      }
    }
    if (!(is.null(time3) || is.null(time4)) ) {
      if (length(time3) > 1) {
      if (time3[1] < time4[1] && time3[2] < time4[1]) {
        y_output[i] <- 1
      }
    }
  } }
  return(y_output)
}

print("order2")
for (var_num1 in 4:6){
  if (var_num1 == 4){
    for (var_num2 in 5:6){
      y_test <- order2(x=x_out, var_num1 = var_num1, var_num2 = var_num2)
      print(c("var1 = ",var_num1,"; var2 = ",var_num2,"; prop = ",sum(y_test[!is.na(y_test)]) / length(y_test[!is.na(y_test)])))
      print(table(y_test, useNA = "always"))
    }
  }
  if (var_num1 == 5){
    for (var_num2 in c(4,6)){
      y_test <- order2(x=x_out, var_num1 = var_num1, var_num2 = var_num2)
      print(c("var1 = ",var_num1,"; var2 = ",var_num2,"; prop = ",sum(y_test[!is.na(y_test)]) / length(y_test[!is.na(y_test)])))
      print(table(y_test, useNA = "always"))
    }
  }
  if (var_num1 == 6){
    for (var_num2 in 4:5){
      y_test <- order2(x=x_out, var_num1 = var_num1, var_num2 = var_num2)
      print(c("var1 = ",var_num1,"; var2 = ",var_num2,"; prop = ",sum(y_test[!is.na(y_test)]) / length(y_test[!is.na(y_test)])))
      print(table(y_test, useNA = "always"))
    }
  }
}

order2_64 <- function(x) {
  return(order2(x=x, var_num1 = 6, var_num2 = 4))
}

order2_46 <- function(x) {
  return(order2(x=x, var_num1 = 4, var_num2 = 6))
}


order2_45 <- function(x) {
  return(order2(x=x, var_num1 = 4, var_num2 = 5))
}

order2_54 <- function(x) {
  return(order2(x=x, var_num1 = 5, var_num2 = 4))
}

order3 <- function(x, var_num1, var_num2) {  ## two variables and first event of one must come after first of other but before second of other
  n_people <- dim(x)[1]
  n_time <- dim(x)[3]
  y_output <- array(0, n_people)
  for (i in seq_len(n_people)) {
    time3 <- c()
    time4 <- c()
    for (j in seq_len(n_time)) {
      if (x[i, var_num1, j] > 0) {
        time3 <- c(time3, j)
      }
      if (x[i, var_num2, j] > 0) {
        time4 <- c(time4, j)
      }
    }
    if (!(is.null(time3) || is.null(time4)) ) {
      if (length(time4) > 1) {
        if (time3[1] >= time4[1] && time3[1] <= time4[2]) {
          y_output[i] <- 1
        }
      }
    } else{y_output[i] <- NA}}
  return(y_output)
}

print("order3")
for (var_num1 in 4:6){
  if (var_num1 == 4){
    for (var_num2 in 5:6){
      y_test <- order3(x=x_out, var_num1 = var_num1, var_num2 = var_num2)
      print(c("var1 = ",var_num1,"; var2 = ",var_num2,"; prop = ",sum(y_test[!is.na(y_test)]) / length(y_test[!is.na(y_test)])))
      print(table(y_test, useNA = "always"))
    }
  }
  if (var_num1 == 5){
    for (var_num2 in c(4,6)){
      y_test <- order3(x=x_out, var_num1 = var_num1, var_num2 = var_num2)
      print(c("var1 = ",var_num1,"; var2 = ",var_num2,"; prop = ",sum(y_test[!is.na(y_test)]) / length(y_test[!is.na(y_test)])))
      print(table(y_test, useNA = "always"))
    }
  }
  if (var_num1 == 6){
    for (var_num2 in 4:5){
      y_test <- order3(x=x_out, var_num1 = var_num1, var_num2 = var_num2)
      print(c("var1 = ",var_num1,"; var2 = ",var_num2,"; prop = ",sum(y_test[!is.na(y_test)]) / length(y_test[!is.na(y_test)])))
      print(table(y_test, useNA = "always"))
    }
  }
}

order3_45 <- function(x) {
  return(order3(x=x, var_num1 = 4, var_num2 = 5))
}

order3_54 <- function(x) {
  return(order3(x=x, var_num1 = 5, var_num2 = 4))
}

order3_46 <- function(x) {
  return(order3(x=x, var_num1 = 4, var_num2 = 6))
}

order3_64 <- function(x) {
  return(order3(x=x, var_num1 = 6, var_num2 = 4))
}

order3_56 <- function(x) {
  return(order3(x=x, var_num1 = 5, var_num2 = 6))
}

order3_65 <- function(x) {
  return(order3(x=x, var_num1 = 6, var_num2 = 5))
}


order4 <- function(x, var_num1, var_num2, var_num3) {  ## three variables, first events must occur in specific order - i.e. order one with two and order of the other two mean one way mitigates?
  n_people <- dim(x)[1]
  n_time <- dim(x)[3]
  y_output <- array(0, n_people)
  for (i in seq_len(n_people)) {
    time1 <- c()
    time2 <- c()
    time3 <- c()
    for (j in seq_len(n_time)) {
      if (x[i, var_num1, j] > 0) {
        time1 <- c(time1, j)
      }
      if (x[i, var_num2, j] > 0) {
        time2 <- c(time2, j)
      }
      if (x[i, var_num3, j] > 0) {
        time3 <- c(time3, j)
      }
    }
    if (!(is.null(time2) || is.null(time1)) ) {
        if (time1[1] < time2[1]) {
          y_output[i] <- 1
        }
    } 
    if (!(is.null(time3)) ) {
        y_output[i] <- 0
    }
    # if (!(is.null(time1) || is.null(time3)) ) {
    #   if (time1[1] < time3[1]) {
    #     y_output[i] <- 0
    #   }
    # }
    # if (!(is.null(time1) || is.null(time3)) ) { ## no
    #   if (time1[1] > time3[1]) {
    #     y_output[i] <- 0
    #   }
    # }
    # if (!(is.null(time2) || is.null(time3)) ) { ## no
    #   if (time2[1] < time3[1]) {
    #     y_output[i] <- 0
    #   }
    # }
    # if (!(is.null(time2) || is.null(time3)) ) {
    #   if (time2[1] > time3[1]) {
    #     y_output[i] <- 0
    #   }
    # }
}
  return(y_output)
}

print("order4")
for (var_num1 in 4:6){
  if (var_num1 == 4){
    for (var_num2 in 5:6){
      var_num3 <- 11 - var_num2
      y_test <- order4(x=x_out, var_num1 = var_num1, var_num2 = var_num2, var_num3 = var_num3)
      print(c("var1 = ",var_num1,"; var2 = ",var_num2,"; prop = ",sum(y_test[!is.na(y_test)]) / length(y_test[!is.na(y_test)])))
      print(table(y_test, useNA = "always"))
    }
  }
  if (var_num1 == 5){
    for (var_num2 in c(4,6)){
      var_num3 <- 10 - var_num2
      y_test <- order4(x=x_out, var_num1 = var_num1, var_num2 = var_num2, var_num3 = var_num3)
      print(c("var1 = ",var_num1,"; var2 = ",var_num2,"; prop = ",sum(y_test[!is.na(y_test)]) / length(y_test[!is.na(y_test)])))
      print(table(y_test, useNA = "always"))
    }
  }
  if (var_num1 == 6){
    for (var_num2 in 4:5){
      var_num3 <- 9 - var_num2
      y_test <- order4(x=x_out, var_num1 = var_num1, var_num2 = var_num2, var_num3 = var_num3)
      print(c("var1 = ",var_num1,"; var2 = ",var_num2,"; prop = ",sum(y_test[!is.na(y_test)]) / length(y_test[!is.na(y_test)])))
      print(table(y_test, useNA = "always"))
    }
  }
}

order4_456 <- function(x) {
  return(order4(x=x, var_num1 = 4, var_num2 = 5, var_num3 = 6))
}

order4_546 <- function(x) {
  return(order4(x=x, var_num1 = 5, var_num2 = 4, var_num3 = 6))
}

order4_465 <- function(x) {
  return(order4(x=x, var_num1 = 4, var_num2 = 6, var_num3 = 5))
}

order4_645 <- function(x) {
  return(order4(x=x, var_num1 = 6, var_num2 = 4, var_num3 = 5))
}

order4_564 <- function(x) {
  return(order4(x=x, var_num1 = 5, var_num2 = 6, var_num3 = 4))
}

order4_654 <- function(x) {
  return(order4(x=x, var_num1 = 6, var_num2 = 5, var_num3 = 4))
}


# order3 <- function(x, var_num1, var_num2) {
#   n_people <- dim(x)[1]
#   n_time <- dim(x)[3]
#   y_output <- array(0, n_people)
#   for (i in seq_len(n_people)) {
#     time3 <- c()
#     time4 <- c()
#     for (j in seq_len(n_time)) {
#       if (x[i, var_num1, j] > 0) {
#         time3 <- c(time3, j)
#       }
#       if (x[i, var_num2, j] > 0) {
#         time4 <- c(time4, j)
#       }
#     }
#     if (!(is.null(time3) || is.null(time4)) && length(time3) > 2) {
#       if (time3[1] < time4[1] && time3[2] < time4[1] && time3[3] < time4[1]) {
#         y_output[i] <- 1
#       }
#     } #else { y_output[i] <- NA }
#   }
#   return(y_output)
# }
# 
# print("order3")
# for (var_num1 in 4:6){
#   for (var_num2 in 4:6){
#     y_test <- order3(x=x_out, var_num1 = var_num1, var_num2 = var_num2)
#     print(c("var1 = ",var_num1,"; var2 = ",var_num2,"; prop = ",sum(y_test[!is.na(y_test)]) / length(y_test[!is.na(y_test)])))
#     print(table(y_test, useNA = "always"))
#   } }



timing_nevent_myear <- function(x, n, m, var_num1, var_num2) {
  n_people <- dim(x)[1]
  n_time <- dim(x)[3]
  y_output <- array(0, n_people)
  for (i in seq_len(n_people)) {
    time3 <- c()
    time4 <- c()
    for (j in seq_len(n_time)) {
      if (x[i, var_num1, j] > 0) {
        time3 <- c(time3, j)
      }
      if (x[i, var_num2, j] > 0) {
        time4 <- c(time4, j)
      }
    }
    if (!(is.null(time3) || is.null(time4))) {
      test <- 0
      for (q in seq_along(time3)) {
        if (any(abs(time4 - time3[q]) <= m)) {
          test <- test + 1
        }
      }
      if (test >= n) {
        y_output[i] <- 1
      }
    }
  }
  return(y_output)
}


print("timing")
for (var_num1 in 4:5){
  if (var_num1 ==4){
    for (var_num2 in 5:6){
      for (n in 1:3){
        for (m in 0:5){
          y_test <- timing_nevent_myear(x=x_out, m=m, var_num1 = var_num1, var_num2 = var_num2, n=n)
          if (sum(y_test) / length(y_test) < 0.08 & sum(y_test) / length(y_test) > 0.01){
          print(c("var1 = ",var_num1,"; var2 = ",var_num2,"; n events = ",n,"; m years sep = ",m,"; prop = ",sum(y_test) / length(y_test)))}
        } }}
  }
  # if (var_num1 ==5){
  #   for (var_num2 in c(6)){
  #     for (n in 1:3){
  #       for (m in 0:5){
  #         y_test <- timing_nevent_myear(x=x_out, m=m, var_num1 = var_num1, var_num2 = var_num2, n=n)
  #         print(c("var1 = ",var_num1,"; var2 = ",var_num2,"; n events = ",n,"; m years sep = ",m,"; prop = ",sum(y_test) / length(y_test)))
  #       } }}}
  }


print("timing")
for (var_num1 in 4:5){
  m = 0
  if (var_num1 ==4){
    for (var_num2 in 5:6){
      for (n in 1){
          y_test <- timing_nevent_myear(x=x_out, m=m, var_num1 = var_num1, var_num2 = var_num2, n=n)
          print(c("var1 = ",var_num1,"; var2 = ",var_num2,"; n events = ",n,"; m years sep = ",m,"; prop = ",sum(y_test) / length(y_test)))
        } }
  }
  if (var_num1 ==5){
    for (var_num2 in c(6)){
      for (n in 2:3){
          y_test <- timing_nevent_myear(x=x_out, m=m, var_num1 = var_num1, var_num2 = var_num2, n=n)
          print(c("var1 = ",var_num1,"; var2 = ",var_num2,"; n events = ",n,"; m years sep = ",m,"; prop = ",sum(y_test) / length(y_test)))
        } }}
}

timing_1event_0year_45 <- function(x) {
  return(timing_nevent_myear(x=x, m=0, var_num1 = 4, var_num2 = 5, n=1))
}

timing_1event_0year_46 <- function(x) {
  return(timing_nevent_myear(x=x, m=0, var_num1 = 4, var_num2 = 6, n=1))
}

timing_2event_0year_56 <- function(x) {
  return(timing_nevent_myear(x=x, m=0, var_num1 = 5, var_num2 = 6, n=2))
}


print("timing")
for (var_num1 in 4:5){
  m = 1
  if (var_num1 ==4){
    for (var_num2 in 5:6){
      for (n in 1:2){
        y_test <- timing_nevent_myear(x=x_out, m=m, var_num1 = var_num1, var_num2 = var_num2, n=n)
        print(c("var1 = ",var_num1,"; var2 = ",var_num2,"; n events = ",n,"; m years sep = ",m,"; prop = ",sum(y_test) / length(y_test)))
      } }
  }
  if (var_num1 ==5){
    for (var_num2 in c(6)){
      for (n in 3:4){
        y_test <- timing_nevent_myear(x=x_out, m=m, var_num1 = var_num1, var_num2 = var_num2, n=n)
        print(c("var1 = ",var_num1,"; var2 = ",var_num2,"; n events = ",n,"; m years sep = ",m,"; prop = ",sum(y_test) / length(y_test)))
      } }}
}


timing_2event_1year_45 <- function(x) {
  return(timing_nevent_myear(x=x, m=1, var_num1 = 4, var_num2 = 5, n=2))
}

timing_1event_1year_46 <- function(x) {
  return(timing_nevent_myear(x=x, m=1, var_num1 = 4, var_num2 = 6, n=1))
}

timing_4event_1year_56 <- function(x) {
  return(timing_nevent_myear(x=x, m=1, var_num1 = 5, var_num2 = 6, n=4))
}

print("timing")
for (var_num1 in 4:5){
  m = 4
  if (var_num1 ==4){
    for (var_num2 in 5:6){
      for (n in 2:4){
        y_test <- timing_nevent_myear(x=x_out, m=m, var_num1 = var_num1, var_num2 = var_num2, n=n)
        print(c("var1 = ",var_num1,"; var2 = ",var_num2,"; n events = ",n,"; m years sep = ",m,"; prop = ",sum(y_test) / length(y_test)))
      } }
  }
  if (var_num1 ==5){
    for (var_num2 in c(6)){
      for (n in 2:6){
        y_test <- timing_nevent_myear(x=x_out, m=m, var_num1 = var_num1, var_num2 = var_num2, n=n)
        print(c("var1 = ",var_num1,"; var2 = ",var_num2,"; n events = ",n,"; m years sep = ",m,"; prop = ",sum(y_test) / length(y_test)))
      } }}
}


timing_2event_4year_45 <- function(x) {
  return(timing_nevent_myear(x=x, m=4, var_num1 = 4, var_num2 = 5, n=2))
}

timing_2event_4year_46 <- function(x) {
  return(timing_nevent_myear(x=x, m=4, var_num1 = 4, var_num2 = 6, n=2))
}

timing_6event_4year_56 <- function(x) {
  return(timing_nevent_myear(x=x, m=4, var_num1 = 5, var_num2 = 6, n=6))
}



timing_1event_0year_46 <- function(x) {
  return(timing_nevent_myear(x=x, m=0, var_num1 = 4, var_num2 = 6, n=1))
}
timing_1event_1year_46 <- function(x) {
  return(timing_nevent_myear(x=x, m=1, var_num1 = 4, var_num2 = 6, n=1))
}
timing_1event_2year_46 <- function(x) {
  return(timing_nevent_myear(x=x, m=2, var_num1 = 4, var_num2 = 6, n=1))
}
timing_1event_4year_46 <- function(x) {
  return(timing_nevent_myear(x=x, m=4, var_num1 = 4, var_num2 = 6, n=1))
}

timing_2event_0year_45 <- function(x) {
  return(timing_nevent_myear(x=x, m=0, var_num1 = 4, var_num2 = 5, n=2))
}
timing_2event_1year_45 <- function(x) {
  return(timing_nevent_myear(x=x, m=1, var_num1 = 4, var_num2 = 5, n=2))
}
timing_2event_2year_45 <- function(x) {
  return(timing_nevent_myear(x=x, m=2, var_num1 = 4, var_num2 = 5, n=2))
}
timing_2event_4year_45 <- function(x) {
  return(timing_nevent_myear(x=x, m=4, var_num1 = 4, var_num2 = 5, n=2))
}


##########################################################################################

# source("Data_simulation/LCP_rule_functions.R")

# function_names <- c(
#   "repeat2","repeat3", "repeat4", "repeat2_2",
#   "order1","order2", "order3", "order4",
#   "timing1_0","timing2_0", "timing1_1","timing2_1", "timing1_2", "timing2_2", "timing3_2", "timing4_2","timing1_3","timing2_3","timing3_3","timing4_3","timing1_4","timing2_4","timing3_4","timing4_4","critical5",
#   "critical10", "sensitive1", "sensitive2","sensitive3","sensitive4", "weighted2", "weighted3"
# )

function_names <- c(
# Period
"critical_1", "critical_2", "critical_1_var6",
"sensitive_var4_n2_p5", "sensitive_var5_n3_p2", "sensitive_var6_n2_p2", "sensitive_var6_n3_p5",
"weighted_var4_n1", "weighted_var4_n15", "weighted_var6_n15",

# Repeats
"repeat_var4_r2", "repeat_var4_r3", "repeat_var6_r3", "repeat_var6_r4",
"repeat2_2_var4", "repeat2_2_var6",

## Order
"order1diff_var4_var6", "order1diff_var4_var5",
"order2_64", "order2_45", "order2_54", "order1diff_var6_var4",

## Timing
"timing_1event_0year_45",  "timing_1event_0year_46", "timing_2event_0year_56",
"timing_2event_1year_45", "timing_1event_1year_46", "timing_4event_1year_56", "timing_2event_4year_45", "timing_2event_4year_46", "timing_6event_4year_56" ,

#timing_1event_0year_46,
#timing_1event_1year_46 ,
"timing_1event_2year_46" ,"timing_1event_4year_46" ,"timing_2event_0year_45" ,
#timing_2event_1year_45,
"timing_2event_2year_45" #,timing_2event_4year_45
)

# paper_names <- c(
#   "Repeats1", "Repeats2", "Repeats3",
#   "Order1", "Order2", "Order3",
#   "Timing1", "Timing2", "Timing3", "Timing4",
#   "Period1", "Period2", "Period3", "Period4"
# )

# paper_names <- paste0(paper_names, "_depd")

paper_names <- c(
  # Period
  "critical_1", "critical_2", "critical_1_var6",
  "sensitive_var4_n2_p5", "sensitive_var5_n3_p2", "sensitive_var6_n2_p2", "sensitive_var6_n3_p5",
  "weighted_var4_n1", "weighted_var4_n15", "weighted_var6_n15",
  
  # Repeats
  "repeat_var4_r2", "repeat_var4_r3", "repeat_var6_r3", "repeat_var6_r4",
  "repeat2_2_var4", "repeat2_2_var6",
  
  ## Order
  "order1diff_var4_var6", "order1diff_var4_var5",
  "order2_64", "order2_45", "order2_54", "order1diff_var6_var4",
  
  ## Timing
  "timing_1event_0year_45",  "timing_1event_0year_46", "timing_2event_0year_56",
  "timing_2event_1year_45", "timing_1event_1year_46", "timing_4event_1year_56", "timing_2event_4year_45", "timing_2event_4year_46", "timing_6event_4year_56" 
)


prop_table <- c()


for (i in seq_along(function_names)) {
  print(function_names[i])
  
  y_outcheck <- eval(parse(text = (paste0(function_names[i], "(x_out,y_out)"))))
  
  # name <- paste("data_", paper_names[i], sep = "")
  name <- paste("data_",function_names[i], sep = "")
  prop_table <- rbind(prop_table, c(name, sum(y_outcheck) / length(y_outcheck)))
  print(sum(y_outcheck) / length(y_outcheck))
  ## moving data to python
  
  
  ## moving data to python
  # np$save(paste0("Data_simulation/Simulated_Data/data_depd_rand", randnum, "_X.npy"), r_to_py(x_out))
  # np$save(paste0("Data_simulation/Simulated_Data/dedp_", name, "rand", randnum, "_Y.npy"), r_to_py(y_outcheck))
}
names(prop_table) <- c("Data", "Proportion 1s")
# write.csv(prop_table, paste0("Data_simulation/Simulated_Data/depd_data_rand", randnum, ".csv"))

