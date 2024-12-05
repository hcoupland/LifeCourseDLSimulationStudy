### File to prepare data split into test sets to send to others
library(list)
library(dplyr)
library(plyr)
library(reshape)
library(stringr)
library(egg)
library(abind)
library(arrow)

setwd("C:/Users/hlc17/Documents/PostDoc/Simulation paper/Downloads from dide2/models")
var_names <- c("Parental origin","Mother's age at first birth","Parental diabetes","Loss", "SES", "Dynamic")
n_vars <- length(var_names)
n_years <- 16

stoc <- "stoc10"
model <- "XGBoost"
data <- "order2_64"
rand <- "randtr7"


filenames <- list.files(".", pattern="*.csv", full.names=TRUE)
filenames <- grep(stoc,filenames,fixed=TRUE,value=TRUE)
filenames <- grep(model,filenames,fixed=TRUE,value=TRUE)
filenames <- grep(data,filenames,fixed=TRUE,value=TRUE)
filenames <- grep(rand,filenames,fixed=TRUE,value=TRUE)


# ## Need to pick one with the biggest date
# # Extract date and time from filenames
# if (length(filenames)>7){
#   datetime_str <- str_extract(filenames, "\\d{8}-\\d{6}")
#   datetime <- as.POSIXct(datetime_str, format="%Y%m%d-%H%M%S")
#
#   # Bind datetime with filenames and sort by datetime
#   sorted_files <- filenames[order(datetime, decreasing = TRUE)]
#
#   # Select only the 5 most recent files
#   most_recent_files <- head(sorted_files, 5)
#
#   # Display the most recent files
#   filenames <- most_recent_files
# }

Xtrain<-grep("Xtrain",filenames,fixed=TRUE,value=TRUE)
Xvalid<-grep("Xvalid",filenames,fixed=TRUE,value=TRUE)
Xtest<-grep("Xtest",filenames,fixed=TRUE,value=TRUE)
Ytrain<-grep("Ytrain",filenames,fixed=TRUE,value=TRUE)
Yvalid<-grep("Yvalid",filenames,fixed=TRUE,value=TRUE)
Ytest<-grep("Ytest",filenames,fixed=TRUE,value=TRUE)
params<-grep("params",filenames,fixed=TRUE,value=TRUE)

Xtrain<-lapply(Xtrain,read.csv)[[1]]
Xtrain<-aperm(array(unlist(Xtrain),dim=c(nrow(Xtrain),n_years,n_vars)),c(1,3,2))
Xvalid<-lapply(Xvalid,read.csv)[[1]]
Xvalid<-aperm(array(unlist(Xvalid),dim=c(nrow(Xvalid),n_years,n_vars)),c(1,3,2))
Xtest<-lapply(Xtest,read.csv)[[1]]
Xtest<-aperm(array(unlist(Xtest),dim=c(nrow(Xtest),n_years,n_vars)),c(1,3,2))

Ytrain<-lapply(Ytrain,read.csv)[[1]]
Yvalid<-lapply(Yvalid,read.csv)[[1]]
Ytest<-lapply(Ytest,read.csv)[[1]]

params<-lapply(params,read.csv)[[1]]
params <- params %>%
  select(epochs, ESPatience, gamma, max_depth, eta, subsample, colsample_bytree, n_estimators, min_child_weight, reg_alpha, reg_lambda)

# Example 3D array: (num_people, num_features, num_timepoints)
# For demonstration, let's create a sample array
array_data <- Xtrain

# Get dimensions
num_people <- dim(array_data)[1]
num_features <- dim(array_data)[2]
num_timepoints <- dim(array_data)[3]

# Create column names for the resulting data frame
feature_time_names <- unlist(lapply(1:num_features, function(f) {
  paste0("feature", f, "_time", 1:num_timepoints)
}))

# Reshape the array
reshaped_data <- data.frame(
  ID = rep(1:num_people, each = 1), # Assign unique IDs to each person
  do.call(rbind, lapply(1:num_people, function(i) {
    as.vector(t(array_data[i, , ])) # Flatten feature-time combinations for each person
  }))
)

# Assign the correct column names
colnames(reshaped_data) <- c("ID", feature_time_names)

# Print the reshaped data
print(reshaped_data)

X_train_df <- reshaped_data

array_data <- Xvalid

# Get dimensions
num_people <- dim(array_data)[1]
num_features <- dim(array_data)[2]
num_timepoints <- dim(array_data)[3]

# Create column names for the resulting data frame
feature_time_names <- unlist(lapply(1:num_features, function(f) {
  paste0("feature", f, "_time", 1:num_timepoints)
}))

# Reshape the array
reshaped_data <- data.frame(
  ID = rep(1:num_people, each = 1), # Assign unique IDs to each person
  do.call(rbind, lapply(1:num_people, function(i) {
    as.vector(t(array_data[i, , ])) # Flatten feature-time combinations for each person
  }))
)

# Assign the correct column names
colnames(reshaped_data) <- c("ID", feature_time_names)

# Print the reshaped data
print(reshaped_data)

X_valid_df <- reshaped_data


array_data <- Xtest

# Get dimensions
num_people <- dim(array_data)[1]
num_features <- dim(array_data)[2]
num_timepoints <- dim(array_data)[3]

# Create column names for the resulting data frame
feature_time_names <- unlist(lapply(1:num_features, function(f) {
  paste0("feature", f, "_time", 1:num_timepoints)
}))

# Reshape the array
reshaped_data <- data.frame(
  ID = rep(1:num_people, each = 1), # Assign unique IDs to each person
  do.call(rbind, lapply(1:num_people, function(i) {
    as.vector(t(array_data[i, , ])) # Flatten feature-time combinations for each person
  }))
)

# Assign the correct column names
colnames(reshaped_data) <- c("ID", feature_time_names)

# Print the reshaped data
print(reshaped_data)

X_test_df <- reshaped_data

X_valid_df$ID <- X_valid_df$ID + max(X_train_df$ID)
X_test_df$ID <- X_test_df$ID + max(X_valid_df$ID)

### NICE
X_train_df$output <- Ytrain$X0
X_valid_df$output <- Yvalid$X0
X_test_df$output <- Ytest$X0

## Then have to write all as parquet files
filepath <- "C:/Users/hlc17/Documents/PostDoc/Simulation paper/Rough work"
parquet = tempfile(fileext = ".parquet", tmpdir = filepath, pattern = "data_train")
write_parquet(X_train_df, sink = parquet)
parquet = tempfile(fileext = ".parquet", tmpdir = filepath, pattern = "data_valid")
write_parquet(X_valid_df, sink = parquet)
parquet = tempfile(fileext = ".parquet", tmpdir  = filepath, pattern = "data_test")
write_parquet(X_test_df, sink = parquet)
parquet = tempfile(fileext = ".parquet", tmpdir = filepath, pattern = "params")
write_parquet(params, sink = parquet)

