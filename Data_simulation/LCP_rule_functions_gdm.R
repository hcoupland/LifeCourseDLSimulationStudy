## Sheet containing functions that calculate the different LCP rules

# Period rules 1 and 2
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

critical_1 <- function(x){
  return(sensitiven(x=x, time_start=1, period=1, var_num = 4, n=1))
}

sensitive_var4_n2_p5 <- function(x){
  return(sensitiven(x=x, time_start=1, period=5, var_num = 4, n=2))
}

# Period 3 rule
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

weighted_var4_n1 <- function(x) {
  return(weightedn(x=x_out, n=1, var_num = 4))
}

# Repeats rules 1 and 3
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

repeat_var6_r4 <- function(x) {
  return(repeatn(x=x,  n=4, var_num = 6))
}

repeat2_var4_var5 <- function(x) {
  return(as.numeric(repeatn(x = x, n = 2, var_num = 5) == 1 & repeatn(x = x, n = 2, var_num = 4) == 1))
}

# Repeats rule 2
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


repeat2_2_var6 <- function(x) {
  return(repeat2_2(x=x, var_num = 6))
}

# Order 1 rule
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
    if (!(is.null(time1) || is.null(time2))) {
      if (time1[1] < time2[1]) {
        y_output[i] <- 1
      }
    }  else if (!is.null(time1)){
      y_output[i] <- 0#1
    }   else if (!is.null(time2)){
      y_output[i] <- 0
    }
  }
  return(y_output)
}

order1diff_var6_var4 <- function(x) {
  return( order1diff(x=x, var_num1 = 6, var_num2 = 4))
}

# Order 2 rule
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

order2_64 <- function(x) {
  return(order2(x=x, var_num1 = 6, var_num2 = 4))
}

# Order 3 rule
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
}
  return(y_output)
}


order4_645 <- function(x) {
  return(order4(x=x, var_num1 = 6, var_num2 = 4, var_num3 = 5))
}

# Timing 1, 2 and 3 rules
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


timing_1event_0year_45 <- function(x) {
  return(timing_nevent_myear(x=x, m=0, var_num1 = 4, var_num2 = 5, n=1))
}

timing_2event_1year_45 <- function(x) {
  return(timing_nevent_myear(x = x, m = 1, var_num1 = 4, var_num2 = 5, n = 2))
}

timing_2event_4year_45 <- function(x) {
  return(timing_nevent_myear(x = x, m = 4, var_num1 = 4, var_num2 = 5, n = 2))
}