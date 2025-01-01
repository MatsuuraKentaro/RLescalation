#' @importFrom R6 R6Class
#' @importFrom utils globalVariables

globalVariables("DoseEscalationEnv")

# State s is the vector described in Sect. 2.2 of the original paper, 
# plus the information whether it is the final cohort or not.
compute_state <- function(current_dose, J, data_Ns, data_DLTs, N_total) {
  stopifnot(length(data_Ns) == length(data_DLTs))
  stopifnot(all(data_Ns >= data_DLTs))

  is_final <- ifelse(sum(data_Ns) == N_total, 1.0, 0.0)
  
  state <- as.array(c(
    (current_dose - 1) / J,
    data_Ns / N_total,
    data_DLTs / N_total,
    sum(data_Ns) / N_total,
    sum(data_DLTs) / N_total,
    is_final
  ))
  state
}

is_apple_silicon <- function() {
  sys_info <- Sys.info()
  sys_info["sysname"] == "Darwin" && sys_info["machine"] == "arm64"
}
