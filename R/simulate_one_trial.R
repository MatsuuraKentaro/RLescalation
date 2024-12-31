#' Simulate One Trial Using an Obtained Optimal Dose Escalation Rule
#'
#' @param escalation_rule An object of class \link[RLescalation]{EscalationRule}
#'        specifying an obtained optimal dose escalation rule.
#' @param prob_true A numeric vector specifying the true DLT probabilities.
#' @param seed An integer value. Random seed for data generation in this trial.
#'        
#' @returns A data frame which contains the cohort ID, the assigned dose, 
#'        the number of assigned patients, the number of DLTs, and the recommended 
#'        action including down, stay, up, MTD_1, ..., MTD_J, no_MTD, 
#'        and fail to determine MTD.
#' 
#' @examples
#' library(RLescalation)
#' 
#' \dontrun{
#' escalation_rule <- learn_escalation_rule(
#'   J = 6, target = 0.25, epsilon = 0.04, delta = 0.1,
#'   N_total = 36, N_cohort = 3, seed = 123,
#'   rl_config = rl_config_set(iter = 1000)
#' )}
#' 
#' prob_true <- c(0.03, 0.13, 0.17, 0.19, 0.26, 0.31)
#' 
#' # Simulate one trial using the obtained `escalation_rule`
#' \dontrun{
#' sim_one <- simulate_one_trial(escalation_rule, prob_true, seed = 123)}
#' 
#' @importFrom stats rbinom
#' 
#' @export
simulate_one_trial <- function(
    escalation_rule, prob_true, seed = NULL) {

  prob_true <- as.double(prob_true)
  J <- escalation_rule$policy$config$env_config$J
  stopifnot("'prob_true' must have the same length of doses" = length(prob_true) == J)

  N_total <- escalation_rule$policy$config$env_config$N_total
  N_cohort <- escalation_rule$policy$config$env_config$N_cohort
  num_cohorts <- N_total %/% N_cohort
  
  set.seed(seed)
  
  sim_Ns <- rep(0L, J)
  sim_DLTs <- rep(0L, J)
  draw_dose <- 1L
  sim_list <- list()

  for (cohortID in seq_len(num_cohorts)) {
    draw_DLT <- rbinom(n = 1L, size = N_cohort, prob = prob_true[draw_dose])
    sim_Ns[draw_dose] <- sim_Ns[draw_dose] + N_cohort
    sim_DLTs[draw_dose] <- sim_DLTs[draw_dose] + draw_DLT
    current_dose <- draw_dose
    
    next_action <- escalation_rule$opt_action(current_dose, sim_Ns, sim_DLTs)
    
    sim_list[[length(sim_list) + 1L]] <- data.frame(
      cohortID = cohortID, dose = draw_dose, N = N_cohort, DLT = draw_DLT, 
      recommended = next_action)
    
    if (next_action == "no_MTD") break
    
    if (next_action == "down") {
      draw_dose <- current_dose - 1L
    } else if (next_action == "stay") {
      draw_dose <- current_dose
    } else if (next_action == "up") {
      draw_dose <- current_dose + 1L
    }
  }
  
  do.call(rbind, sim_list)
}
