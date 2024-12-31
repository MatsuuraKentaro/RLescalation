#' Compute DLT Probability Scenarios for Reinforcement Learning
#' 
#' Compute the scenarios described in Sect. 2.2 of the original paper.
#'
#' @param J A positive integer value. The number of doses.
#' @param target A positive numeric value. The target DLT probability.
#' @param epsilon A positive numeric value. The acceptable range of target DLT 
#'        probabilities is defined as \[`target` - `epsilon`, `target` + `epsilon`\].
#' @param delta A positive numeric value. The unacceptable ranges of target DLT 
#'        probabilities are defined as \[0, `target` - `delta`\] and 
#'        \[`target` + `delta`, 1\].
#'
#' @return A named list of three elements:
#'         - prob: a list of DLT probability scenarios
#'         - MTD: a list of true MTD indices (Note that `-1` means "no MTD")
#'         - weight: a vector of weights for each scenario
#' 
#' @examples
#' scenarios <- compute_rl_scenarios(J = 6, target = 0.25, epsilon = 0.04, delta = 0.1)
#' print(scenarios)
#'
#' @importFrom stats plogis
#'
#' @export
compute_rl_scenarios <- function(J, target, epsilon, delta) {
  ## check arguments
  J <- as.integer(J)
  stopifnot(length(J) == 1L, J >= 1L)
  
  target <- as.double(target)
  epsilon <- as.double(epsilon)
  delta <- as.double(delta)
  stopifnot(length(target) == 1L, 0 < target, target < 1)
  stopifnot(length(epsilon) == 1L, 0 < epsilon, epsilon < 1)
  stopifnot(length(delta) == 1L, 0 < delta, delta < 1)
  stopifnot(epsilon < delta)

  ## compute scenarios  
  doses <- 1L:J
  
  func_root <- function(x, target, j, j_add, p_add1, p_add2) {
    return(c(plogis(x[1L] + x[2L]*j) - (target + p_add1),
             plogis(x[1L] + x[2L]*(j + j_add)) - (target + p_add2)))
  }
  
  middle_list <- lapply(c(1L:J, 0L), function(j) {
    sol <- nleqslv::nleqslv(c(-3.0, 1.0), func_root, jac = NULL, target, j, 1L, 0.0, delta)
    probs <- plogis(sol$x[1L] + sol$x[2L]*doses)
    pmin(pmax(probs, 0.1), 0.8)
  })
  
  lower_list <- lapply(c(1L:J, 0L), function(j) {
    sol <- nleqslv::nleqslv(c(-3.0, 1.0), func_root, jac = NULL, target, j, 1L, -epsilon, delta)
    probs <- plogis(sol$x[1L] + sol$x[2L]*doses)
    pmin(pmax(probs, 0.1), 0.8)
  })
  
  higher_list <- lapply(1L:J, function(j) {
    sol <- nleqslv::nleqslv(c(-3.0, 1.0), func_root, jac = NULL, target, j, -1L, epsilon, -delta)
    probs <- plogis(sol$x[1L] + sol$x[2L]*doses)
    pmin(pmax(probs, 0.1), 0.8)
  })
  
  prob_list <- c(middle_list, lower_list, higher_list)
  MTD_list <- as.list(rep(c(1L:J, -1L), length.out = length(prob_list)))
  weight <- rep(1L, length(prob_list))
  
  list(prob = prob_list, MTD = MTD_list, weight = weight)
}
