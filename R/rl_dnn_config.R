#' DNN Configuration for Reinforcement Learning
#'
#' DNN (deep neural network) configuration for reinforcement learning.
#' For detail, see Section 3.1 of the original paper.
#'
#' @param fcnet_hiddens A positive integer vector. Numbers of units of the
#'        intermediate layers.
#' @param fcnet_activation A character value specifying the activation function.
#'        Possible values are "ReLU" (default), "tanh", "Swish" (or "SiLU"), or
#'        "linear".
#' @param ... Other configurations. See source code of RLlib.
#'        https://github.com/ray-project/ray/blob/master/rllib/models/catalog.py
#'
#' @return A list of DNN configuration parameters
#' 
#' @examples
#' \dontrun{
#' escalation_rule <- learn_escalation_rule(
#'   J = 6, target = 0.25, epsilon = 0.04, delta = 0.1,
#'   N_total = 36, N_cohort = 3, seed = 123,
#'   rl_config = rl_config_set(
#'     iter = 1000, 
#'     # We change the DNN model
#'     model = rl_dnn_config(fcnet_hiddens = c(512L, 512L), fcnet_activation = "tanh")
#'   )
#' )} 
#'
#' @export
rl_dnn_config <- function(
    fcnet_hiddens = c(256L, 256L),
    fcnet_activation = c("relu", "tanh", "swish", "silu", "linear"), ...) {

  fcnet_hiddens <- as.integer(fcnet_hiddens)
  fcnet_activation <- tolower(fcnet_activation)
  fcnet_activation <- match.arg(fcnet_activation)

  config <- list(fcnet_hiddens = fcnet_hiddens, fcnet_activation = fcnet_activation)
  other_config <- list(...)

  append(config, other_config)
}
