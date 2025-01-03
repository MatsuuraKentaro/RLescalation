% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/rl_dnn_config.R
\name{rl_dnn_config}
\alias{rl_dnn_config}
\title{DNN Configuration for Reinforcement Learning}
\usage{
rl_dnn_config(
  fcnet_hiddens = c(256L, 256L),
  fcnet_activation = c("relu", "tanh", "swish", "silu", "linear"),
  ...
)
}
\arguments{
\item{fcnet_hiddens}{A positive integer vector. Numbers of units of the
intermediate layers.}

\item{fcnet_activation}{A character value specifying the activation function.
Possible values are "ReLU" (default), "tanh", "Swish" (or "SiLU"), or
"linear".}

\item{...}{Other configurations. See source code of RLlib.
https://github.com/ray-project/ray/blob/master/rllib/models/catalog.py}
}
\value{
A list of DNN configuration parameters
}
\description{
DNN (deep neural network) configuration for reinforcement learning.
For detail, see Section 3.1 of the original paper.
}
\examples{
\dontrun{
escalation_rule <- learn_escalation_rule(
  J = 6, target = 0.25, epsilon = 0.04, delta = 0.1,
  N_total = 36, N_cohort = 3, seed = 123,
  rl_config = rl_config_set(
    iter = 1000, 
    # We change the DNN model
    model = rl_dnn_config(fcnet_hiddens = c(512L, 512L), fcnet_activation = "tanh")
  )
)} 

}
