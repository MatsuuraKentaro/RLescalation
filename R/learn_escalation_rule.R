#' Build an Optimal Dose Escalation Rule using Reinforcement Learning
#'
#' @param J A positive integer value. The number of doses.
#' @param target A positive numeric value. The target DLT probability.
#' @param epsilon A positive numeric value. The acceptable range of target DLT 
#'        probabilities is defined as \[`target` - `epsilon`, `target` + `epsilon`\].
#' @param delta A positive numeric value. The unacceptable ranges of target DLT 
#'        probabilities are defined as \[0, `target` - `delta`\] and 
#'        \[`target` + `delta`, 1\].
#' @param N_total A positive integer value. The total number of patients.
#' @param N_cohort A positive integer value. The number of patients for each cohort.
#' @param seed An integer value. Random seed for reinforcement learning.
#' @param rl_config A list. Other settings for reinforcement learning. See
#'        \link{rl_config_set} for details.
#' @param rl_scenarios A list. Scenarios used for reinforcement learning. 
#'        Default is `NULL` (use scenarios in the Sect. 2.2 of the original paper).
#'        See \link{compute_rl_scenarios} for details.      
#' @param output_dir A character value. Directory name or path to store the
#'        built escalation rule. Default is the current datetime.
#' @param output_base_dir A character value. Parent directory path where the
#'        built escalation rule will be stored. Valid only if 'output_dir' does
#'        not contain '/'. Default is "escalation_rules".
#' @param checkpoint_dir A character value. Parent directory path to save
#'        checkpoints. It enables you to resume learning from that point onwards.
#'        Default is "checkpoints".
#'
#' @returns An \link{EscalationRule} object.
#' 
#' @examples
#' library(RLescalation)
#' 
#' # We obtain an optimal dose escalation rule by executing `learn_escalation_rule()`.
#' \dontrun{
#' escalation_rule <- learn_escalation_rule(
#'   J = 6, target = 0.25, epsilon = 0.04, delta = 0.1,
#'   N_total = 36, N_cohort = 3, seed = 123,
#'   rl_config = rl_config_set(iter = 1000)
#' )}
#' 
#' @importFrom glue glue
#'
#' @export
learn_escalation_rule <- function(
    J, target, epsilon, delta, N_total, N_cohort,
    seed = NULL, rl_config = rl_config_set(), rl_scenarios = NULL,
    output_dir = format(Sys.time(), "%Y%m%d_%H%M%S"),
    output_base_dir = "escalation_rules", checkpoint_dir = "checkpoints") {

  # -------------------------------------------------------------------------
  # Check arguments ---------------------------------------------------------
  # -------------------------------------------------------------------------
  J <- as.integer(J)
  N_total <- as.integer(N_total)
  N_cohort <- as.integer(N_cohort)
  stopifnot(length(J) == 1L, J >= 1L)
  stopifnot(length(N_total) == 1L, N_total >= 1L)
  stopifnot(length(N_cohort) == 1L, N_cohort >= 1L)
  stopifnot(N_total %% N_cohort == 0.)
  
  target <- as.double(target)
  epsilon <- as.double(epsilon)
  delta <- as.double(delta)
  stopifnot(length(target) == 1L, 0 < target, target < 1)
  stopifnot(length(epsilon) == 1L, 0 < epsilon, epsilon < 1)
  stopifnot(length(delta) == 1L, 0 < delta, delta < 1)
  stopifnot(epsilon < delta)
  
  stopifnot(is.null(seed) || length(seed) == 1L)
  if (!is.null(seed)) {
    seed <- as.integer(seed)
  }

  # TODO
  stopifnot(is.null(rl_scenarios) || (length(rl_scenarios) == 3L && all(names(rl_scenarios) == c("prob", "MTD", "weight"))))
  if (is.null(rl_scenarios)) {
    rl_scenarios <- compute_rl_scenarios(J, target, epsilon, delta)
  }
  
  # TODO
  if (!grepl("/", output_dir)) {
    # If 'output_dir' does not contain '/', it is a directory name.
    output_path <- file.path(output_base_dir, output_dir)
  }
  output_checkpoint_path <- file.path(checkpoint_dir, basename(output_path))

  # -------------------------------------------------------------------------
  # Execute reinforcement learning ------------------------------------------
  # -------------------------------------------------------------------------
  reticulate::source_python(system.file("python/DoseEscalationEnv.py", package = "RLescalation"))

  # Convert R list of vector to Python List of List.
  # e.g., list(c(1, 2), -1, c(3, 4, 5)) => [[0, 1], [-1], [2, 3, 4]]
  # - reticulate::r_to_py() converts list(c(1, 2), -1, c(3, 4, 5)) into [[1, 2], -1, [3, 4, 5]],
  #   so when the length of vector (MTDs) is 1, we need add list()
  # - x == -1 stands for "no_MTD", so we preserve it
  MTD_scenarios <- lapply(rl_scenarios$MTD, function(x) {
    if (length(x) == 1) {
      list(as.integer(ifelse(x == -1L, x, x - 1L)))
    } else {
      as.integer(ifelse(x == -1L, x, x - 1L))
    }
  })
  weight_scenarios <- rl_scenarios$weight / sum(rl_scenarios$weight)
  
  env_config <- list(
    J = J, N_total = N_total, N_cohort = N_cohort,
    prob_scenarios = rl_scenarios$prob,
    MTD_scenarios = MTD_scenarios,
    weight_scenarios = weight_scenarios
  )

  ppo <- reticulate::import("ray.rllib.algorithms.ppo")

  N_update <- rl_config$iter
  save_start_iter <- rl_config$save_start_iter
  save_every_iter <- rl_config$save_every_iter
  digits <- floor(log10(N_update)) + 1L
  num_env_runners <- rl_config$cores
  rl_config[c("iter", "save_start_iter", "save_every_iter", "cores")] <- NULL

  config <- ppo$PPOConfig()$
    environment(env = DoseEscalationEnv, env_config = env_config)$
    env_runners(num_env_runners = num_env_runners, num_envs_per_env_runner = 1L)$
    framework(framework = "torch")$
    debugging(seed = seed)  # Note: NULL is converted to None in Python.
  config <- do.call(config$training, rl_config)
  algo <- config$build()

  result <- train_algo(algo, n_start = 1L, N_update,
                       output_path, output_checkpoint_path,
                       save_start_iter, save_every_iter)

  # Create EscalationRule object
  escalation_rule <- EscalationRule$new(dir = output_path)
  info <- list(call = match.call(), iterations = N_update)
  default_arguments <- formals()
  input <- Map(eval, as.list(info$call)[-1L])
  default_arguments[names(input)] <- input
  input <- default_arguments
  escalation_rule$set_info(info, input, result$episode_data, result$checkpoints)

  escalation_rule
}
