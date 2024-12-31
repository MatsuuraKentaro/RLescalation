#' EscalationRule Class
#'
#' @description
#' This class represents an escalation rule that generates a next escalation.
#'
#' @field policy The RLlib policy that is a Python object.
#' @field dir Directory path of the escalation rule (policy).
#' @field dirpath Full path to the directory of the escalation rule.
#' @field created_at Created time of this object.
#' @field info Information when learning the escalation rule.
#' @field input Inputs for learning the escalation rule.
#' @field log The log of scores during the learning of the escalation rule.
#' @field checkpoints The integer vector of iteration counts for checkpoints.
#' @field checkpoints_paths The paths to the directories where each checkpoint is stored.
#'
#' @export
EscalationRule <- R6Class(
  "EscalationRule",

  public = list(
    policy = NULL,
    dir = NULL,
    dirpath = NULL,
    created_at = NULL,

    info = NULL,
    input = NULL,
    log = NULL,
    checkpoints = NULL,
    checkpoints_paths = NULL,

    #' @description
    #' Create a new EscalationRule object.
    #'
    #' @param dir A character value. A directory name or path where an
    #'        escalation rule is outputted. By default, the latest escalation
    #'        rule is searched in 'base_dir'.
    #' @param base_dir A character value. A directory path that is used as the
    #'        parent directory if the 'dir' argument is a directory name and is
    #'        not used otherwise.
    initialize = function(dir = "latest", base_dir = "escalation_rules") {
      # Check arguments
      stopifnot(length(dir) == 1L)
      stopifnot(length(base_dir) == 1L)

      # Identify the specified directory path
      if (dir == "latest") {
        # Search the latest directory in 'base_dir'
        df <- file.info(dir(base_dir, full.names = TRUE))
        df <- df[df$isdir, , drop = FALSE]
        df <- df[df$mtime == max(df$mtime), , drop = FALSE]
        stopifnot("Cannot identify the latest escalation rule" = nrow(df) == 1L)
        dir <- rownames(df)
      } else if (!grepl("/|\\\\", dir)) {
        # If 'dir' does not contain '/' or '\\', it is a directory name.
        dir <- file.path(base_dir, dir)
      }

      # If dir is a checkpoint directory, change it to its policy directory
      if ("policies" %in% list.dirs(dir, full.names = FALSE, recursive = FALSE)) {
        dir <- file.path(dir, "policies", "default_policy")
      }

      # Restore policy object
      policy_lib <- reticulate::import("ray.rllib.policy.policy")
      policy <- policy_lib$Policy$from_checkpoint(dir)

      # Compress the policy directory and keep it in binary format in this object
      compressed_policy_file <- tempfile(fileext = ".zip")
      zip(zipfile = compressed_policy_file, files = list.files(dir, full.names = TRUE))
      private$policy_binary <- readBin(compressed_policy_file, what = "raw",
                                       n = file.info(compressed_policy_file)$size)

      self$policy <- policy
      self$dir <- dir
      self$dirpath <- normalizePath(dir)
      self$created_at <- format(Sys.time(), format = "%Y-%m-%d %H:%M:%S")
    },

    #' @description
    #' Compute optimal action probabilities using the obtained escalation rule 
    #' for data of N and DLT.
    #'
    #' @param current_dose An integer value. This is the current dose index, 
    #'        which is within `1:J`. 
    #' @param data_Ns A numeric vector. The cumulative number of patients  
    #'        assigned to each dose in your clinical trial.
    #' @param data_DLTs A numeric vector. The cumulative number of DLTs 
    #'        corresponding to each dose for the 'data_Ns' argument.
    #'
    #' @return A character that represents the optimal action. One of the followings:
    #'         down, stay, up, MTD_1, ..., MTD_J, no_MTD
    #'
    #' @importFrom glue glue
    opt_action = function(current_dose, data_Ns, data_DLTs) {
      # If policy has been reset, reload it from the directory where it is stored
      if (is.null(self$policy$config)) {
        # If the directory where policy is stored cannot be found, restore it from the binary data
        if (!dir.exists(self$dir)) {
          compressed_policy_file <- tempfile(fileext = ".zip")
          writeBin(private$policy_binary, compressed_policy_file)
          unzip(compressed_policy_file)
          message(glue("Created escalation rule directory '{self$dir}'."))
        }
        policy_lib <- reticulate::import("ray.rllib.policy.policy")
        self$policy <- policy_lib$Policy$from_checkpoint(self$dir)
      }

      # Extract the clinical trial settings from the escalation rule (policy)
      policy <- self$policy
      env_config <- policy$config$env_config
      J <- env_config$J
      target <- env_config$target
      N_total <- env_config$N_total

      # Check arguments
      current_dose <- as.integer(current_dose)
      stopifnot(length(current_dose) == 1L, current_dose >= 1L, current_dose <= J)
      stopifnot(length(data_Ns) == length(data_DLTs), length(data_Ns) == J)
      data_Ns   <- as.numeric(data_Ns)
      data_DLTs <- as.numeric(data_DLTs)
      stopifnot(all(data_Ns >= data_DLTs))
      
      # Obtain the logits of next actions
      state <- compute_state(current_dose, J, data_Ns, data_DLTs, N_total)
      logits <- policy$compute_single_action(state, full_fetch = TRUE)[[3L]]$action_dist_inputs
      names(logits) <- c(c("down", "stay", "up"), paste0("MTD_", 1L:J))
      
      # Determine the next action (max over all), move (max over down/stay/up), and MTD (max over MTD_1, ..., MTD_J)
      next_action <- names(which.max(logits))
      move <- names(which.max(logits[1L:3L]))
      MTD <- names(which.max(logits[(3L+1L):(3L+J)]))

      if (current_dose == 1L && next_action == "down") {
        return("no_MTD")
      }
      
      if (sum(data_Ns) == N_total) {
        # Return MTD instead of next_action because MTD should be determined
        return(MTD)
      } 
      
      # Return escalation instead of next_action because the trial should be continued
      if (move == "down") {
        recommended <- ifelse(current_dose == 1L, "stay", "down")
      } else if (move == "stay") {
        recommended <- "stay"
      } else if (move == "up") {
        recommended <- ifelse(current_dose == J, "stay", "up")
      }
      
      return(recommended)
    },

    #' @description
    #' Resume learning the escalation rule. This function updates the original
    #' EscalationRule object.
    #'
    #' @param iter A number of additional iterations.
    #'
    #' @return An updated \link{EscalationRule} object.
    resume_learning = function(iter) {
      checkpoint_path <- tail(self$checkpoints_paths, 1L)
      algorithm <- reticulate::import("ray.rllib.algorithms.algorithm")
      algo <- algorithm$Algorithm$from_checkpoint(checkpoint_path)

      output_path <- self$dirpath
      output_checkpoint_path <- sub("^(.*)_\\d+$", "\\1", checkpoint_path)
      save_start_iter <- self$input$rl_config$save_start_iter
      save_every_iter <- self$input$rl_config$save_every_iter
      N_update <- self$info$iterations + iter

      n_start <- self$info$iterations + 1L

      result <- train_algo(algo, n_start, N_update,
                           output_path, output_checkpoint_path,
                           save_start_iter, save_every_iter)

      checkpoints <- c(self$checkpoints_paths, result$checkpoints)
      episode_data <- rbind(self$info$log, result$episode_data)

      self$info$iterations <- N_update
      self$log <- episode_data
      private$set_checkpoints(checkpoints)

      invisible(self)
    },

    #' @description
    #' Set information when learning the escalation rule.
    #'
    #' @param info Information when learning the escalation rule.
    #' @param input Inputs for learning the escalation rule.
    #' @param log The log of scores during the learning of the escalation rule.
    #' @param checkpoints The paths to the directories where each checkpoint is stored.
    set_info = function(info, input, log, checkpoints) {
      self$info <- info
      self$input <- input
      self$log <- log
      private$set_checkpoints(checkpoints)
    },

    #' @description
    #' Print function for EscalationRule object
    #'
    #' @importFrom glue glue
    print = function() {
      print(glue("<EscalationRule>"))
      print(glue("dir: {self$dir}"))
      print(glue("created at: {self$created_at}"))
      if (!is.null(self$info)) {
        print(glue("call:"))
        print(glue("{deparse(self$info$call)}"))
        print(glue("iterations: {self$info$iterations}"))
        checkpoints <- paste0(self$checkpoints, collapse = ", ")
        print(glue("checkpoints: {checkpoints}"))
      }
    }
  ),

  private = list(
    policy_binary = NULL,

    set_checkpoints = function(checkpoints_paths) {
      self$checkpoints_paths = checkpoints_paths
      self$checkpoints <- as.integer(sub(".*_(\\d+)$", "\\1", checkpoints_paths))
    }
  )
)
