# This file is part of the R package "aifeducation".
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as published by
# the Free Software Foundation.
#
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>


#' @title Base class for most objects
#' @description
#' Objects of this class containing fields and methods used in several other classes in 'AI for Education'.
#'
#' This class is **not** designed for a direct application and should only be used by developers.
#'
#' @return A new object of this class.
#' @family R6 Classes for Developers
#' @export
AIFEMaster <- R6::R6Class(
  classname = "AIFEMaster",
  public = list(
    #' @field last_training ('list()')\cr
    #'   List for storing the history, the configuration, and the results of the last
    #'   training. This information will be overwritten if a new training is started.
    #'
    #' * `last_training$start_time`: Time point when training started.
    #' * `last_training$learning_time`: Duration of the training process.
    #' * `last_training$finish_time`: Time when the last training finished.
    #' * `last_training$history`: History of the last training.
    #' * `last_training$data`: Object of class `table` storing the initial frequencies of the passed data.
    #' * `last_training$config`: List storing the configuration used for the last training.
    #'
    last_training = list(
      start_time = NA,
      learning_time = NULL,
      finish_time = NULL,
      history = list(),
      data = NULL,
      config = list()
    ),
    # General Information set and get--------------------------------------------
    #' @description Method for requesting the model information.
    #' @return `list` of all relevant model information.
    get_model_info = function() {
      return(list(
        model_license = private$model_info$model_license,
        model_name = private$model_info$model_name,
        model_id = private$model_info$model_id,
        model_name_root = private$model_info$model_name_root,
        model_label = private$model_info$model_label,
        model_date = private$model_info$model_date,
        model_language = private$model_info$model_language
      ))
    },
    #---------------------------------------------------------------------------
    #' @description Method for setting publication information of the model.
    #' @param authors List of authors.
    #' @param citation Free text citation.
    #' @param url URL of a corresponding homepage.
    #' @return Function does not return a value. It is used for setting the private members for publication information.
    set_publication_info = function(authors,
                                    citation,
                                    url = NULL) {
      private$publication_info$developed_by$authors <- authors
      private$publication_info$developed_by$citation <- citation
      private$publication_info$developed_by$url <- url
    },
    #--------------------------------------------------------------------------
    #' @description Method for requesting the bibliographic information of the model.
    #' @return `list` with all saved bibliographic information.
    get_publication_info = function() {
      return(private$publication_info)
    },
    #--------------------------------------------------------------------------
    #' @description Method for setting the license of the model.
    #' @param license `string` containing the abbreviation of the license or the license text.
    #' @return Function does not return a value. It is used for setting the private member for the software license of
    #'   the model.
    set_model_license = function(license = "CC BY") {
      private$model_info$model_license <- license
    },
    #' @description Method for getting the license of the model.
    #' @param license `string` containing the abbreviation of the license or the license text.
    #' @return `string` representing the license for the model.
    get_model_license = function() {
      return(private$model_info$model_license)
    },
    #--------------------------------------------------------------------------
    #' @description Method for setting the license of the model's documentation.
    #' @param license `string` containing the abbreviation of the license or the license text.
    #' @return Function does not return a value. It is used for setting the private member for the documentation license
    #'   of the model.
    set_documentation_license = function(license = "CC BY") {
      private$model_description$license <- license
    },
    #' @description Method for getting the license of the model's documentation.
    #' @param license `string` containing the abbreviation of the license or the license text.
    #' @return Returns the license as a `string`.
    get_documentation_license = function() {
      return(private$model_description$license)
    },
    #--------------------------------------------------------------------------
    #' @description Method for setting a description of the model.
    #' @param eng `string` A text describing the training, its theoretical and empirical background, and output in
    #'   English.
    #' @param native `string` A text describing the training , its theoretical and empirical background, and output in
    #'   the native language of the model.
    #' @param abstract_eng `string` A text providing a summary of the description in English.
    #' @param abstract_native `string` A text providing a summary of the description in the native language of the
    #'   model.
    #' @param keywords_eng `vector` of keyword in English.
    #' @param keywords_native `vector` of keyword in the native language of the model.
    #' @return Function does not return a value. It is used for setting the private members for the description of the
    #'   model.
    set_model_description = function(eng = NULL,
                                     native = NULL,
                                     abstract_eng = NULL,
                                     abstract_native = NULL,
                                     keywords_eng = NULL,
                                     keywords_native = NULL) {
      if (!is.null(eng)) {
        private$model_description$eng <- eng
      }
      if (!is.null(native)) {
        private$model_description$native <- native
      }

      if (!is.null(abstract_eng)) {
        private$model_description$abstract_eng <- abstract_eng
      }
      if (!is.null(abstract_native)) {
        private$model_description$abstract_native <- abstract_native
      }

      if (!is.null(keywords_eng)) {
        private$model_description$keywords_eng <- keywords_eng
      }
      if (!is.null(keywords_native)) {
        private$model_description$keywords_native <- keywords_native
      }
    },
    #' @description Method for requesting the model description.
    #' @return `list` with the description of the classifier in English and the native language.
    get_model_description = function() {
      return(private$model_description)
    },
    #---------------------------------------------------------------------------
    #' @description Method for requesting a summary of the R and python packages' versions used for creating the model.
    #' @return Returns a `list` containing the versions of the relevant R and python packages.
    get_package_versions = function() {
      return(
        list(
          r_package_versions = private$r_package_versions,
          py_package_versions = private$py_package_versions
        )
      )
    },
    #---------------------------------------------------------------------------
    #' @description Method for requesting a summary of tracked energy consumption during training and an estimate of the
    #'   resulting CO2 equivalents in kg.
    #' @param track_mode `r get_param_doc_desc("track_mode")`
    #' @return Returns a `list` containing the tracked energy consumption, CO2 equivalents in kg, information on the
    #'   tracker used, and technical information on the training infrastructure.
    get_sustainability_data = function(track_mode = "training") {
      if (track_mode == "training") {
        return(private$sustainability$track_log)
      } else if (track_mode == "inference") {
        return(private$sustainability_inference)
      }
    },
    #---------------------------------------------------------------------------
    #' @description Method for requesting the machine learning framework used for the model.
    #' @return Returns a `string` describing the machine learning framework used for the classifier.
    get_ml_framework = function() {
      return(private$ml_framework)
    },

    #-------------------------------------------------------------------------
    #' @description Method for checking if the model was successfully configured. An object can only be used if this
    #'   value is `TRUE`.
    #' @return `bool` `TRUE` if the model is fully configured. `FALSE` if not.
    is_configured = function() {
      return(private$configured)
    },
    #--------------------------------------------------------------------------
    #' @description Check if the [TEFeatureExtractor] is trained.
    #' @return Returns `TRUE` if the object is trained and `FALSE` if not.
    is_trained = function() {
      return(private$trained)
    },
    #--------------------------------------------------------------------------
    #' @description Method for requesting all private fields and methods. Used for loading and updating an object.
    #' @return Returns a `list` with all private fields and methods.
    get_private = function() {
      return(private)
    },
    #--------------------------------------------------------------------------
    #' @description Return all fields.
    #' @return Method returns a `list` containing all public and private fields
    #' of the object.
    get_all_fields = function() {
      public_list <- NULL
      private_list <- NULL

      for (entry in names(self)) {
        if (!is.function(self[[entry]]) & !is.environment(self[[entry]])) {
          public_list[entry] <- list(self[[entry]])
        }
      }

      for (entry in names(private)) {
        if (!is.function(private[[entry]]) & !is.environment(private[[entry]])) {
          private_list[entry] <- list(private[[entry]])
        }
      }

      return(
        list(
          public = public_list,
          private = private_list
        )
      )
    },
    #--------------------------------------------------------------------------
    #' @description Method for requesting the model configuration.
    #' @return Returns a `list` with all configuration parameters used during configuration.
    get_model_config = function() {
      return(private$model_config)
    }
  ),
  private = list(
    model = NULL,
    model_config = list(),
    ml_framework = NA,
    sustainability_tracker = NA,
    sustainability = list(
      sustainability_tracked = FALSE,
      track_log = data.frame()
    ),
    sustainability_inference = data.frame(),
    dir_checkpoint = NULL,
    model_info = list(
      model_license = NA,
      model_name = NA,
      model_name_root = NA,
      model_id = NA,
      name_root = NA,
      model_label = NA,
      model_date = NA
    ),
    publication_info = list(
      developed_by = list(
        authors = NULL,
        citation = NULL,
        url = NULL
      )
    ),
    model_description = list(
      eng = NULL,
      native = NULL,
      abstract_eng = NULL,
      abstract_native = NULL,
      keywords_eng = NULL,
      keywords_native = NULL,
      license = NA
    ),
    r_package_versions = list(
      aifeducation = NA,
      reticulate = NA
    ),
    py_package_versions = list(
      tensorflow = NA,
      torch = NA,
      keras = NA,
      numpy = NA
    ),
    log_config = list(
      log_dir = NULL,
      log_write_interval = 10L,
      log_state_file = NULL,
      log_loss_file = NULL
    ),
    log_state = list(
      last_log = NULL,
      value_top = 0L,
      total_top = 1L,
      message_top = NA
    ),

    # Object stores estimates for calculating learning rates
    lr_statistics = data.frame(),

    # Variable for checking if the object is successfully configured. Only if
    # this is TRUE the object can be used
    configured = FALSE,

    # Variable for checking if the object has been successfully trained.
    trained = FALSE,

    #--------------------------------------------------------------------------
    # Method for setting the model info
    set_model_info = function(model_name, label, model_date) {
      private$model_info$model_name <- model_name
      private$model_info$model_label <- label
      private$model_info$model_date <- model_date
    },
    #-----------------------------------------------------------------------
    load_reload_python_scripts = function() {
      return(NULL)
    },
    #-------------------------------------------------------------------------
    # Method for setting configured to TRUE
    set_configuration_to_TRUE = function() {
      private$configured <- TRUE
    },
    #-------------------------------------------------------------------------
    # Method for checking if the configuration is done successfully
    check_config_for_TRUE = function() {
      if (!private$configured) {
        stop("The object is not configured. Please call the method configure.")
      }
    },
    #--------------------------------------------------------------------------
    # Method for checking if the configuration is already done
    check_config_for_FALSE = function() {
      if (private$configured) {
        stop("The object is configured. Please create a new object if you would like to change the object's
             configuration.")
      }
    },
    #-------------------------------------------------------------------------
    check_for_untrained = function() {
      if (self$is_trained()) {
        stop("The model has already been trained and cant't be modified. Please create a new model if you need antoher training run.")
      }
    },
    #---------------------------------------------------------------------------
    check_for_trained = function() {
      if (!self$is_trained()) {
        stop("The model has not been trained. Please train the model before you use it.")
      }
    },
    #--------------------------------------------------------------------------
    set_package_versions = function() {
      private$r_package_versions$aifeducation <- packageVersion("aifeducation")
      private$r_package_versions$reticulate <- packageVersion("reticulate")

      private$py_package_versions$torch <- torch["__version__"]
      private$py_package_versions$numpy <- np$version$short_version
    },
    #-------------------------------------------------------------------------
    load_base_config_and_docs_general = function(config_public, config_private) {
      # Set configuration of the core model
      private$model_config <- config_private$model_config

      # Set model info
      private$set_model_info(
        model_name = config_private$model_info$model_name,
        label = config_private$model_info$model_label,
        model_date = config_private$model_info$model_date
      )

      # Set last training
      self$last_training$config <- config_public$last_training$config
      self$last_training$start_time <- config_public$last_training$start_time
      self$last_training$learning_time <- config_public$last_training$learning_time
      self$last_training$finish_time <- config_public$last_training$finish_time
      self$last_training$history <- config_public$last_training$history
      self$last_training$data <- config_public$last_training$data

      # Set license
      self$set_model_license(config_private$model_info$model_license)
      self$set_documentation_license(config_private$model_description$license)

      # Set description and documentation
      self$set_model_description(
        eng = config_private$model_description$eng,
        native = config_private$model_description$native,
        abstract_eng = config_private$model_description$abstract_eng,
        abstract_native = config_private$model_description$abstract_native,
        keywords_eng = config_private$model_description$keywords_eng,
        keywords_native = config_private$model_description$keywords_native
      )

      # Set publication info
      self$set_publication_info(
        authors = config_private$publication_info$developed_by$authors,
        citation = config_private$publication_info$developed_by$citation,
        url = config_private$publication_info$developed_by$url
      )

      # Get and set original package versions
      private$r_package_versions$aifeducation <- config_private$r_package_versions$aifeducation
      private$r_package_versions$reticulate <- config_private$r_package_versions$reticulate

      private$py_package_versions$torch <- config_private$py_package_versions$torch
      private$py_package_versions$numpy <- config_private$py_package_versions$numpy
    },
    #--------------------------------------------------------------------------
    init_and_start_sustainability_tracker = function(trace,
                                                     country_iso_code,
                                                     region,
                                                     measure_power_secs,
                                                     sustain_log_level) {
      # Trace
      print_message(
        msg = "Start Sustainability Tracking",
        trace = trace
      )

      if (check_versions(a = get_py_package_version("codecarbon"), operator = ">=", b = "2.8.0")) {
        path_look_file <- codecarbon$lock$LOCKFILE
        if (file.exists(path_look_file)) {
          unlink(path_look_file)
        }
      }

      private$sustainability_tracker <- codecarbon$OfflineEmissionsTracker(
        country_iso_code = country_iso_code,
        region = region,
        tracking_mode = "machine",
        log_level = sustain_log_level,
        measure_power_secs = measure_power_secs,
        save_to_file = FALSE,
        save_to_api = FALSE,
        allow_multiple_runs = FALSE
      )
      private$sustainability_tracker$start()
    },
    #---------------------------------------------------------------------------
    stop_sustainability_tracker = function(trace, task) {
      # Trace
      print_message(
        msg = "Stop Sustainability Tracking",
        trace = trace
      )

      private$sustainability_tracker$stop()

      sustain_data <- as.data.frame(
        summarize_tracked_sustainability(
          sustainability_tracker = private$sustainability_tracker,
          task = task
        )
      )
      return(sustain_data)
    },
    #------------------------------------------------------------------------------
    init_and_start_sustainability_tracking = function() {
      if (self$last_training$config$sustain_track) {
        private$init_and_start_sustainability_tracker(
          trace = self$last_training$config$trace,
          country_iso_code = self$last_training$config$sustain_iso_code,
          region = self$last_training$config$sustain_region,
          measure_power_secs = self$last_training$config$sustain_interval,
          sustain_log_level = self$last_training$config$sustain_log_level
        )
      }
    },
    #---------------------------------------------------------------------------
    stop_sustainability_tracking = function(task = NA) {
      if (self$last_training$config$sustain_track) {
        sustain_data <- private$stop_sustainability_tracker(
          trace = self$last_training$config$trace,
          task = task
        )

        private$sustainability$sustainability_tracked <- TRUE

        if (is.null_or_na(private$sustainability$track_log)) {
          private$sustainability$track_log <- sustain_data
        } else {
          private$sustainability$track_log <- rbind(
            private$sustainability$track_log,
            sustain_data
          )
        }
      }
    },
    #--------------------------------------------------------------------------
    stop_sustainability_tracking_inference = function(trace, task) {
      # Trace
      print_message(
        msg = "Stop Sustainability Tracking",
        trace = trace
      )
      private$sustainability_tracker$stop()
      return(
        as.data.frame(
          summarize_tracked_sustainability(
            sustainability_tracker = private$sustainability_tracker,
            task = task
          )
        )
      )
    },
    #-------------------------------------------------------------------------
    columnes_sustainability = function() {
      return(c(
        "sustainability_data.duration_sec",
        "sustainability_data.co2eq_kg",
        "sustainability_data.cpu_energy_kwh",
        "sustainability_data.gpu_energy_kwh",
        "sustainability_data.ram_energy_kwh",
        "sustainability_data.total_energy_kwh",
        "technical.tracker",
        "technical.py_package_version",
        "technical.cpu_count",
        "technical.cpu_model",
        "technical.gpu_count",
        "technical.gpu_model",
        "technical.ram_total_size",
        "region.country_name",
        "region.country_iso_code",
        "region.region"
      ))
    },
    #-------------------------------------------------------------------------
    columnes_sustainability_training = function() {
      return(c(
        "sustainability_tracked",
        "date",
        "task",
        private$columnes_sustainability()
      ))
    },
    #-------------------------------------------------------------------------
    columnes_sustainability_inference = function() {
      return(c(
        "sustainability_tracked",
        "date",
        "task",
        "data",
        "batch",
        "min_seq_len",
        "mean_seq_len",
        "sd_seq_len",
        "max_seq_len",
        private$columnes_sustainability()
      ))
    },
    #-------------------------------------------------------------------------
    columnes_flops_estimates = function() {
      return(c(
        "date",
        "approach",
        "package",
        "version",
        "n_parameter",
        "batch_size",
        "n_batches",
        "n_epochs",
        "flops_bp_1",
        "flops_bp_2",
        "flops_bp_3",
        "flops_bp_4",
        "flops_counted"
      ))
    },
    #--------------------------------------------------------------------------
    check_and_update_column_names = function(data_frame, type) {
      if (type == "sustain_training") {
        column_names <- private$columnes_sustainability_training()
      } else if (type == "sustain_inference") {
        column_names <- private$columnes_sustainability_inference()
      } else if (type == "flops") {
        column_names <- private$columnes_flops_estimates()
      } else {
        stop("Type not implemented.")
      }

      tmp_data_frame <- data_frame
      add_vector <- vector(length = nrow(data_frame))
      add_vector <- NA
      for (clmn_name in setdiff(x = column_names, y = colnames(data_frame))) {
        tmp_data_frame[clmn_name] <- add_vector
      }
      return(tmp_data_frame)
    },
    #-------------------------------------------------------------------------
    # Method for loading sustainability data
    load_sustainability_data = function(model_dir) {
      sustainability_datalog_path <- file.path(model_dir, "sustainability.csv")
      if (file.exists(sustainability_datalog_path)) {
        private$sustainability$track_log <- private$check_and_update_column_names(
          data_frame = read.csv(sustainability_datalog_path),
          type = "sustain_training"
        )
        private$sustainability$sustainability_tracked <- TRUE
      } else {
        private$sustainability$sustainability_tracked <- FALSE
        private$sustainability$track_log <- data.frame()
      }
    },
    #-------------------------------------------------------------------------
    # Method for saving sustainability data
    save_sustainability_data = function(dir_path, folder_name) {
      save_location <- file.path(dir_path, folder_name)
      create_dir(dir_path, trace = TRUE, msg_fun = FALSE)
      sustain_matrix <- private$sustainability$track_log
      if (nrow(sustain_matrix) > 0L) {
        write.csv(
          x = sustain_matrix,
          file = file.path(save_location, "sustainability.csv"),
          row.names = FALSE
        )
      }
    },
    #-------------------------------------------------------------------------
    # Method for loading sustainability data inference
    load_sustainability_data_inference = function(model_dir) {
      sustainability_datalog_path <- file.path(model_dir, "sustainability_inf.csv")
      if (file.exists(sustainability_datalog_path)) {
        private$sustainability_inference <- private$check_and_update_column_names(
          data_frame = read.csv(sustainability_datalog_path),
          type = "sustain_inference"
        )
      } else {
        private$sustainability_inference <- data.frame()
      }
    },
    #-------------------------------------------------------------------------
    # Method for saving sustainability data inference
    save_sustainability_data_inference = function(dir_path, folder_name) {
      save_location <- file.path(dir_path, folder_name)
      create_dir(dir_path, trace = TRUE, msg_fun = FALSE)
      sustain_matrix <- private$sustainability_inference
      if (nrow(sustain_matrix) > 0L) {
        write.csv(
          x = sustain_matrix,
          file = file.path(save_location, "sustainability_inf.csv"),
          row.names = FALSE
        )
      }
    },
    #-------------------------------------------------------------------------
    # Method for loading flops estimates
    load_flops_estimates = function(model_dir) {
      datalog_path <- file.path(model_dir, "flops_estimates.csv")
      if (file.exists(datalog_path)) {
        private$flops_estimates <- private$check_and_update_column_names(
          data_frame = read.csv(datalog_path),
          type = "flops"
        )
      } else {
        private$flops_estimates <- data.frame()
      }
    },
    #-------------------------------------------------------------------------
    # Method for saving flops estimates
    save_flops_estimates = function(dir_path, folder_name) {
      save_location <- file.path(dir_path, folder_name)
      create_dir(dir_path, trace = TRUE, msg_fun = FALSE)
      sustain_matrix <- private$flops_estimates
      if (nrow(sustain_matrix) > 0L) {
        write.csv(
          x = sustain_matrix,
          file = file.path(save_location, "flops_estimates.csv"),
          row.names = FALSE
        )
      }
    },
    #---------------------------------------------------------------------------
    create_checkpoint_directory = function() {
      # Create a directory for the package
      tmp_dir <- create_and_get_tmp_dir()

      # Create a folder for the current task
      private$dir_checkpoint <- file.path(tmp_dir, generate_id(16L))
      create_dir(dir = private$dir_checkpoint, trace = FALSE)
    },
    #--------------------------------------------------------------------------
    clean_checkpoint_directory = function() {
      unlink(
        x = private$dir_checkpoint,
        recursive = TRUE,
        force = FALSE
      )
    },
    #--------------------------------------------------------------------------
    save_all_args = function(args, group = "training") {
      if (group %in% c("configure", "training")) {
        if (group == "training") {
          for (arg in names(args)) {
            if (!R6::is.R6(args[[arg]]) &
              !is.factor(args[[arg]]) &
              !arg %in% c("log_dir", "log_write_interval")) {
              self$last_training$config[arg] <- list(args[[arg]])
            }
          }
        } else if (group == "configure") {
          for (arg in names(args)) {
            if (!R6::is.R6(args[[arg]]) &
              !is.factor(args[[arg]]) &
              !arg %in% c(
                "log_dir", "log_write_interval",
                "base_model", "model_language", "model_label",
                "model_name", "tokenizer", "text_embeddings"
              )) {
              private$model_config[arg] <- list(args[[arg]])
            }
          }
        } else {
          stop("Argument 'group' must be 'configure' or 'training'.")
        }
      }
    },
    #------------------------------------------------------------------------------
    load_config_file = function(dir_path) {
      config_file <- load_R_config_state(dir_path)
      # Load public fields
      params_self <- names(self)
      for (param in params_self) {
        if (!is.function(self[[param]])) {
          param_in_file_public <- config_file$public[[param]]
          param_in_file_private <- config_file$private[[param]]

          # Check if values are in public
          if (!is.null_or_na(param_in_file_public)) {
            self[[param]] <- param_in_file_public
          }

          # check if values are in private
          if (!is.null_or_na(param_in_file_private)) {
            self[[param]] <- param_in_file_private
          }
        }
      }

      # Load private fields
      params_private <- names(private)
      for (param in params_private) {
        if (!is.function(private[[param]])) {
          param_in_file_public <- config_file$public[[param]]
          param_in_file_private <- config_file$private[[param]]

          # Check if values are in public
          if (!is.null_or_na(param_in_file_public)) {
            private[[param]] <- param_in_file_public
          }

          # check if values are in private
          if (!is.null_or_na(param_in_file_private)) {
            private[[param]] <- param_in_file_private
          }
        }
      }

      # Check Model config
      # Necessary for objects saved with aifeducation lower 1.1.2
      # These objects save the model config in the public part of the model
      if (!is.null(config_file$public[["model_config"]])) {
        private$model_config <- config_file$public[["model_config"]]
      }

      # Check if the model parameter for configuration are saved on other fields
      config_params <- setdiff(
        x = rlang::fn_fmls_names(self$configure),
        y = c("model_name", "model_label", "model_language", "text_embeddings", "features")
      )

      require_update <- check_versions(
        a = self$get_package_versions()$r_package_versions$aifeducation,
        operator = "<",
        b = "1.1.2"
      )
      if (require_update) {
        for (config_param in config_params) {
          # Search in public
          for (i in seq_along(config_file$public)) {
            current_entry <- config_file$public[[i]]
            if (is.list(current_entry)) {
              current_entry <- config_file$public[[i]][[config_param]]
            } else {
              current_entry <- NULL
            }

            if (!is.null_or_na(current_entry)) {
              private$model_config[config_param] <- list(current_entry)
            }
          }
          # Search in private
          for (i in seq_along(config_file$private)) {
            current_entry <- config_file$private[[i]]
            if (is.list(current_entry)) {
              current_entry <- config_file$private[[i]][[config_param]]
            } else {
              current_entry <- NULL
            }
            if (!is.null_or_na(current_entry)) {
              private$model_config[config_param] <- list(current_entry)
            }
          }
        }
      }

      require_update <- check_versions(
        a = self$get_package_versions()$r_package_versions$aifeducation,
        operator = "<",
        b = "9.9.9"
      )
      if (require_update) {
        # Update values for extensions
        # This in important for all cases that introduce new and additional parameters
        param_dict <- get_param_dict()
        config_params <- setdiff(
          x = rlang::fn_fmls_names(self$configure),
          y = c("model_name", "model_label", "model_language", "text_embeddings", "features", "feature_extractor")
        )
        if (is.function(self$configure)) {
          param_names_new <- config_params
          for (param in param_names_new) {
            if (is_valid_and_exportable_param(arg_name = param, param_dict = param_dict)) {
              if (is.null(private$model_config[[param]])) {
                if (!is.null(param_dict[[param]]$default_historic)) {
                  private$model_config[param] <- list(param_dict[[param]]$default_historic)
                } else {
                  warning("Historic default for ", param, " is missing in parameter dictionary.")
                }
              }
            }
            # Necessary for objects saved with aifeducation lower 1.1.0
            # Values were changed to upper and lower cases
            private$model_config[param] <- list(update_values_to_new_1.1.0(private$model_config[[param]]))
          }
        }
      }
    }
  )
)

#' @title Base class for objects using a pytorch model as core model.
#' @description
#' Objects of this class containing fields and methods used in several other classes in 'AI for Education'.
#'
#' This class is **not** designed for a direct application and should only be used by developers.
#'
#' @return A new object of this class.
#' @family R6 Classes for Developers
#' @export
AIFEBaseModel <- R6::R6Class(
  classname = "AIFEBaseModel",
  inherit = AIFEMaster,
  public = list(
    #---------------------------------------------------------------------------
    #' @description Method for counting the trainable parameters of a model.
    #' @return Returns the number of trainable parameters of the model.
    count_parameter = function() {
      iterator <- reticulate::as_iterator(private$model$parameters())
      iteration_finished <- FALSE
      count <- 0L
      while (!iteration_finished) {
        iter_results <- reticulate::iter_next(it = iterator)
        if (is.null(iter_results)) {
          iteration_finished <- TRUE
        } else {
          if (iter_results$requires_grad) {
            count <- count + iter_results$numel()
          }
        }
      }
      return(count)
    }
  ),
  private = list(
    #-------------------------------------------------------------------------
    prepare_history_data = function(history) {
      # Provide rownames for the history
      for (i in seq_along(history)) {
        if (!is.null(history[[i]])) {
          if (nrow(history[[i]]) == 2L) {
            rownames(history[[i]]) <- c("train", "val")
          } else if (nrow(history[[i]]) == 3L) {
            rownames(history[[i]]) <- c("train", "val", "test")
          }

          if (is.matrix(history[[i]])) {
            # Replace value -100 with the last value
            # Max index for replacements
            index_max <- ncol(history[[i]])
            for (j in seq_len(nrow(history[[i]]))) {
              # Check if -100 occurs in the row
              includes_m_100 <- (history[[i]][j, ] == -100L)

              # if at least one -100 occurs
              if (sum(includes_m_100) > 0L && !anyNA(includes_m_100)) {
                # min index for replacements
                index_min <- min(which(includes_m_100))
                # replace
                history[[i]][j, index_min:index_max] <- history[[i]][j, (index_min - 1L)]
              }
            }
          }
        }
      }
      return(history)
    }
  )
)
