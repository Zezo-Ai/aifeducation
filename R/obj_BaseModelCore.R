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

#' @title Abstract class for all BaseModels
#' @description This class contains all methods shared by all BaseModels.
#' @return `r get_description("return_object")`
#' @family R6 Classes for Developers
#' @export
BaseModelCore <- R6::R6Class(
  classname = "BaseModelCore",
  inherit = AIFEBaseModel,
  private = list(
    model_type = NULL,
    adjust_max_sequence_length = 0L,
    return_token_type_ids = FALSE,
    sequence_mode = "equal",
    model_info = list(),
    flops_estimates = data.frame(),
    publication_info = list(
      developed_by = list(
        authors = NULL,
        citation = NULL,
        url = NULL
      ),
      modified_by = list(
        authors = NULL,
        citation = NULL,
        url = NULL
      )
    ),

    #-------------------------------------------------------------------------
    load_reload_python_scripts = function() {
      load_py_scripts(
        c(
          "py_log.py",
          "datasets_transformer_compute_vocabulary.py",
          "datasets_transformer_prepare_data.py",
          "pytorch_transformer_callbacks.py",
          "pytorch_base_models_training_loops.py",
          "data_collator.py"
        )
      )
    },
    #--------------------------------------------------------------------------
    # Method for loading training history
    load_training_history = function(model_dir) {
      training_datalog_path <- file.path(model_dir, "history.log")
      if (file.exists(training_datalog_path)) {
        self$last_training$history <- utils::read.csv2(file = training_datalog_path)
      } else {
        self$last_training$history <- NA
      }
    },
    #--------------------------------------------------------------------------
    # Method for saving training history
    save_training_history = function(dir_path, folder_name) {
      if (!is.null_or_na(self$last_training$history)) {
        save_location <- file.path(dir_path, folder_name)
        create_dir(dir_path, trace = TRUE, msg_fun = FALSE)
        write.csv2(
          x = self$last_training$history,
          file = file.path(save_location, "history.log"),
          row.names = FALSE,
          quote = FALSE
        )
      }
    },
    #--------------------------------------------------------------------------
    save_tokenizer = function(dir_path, folder_name) {
      save_location <- file.path(dir_path, folder_name)
      create_dir(dir_path = save_location, trace = FALSE)
      save_to_disk(
        object = self$Tokenizer,
        dir_path = save_location,
        folder_name = "tokenizer"
      )
    },
    #--------------------------------------------------------------------------
    load_tokenizer = function(dir_path) {
      load_location <- file.path(dir_path, "tokenizer")
      self$Tokenizer <- load_from_disk(load_location)
    },
    #--------------------------------------------------------------------------
    load_BaseModel = function(dir_path) {},
    #--------------------------------------------------------------------------
    set_model_config_from_hf = function() {
      tmp_args <- rlang::fn_fmls_names(self$configure)
      for (arg in intersect(x = tmp_args, y = names(private$model$config))) {
        private$model_config[arg] <- list(private$model$config[arg])
      }
    },
    #--------------------------------------------------------------------------
    set_up_logger = function(log_dir, log_write_interval) {
      private$log_config$log_dir <- log_dir
      private$log_config$log_write_interval <- log_write_interval

      private$log_config$log_state_file <- file.path(private$log_config$log_dir, "aifeducation_state.log")
      private$log_config$log_loss_file <- file.path(private$log_config$log_dir, "aifeducation_loss.log")
    },
    #--------------------------------------------------------------------------
    update_logger = function(message) {
      private$log_state$value_top <- private$log_state$value_top + 1L

      private$log_state$last_log <- py$write_log_py(
        log_file = private$log_config$log_state_file,
        value_top = private$log_state$value_top,
        total_top = private$log_state$total_top,
        message_top = message,
        last_log = private$log_state$last_log,
        write_interval = private$log_config$log_write_interval
      )
    },
    #--------------------------------------------------------------------------
    prepare_data_for_training = function(raw_text_dataset) {
      # Update Logger
      private$update_logger("Prepare Data for Training")

      # Trace
      print_message(
        msg = "Prepare Data for Training",
        trace = self$last_training$config$trace
      )

      tokenized_texts_raw <- tokenize_dataset(
        dataset = raw_text_dataset,
        tokenizer = self$Tokenizer$get_tokenizer(),
        max_length = self$last_training$config$max_sequence_length - private$adjust_max_sequence_length,
        log_file = private$log_config$log_state_file,
        write_interval = private$log_config$log_write_interval,
        value_top = private$log_state$value_top,
        total_top = private$log_state$total_top,
        message_top = "Tokenize Text"
      )

      length_vector <- extract_column_from_py_dataset(tokenized_texts_raw, "length")
      if (self$last_training$config$full_sequences_only) {
        relevant_indices <- which(length_vector == self$last_training$config$max_sequence_length)
      } else {
        relevant_indices <- which(
          length_vector <= self$last_training$config$max_sequence_length &
            length_vector >= self$last_training$config$min_seq_len
        )
      }

      if (length(relevant_indices) > 0L) {
        tokenized_texts_raw <- tokenized_texts_raw$select(as.integer(relevant_indices - 1L))
      }

      relevant_columns <- c("input_ids", "attention_mask", "labels")
      if (private$return_token_type_ids) {
        relevant_columns <- append(relevant_columns, "token_type_ids")
      }
      if (self$last_training$config$whole_word) {
        relevant_columns <- append(relevant_columns, "word_ids")
      }
      tokenized_texts_raw <- tokenized_texts_raw$select_columns(relevant_columns)

      tokenized_texts_raw$set_format(type = "torch")
      tokenized_texts_raw <- tokenized_texts_raw$train_test_split(
        test_size = self$last_training$config$val_size
      )
      return(tokenized_texts_raw)
    },
    #--------------------------------------------------------------------------
    create_data_collator = function() {
      # Update Logger
      private$update_logger("Create Data Collator")

      # Trace
      print_message(
        msg = "Create Data Collator",
        trace = self$last_training$config$trace
      )

      if (self$last_training$config$whole_word) {
        tmp_data_collator <- py$DataCollatorForWholeWordMask(
          tokenizer = self$Tokenizer$get_tokenizer(),
          mlm_probability = self$last_training$config$p_mask,
          pad_input = FALSE
        )
      } else {
        tmp_data_collator <- transformers$DataCollatorForLanguageModeling(
          tokenizer = self$Tokenizer$get_tokenizer(),
          mlm = TRUE,
          mlm_probability = self$last_training$config$p_mask,
          return_tensors = "pt"
        )
      }
      return(tmp_data_collator)
    },
    #---------------------------------------------------------------------------
    create_trainer = function(tokenized_dataset, data_collator) {
      # Update Logger
      private$update_logger("Create Trainer")

      # Trace
      print_message(
        msg = "Create Trainer",
        trace = self$last_training$config$trace
      )

      # Trace
      msg <- ifelse(self$last_training$config$whole_word, "Using Whole Word Masking", "Using Token Masking")
      print_message(msg, self$last_training$config$trace)

      create_logger <- py$create_AIFETransformerCSVLogger_PT
      logger_args <- list(
        loss_file = private$log_config$log_loss_file,
        log_file = private$log_config$log_state_file,
        value_top = private$log_state$value_top,
        total_top = private$log_state$total_top,
        message_top = "Training...",
        min_step = 1L
      )
      logger <- do.call(create_logger, logger_args)

      if (check_versions(a = get_py_package_version("transformers"), operator = ">=", b = "5.0.0")) {
        training_args <- transformers$TrainingArguments(
          output_dir = private$dir_checkpoint,
          eval_strategy = "epoch",
          num_train_epochs = as.integer(self$last_training$config$n_epoch),
          logging_strategy = "epoch",
          save_strategy = "epoch",
          save_total_limit = 1L,
          load_best_model_at_end = TRUE,
          optim = "adamw_torch",
          learning_rate = self$last_training$config$learning_rate,
          per_device_train_batch_size = as.integer(self$last_training$config$batch_size),
          per_device_eval_batch_size = as.integer(self$last_training$config$batch_size),
          auto_find_batch_size = FALSE,
          report_to = "none",
          log_level = "error",
          disable_tqdm = !self$last_training$config$pytorch_trace,
          dataloader_pin_memory = torch$cuda$is_available(),
          remove_unused_columns = FALSE
        )
      } else if (
        check_versions(a = get_py_package_version("transformers"), operator = ">=", b = "4.46.0") &&
          check_versions(a = get_py_package_version("transformers"), operator = "<", b = "5.0.0")
      ) {
        training_args <- transformers$TrainingArguments(
          output_dir = private$dir_checkpoint,
          overwrite_output_dir = TRUE,
          eval_strategy = "epoch",
          num_train_epochs = as.integer(self$last_training$config$n_epoch),
          logging_strategy = "epoch",
          save_strategy = "epoch",
          save_total_limit = 1L,
          load_best_model_at_end = TRUE,
          optim = "adamw_torch",
          learning_rate = self$last_training$config$learning_rate,
          per_device_train_batch_size = as.integer(self$last_training$config$batch_size),
          per_device_eval_batch_size = as.integer(self$last_training$config$batch_size),
          save_safetensors = TRUE,
          auto_find_batch_size = FALSE,
          report_to = "none",
          log_level = "error",
          disable_tqdm = !self$last_training$config$pytorch_trace,
          dataloader_pin_memory = torch$cuda$is_available(),
          remove_unused_columns = FALSE
        )
      } else {
        training_args <- transformers$TrainingArguments(
          output_dir = private$dir_checkpoint,
          overwrite_output_dir = TRUE,
          evaluation_strategy = "epoch",
          num_train_epochs = as.integer(self$last_training$config$n_epoch),
          logging_strategy = "epoch",
          save_strategy = "epoch",
          save_total_limit = 1L,
          load_best_model_at_end = TRUE,
          optim = "adamw_torch",
          learning_rate = self$last_training$config$learning_rate,
          per_device_train_batch_size = as.integer(self$last_training$configbatch_size),
          per_device_eval_batch_size = as.integer(self$last_training$config$batch_size),
          save_safetensors = TRUE,
          auto_find_batch_size = FALSE,
          report_to = "none",
          log_level = "error",
          disable_tqdm = !self$last_training$config$pytorch_trace,
          remove_unused_columns = FALSE
        )
      }

      if (check_versions(a = get_py_package_version("transformers"), operator = ">=", b = "4.46.0")) {
        tmp_trainer <- transformers$Trainer(
          model = private$model,
          train_dataset = tokenized_dataset$train,
          eval_dataset = tokenized_dataset$test,
          args = training_args,
          data_collator = data_collator,
          processing_class = self$Tokenizer$get_tokenizer()
        )
      } else {
        tmp_trainer <- transformers$Trainer(
          model = private$model,
          train_dataset = tokenized_dataset$train,
          eval_dataset = tokenized_dataset$test,
          args = training_args,
          data_collator = data_collator,
          tokenizer = self$Tokenizer$get_tokenizer()
        )
      }

      tmp_trainer$remove_callback(transformers$integrations$CodeCarbonCallback)
      if (!as.logical(self$last_training$config$pytorch_trace)) {
        tmp_trainer$remove_callback(transformers$PrinterCallback)
        tmp_trainer$remove_callback(transformers$ProgressCallback)
      }

      # Add Callbacks
      tmp_trainer$add_callback(logger)

      return(tmp_trainer)
    },
    #---------------------------------------------------------------------------
    calc_flops_architecture_based_iternal = function(batch_size, n_batches, n_epochs) {
      # Trace
      print_message(
        msg = "Calculate Flops Based on Architecture",
        trace = self$last_training$config$trace
      )

      results <- self$calc_flops_architecture_based(
        batch_size = batch_size,
        n_batches = n_batches,
        n_epoch = n_epochs
      )

      private$flops_estimates <- rbind(
        private$flops_estimates,
        results
      )
    },
    #----------------------------------------------------------------------------
    start_training = function(trainer) {
      # Update Logger
      private$update_logger("Training")

      # Trace
      print_message(
        msg = "Start Training",
        trace = self$last_training$config$trace
      )

      if (torch$cuda$is_available()) {
        torch$cuda$empty_cache()
      }
      trainer$train()
    },
    #---------------------------------------------------------------------------
    config_dataset_prograss_bar = function() {
      if (self$last_training$config$pytorch_trace) {
        datasets$enable_progress_bars()
      } else {
        datasets$disable_progress_bars()
      }
    },
    #--------------------------------------------------------------------------
    check_arg_combinations = function(args) {
      # Placeholder for the child classes
    },
    #---------------------------------------------------------------------------
    do_configuration = function(args) {
      # Load or reload python scripts
      private$load_reload_python_scripts()

      # Check if the object is not configured
      private$check_config_for_FALSE()

      # Check arguments
      check_all_args(args = args)

      # Check argument combinations
      private$check_arg_combinations(args = args)

      # Save args
      private$save_all_args(args = args, group = "configure")

      # Create the model
      configuration <- private$create_model(args)

      # Create the tokenizer
      self$Tokenizer <- args$tokenizer$clone(deep = TRUE)

      # Set package versions
      private$set_package_versions()

      # Prevent the object from modification
      private$set_configuration_to_TRUE()
    },
    #--------------------------------------------------------------------------
    do_training = function(args) {
      # Check if the model is configured
      private$check_config_for_TRUE()

      # Check all arguments
      check_all_args(args = args)

      # Save args
      private$save_all_args(args = args, group = "training")

      # Check arg combinations
      if (!inherits(self$Tokenizer, "WordPieceTokenizer") & self$last_training$config$whole_word) {
        print_message(
          msg = "Whole word masking is only available for WordPieceTokenizer. Set whole_word to 'FALSE'.",
          trace = TRUE
        )
        self$last_training$config$whole_word <- FALSE
      }

      # Load or reload python scripts
      private$load_reload_python_scripts()

      # set up logger
      private$set_up_logger(log_dir = args$log_dir, log_write_interval = args$log_write_interval)
      private$log_state$value_top <- 0L
      private$log_state$total_top <- 6L

      # Update logger
      private$update_logger("Prepare Process")

      # Configurate datasets progress bar
      private$config_dataset_prograss_bar()

      # Check and create temporary directory for checkpoints
      private$create_checkpoint_directory()

      # Start Sustainability Tracking
      private$init_and_start_sustainability_tracking()

      # Prepare Data for Training
      prepared_data <- private$prepare_data_for_training(raw_text_dataset = args$text_dataset$get_dataset())

      # Calculate Flops based on architecture-approach
      if (private$model_type != "longformer") {
        private$calc_flops_architecture_based_iternal(
          batch_size = self$last_training$config$batch_size,
          n_batches = ceiling(prepared_data$train$num_rows / self$last_training$config$batch_size),
          n_epochs = self$last_training$config$n_epoch
        )
      }

      # Create Data Collator
      data_collator <- private$create_data_collator()

      # Create Trainer
      trainer <- private$create_trainer(
        tokenized_dataset = prepared_data,
        data_collator = data_collator
      )

      # Start Training
      private$start_training(trainer)

      # Save training history
      history_log <- pandas$DataFrame(trainer$state$log_history)
      self$last_training$history <- clean_pytorch_log_transformers(history_log)

      # Stop sustainability tracking if requested
      private$stop_sustainability_tracking(task = "training")

      # Clean temporary directory
      private$clean_checkpoint_directory()

      # Update logger
      private$update_logger("Finish")

      # Trace
      print_message(
        msg = "Finish",
        trace = self$last_training$config$trace
      )
    }
  ),
  public = list(
    #' @field Tokenizer ('TokenizerBase')\cr
    #' Objects of class `TokenizerBase`.
    Tokenizer = NULL,

    #--------------------------------------------------------------------------
    #' @description Creates BaseModel from a pretrained model
    #' @param model_dir `r get_description("model_dir")`
    #' @param tokenizer_dir `r get_param_doc_desc("tokenizer_dir")`
    #' @return `r get_description("return_object")`
    create_from_hf = function(model_dir = NULL, tokenizer_dir = NULL) {
      if (is.null(tokenizer_dir)) {
        tokenizer_dir <- model_dir
      }

      # Load the BaseModel
      tmp_model <- private$load_BaseModel(model_dir)
      # transformers$AutoModelForMaskedLM$from_pretrained(model_dir)

      # Check if the model is the correct model type
      detected_model_type <- detect_base_model_type(tmp_model)
      if (detected_model_type != private$model_type) {
        stop("Detected ", detected_model_type, " but expected ", private$model_type, ".")
      }

      # Add model to the R6 class
      private$model <- tmp_model

      # Set Model Config
      private$set_model_config_from_hf()

      # Load Sustainability Data
      private$load_sustainability_data(model_dir = model_dir)

      # Load Sustainability Data Inference
      private$load_sustainability_data_inference(model_dir = model_dir)

      # Load training history
      private$load_training_history(model_dir = model_dir)

      # Create and Load the Tokenizer
      tokenizer <- HuggingFaceTokenizer$new()
      tokenizer$create_from_hf(tokenizer_dir)
      self$Tokenizer <- tokenizer

      # Set configured to TRUE to avoid changes in the model
      private$set_configuration_to_TRUE()
    },
    #--------------------------------------------------------------------------
    #' @description Traines a BaseModel
    #' @param text_dataset `r get_param_doc_desc("text_dataset")`
    #' @param p_mask `r get_param_doc_desc("p_mask")`
    #' @param whole_word `r get_param_doc_desc("whole_word")`
    #' @param val_size `r get_param_doc_desc("val_size")`
    #' @param n_epoch `r get_param_doc_desc("n_epoch")`
    #' @param batch_size `r get_param_doc_desc("batch_size")`
    #' @param max_sequence_length `r get_param_doc_desc("max_sequence_length")`
    #' @param full_sequences_only `r get_param_doc_desc("full_sequences_only")`
    #' @param min_seq_len `r get_param_doc_desc("min_seq_len")`
    #' @param learning_rate `r get_param_doc_desc("learning_rate")`
    #' @param sustain_track `r get_param_doc_desc("sustain_track")`
    #' @param sustain_iso_code `r get_param_doc_desc("sustain_iso_code")`
    #' @param sustain_region `r get_param_doc_desc("sustain_region")`
    #' @param sustain_interval `r get_param_doc_desc("sustain_interval")`
    #' @param sustain_log_level `r get_param_doc_desc("sustain_log_level")`
    #' @param trace `r get_param_doc_desc("trace")`
    #' @param pytorch_trace `r get_param_doc_desc("pytorch_trace")`
    #' @param log_dir `r get_param_doc_desc("log_dir")`
    #' @param log_write_interval `r get_param_doc_desc("log_write_interval")`
    #' @return `r get_description("return_nothing")`
    train = function(text_dataset,
                     p_mask = 0.15,
                     whole_word = TRUE,
                     val_size = 0.1,
                     n_epoch = 1L,
                     batch_size = 12L,
                     max_sequence_length = 250L,
                     full_sequences_only = FALSE,
                     min_seq_len = 50L,
                     learning_rate = 3e-3,
                     sustain_track = FALSE,
                     sustain_iso_code = NULL,
                     sustain_region = NULL,
                     sustain_interval = 15L,
                     sustain_log_level = "warning",
                     trace = TRUE,
                     pytorch_trace = 1L,
                     log_dir = NULL,
                     log_write_interval = 2L) {
      private$do_training(args = get_called_args(n = 1L))
    },
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
    },
    #--------------------------------------------------------------------------
    #' @description Method for requesting a plot of the training history.
    #' This method requires the *R* package 'ggplot2' to work.
    #' @param x_min `r get_param_doc_desc("x_min")`
    #' @param x_max `r get_param_doc_desc("x_max")`
    #' @param y_min `r get_param_doc_desc("y_min")`
    #' @param y_max `r get_param_doc_desc("y_max")`
    #' @param ind_best_model `r get_param_doc_desc("ind_best_model")`
    #' @param text_size `r get_param_doc_desc("text_size")`
    #' @return Returns a plot of class `ggplot` visualizing the training process.
    plot_training_history = function(x_min = NULL, x_max = NULL, y_min = NULL, y_max = NULL, ind_best_model = TRUE, text_size = 10L) {
      requireNamespace("ggplot2")
      plot_data <- self$last_training$history

      if (is.null_or_na(x_min)) {
        x_min <- 1L
      }
      if (is.null_or_na(x_max)) {
        x_max <- nrow(plot_data)
      }

      if (is.null_or_na(y_min)) {
        y_min <- min(plot_data[, c("loss", "val_loss")])
      }
      if (is.null_or_na(y_max)) {
        y_max <- max(plot_data[, c("loss", "val_loss")])
      }

      tmp_colnames <- c("epoch", "val_loss", "loss")
      cols_exist <- sum(tmp_colnames %in% colnames(plot_data)) == length(tmp_colnames)

      if (cols_exist) {
        tmp_plot <- ggplot2::ggplot(data = plot_data) +
          ggplot2::geom_line(ggplot2::aes(x = .data$epoch, y = .data$loss, color = "train")) +
          ggplot2::geom_line(ggplot2::aes(x = .data$epoch, y = .data$val_loss, color = "validation"))

        if (ind_best_model) {
          best_state_point <- get_best_state_point(
            plot_data = plot_data,
            measure = "loss"
          )
          tmp_plot <- add_point(
            plot_object = tmp_plot,
            x = best_state_point$epoch,
            y = best_state_point$value,
            type = "segment",
            state = "best"
          )
        }

        tmp_plot <- tmp_plot + ggplot2::theme_classic() +
          ggplot2::ylab("value") +
          ggplot2::xlab("epoch") +
          ggplot2::coord_cartesian(ylim = c(y_min, y_max), xlim = c(x_min, x_max)) +
          ggplot2::scale_color_manual(
            values = c(
              train = "red",
              validation = "blue",
              test = "darkgreen"
            )
          ) +
          ggplot2::theme(
            text = ggplot2::element_text(size = text_size),
            legend.position = "bottom"
          )

        if (ind_best_model) {
          tmp_plot <- tmp_plot + ggplot2::scale_linetype_manual(
            values = c(best = 5L, final = 3L)
          )
        }
        return(tmp_plot)
      } else {
        warning("Data for the training history is not available.")
        return(NULL)
      }
    },
    #--------------------------------------------------------------------------
    #' @description Method for receiving the special tokens of the model
    #' @return Returns a `matrix` containing the special tokens in the rows
    #' and their type, token, and id in the columns.
    get_special_tokens = function() {
      return(self$Tokenizer$get_special_tokens())
    },
    #---------------------------------------------------------------------------
    #' @description Tokenizer statistics
    #' @return Returns a `data.frame` containing the tokenizer's statistics.
    get_tokenizer_statistics = function() {
      return(self$Tokenizer$get_tokenizer_statistics())
    },
    # Fill Mask------------------------------------------------------------------
    #' @description Method for calculating tokens behind mask tokens.
    #' @param masked_text `r get_param_doc_desc("masked_text")`
    #' @param n_solutions `r get_param_doc_desc("n_solutions")`
    #' @return Returns a `list` containing a `data.frame` for every
    #' mask. The `data.frame` contains the solutions in the rows and reports
    #' the score, token id, and token string in the columns.
    fill_mask = function(masked_text, n_solutions = 5L) {
      # Arugment checking
      check_type(object = masked_text, type = "string", FALSE)
      check_type(object = n_solutions, type = "int", FALSE)

      framework <- "pt"
      private$model$to("cpu")
      private$model$eval()

      if (private$model_type != "mpnet") {
        run_py_file("FillMaskForMPLM.py")
        fill_mask_pipeline_class <- py$FillMaskPipelineForMPLM
      } else {
        fill_mask_pipeline_class <- transformers$FillMaskPipeline
      }

      if (check_versions(a = get_py_package_version("transformers"), operator = ">=", b = "5.0.0")) {
        fill_mask_pipeline <- fill_mask_pipeline_class(
          model = private$model,
          tokenizer = self$Tokenizer$get_tokenizer(),
          num_workers = 1L,
          binary_output = FALSE,
          top_k = as.integer(n_solutions),
          tokenizer_kwargs = reticulate::dict(
            list(
              return_token_type_ids = private$return_token_type_ids,
              max_length = as.integer(private$model$config$max_position_embeddings - private$adjust_max_sequence_length),
              truncation = "longest_first"
            )
          )
        )
      } else {
        fill_mask_pipeline <- fill_mask_pipeline_class(
          model = private$model,
          tokenizer = self$Tokenizer$get_tokenizer(),
          framework = "pt",
          num_workers = 1L,
          binary_output = FALSE,
          top_k = as.integer(n_solutions),
          tokenizer_kwargs = reticulate::dict(
            list(
              return_token_type_ids = private$return_token_type_ids,
              max_length = as.integer(private$model$config$max_position_embeddings - private$adjust_max_sequence_length),
              truncation = "longest_first"
            )
          )
        )
      }


      special_tokens <- self$Tokenizer$get_special_tokens()
      mask_token <- special_tokens[special_tokens[, "type"] == "mask_token", "token"]

      n_mask_tokens <- ncol(stringi::stri_extract_all_fixed(
        str = masked_text,
        pattern = mask_token,
        simplify = TRUE
      ))

      if (n_mask_tokens == 0L) {
        stop("There is no masking token. Please check your input.")
      }

      solutions <- as.list(fill_mask_pipeline(masked_text))

      solutions_list <- NULL

      if (n_mask_tokens == 1L) {
        solution_data_frame <- matrix(
          nrow = length(solutions),
          ncol = 3L
        )
        colnames(solution_data_frame) <- c(
          "score",
          "token",
          "token_str"
        )
        for (i in seq_along(solutions)) {
          solution_data_frame[i, "score"] <- solutions[[i]]$score
          solution_data_frame[i, "token"] <- solutions[[i]]$token
          solution_data_frame[i, "token_str"] <- solutions[[i]]$token_str
        }
        solution_data_frame <- as.data.frame(solution_data_frame)
        solution_data_frame$score <- as.numeric(solution_data_frame$score)
        solutions_list[length(solutions_list) + 1L] <- list(solution_data_frame)
      } else {
        for (j in seq_along(solutions)) {
          solution_data_frame <- matrix(
            nrow = length(solutions[[j]]),
            ncol = 3L
          )
          colnames(solution_data_frame) <- c(
            "score",
            "token",
            "token_str"
          )
          for (i in seq_along(solutions[[j]])) {
            solution_data_frame[i, "score"] <- solutions[[j]][[i]]$score
            solution_data_frame[i, "token"] <- solutions[[j]][[i]]$token
            solution_data_frame[i, "token_str"] <- solutions[[j]][[i]]$token_str
          }
          solution_data_frame <- as.data.frame(solution_data_frame)
          solution_data_frame$score <- as.numeric(solution_data_frame$score)
          solutions_list[length(solutions_list) + 1L] <- list(solution_data_frame)
        }
      }

      return(solutions_list)
    },
    #--------------------------------------------------------------------------
    #' @description Method for saving a model on disk.
    #' @param dir_path `r get_description("save_dir")`
    #' @param folder_name `r get_param_doc_desc("folder_name")`
    #' @return `r get_description("return_save_on_disk")`
    save = function(dir_path, folder_name) {
      save_location <- file.path(dir_path, folder_name)
      create_dir(dir_path = save_location, trace = FALSE)

      # Save BaseModel
      private$model$save_pretrained(
        save_directory = save_location,
        safe_serilization = TRUE
      )

      # Save Tokenizer
      private$save_tokenizer(dir_path = dir_path, folder_name = folder_name)

      # Save Sustainability Data
      private$save_sustainability_data(dir_path = dir_path, folder_name = folder_name)

      # Save Sustainability Data Inference
      private$save_sustainability_data_inference(dir_path = dir_path, folder_name = folder_name)

      # Save training history
      private$save_training_history(dir_path = dir_path, folder_name = folder_name)

      # Save Flops Estimates
      private$save_flops_estimates(dir_path = dir_path, folder_name = folder_name)
    },
    #--------------------------------------------------------------------------
    #' @description Loads an object from disk
    #' and updates the object to the current version of the package.
    #' @param dir_path `r get_description("load_dir")`
    #' @return `r get_description("return_load_on_disk")`
    load_from_disk = function(dir_path) {
      # Load private and public config files
      private$load_config_file(dir_path)

      # Load or reload python scripts
      private$load_reload_python_scripts()

      # Load BaseModel
      private$load_BaseModel(dir_path = dir_path)

      # Load Tokenizer
      private$load_tokenizer(dir_path = dir_path)

      # Load Sustainability Data
      private$load_sustainability_data(model_dir = dir_path)

      # Load Sustainability Data Inference
      private$load_sustainability_data_inference(model_dir = dir_path)

      # Load training history
      private$load_training_history(model_dir = dir_path)

      # Load Flops Estimates
      private$load_flops_estimates(model_dir = dir_path)

      # Set configured to TRUE
      private$set_configuration_to_TRUE()
    },
    #--------------------------------------------------------------------------
    #' @description Get 'PyTorch' model
    #' @return Returns the underlying 'PyTorch' model.
    get_model = function() {
      return(private$model)
    },
    #--------------------------------------------------------------------------
    #' @description Type of the underlying model.
    #' @return Returns a `string` describing the model's architecture.
    get_model_type = function() {
      return(private$model_type)
    },
    #--------------------------------------------------------------------------
    #' @description Size of the final layer.
    #' @return Returns an `int` describing the number of dimensions of the last
    #' hidden layer.
    get_final_size = function() {
      return(private$model$config$hidden_size)
    },
    #--------------------------------------------------------------------------
    #' @description Number of layers.
    #' @return Returns an `int` describing the number of layers available for
    #' embedding.
    get_n_layers = function() {
      return(private$model$config$num_hidden_layers)
    },
    #--------------------------------------------------------------------------
    #' @description Flop estimates
    #' @return Returns a `data.frame` containing statistics about the flops.
    get_flops_estimates = function() {
      return(private$flops_estimates)
    },
    #--------------------------------------------------------------------------
    #' @description Method for setting the bibliographic information of the model.
    #' @param type `string` Type of information which should be changed/added.
    #' `developer`, and `modifier` are possible.
    #' @param authors List of people.
    #' @param citation `string` Citation in free text.
    #' @param url `string` Corresponding URL if applicable.
    #' @return Function does not return a value. It is used to set the private
    #' members for publication information of the model.
    set_publication_info = function(type,
                                    authors,
                                    citation,
                                    url = NULL) {
      if (type == "developer") {
        private$publication_info$developed_by$authors <- authors
        private$publication_info$developed_by$citation <- citation
        private$publication_info$developed_by$url <- url
      } else if (type == "modifier") {
        private$publication_info$modified_by$authors <- authors
        private$publication_info$modified_by$citation <- citation
        private$publication_info$modified_by$url <- url
      }
    },
    #--------------------------------------------------------------------------
    #' @description Calculates the energy consumption for inference of the given task.
    #' @param text_dataset `r get_param_doc_desc("text_dataset")`
    #' @param n_samples `r get_param_doc_desc("n_samples")`
    #' @param sustain_iso_code `r get_param_doc_desc("sustain_iso_code")`
    #' @param sustain_region `r get_param_doc_desc("sustain_region")`
    #' @param sustain_interval `r get_param_doc_desc("sustain_interval")`
    #' @param sustain_log_level `r get_param_doc_desc("sustain_log_level")`
    #' @param trace `r get_param_doc_desc("trace")`
    #' @return Returns nothing. Method saves the statistics internally.
    #' The statistics can be accessed with the method `get_sustainability_data("inference")`
    estimate_sustainability_inference_fill_mask = function(text_dataset = NULL,
                                                           n_samples = NULL,
                                                           sustain_iso_code = NULL,
                                                           sustain_region = NULL,
                                                           sustain_interval = 15L,
                                                           sustain_log_level = "warning",
                                                           trace = TRUE) {
      # Prepare Data
      print_message(
        msg = "Prepare Data",
        trace = trace
      )

      n_cases <- text_dataset$n_rows()
      sample_size <- min(n_cases, n_samples)
      random_sample <- sample(
        x = seq.int(from = 1L, to = n_cases),
        size = sample_size,
        replace = FALSE
      ) - 1L

      # Prepare Data
      print_message(
        msg = "Add Masking Token",
        trace = trace
      )
      mask_token <- self$Tokenizer$get_special_tokens()["mask_token", "token"]

      selected_data <- text_dataset$select(random_sample)
      selected_texts <- extract_column_from_py_dataset(selected_data, "text")
      selected_texts_with_mask <- paste(mask_token, selected_texts)

      # Start Tracking
      private$init_and_start_sustainability_tracker(
        trace = trace,
        country_iso_code = sustain_iso_code,
        region = sustain_region,
        measure_power_secs = sustain_interval,
        sustain_log_level = sustain_log_level
      )

      for (i in 1L:sample_size) {
        self$fill_mask(masked_text = selected_texts_with_mask[i], n_solutions = 1L)
      }

      # Stop Tracking
      results <- private$stop_sustainability_tracker(
        trace = trace,
        task = "FillMask"
      )

      # Add additional information
      results$data <- "empirical data"
      results$n <- sample_size
      results$batch <- 1L
      results$min_seq_len <- NA
      results$mean_seq_len <- NA
      results$sd_seq_len <- NA
      results$max_seq_len <- NA

      if (is.null_or_na(private$sustainability_inference)) {
        private$sustainability_inference <- results
      } else {
        private$sustainability_inference <- rbind(
          private$sustainability_inference,
          results
        )
      }
    },
    #--------------------------------------------------------------------------
    #' @description Calculates FLOPS based on model's architecture.
    #' @param batch_size `r get_param_doc_desc("batch_size")`
    #' @param n_batches `r get_param_doc_desc("n_batches")`
    #' @param n_epoch `r get_param_doc_desc("n_epoch")`
    #' @return Returns a `data.frame` storing the estimates.
    calc_flops_architecture_based = function(batch_size, n_batches, n_epoch) {
      tokenizer <- self$Tokenizer$get_tokenizer()
      max_seq_len <- self$get_model_config()$max_position_embeddings

      # Tokens without special tokens
      possible_tokens <- setdiff(
        x = names(self$Tokenizer$get_tokenizer()$get_vocab()),
        y = self$get_special_tokens()[, "token"]
      )

      generated_texts <- vector(length = batch_size)
      for (i in seq_along(generated_texts)) {
        generated_texts[i] <- paste(sample(
          x = possible_tokens,
          size = max_seq_len,
          replace = TRUE
        ), collapse = " ")
      }

      res_colnames <- private$columnes_flops_estimates()
      results <- matrix(
        nrow = 1L,
        ncol = length(res_colnames)
      )
      colnames(results) <- res_colnames
      results <- as.data.frame(results)

      bp_factors <- c(1.0, 2.0, 3.0, 4.0)

      tokenized_texts <- tokenizer(
        text = generated_texts,
        truncation = TRUE,
        max_length = as.integer(max_seq_len - private$adjust_max_sequence_length),
        return_tensors = "pt",
        return_token_type_ids = private$return_token_type_ids,
        padding = TRUE
      )


      for (bp_factor in bp_factors) {
        est_flops <- calflops$calculate_flops(
          model = self$get_model(),
          input_shape = NULL,
          transformer_tokenizer = NULL,
          # args=[],
          kwargs = tokenized_texts,
          forward_mode = "forward",
          include_backPropagation = TRUE,
          compute_bp_factor = bp_factor,
          print_results = FALSE,
          print_detailed = FALSE,
          output_as_string = FALSE,
          output_precision = 2.0,
          output_unit = NULL,
          ignore_modules = NULL
        )

        results[1L, "n_parameter"] <- est_flops[[3L]]
        results[1L, "batch_size"] <- batch_size
        results[1L, paste0("flops_bp_", bp_factor)] <- est_flops[[1L]] * n_batches * n_epoch
      }
      results[1L, "approach"] <- "architecture-based"

      results[1L, "n_batches"] <- n_batches
      results[1L, "n_epochs"] <- n_epoch

      results[1L, "package"] <- "calflops"
      if (is_venv()) {
        results[1L, "version"] <- get_py_package_version("calflops")
      } else {
        results[1L, "version"] <- NA
      }

      results[1L, "date"] <- get_time_stamp()

      return(results)
    }
  )
)
