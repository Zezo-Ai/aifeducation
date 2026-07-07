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


#' @title Base class for models using neural nets
#' @description Abstract class for all models that do not rely on the python library 'transformers'.
#' All models of this class require text embeddings as input. These are provided as
#' objects of class [EmbeddedText] or [LargeDataSetForTextEmbeddings].
#'
#' Objects of this class containing fields and methods used in several other classes in 'AI for Education'.
#'
#' This class is **not** designed for a direct application and should only be used by developers.
#'
#' @return A new object of this class.
#' @family R6 Classes for Developers
#' @export
ModelsBasedOnTextEmbeddings <- R6::R6Class(
  classname = "ModelsBasedOnTextEmbeddings",
  inherit = AIFEBaseModel,
  public = list(
    #--------------------------------------------------------------------------
    #' @description Method for requesting the text embedding model information.
    #' @return `list` of all relevant model information on the text embedding model underlying the model.
    get_text_embedding_model = function() {
      return(private$text_embedding_model)
    },
    #--------------------------------------------------------------------------
    #' @description Method for requesting the name (unique id) of the underlying text embedding model.
    #' @return Returns a `string` describing name of the text embedding model.
    get_text_embedding_model_name = function() {
      return(private$text_embedding_model$model$model_name)
    },
    #--------------------------------------------------------------------------
    # Check Embedding Model compatibility of the text embedding
    #' @description Method for checking if the provided text embeddings are created with the same [TextEmbeddingModel]
    #'   as the model.
    #' @param text_embeddings Object of class [EmbeddedText] or [LargeDataSetForTextEmbeddings].
    #' @return `TRUE` if the underlying [TextEmbeddingModel] are the same. `FALSE` if the models differ.
    check_embedding_model = function(text_embeddings) {
      # Check object type
      private$check_embeddings_object_type(text_embeddings, strict = TRUE)

      # Check original text embedding model
      embedding_model_config <- text_embeddings$get_model_info()
      check <- "model_name"

      if (
        !is.null_or_na(embedding_model_config[[check]]) &
          !is.null_or_na(private$text_embedding_model$model[[check]])
      ) {
        if (embedding_model_config[[check]] != private$text_embedding_model$model[[check]]) {
          stop("The TextEmbeddingModel that generated the data_embeddings is not
               the same as the TextEmbeddingModel when generating the classifier.")
        }
      }
    },
    #-------------------------------------------------------------------------
    #' @description Method for saving a model.
    #' @param dir_path `string` Path of the directory where the model should be saved.
    #' @param folder_name `string` Name of the folder that should be created within the directory.
    #' @return Function does not return a value. It saves the model to disk.
    #' @importFrom utils write.csv
    save = function(dir_path, folder_name) {
      # Save pytorch model
      private$save_pytorch_model(dir_path = dir_path, folder_name = folder_name)

      # Saving Sustainability Data
      private$save_sustainability_data(dir_path = dir_path, folder_name = folder_name)
    },
    #--------------------------------------------------------------------------
    #' @description loads an object from disk and updates the object to the current version of the package.
    #' @param dir_path Path where the object set is stored.
    #' @return Method does not return anything. It loads an object from disk.
    load_from_disk = function(dir_path) {
      # Set configuration state
      private$set_configuration_to_TRUE()

      # Load R file with configuration and other data
      config_file <- load_R_config_state(dir_path)

      # load information of the text embedding model
      private$load_config_and_docs_textembeddingmodel(
        config_public = config_file$public,
        config_private = config_file$private
      )

      # Call the core method which loads data common for all models.
      # These are documentations, licenses, model's name and label etc.
      private$load_config_file(dir_path)

      # private$load_base_config_and_docs_general(
      #  config_public = config_file$public,
      #  config_private = config_file$private
      # )

      # Check and update model_config
      # Call this method to add parameters that where added in later version
      # which are missing in the old model
      # private$update_model_config()

      # Check and update pad value if necessary
      private$update_pad_value()

      # Create and load AI model
      private$load_pytorch_model(dir_path = dir_path)

      # Load sustainability data
      private$load_sustainability_data(model_dir = dir_path)

      # Set training status
      private$trained <- config_file$private$trained
    },
    #--------------------------------------------------------------------------
    #' @description Method for requesting a plot of the training history.
    #' This method requires the *R* package 'ggplot2' to work.
    #' @param final_training `bool` If `FALSE` the values of the performance estimation are used. If `TRUE` only
    #' the epochs of the final training are used.
    #' @param add_min_max `r get_param_doc_desc("add_min_max")`
    #' @param pl_step `int` Number of the step during pseudo labeling to plot. Only relevant if the model was trained
    #' with active pseudo labeling.
    #' @param measure Measure to plot.
    #' @param x_min `r get_param_doc_desc("x_min")`
    #' @param x_max `r get_param_doc_desc("x_max")`
    #' @param y_min `r get_param_doc_desc("y_min")`
    #' @param y_max `r get_param_doc_desc("y_max")`
    #' @param ind_best_model `r get_param_doc_desc("ind_best_model")`
    #' @param ind_selected_model `r get_param_doc_desc("ind_selected_model")`
    #' @param text_size `r get_param_doc_desc("text_size")`
    #' @return Returns a plot of class `ggplot` visualizing the training process.
    plot_training_history = function(final_training = FALSE,
                                     pl_step = NULL,
                                     measure = "loss",
                                     ind_best_model = TRUE,
                                     ind_selected_model = TRUE,
                                     x_min = NULL,
                                     x_max = NULL,
                                     y_min = NULL,
                                     y_max = NULL,
                                     add_min_max = TRUE,
                                     text_size = 10L) {
      requireNamespace("ggplot2")
      data_prepared <- private$prepare_training_history(
        final = final_training,
        pl_step = pl_step
      )
      plot_data_all <- data_prepared$aggregated
      # Select the performance measure to display
      plot_data <- plot_data_all[[measure]]
      checkpoints <- data_prepared[["checkpoints"]]

      # Create Plot
      if (measure == "loss") {
        y_label <- "loss"
      } else if (measure == "accuracy") {
        y_label <- "Accuracy"
      } else if (measure == "balanced_accuracy") {
        y_label <- "Balanced Accuracy"
      } else if (measure == "avg_iota") {
        y_label <- "Average Iota"
      } else if (measure == "s_avg_iota") {
        y_label <- "Smoothed Average Iota"
      }

      # set x_min and x_max if they are NULL
      if (is.null_or_na(x_min)) {
        x_min <- 1L
      }
      if (is.null_or_na(x_max)) {
        x_max <- nrow(plot_data)
      }
      plot_data <- plot_data[x_min:x_max, ]

      # Set y_min and y_max if they are NULL
      data_colnames <- setdiff(x = colnames(plot_data), y = "epoch")
      if (!add_min_max) {
        data_colnames <- setdiff(
          x = data_colnames,
          y = c(
            "train_min",
            "train_max",
            "validation_min",
            "validation_max"
          )
        )
        if ("test_mean" %in% data_colnames) {
          data_colnames <- setdiff(
            x = data_colnames,
            y = c(
              "test_min",
              "test_max"
            )
          )
        }
      }
      if (is.null_or_na(y_min)) {
        y_min <- min(plot_data[, data_colnames], na.rm = TRUE)
      }
      if (is.null_or_na(y_max)) {
        y_max <- max(plot_data[, data_colnames], na.rm = TRUE)
      }

      tmp_plot <- ggplot2::ggplot(data = plot_data) +
        ggplot2::geom_line(ggplot2::aes(x = .data$epoch, y = .data$train_mean, color = "train")) +
        ggplot2::geom_line(ggplot2::aes(x = .data$epoch, y = .data$validation_mean, color = "validation"))

      if (add_min_max) {
        tmp_plot <- tmp_plot +
          ggplot2::geom_line(ggplot2::aes(x = .data$epoch, y = .data$train_min, color = "train")) +
          ggplot2::geom_line(ggplot2::aes(x = .data$epoch, y = .data$train_max, color = "train")) +
          ggplot2::geom_ribbon(
            ggplot2::aes(
              x = .data$epoch,
              ymin = .data$train_min,
              ymax = .data$train_max
            ),
            alpha = 0.25,
            fill = "red"
          ) +
          ggplot2::geom_line(ggplot2::aes(x = .data$epoch, y = .data$validation_min, color = "validation")) +
          ggplot2::geom_line(ggplot2::aes(x = .data$epoch, y = .data$validation_max, color = "validation")) +
          ggplot2::geom_ribbon(
            ggplot2::aes(
              x = .data$epoch,
              ymin = .data$validation_min,
              ymax = .data$validation_max
            ),
            alpha = 0.25,
            fill = "blue"
          )
      }

      if ("test_mean" %in% colnames(plot_data)) {
        tmp_plot <- tmp_plot +
          ggplot2::geom_line(ggplot2::aes(x = .data$epoch, y = .data$test_mean, color = "test"))
        if (add_min_max) {
          tmp_plot <- tmp_plot +
            ggplot2::geom_line(ggplot2::aes(x = .data$epoch, y = .data$test_min, color = "test")) +

            ggplot2::geom_line(ggplot2::aes(x = .data$epoch, y = .data$test_max, color = "test")) +
            ggplot2::geom_ribbon(
              ggplot2::aes(
                x = .data$epoch,
                ymin = .data$test_min,
                ymax = .data$test_max
              ),
              alpha = 0.25,
              fill = "darkgreen"
            )
        }
      }

      if (final_training) {
        if (ind_best_model) {
          best_state_point <- get_best_state_point(
            plot_data = plot_data,
            measure = measure
          )
          tmp_plot <- add_point(
            plot_object = tmp_plot,
            x = best_state_point$epoch,
            y = best_state_point$value,
            type = "segment",
            state = "best"
          )
        } else {
          best_state_point <- list(
            epoch = NULL,
            value = NULL
          )
        }

        if (ind_selected_model) {
          selected_state_point <- get_used_state_point(
            plot_data = plot_data_all,
            measure = measure,
            checkpoints = checkpoints
          )
          tmp_plot <- add_point(
            plot_object = tmp_plot,
            x = selected_state_point$epoch,
            y = selected_state_point$value,
            type = "segment",
            state = "final"
          )
        } else {
          selected_state_point <- list(
            epoch = NULL,
            value = NULL
          )
        }

        if (final_training || ind_selected_model) {
          tmp_plot <- add_breaks(
            plot_object = tmp_plot,
            x_min = x_min,
            x_max = x_max,
            y_min = y_min,
            y_max = y_max,
            special_x = c(selected_state_point$epoch, best_state_point$epoch),
            special_y = c(selected_state_point$value, best_state_point$value)
          )
        }
      } else {
        if (ind_best_model) {
          best_states <- get_best_states_from_folds(
            data_folds = data_prepared$folds,
            measure = measure
          )
          tmp_plot <- add_point(
            plot_object = tmp_plot,
            x = best_states$epochs,
            y = best_states$values,
            type = "point",
            state = "best"
          )
        }
        if (ind_selected_model) {
          selected_states <- get_selected_states_from_folds(
            data_folds = data_prepared$folds,
            measure = measure,
            checkpoints = checkpoints
          )
          tmp_plot <- add_point(
            plot_object = tmp_plot,
            x = selected_states$epochs,
            y = selected_states$values,
            type = "point",
            state = "final"
          )
        }
      }

      tmp_plot <- tmp_plot + ggplot2::theme_classic() +
        ggplot2::ylab(y_label) +
        ggplot2::xlab("epoch") +
        ggplot2::coord_cartesian(ylim = c(y_min, y_max), xlim = c(x_min, x_max)) +
        ggplot2::scale_color_manual(
          values = c(
            train = "red",
            validation = "blue",
            test = "darkgreen"
          )
        ) + ggplot2::theme(
          text = ggplot2::element_text(size = text_size),
          legend.position = "bottom"
        )

      if (ind_best_model || ind_selected_model) {
        if (final_training) {
          tmp_plot <- tmp_plot + ggplot2::scale_linetype_manual(
            values = c(best = 5L, final = 3L)
          )
        } else {
          tmp_plot <- tmp_plot + ggplot2::scale_shape_manual(
            values = c(best = 16L, final = 15L)
          )
        }
      }

      return(tmp_plot)
    },
    #--------------------------------------------------------------------------
    #' @description Method for requesting a plot that shows the analysis for selecting
    #' a good learning rate.
    #' @return Returns a plot of class `ggplot` visualizing the training process.
    plot_learning_rate = function() {
      plot_data <- private$lr_statistics
      if (is.null_or_na(plot_data)) {
        stop("Calculation of learning rates was not requested during training.")
      } else {
        if (nrow(plot_data) <= 0L) {
          stop("Calculation of learning rates was not requested during training.")
        } else {
          plot_data$lr_rate <- as.factor(plot_data$lr_rate)
          # plot_data$rel_improvment=(plot_data$delta)/plot_data$start_loss
          relevant_range <- subset(
            x = plot_data,
            subset = plot_data$best_range
          )
          selected <- which(plot_data$selected)
          if (length(selected) <= 1L) {
            selected <- c(selected, selected)
          }

          # Create plot
          tmp_plot <- ggplot2::ggplot(data = plot_data) +
            ggplot2::geom_bar(
              stat = "identity",
              ggplot2::aes(
                x = lr_rate,
                y = delta
              )
            ) +
            ggplot2::geom_point(
              ggplot2::aes(
                x = lr_rate,
                y = smoothed_delta
              )
            ) +
            ggplot2::xlab("Learning Rate") +
            ggplot2::ylab("Change in %") +
            ggplot2::geom_vline(xintercept = plot_data$lr_rate[selected[1L]]) +
            ggplot2::geom_vline(xintercept = plot_data$lr_rate[selected[2L]]) +
            ggplot2::coord_flip() +
            ggplot2::theme_classic()
          return(tmp_plot)
        }
      }
    },
    #----------------------------------------------------------------------------
    #' @description Method for requesting the learning rate statistics.
    #' @return Returns a `data.frame` with the following columns
    #' * index: Index of the learning rate
    #' * lr_rate: Learning rate as `factor`.
    #' * start_loss: Loss before any training.
    #' * final_loss: Loss after the complete training.
    #' * delta: Normalized loss as (final_loss-start_loss)/start_loss.
    #' * smoothed_delta: Smoothed normalized loss.
    #' * best_range: `bool` indicating if the learning rate is part of the best range.
    #' * selected: `bool` indicating if the learning rate is selected for further training.
    #'
    get_lr_statistics = function() {
      return(private$lr_statistics)
    }
  ),
  private = list(
    text_embedding_model = list(
      model = list(),
      times = NA,
      features = NA
    ),
    #------------------------------------------------------------------------------
    load_config_and_docs_textembeddingmodel = function(config_public, config_private) {
      if (is.null(config_private$text_embedding_model$pad_value)) {
        pad_value <- 0L
      } else {
        pad_value <- config_private$text_embedding_model$pad_value
      }

      private$set_text_embedding_model(
        model_info = config_private$text_embedding_model$model,
        feature_extractor_info = config_private$text_embedding_model$feature_extractor,
        times = config_private$text_embedding_model$times,
        features = config_private$text_embedding_model$features,
        pad_value = pad_value
      )
    },
    #---------------------------------------------------------------------------
    check_embeddings_object_type = function(embeddings, strict = TRUE) {
      if (strict) {
        if (
          !(inherits(embeddings, "EmbeddedText")) &
            !(inherits(embeddings, "LargeDataSetForTextEmbeddings"))
        ) {
          stop("text_embeddings must be of class EmbeddedText or LargeDataSetForTextEmbeddings.")
        }
      } else {
        if (
          !(inherits(embeddings, "EmbeddedText")) &
            !(inherits(embeddings, "LargeDataSetForTextEmbeddings")) &
            !(inherits(embeddings, "array")) &
            !(inherits(embeddings, "datasets.arrow_dataset.Dataset"))
        ) {
          stop("text_embeddings must be of class EmbeddedText, LargeDataSetForTextEmbeddings,
               datasets.arrow_dataset.Dataset or array.")
        }
      }
    },
    #-------------------------------------------------------------------------
    check_single_prediction = function(embeddings) {
      if (
        inherits(embeddings, "EmbeddedText") |
          inherits(embeddings, "LargeDataSetForTextEmbeddings")
      ) {
        if (embeddings$n_rows() > 1L) {
          single_prediction <- FALSE
        } else {
          single_prediction <- TRUE
        }
      } else if (inherits(embeddings, "array")) {
        if (nrow(embeddings) > 1L) {
          single_prediction <- FALSE
        } else {
          single_prediction <- TRUE
        }
      } else if (inherits(embeddings, "datasets.arrow_dataset.Dataset")) {
        single_prediction <- FALSE
      }
      return(single_prediction)
    },
    #--------------------------------------------------------------------------
    prepare_embeddings_as_dataset = function(embeddings) {
      if (inherits(embeddings, "datasets.arrow_dataset.Dataset")) {
        prepared_dataset <- embeddings
      } else if (inherits(embeddings, "EmbeddedText")) {
        prepared_dataset <- datasets$Dataset$from_dict(
          reticulate::dict(
            list(
              id = rownames(embeddings$embeddings),
              input = np$squeeze(
                np$split(
                  reticulate::np_array(embeddings$embeddings),
                  as.integer(nrow(embeddings$embeddings)),
                  axis = 0L
                )
              )
            ),
            convert = FALSE
          )
        )
      } else if (inherits(embeddings, "array")) {
        prepared_dataset <- datasets$Dataset$from_dict(
          reticulate::dict(
            list(
              id = rownames(embeddings),
              input = np$squeeze(np$split(reticulate::np_array(embeddings), as.integer(nrow(embeddings)), axis = 0L))
            ),
            convert = FALSE
          )
        )
      } else if (inherits(embeddings, "LargeDataSetForTextEmbeddings")) {
        prepared_dataset <- embeddings$get_dataset()
      }
      return(prepared_dataset)
    },
    #-------------------------------------------------------------------------
    prepare_embeddings_as_np_array = function(embeddings) {
      if (inherits(embeddings, "EmbeddedText")) {
        prepared_dataset <- embeddings$embeddings
        tmp_np_array <- np$array(prepared_dataset)
      } else if (inherits(embeddings, "array")) {
        prepared_dataset <- embeddings
        tmp_np_array <- np$array(prepared_dataset)
      } else if (inherits(embeddings, "datasets.arrow_dataset.Dataset")) {
        # prepared_dataset <- embeddings$set_format("np")
        tmp_np_array <- extract_column_from_py_dataset(embeddings, "input")
      } else if (inherits(embeddings, "LargeDataSetForTextEmbeddings")) {
        # prepared_dataset <- embeddings$get_dataset()
        # prepared_dataset$set_format("np")
        tmp_np_array <- extract_column_from_py_dataset(embeddings$get_dataset(), "input")
      }
      tmp_np_array <- reticulate::np_array(tmp_np_array)
      if (!numpy_writeable(tmp_np_array)) {
        warning("Numpy array is not writable")
      }
      return(tmp_np_array)
    },
    #--------------------------------------------------------------------------
    get_rownames_from_embeddings = function(embeddings) {
      if (inherits(embeddings, "EmbeddedText")) {
        return(rownames(embeddings$embeddings))
      } else if (inherits(embeddings, "array")) {
        return(rownames(embeddings))
      } else if (inherits(embeddings, "datasets.arrow_dataset.Dataset")) {
        return(extract_column_from_py_dataset(embeddings, "id"))
      } else if (inherits(embeddings, "LargeDataSetForTextEmbeddings")) {
        embeddings$get_ids()
      }
    },
    #--------------------------------------------------------------------------
    set_text_embedding_model = function(model_info,
                                        feature_extractor_info,
                                        times,
                                        features,
                                        pad_value) {
      private$text_embedding_model["model"] <- list(model_info)
      private$text_embedding_model["feature_extractor"] <- feature_extractor_info
      private$text_embedding_model["times"] <- times
      private$text_embedding_model["features"] <- features
      private$text_embedding_model["pad_value"] <- pad_value
    },
    set_up_logger = function(log_dir, log_write_interval) {
      private$log_config$log_dir <- log_dir
      private$log_config$log_state_file <- file.path(private$log_config$log_dir, "aifeducation_state.log")
      private$log_config$log_write_interval <- log_write_interval
    },
    #-------------------------------------------------------------------------
    # This Method updates the model config in the case that new parameters have been
    # introduced
    update_model_config = function() {
      # Check if an update is necessary
      current_pkg_version <- self$get_package_versions()$r_package_versions$aifeducation
      if (is.null_or_na(current_pkg_version)) {
        need_update <- TRUE
      } else {
        if (check_versions(
          a = packageVersion("aifeducation"),
          operator = ">",
          b = self$get_package_versions()$r_package_versions$aifeducation
        )) {
          need_update <- TRUE
        } else {
          need_update <- FALSE
        }
      }

      # check if an update of values is necessary. This is the case if the model
      # was created with an older version of aifeducation compared to 1.1.0
      # Update values to the new values introduced in version 1.1.0
      if (is.null_or_na(current_pkg_version)) {
        update_values <- TRUE
      } else {
        if (check_versions(
          a = "1.1.0",
          operator = ">",
          b = self$get_package_versions()$r_package_versions$aifeducation
        )) {
          update_values <- TRUE
        } else {
          update_values <- FALSE
        }
      }

      if (need_update) {
        param_dict <- get_param_dict()
        if (is.function(self$configure)) {
          param_names_new <- rlang::fn_fmls_names(self$configure)
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
            if (update_values) {
              private$model_config[param] <- list(update_values_to_new_1.1.0(private$model_config[[param]]))
            }
          }

          # Update Package version for the model
          private$r_package_versions$aifeducation <- packageVersion("aifeducation")
        } else {
          warning("Class does not have a method `configure`.")
        }
      }
    },
    #-------------------------------------------------------------------------
    update_pad_value = function() {
      current_pkg_version <- self$get_package_versions()$r_package_versions$aifeducation
      if (is.na(current_pkg_version)) {
        need_update <- TRUE
      } else {
        if (check_versions(
          a = packageVersion("aifeducation"),
          operator = ">",
          b = self$get_package_versions()$r_package_versions$aifeducation
        )) {
          need_update <- TRUE
        } else {
          need_update <- FALSE
        }
      }

      if (need_update) {
        if (is.null_or_na(private$text_embedding_model[["pad_value"]])) {
          private$text_embedding_model["pad_value"] <- 0L
        }
      }
    },
    #--------------------------------------------------------------------------
    generate_model_id = function(name) {
      if (is.null(name)) {
        return(paste0("mbote_", generate_id(16L)))
      } else {
        return(name)
      }
    },
    #--------------------------------------------------------------------------
    #'  Prepare history data of objects
    #'  Function for preparing the history data of a model in order to be plotted in AI for Education - Studio.
    #'
    #'  final `bool` If `TRUE` the history data of the final training is used for the data set.
    #'  pl_step `int` If `use_pl=TRUE` select the step within pseudo labeling for which the data should be prepared.
    #'  Returns a named `list` with the training history data of the model. The
    #' reported measures depend on the provided model.
    #'
    #'  Utils Studio Developers
    #'  internal
    prepare_training_history = function(final = FALSE,
                                        pl_step = NULL) {
      plot_data <- self$last_training$history

      # if (length(plot_data) <= 1L) {
      #  plot_data[[1L]] <- list(loss = plot_data[[1L]])
      # }

      if (is.null_or_na(final)) {
        final <- FALSE
      }

      if (is.null_or_na(self$last_training$config$use_pl)) {
        use_pl <- FALSE
      } else {
        use_pl <- self$last_training$config$use_pl
      }

      if (use_pl & is.null_or_na(pl_step)) {
        stop("Model was trained with pseudo labeling. Please provide a pl_step.")
      }

      # Get standard statistics
      n_epochs <- self$last_training$config$epochs
      index_final <- length(self$last_training$history)

      # Get information about the existence of a training, validation, and test data set
      # Get Number of folds for the request
      if (!final) {
        n_folds <- length(self$last_training$history)
        if (n_folds > 1L) {
          n_folds <- n_folds - 1L
        }

        if (!use_pl) {
          measures <- names(plot_data[[1L]])
          n_sample_type <- nrow(plot_data[[1L]][[measures[1L]]])
          checkpoints <- list()
          for (i in seq.int(n_folds)) {
            checkpoints[i] <- list(
              plot_data[[i]][["checkpoints"]]
            )
          }
        } else {
          measures <- names(plot_data[[1L]][[1L]])
          n_sample_type <- nrow(plot_data[[1L]][[as.numeric(pl_step)]][[measures[1L]]])
          checkpoints <- list()
          for (i in seq.int(n_folds)) {
            checkpoints[i] <- list(
              plot_data[[i]][[as.numeric(pl_step)]][["checkpoints"]]
            )
          }
        }
      } else {
        n_folds <- 1L
        if (!use_pl) {
          measures <- names(plot_data[[index_final]])
          n_sample_type <- nrow(plot_data[[index_final]][[measures[1L]]])
          checkpoints <- plot_data[[index_final]][["checkpoints"]]
        } else {
          measures <- names(plot_data[[index_final]][[1L]])
          n_sample_type <- nrow(plot_data[[index_final]][[as.numeric(pl_step)]][[measures[1L]]])
          checkpoints <- plot_data[[index_final]][[as.numeric(pl_step)]][["checkpoints"]]
        }
      }

      if (n_sample_type == 3L) {
        sample_type_name <- c("train", "validation", "test")
      } else {
        sample_type_name <- c("train", "validation")
      }

      # Create array for saving the data-------------------------------------------
      result_list <- NULL
      results_folds <- NULL
      for (j in seq_along(measures)) {
        measure <- measures[j]
        measure_array <- array(
          dim = c(
            n_folds,
            n_sample_type,
            n_epochs
          ),
          dimnames = list(fold = NULL, sample_type = sample_type_name, epoch = NULL)
        )

        final_data_measure <- matrix(
          data = NA,
          nrow = n_epochs,
          ncol = 3L * n_sample_type + 1L
        )
        colnames(final_data_measure) <- c(
          "epoch",
          paste0(
            sample_type_name,
            c(
              rep("_min", times = n_sample_type),
              rep("_mean", times = n_sample_type),
              rep("_max", times = n_sample_type)
            )
          )
        )
        final_data_measure[, "epoch"] <- seq.int(from = 1L, to = n_epochs)

        if (!final) {
          n_matrix_folds <- n_folds
        } else {
          n_matrix_folds <- 1L
        }

        fold_values_train <- matrix(
          data = NA,
          nrow = n_epochs,
          ncol = n_matrix_folds
        )
        fold_values_val <- fold_values_train
        fold_values_test <- fold_values_train

        if (!final) {
          for (i in 1L:n_folds) {
            if (!use_pl) {
              measure_array[i, , ] <- plot_data[[i]][[measure]]
            } else {
              measure_array[i, , ] <- plot_data[[i]][[as.numeric(pl_step)]][[measure]]
            }
          }
        } else {
          if (!use_pl) {
            measure_array[1L, , ] <- plot_data[[index_final]][[measure]]
          } else {
            measure_array[1L, , ] <- plot_data[[index_final]][[as.numeric(pl_step)]][[measure]]
          }
        }

        for (i in 1L:n_epochs) {
          final_data_measure[i, "train_min"] <- min(measure_array[, "train", i])
          final_data_measure[i, "train_mean"] <- mean(measure_array[, "train", i])
          final_data_measure[i, "train_max"] <- max(measure_array[, "train", i])

          final_data_measure[i, "validation_min"] <- min(measure_array[, "validation", i])
          final_data_measure[i, "validation_mean"] <- mean(measure_array[, "validation", i])
          final_data_measure[i, "validation_max"] <- max(measure_array[, "validation", i])

          if (n_sample_type == 3L) {
            final_data_measure[i, "test_min"] <- min(measure_array[, "test", i])
            final_data_measure[i, "test_mean"] <- mean(measure_array[, "test", i])
            final_data_measure[i, "test_max"] <- max(measure_array[, "test", i])
          }
          if (!final) {
            for (k in 1L:n_folds) {
              fold_values_train[i, k] <- measure_array[k, "train", i]
              fold_values_val[i, k] <- measure_array[k, "validation", i]
              if (n_sample_type == 3L) {
                fold_values_test[i, k] <- measure_array[k, "test", i]
              }
            }
          } else {
            fold_values_train[i, 1L] <- measure_array[, "train", i]
            fold_values_val[i, 1L] <- measure_array[, "validation", i]
          }
        }

        result_list[j] <- list(final_data_measure)
        results_folds[j] <- list(
          list(
            folds_train = fold_values_train,
            folds_val = fold_values_val,
            folds_test = fold_values_test
          )
        )
      }

      # Finalize data---------------------------------------------------------------
      names(result_list) <- measures
      names(results_folds) <- measures
      return(
        list(
          aggregated = result_list,
          folds = results_folds,
          checkpoints = checkpoints
        )
      )
    },
    #---------------------------------------------------------------------------
    save_pytorch_model = function(dir_path, folder_name) {
      save_location <- file.path(dir_path, folder_name)

      save_format <- "safetensors"

      if (save_format == "safetensors" & !reticulate::py_module_available("safetensors")) {
        warning("Python library 'safetensors' is not available. Using
                 standard save format for pytorch.")
        save_format <- "pt"
      }

      create_dir(save_location, FALSE)
      private$model$to("cpu", dtype = torch$float32)
      if (save_format == "safetensors") {
        file_path <- paste0(save_location, "/", "model_data", ".safetensors")
        safetensors$torch$save_model(model = private$model, filename = file_path)
      } else if (save_format == "pt") {
        file_path <- paste0(save_location, "/", "model_data", ".pt")
        torch$save(private$model$state_dict(), file_path)
      }
    },
    #---------------------------------------------------------------------------
    load_pytorch_model = function(dir_path) {
      # Load python scripts


      # Load the model---------------------------------------------------------
      path_pt <- paste0(dir_path, "/", "model_data", ".pt")
      path_safe_tensors <- paste0(dir_path, "/", "model_data", ".safetensors")
      private$init_model()
      private$model$to("cpu", dtype = torch$float32)

      if (file.exists(path_safe_tensors)) {
        safetensors$torch$load_model(
          model = private$model,
          filename = path_safe_tensors,
          device = "cpu"
        )
      } else {
        if (file.exists(paths = path_pt)) {
          private$model$load_state_dict(torch$load(path_pt))
        } else {
          stop("There is no compatible model file in the choosen directory.
                     Please check path. Please note that classifiers have to be loaded with
                     the same framework as during creation.")
        }
      }
    },
    #-------------------------------------------------------------------------
    calculate_learning_rate = function(data_manager) {
      if (self$last_training$config$lr_rate == 0.0 || self$last_training$config$lr_min == 0.0) {
        print_message(
          msg = "Estimating Learning Rates",
          trace = self$last_training$config$trace
        )
        if (is_on_CI()) {
          total_epochs <- 2L
        } else {
          total_epochs <- 20L
        }
        estimates <- private$estimate_learning_rates(
          data_manager,
          total_epochs = total_epochs
        )
        private$lr_statistics <- private$select_learning_rates(
          estimates,
          total_epochs = total_epochs
        )
      }
    },
    #--------------------------------------------------------------------------
    select_learning_rates = function(lr_estimation_results, total_epochs) {
      lr_estimation_results <- t(lr_estimation_results)
      lr_estimation_results <- lr_estimation_results[order(lr_estimation_results[, 1], decreasing = TRUE), ]
      colnames(lr_estimation_results) <- c("lr_rate", "n_improvments", "start_loss", "final_loss")
      lr_estimation_results <- as.data.frame(lr_estimation_results)
      lr_estimation_results <- subset(
        x = lr_estimation_results,
        subset = lr_estimation_results$lr_rate != 0.0
      )
      lr_estimation_results$index <- seq.int(from = 1L, to = nrow(lr_estimation_results))
      lr_estimation_results$selected <- FALSE
      # Smooth data distribution
      lr_estimation_results$delta <- (lr_estimation_results$final_loss - lr_estimation_results$start_loss) / lr_estimation_results$start_loss * 100
      smoothed_fct <- loess(
        formula= delta ~ index,
        data = lr_estimation_results,
        span = 0.85
        )
      lr_estimation_results$smoothed_delta <- predict(smoothed_fct)
      lr_estimation_results$smoothed_delta_imp <- lr_estimation_results$smoothed_delta < 0.0
      # Calculate gradient and calculate turning point
      gradient <- vector(length = nrow(lr_estimation_results))
      gradient[] <- NA
      for (i in nrow(lr_estimation_results):2L) {
        gradient[i - 1L] <- lr_estimation_results$smoothed_delta[i - 1L] - lr_estimation_results$smoothed_delta[i]
      }
      lr_estimation_results$gradient <- gradient

      turning_points <- vector(length = nrow(lr_estimation_results))
      for (i in (nrow(lr_estimation_results) - 1L):2L) {
        condition_one <- lr_estimation_results$gradient[i - 1L] <= lr_estimation_results$gradient[i]
        if (condition_one) {
          turning_points[i] <- TRUE
        } else {
          turning_points[i] <- FALSE
        }
      }
      lr_estimation_results$turning_points <- turning_points
      # identify stable range
      range_length <- vector(length = nrow(lr_estimation_results))
      for (i in 1:nrow(lr_estimation_results)) {
        continious <- TRUE
        counter <- 0
        for (j in i:nrow(lr_estimation_results)) {
          if (lr_estimation_results$smoothed_delta_imp[j] && continious) {
            counter <- counter + 1
          } else {
            continious <- FALSE
          }
        }
        range_length[i] <- counter
      }
      best_range <- vector(length = length(range_length))
      best_range[] <- FALSE
      start_idx <- which(range_length == max(range_length))
      range <- range_length[start_idx]
      best_range[start_idx:(start_idx + range - 1)] <- TRUE
      lr_estimation_results$best_range <- best_range
      relevant_range <- subset(
        x = lr_estimation_results,
        subset = (lr_estimation_results$best_range & lr_estimation_results$smoothed_delta_imp)
      )
      # Selecht the best values from the increasing side of the graph
      if (nrow(relevant_range) > 0L) {
        best_idx <- min(which(relevant_range$delta == min(relevant_range$delta)))
        relevant_range <- relevant_range[best_idx:nrow(relevant_range), ]

        relevant_range <- relevant_range[order(relevant_range$lr_rate, decreasing = TRUE), ]

        if (any(relevant_range$turning_points)) {
          best_idx <- min(which(relevant_range$turning_points))
        } else {
          best_idx <- min(which(relevant_range$delta == min(relevant_range$delta)))
        }

        best <- relevant_range$lr_rate[best_idx]

        lr_idx <- max(which(relevant_range$delta == max(relevant_range$delta)))
        lr <- relevant_range$lr_rate[lr_idx]

        if (best > lr) {
          self$last_training$config$lr_rate <- best
          self$last_training$config$lr_min <- lr
        } else {
          self$last_training$config$lr_rate <- lr
          self$last_training$config$lr_min <- best
        }

        lr_estimation_results$selected[which(lr_estimation_results$lr_rate == self$last_training$config$lr_rate)] <- TRUE
        lr_estimation_results$selected[which(lr_estimation_results$lr_rate == self$last_training$config$lr_min)] <- TRUE

        print_message(
          msg = paste0(
            "Set lr_rate to ",
            format(self$last_training$config$lr_rate, scientific = TRUE),
            " and lr_min to ",
            format(self$last_training$config$lr_min, scientific = TRUE),
            "."
          ),
          trace = self$last_training$config$trace
        )
      } else {
        print_message(
          msg = paste0(
            "No good learning rates could be identified. ",
            "Set lr_rate to ",
            format(self$last_training$config$lr_rate, scientific = TRUE),
            " and lr_min to ",
            format(self$last_training$config$lr_min, scientific = TRUE),
            "."
          ),
          trace = self$last_training$config$trace
        )
        self$last_training$config$lr_rate <- 1e-3
        self$last_training$config$lr_min <- 1e-4
      }
      # Reduce to relevant columns
      lr_estimation_results <- lr_estimation_results[c(
        "index",
        "lr_rate",
        "start_loss",
        "final_loss",
        "delta",
        "smoothed_delta",
        "best_range",
        "selected"
      )]
      return(lr_estimation_results)
    },
    #--------------------------------------------------------------------------
    save_init_weights = function() {
      torch$save(
        obj = private$model$state_dict(),
        f = file.path(private$dir_checkpoint, "init_weights.pt")
      )
    },
    #--------------------------------------------------------------------------
    load_init_weights = function() {
      private$model$load_state_dict(
        torch$load(
          file.path(private$dir_checkpoint, "init_weights.pt"),
          weights_only = TRUE
        )
      )
    }
  )
)
