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

#' @title Abstract class for all classifiers that use numerical representations of texts instead of words.
#' @description Base class for classifiers relying on [EmbeddedText] or [LargeDataSetForTextEmbeddings] generated with a [TextEmbeddingModel].
#'
#' Objects of this class containing fields and methods used in several other classes in 'AI for Education'.
#'
#' This class is **not** designed for a direct application and should only be used by developers.
#'
#' @return A new object of this class.
#' @family R6 Classes for Developers
#' @export
ClassifiersBasedOnTextEmbeddings <- R6::R6Class(
  classname = "ClassifiersBasedOnTextEmbeddings",
  inherit = ModelsBasedOnTextEmbeddings,
  public = list(
    #' @field feature_extractor ('list()')\cr
    #'   List for storing information and objects about the feature_extractor.
    feature_extractor = list(),

    #' @field reliability ('list()')\cr
    #'
    #'   List for storing central reliability measures of the last training.
    #'
    #'   * `reliability$test_metric`: Array containing the reliability measures for the test data for
    #'   every fold and step (in case of pseudo-labeling).
    #'   * `reliability$test_metric_mean`: Array containing the reliability measures for the test data.
    #'   The values represent the mean values for every fold.
    #'   * `reliability$raw_iota_objects`: List containing all iota_object generated with the package `iotarelr`
    #'   for every fold at the end of the last training for the test data.
    #'
    #'
    #'   * `reliability$raw_iota_objects$iota_objects_end`: List of objects with class `iotarelr_iota2` containing the
    #'   estimated iota reliability of the second generation for the final model for every fold for the test data.
    #'   *  `reliability$raw_iota_objects$iota_objects_end_free`: List of objects with class `iotarelr_iota2` containing
    #'   the estimated iota reliability of the second generation for the final model for every fold for the test data.
    #'   Please note that the model is estimated without forcing the Assignment Error Matrix to be in line with the
    #'   assumption of weak superiority.
    #'   * `reliability$iota_object_end`: Object of class `iotarelr_iota2` as a mean of the individual objects
    #'   for every fold for the test data.
    #'   * `reliability$iota_object_end_free`: Object of class `iotarelr_iota2` as a mean of the individual objects
    #'   for every fold. Please note that the model is estimated without forcing the Assignment Error Matrix to be in
    #'   line with the assumption of weak superiority.
    #'
    #'
    #'   * `reliability$standard_measures_end`: Object of class `list` containing the final measures for precision,
    #'   recall, and f1 for every fold.
    #'   * `reliability$standard_measures_mean`: `matrix` containing the mean measures for precision, recall, and f1.
    #'
    reliability = list(
      test_metric = NULL,
      test_metric_mean = NULL,
      raw_iota_objects = list(
        iota_objects_end = NULL,
        iota_objects_end_free = NULL
      ),
      iota_object_end = NULL,
      iota_object_end_free = NULL,
      standard_measures_end = NULL,
      standard_measures_mean = NULL
    ),
    #---------------------------------------------------------------------------
    #' @description Method for predicting new data with a trained neural net.
    #' @param newdata Object of class [TextEmbeddingModel] or [LargeDataSetForTextEmbeddings] for which predictions
    #'   should be made. In addition, this method allows to use objects of class `array` and
    #'   `datasets.arrow_dataset.Dataset`. However, these should be used only by developers.
    #' @param ml_trace `int` `ml_trace=0` does not print any information on the process from the machine learning
    #'   framework.
    #' @param batch_size `int` Size of batches.
    #' @return Returns a `data.frame` containing the predictions and the probabilities of the different labels for each
    #'   case.
    predict = function(newdata,
                       batch_size = 32L,
                       ml_trace = 1L) {
      # Check arguments
      check_type(object = batch_size, object_name = "batch_size", type = "int", FALSE)
      check_type(object = ml_trace, object_name = "ml_trace", type = "int", FALSE)

      # Check if the embeddings must be compressed before passing to the model
      requires_compression <- self$requires_compression(newdata)

      # Check input for compatible text embedding models and feature extractors
      if (
        inherits(newdata, "EmbeddedText") |
          inherits(newdata, "LargeDataSetForTextEmbeddings")
      ) {
        self$check_embedding_model(text_embeddings = newdata, require_compressed = FALSE)
      } else {
        private$check_embeddings_object_type(newdata, strict = FALSE)
        if (requires_compression) {
          stop("Objects of class datasets.arrow_dataset.Dataset must be provided in
               compressed form.")
        }
      }
      # Apply feature extractor if it is part of the model
      if (requires_compression) {
        if (inherits(newdata, "EmbeddedText")) {
          newdata <- newdata$convert_to_LargeDataSetForTextEmbeddings()
        } else {
          newdata <- newdata
        }

        # Returns a data set
        newdata <- self$feature_extractor$extract_features_large(
          data_embeddings = newdata,
          batch_size = as.integer(batch_size)
        )
      }

      # Load Custom Model Scripts
      private$load_reload_python_scripts()

      # Check number of cases in the data
      single_prediction <- private$check_single_prediction(newdata)

      # Get current row names/name of the cases
      current_row_names <- private$get_rownames_from_embeddings(newdata)

      # If at least two cases are part of the data set---------------------------
      if (!single_prediction) {
        # Returns a data set object
        prediction_data <- private$prepare_embeddings_as_dataset(newdata)

        prediction_data$set_format("torch")
        predictions_prob <- py$TeClassifierBatchPredict(
          model = private$model,
          dataset = prediction_data,
          batch_size = as.integer(batch_size)
        )
        predictions_prob <- tensor_to_numpy(predictions_prob)
        predictions <- max.col(predictions_prob) - 1L


        # In the case the data has one single row-------------------------------
      } else {
        prediction_data <- private$prepare_embeddings_as_np_array(newdata)

        if (torch$cuda$is_available()) {
          device <- "cuda"
          dtype <- torch$double
          private$model$to(device, dtype = dtype)
          private$model$eval()
          input <- torch$from_numpy(prediction_data)
          predictions_prob <- private$model(input$to(device, dtype = dtype),
            prediction_mode = TRUE
          )
          predictions_prob <- tensor_to_numpy(predictions_prob)
        } else {
          device <- "cpu"
          dtype <- torch$float
          private$model$to(device, dtype = dtype)
          private$model$eval()
          input <- torch$from_numpy(prediction_data)
          predictions_prob <- private$model(input$to(device, dtype = dtype),
            prediction_mode = TRUE
          )
          predictions_prob <- tensor_to_numpy(predictions_prob)
        }
        predictions <- max.col(predictions_prob) - 1L
      }

      # Transforming predictions to target levels------------------------------
      predictions <- as.character(as.vector(predictions))
      for (i in 0L:(length(private$model_config$target_levels) - 1L)) {
        predictions <- replace(
          x = predictions,
          predictions == as.character(i),
          values = private$model_config$target_levels[i + 1L]
        )
      }

      # Transforming to a factor
      predictions <- factor(predictions, levels = private$model_config$target_levels)

      colnames(predictions_prob) <- private$model_config$target_levels
      predictions_prob <- as.data.frame(predictions_prob)
      predictions_prob$expected_category <- predictions
      rownames(predictions_prob) <- current_row_names
      return(predictions_prob)
    },
    #---------------------------------------------------------------------------
    # Check Embedding Model compatibility of the text embedding
    #' @description Method for checking if the provided text embeddings are created with the same [TextEmbeddingModel]
    #'   as the classifier.
    #' @param text_embeddings Object of class [EmbeddedText] or [LargeDataSetForTextEmbeddings].
    #' @param require_compressed `TRUE` if a compressed version of the embeddings are necessary. Compressed embeddings
    #'   are created by an object of class [TEFeatureExtractor].
    #' @return `TRUE` if the underlying [TextEmbeddingModel] is the same. `FALSE` if the models differ.
    check_embedding_model = function(text_embeddings, require_compressed = FALSE) {
      # Check Embeddings Object Type
      private$check_embeddings_object_type(text_embeddings, strict = TRUE)

      # Check original text embedding model.
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

      # Check if a compressed version is necessary and if true if the feature extractor is
      # compatible
      feature_extractor_info <- text_embeddings$get_feature_extractor_info()
      if (require_compressed) {
        if (!is.null(feature_extractor_info$model_name) & !private$model_config$use_fe) {
          stop("Compressed embeddings provided but the classifier does not support
             compressed embeddings.")
        } else if (!is.null(feature_extractor_info$model_name) & private$model_config$use_fe) {
          if (private$text_embedding_model$feature_extractor$model_name != feature_extractor_info$model_name) {
            stop("The feature extractor of the compressed embeddings is not the same as
               the feature extractor during the creation of the classifier.")
          }
        }
      } else {
        if (!is.null(feature_extractor_info$model_name)) {
          stop("Compressed embeddings are provided but uncompressed are needed.")
        }
      }
    },
    #--------------------------------------------------------------------------
    #' @description Method for checking an object of class [TEFeatureExtractor].
    #' @param feature_extractor Object of class [TEFeatureExtractor]
    #' @return This method does nothing returns. It raises an error if
    #'
    #' * the object is `NULL`
    #' * the object does not rely on the same machine learning framework as the classifier
    #' * the object is not trained.
    #'
    check_feature_extractor_object_type = function(feature_extractor) {
      if (!is.null(feature_extractor)) {
        if (!inherits(feature_extractor, "TEFeatureExtractor")) {
          stop("Object passed to feature_extractor must be an object of class
               TEFeatureExtractor or NULL.")
        } else {
          if (!feature_extractor$is_trained()) {
            stop("The supplied feature extractor is not trained. Please
                provide trained feature extractor and try again.")
          }
        }
      }
    },
    #--------------------------------------------------------------------------
    #' @description Method for checking if provided text embeddings must be compressed via a [TEFeatureExtractor] before
    #'   processing.
    #' @param text_embeddings Object of class [EmbeddedText], [LargeDataSetForTextEmbeddings], `array` or
    #'   `datasets.arrow_dataset.Dataset`.
    #' @return Return `TRUE` if a compression is necessary and `FALSE` if not.
    requires_compression = function(text_embeddings) {
      # Check arguments
      check_class(
        object = text_embeddings,
        object_name = "text_embeddings",
        classes = c(
          "EmbeddedText", "LargeDataSetForTextEmbeddings",
          "array", "datasets.arrow_dataset.Dataset"
        ), FALSE
      )

      if (
        inherits(text_embeddings, "EmbeddedText") |
          inherits(text_embeddings, "LargeDataSetForTextEmbeddings")
      ) {
        if (private$model_config$use_fe & !text_embeddings$is_compressed()) {
          return(TRUE)
        } else {
          return(FALSE)
        }
      } else if (inherits(text_embeddings, "array")) {
        if (dim(text_embeddings)[3L] > private$model_config$features) {
          return(TRUE)
        } else {
          return(FALSE)
        }
      } else if (inherits(text_embeddings, "datasets.arrow_dataset.Dataset")) {
        tensors <-extract_column_from_py_dataset(text_embeddings,"input")[1L, , , drop = FALSE]
        if (dim(tensors)[3L] > private$model_config$features) {
          return(TRUE)
        } else {
          return(FALSE)
        }
      }
    },
    #-------------------------------------------------------------------------
    #' @description Method for saving a model.
    #' @param dir_path `string` Path of the directory where the model should be saved.
    #' @param folder_name `string` Name of the folder that should be created within the directory.
    #' @return Function does not return a value. It saves the model to disk.
    save = function(dir_path, folder_name) {
      # Save the classifier
      super$save(
        dir_path = dir_path,
        folder_name = folder_name
      )

      # Save the feature extractor if necessary
      if (private$model_config$use_fe) {
        save_to_disk(
          object = self$feature_extractor,
          dir_path = file.path(dir_path, folder_name),
          folder_name = "feature_extractor"
        )
      }
    },
    #--------------------------------------------------------------------------
    #' @description loads an object from disk and updates the object to the current version of the package.
    #' @param dir_path Path where the object set is stored.
    #' @return Method does not return anything. It loads an object from disk.
    load_from_disk = function(dir_path) {
      # Load common data for these class of models
      super$load_from_disk(dir_path = dir_path)

      # Load reliability data
      private$load_reliability_data(dir_path = dir_path)

      # load FeatureExtractor if it is part of the classifier
      private$load_FeatureExtractor(dir_path = paste0(dir_path))
    },
    #--------------------------------------------------------------------------
    #' @description Method transforms the levels of a factor into numbers corresponding
    #' to the models definition.
    #' @param data_targets `r get_param_doc_desc("data_targets")`
    #' @return Method returns a `factor` containing the numerical representation of
    #'  categories/classes.
    adjust_target_levels = function(data_targets) {
      tmp_data <- as.character(data_targets)
      tmp_data <- factor(
        x = tmp_data,
        levels = private$model_config$target_levels,
        ordered = TRUE
      )
      names(tmp_data) <- names(data_targets)
      return(tmp_data)
    },
    #---------------------------------------------------------------------------
    #' @description Method for requesting a plot of the training history.
    #' This method requires the *R* package 'ggplot2' to work.
    #' @param final_training `r get_param_doc_desc("final_training")`
    #' @param add_min_max `r get_param_doc_desc("add_min_max")`
    #' @param pl_step `int` Number of the step during pseudo labeling to plot. Only relevant if the model was trained
    #' with active pseudo labeling.
    #' @param x_min `r get_param_doc_desc("x_min")`
    #' @param x_max `r get_param_doc_desc("x_max")`
    #' @param y_min `r get_param_doc_desc("y_min")`
    #' @param y_max `r get_param_doc_desc("y_max")`
    #' @param ind_best_model `r get_param_doc_desc("ind_best_model")`
    #' @param ind_selected_model `r get_param_doc_desc("ind_selected_model")`
    #' @param text_size `r get_param_doc_desc("text_size")`
    #' @param measure `string` Measure to plot. Allowed values:
    #' * `"avg_iota"` = Average Iota
    #' * `"loss"` = Loss
    #' * `"accuracy"` = Accuracy
    #' * `"balanced_accuracy"` = Balanced Accuracy
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
      tmp_plot <- super$plot_training_history(
        final_training = final_training,
        pl_step = pl_step,
        measure = measure,
        ind_best_model = ind_best_model,
        ind_selected_model = ind_selected_model,
        x_min = x_min,
        x_max = x_max,
        y_min = y_min,
        y_max = y_max,
        add_min_max = add_min_max,
        text_size = text_size
      )
      return(tmp_plot)
    },
    #' @description Method for requesting a plot the coding stream.
    #' The plot shows how the cases of different categories/classes are
    #' assigned to a the available classes/categories. The visualization
    #' is helpful for analyzing the consequences of coding errors.
    #' @param label_categories_size `double` determining the size of the label for each true and assigned category within the plot.
    #' @param key_size `double` determining the size of the legend.
    #' @param text_size `double` determining the size of the text within the legend.
    #' @return Returns a plot of class `ggplot` visualizing the training process.
    plot_coding_stream = function(label_categories_size = 3L, key_size = 0.5, text_size = 10L) {
      tmp_plot <- iotarelr::plot_iota2_alluvial(
        object = self$reliability$iota_object_end_free,
        label_categories_size = label_categories_size,
        key_size = key_size,
        text_size = text_size
      )
      return(tmp_plot)
    }
  ),
  private = list(
    #--------------------------------------------------------------------------
    init_train = function() {
      # Setting a new ID for the classifier
      private$model_info$model_name <- private$generate_model_id(name = NULL)

      # Initializing Objects for Saving Performance
      metric_names <- get_coder_metrics(
        true_values = NULL,
        predicted_values = NULL,
        return_names_only = TRUE
      )

      self$reliability$test_metric <- matrix(
        nrow = self$last_training$config$n_folds,
        ncol = length(metric_names),
        dimnames = list(
          iterations = NULL,
          metrics = metric_names
        )
      )

      self$reliability$test_metric_mean <- NULL

      self$reliability$iota_objects_end <- NULL
      self$reliability$iota_objects_end_free <- NULL

      self$reliability$iota_object_end <- NULL
      self$reliability$iota_object_end_free <- NULL

      standard_measures_mean_table <- matrix(
        nrow = length(private$model_config$target_levels),
        ncol = 3L,
        data = 0L
      )
      colnames(standard_measures_mean_table) <- c("precision", "recall", "f1")
      rownames(standard_measures_mean_table) <- private$model_config$target_levels

      self$reliability$standard_measures_mean <- standard_measures_mean_table

      # Save start time of training
      self$last_training$start_time <- Sys.time()
    },
    #--------------------------------------------------------------------------
    calculate_test_metric = function(test_data, iteration, type) {
      test_predictions <- self$predict(
        newdata = test_data,
        ml_trace = self$last_training$config$ml_trace,
        batch_size = self$last_training$config$batch_size
      )
      test_pred_cat <- test_predictions$expected_category
      names(test_pred_cat) <- rownames(test_predictions)
      test_pred_cat <- test_pred_cat[extract_column_from_py_dataset(test_data,"id")]
      true_values <- factor(
        x = extract_column_from_py_dataset(test_data,"labels"),
        levels = 0L:(length(private$model_config$target_levels) - 1L),
        labels = private$model_config$target_levels
      )
      names(true_values) <- extract_column_from_py_dataset(test_data,"id")
      test_res <- get_coder_metrics(
        true_values = true_values,
        predicted_values = test_pred_cat
      )

      # Save results
      self$reliability$test_metric[iteration, ] <- test_res
    },
    #--------------------------------------------------------------------------
    calculate_measures_on_categorical_level = function(data_manager, iteration) {
      # Get test data
      data_manager$set_state(
        iteration = iteration,
        step = NULL
      )
      test_data <- data_manager$get_test_dataset()

      if (!is.null(test_data)) {
        # Predict labels
        test_predictions <- self$predict(
          newdata = test_data,
          ml_trace = self$last_training$config$ml_trace,
          batch_size = self$last_training$config$batch_size
        )
        test_pred_cat <- test_predictions$expected_category
        names(test_pred_cat) <- rownames(test_predictions)
        test_pred_cat <- test_pred_cat[extract_column_from_py_dataset(test_data,"id")]

        # Calculate standard measures
        true_values <- factor(
          x = extract_column_from_py_dataset(test_data,"labels"),
          levels = 0L:(length(private$model_config$target_levels) - 1L),
          labels = private$model_config$target_levels
        )
        names(true_values) <- extract_column_from_py_dataset(test_data,"id")
        self$reliability$standard_measures_end[iteration] <- list(
          calc_standard_classification_measures(
            true_values = true_values,
            predicted_values = test_pred_cat
          )
        )

        # Calculate iota objects
        self$reliability$iota_objects_end[iteration] <- list(iotarelr::check_new_rater(
          true_values = factor(
            x = extract_column_from_py_dataset(test_data,"labels"),
            levels = 0L:(length(private$model_config$target_levels) - 1L),
            labels = private$model_config$target_levels
          ),
          assigned_values = test_pred_cat,
          free_aem = FALSE
        ))
        self$reliability$iota_objects_end_free[iteration] <- list(iotarelr::check_new_rater(
          true_values = factor(
            x = extract_column_from_py_dataset(test_data,"labels"),
            levels = 0L:(length(private$model_config$target_levels) - 1L),
            labels = private$model_config$target_levels
          ),
          assigned_values = test_pred_cat,
          free_aem = TRUE
        ))
      } else if (iteration <= data_manager$get_n_folds()) {
        warning("Unable to calculate test scores. There is no test data.")
      }
    },
    #--------------------------------------------------------------------------
    finalize_train = function() {
      # Save Final Information
      self$last_training$date <- get_time_stamp()

      # Finalize measures from content analysis
      test_metric_mean <- vector(length = ncol(self$reliability$test_metric))
      test_metric_mean[] <- 0L
      names(test_metric_mean) <- colnames(self$reliability$test_metric)

      n_mean <- vector(length = ncol(self$reliability$test_metric))
      n_mean[] <- self$last_training$config$n_folds

      for (i in 1L:self$last_training$config$n_folds) {
        for (j in seq_len(ncol(self$reliability$test_metric))) {
          if (!is.na(self$reliability$test_metric[i, j])) {
            test_metric_mean[j] <- test_metric_mean[j] + self$reliability$test_metric[i, j]
          } else {
            n_mean[j] <- n_mean[j] - 1L
          }
        }
      }

      test_metric_mean <- test_metric_mean / n_mean
      test_metric_mean[is.nan(test_metric_mean)] <- NA
      self$reliability$test_metric_mean <- test_metric_mean

      self$last_training$learning_time <- as.numeric(
        difftime(Sys.time(),
          self$last_training$start_time,
          units = "mins"
        )
      )

      # Finalize iota objects
      if (!is.null(self$reliability$iota_objects_end)) {
        self$reliability$iota_object_end <- create_iota2_mean_object(
          iota2_list = self$reliability$iota_objects_end,
          original_cat_labels = private$model_config$target_levels,
          free_aem = FALSE,
          call = "aifeducation::te_classifier_neuralnet"
        )
      } else {
        self$reliability$iota_objects_end <- NULL
      }

      if (!is.null(self$reliability$iota_objects_end_free)) {
        self$reliability$iota_object_end_free <- create_iota2_mean_object(
          iota2_list = self$reliability$iota_objects_end_free,
          original_cat_labels = private$model_config$target_levels,
          free_aem = TRUE,
          call = "aifeducation::te_classifier_neuralnet"
        )
      } else {
        self$reliability$iota_objects_end_free <- NULL
      }

      # Finalize standard measures
      standard_measures <- self$reliability$standard_measures_mean

      for (tmp_cat in private$model_config$target_levels) {
        counter_precision <- 0
        counter_recall <- 0
        counter_f1 <- 0
        for (i in 1L:self$last_training$config$n_folds) {
          val_precision <- self$reliability$standard_measures_end[[i]][as.character(tmp_cat), "precision"]
          val_recall <- self$reliability$standard_measures_end[[i]][as.character(tmp_cat), "recall"]
          val_f1 <- self$reliability$standard_measures_end[[i]][as.character(tmp_cat), "f1"]

          if (!is.nan(val_precision) && !is.na(val_precision)) {
            standard_measures[as.character(tmp_cat), "precision"] <- standard_measures[as.character(tmp_cat), "precision"] +
              val_precision
            counter_precision <- counter_precision + 1
          }

          if (!is.nan(counter_recall) && !is.na(counter_recall)) {
            standard_measures[as.character(tmp_cat), "recall"] <- standard_measures[as.character(tmp_cat), "recall"] +
              val_recall
            counter_recall <- counter_recall + 1
          }

          if (!is.nan(val_f1) && !is.na(val_f1)) {
            standard_measures[as.character(tmp_cat), "f1"] <- standard_measures[as.character(tmp_cat), "f1"] +
              val_f1
            counter_f1 <- counter_f1 + 1
          }
        }
        if (counter_precision == 0) {
          standard_measures[as.character(tmp_cat), "precision"] <- NA
        } else {
          standard_measures[as.character(tmp_cat), "precision"] <- standard_measures[as.character(tmp_cat), "precision"] / counter_precision
        }

        if (counter_recall == 0) {
          standard_measures[as.character(tmp_cat), "recall"] <- NA
        } else {
          standard_measures[as.character(tmp_cat), "recall"] <- standard_measures[as.character(tmp_cat), "recall"] / counter_recall
        }

        if (counter_f1 == 0) {
          standard_measures[as.character(tmp_cat), "f1"] <- NA
        } else {
          standard_measures[as.character(tmp_cat), "f1"] <- standard_measures[as.character(tmp_cat), "f1"] / counter_f1
        }
      }
      self$reliability$standard_measures_mean <- standard_measures
    },
    #--------------------------------------------------------------------------
    train_standard = function(iteration = NULL,
                              data_manager = NULL,
                              inc_synthetic = FALSE) {
      # Print status message to console
      if (self$last_training$config$trace) {
        if (iteration <= self$last_training$config$n_folds) {
          message(
            get_time_stamp(),
            " | ", "Iteration ", iteration, " from ", self$last_training$config$n_folds
          )
        } else {
          message(
            get_time_stamp(),
            " | ", "Final training"
          )
        }
      }

      # Set the state of the DataManager
      data_manager$set_state(
        iteration = iteration,
        step = NULL
      )

      # Generate syntetic cases if requested
      if (inc_synthetic) {
        data_manager$create_synthetic(
          trace = self$last_training$config$trace,
          inc_pseudo_data = FALSE
        )
      }

      # Get the different DataSets
      train_data <- data_manager$get_dataset(
        inc_labeled = TRUE,
        inc_synthetic = inc_synthetic,
        inc_pseudo_data = FALSE,
        inc_unlabeled = FALSE
      )
      val_data <- data_manager$get_val_dataset()
      if (iteration != "final") {
        test_data <- data_manager$get_test_dataset()
      } else {
        test_data <- NULL
      }

      # Print status to console
      if (self$last_training$config$trace) {
        if (iteration <= self$last_training$config$n_folds) {
          message(
            get_time_stamp(),
            " | ", "Iteration ", iteration, " from ", self$last_training$config$n_folds,
            " | ", "Training"
          )
        } else {
          message(
            get_time_stamp(),
            " | ", "Final training",
            " | ", "Training"
          )
        }
      }

      # Start training
      train_history <- private$basic_train(
        train_data = train_data,
        val_data = val_data,
        test_data = test_data,
        reset_model = TRUE,
        use_callback = TRUE,
        log_dir = private$log_config$log_dir,
        log_write_interval = private$log_config$log_write_interval,
        log_top_value = iteration,
        log_top_total = self$last_training$config$n_folds + 1L,
        log_top_message = "Overall"
      )

      # Save history
      self$last_training$history[iteration] <- list(train_history)

      # Calculate test metric
      if (!is.null(test_data)) {
        private$calculate_test_metric(
          test_data = test_data,
          iteration = iteration,
          type = (as.numeric(inc_synthetic)) + 1L
        )
      }
    },
    #--------------------------------------------------------------------------
    train_with_pseudo_labels = function(init_train = TRUE,
                                        iteration = NULL,
                                        data_manager = NULL,
                                        inc_synthetic = FALSE) {
      # If model is not trained than train for the first time
      # Necessary for estimating pseudo labels
      if (init_train) {
        private$train_standard(
          iteration = iteration,
          data_manager = data_manager,
          inc_synthetic = inc_synthetic
        )
      }

      # Get validation and test data for training loop
      val_data <- data_manager$get_val_dataset()
      if (iteration != "final") {
        test_data <- data_manager$get_test_dataset()
      } else {
        test_data <- NULL
      }

      # Start training loop with pseudo labels
      data_manager$set_state(
        iteration = iteration,
        step = NULL
      )

      # Create list for saving training histories per step
      step_histories <- NULL

      for (step in 1L:self$last_training$config$pl_max_steps) {
        # Print status message to console
        if (self$last_training$config$trace) {
          if (iteration <= self$last_training$config$n_folds) {
            message(
              get_time_stamp(),
              " | ", "Iteration ", iteration, " from ", self$last_training$config$n_folds,
              " | ", "Pseudo labeling ", "step ", step, " from ", self$last_training$config$pl_max_steps
            )
          } else {
            message(
              get_time_stamp(),
              " | ", "Final training",
              " | ", "Pseudo labeling ", "step ", step, " from ", self$last_training$config$pl_max_steps
            )
          }
        }

        # Set correct state for the data_manager
        data_manager$set_state(
          iteration = iteration,
          step = step
        )

        # Generate pseudo labels
        pseudo_data <- private$estimate_pseudo_labels(
          unlabeled_data = data_manager$get_unlabeled_data(),
          val_data = val_data,
          current_step = step
        )

        # Save pseudo labels in the data_manager
        data_manager$add_replace_pseudo_data(
          inputs = pseudo_data$input,
          labels = pseudo_data$labels
        )

        # Remove old pseudo data
        rm(pseudo_data)

        # Generate synthetic data if requested
        if (inc_synthetic) {
          data_manager$create_synthetic(
            trace = self$last_training$config$trace,
            inc_pseudo_data = TRUE
          )
        }

        # Request training data
        train_data <- data_manager$get_dataset(
          inc_labeled = TRUE,
          inc_synthetic = inc_synthetic,
          inc_pseudo_data = TRUE,
          inc_unlabeled = FALSE
        )

        # Print status to console
        if (self$last_training$config$trace) {
          if (iteration <= self$last_training$config$n_folds) {
            message(
              get_time_stamp(),
              " | ", "Iteration ", iteration, " from ", self$last_training$config$n_folds,
              " | ", "Training"
            )
          } else {
            message(
              get_time_stamp(),
              " | ", "Final training",
              " | ", "Training"
            )
          }
        }

        # Start training
        train_history <- private$basic_train(
          train_data = train_data,
          val_data = val_data,
          test_data = test_data,
          reset_model = TRUE,
          use_callback = TRUE,
          log_dir = private$log_config$log_dir,
          log_write_interval = private$log_config$log_write_interval,
          log_top_value = iteration,
          log_top_total = self$last_training$config$n_folds + 1L,
          log_top_message = "Overall"
        )

        # Save history
        step_histories[step] <- list(train_history)
      }

      # Save the histories for the complete iteration
      self$last_training$history[iteration] <- list(step_histories)

      # Calculate test metric
      if (!is.null(test_data)) {
        private$calculate_test_metric(
          test_data = test_data,
          iteration = iteration,
          type = 3L
        )
      }
    },
    #--------------------------------------------------------------------------
    estimate_pseudo_labels = function(unlabeled_data,
                                      val_data,
                                      current_step) {
      # Predict pseudo labels for unlabeled data
      predicted_labels <- self$predict(
        newdata = unlabeled_data,
        ml_trace = self$last_training$config$ml_trace,
        batch_size = self$last_training$config$batch_size
      )

      # Create Matrix for saving the results
      new_categories <- matrix(
        nrow = nrow(predicted_labels),
        ncol = 2L
      )
      rownames(new_categories) <- rownames(predicted_labels)
      colnames(new_categories) <- c("cat", "prob")

      # Correct probabilities for reliability on the validation data
      predicted_labels <- private$pseudo_labels_correct_prob(
        predictions = predicted_labels,
        val_data = val_data
      )

      # Gather information for every case. That is the category with the
      # highest probability and save both
      for (i in seq_len(nrow(predicted_labels))) {
        tmp_est_prob <- predicted_labels[i, 1L:(ncol(predicted_labels) - 1L)]
        new_categories[i, 1L] <- which.max(tmp_est_prob) - 1L
        new_categories[i, 2L] <- max(tmp_est_prob)
      }
      new_categories <- as.data.frame(new_categories)

      # Transforming the probabilities to an information index
      new_categories[, 2L] <- abs(
        self$last_training$config$pl_anchor -
          (as.numeric(new_categories[, 2L]) - 1L / length(private$model_config$target_levels)) /
            (1L - 1L / length(private$model_config$target_levels))
      )
      new_categories <- as.data.frame(new_categories)

      # Reducing the new categories to the desired range
      condition <- (
        new_categories[, 2L] >= self$last_training$config$pl_min &
          new_categories[, 2L] <= self$last_training$config$pl_max
      )
      new_categories <- subset(new_categories, condition)

      # Calculate number of cases to include
      bpl_inc_ratio <- current_step / self$last_training$config$pl_max_steps
      n_cases_to_include <- nrow(new_categories) * bpl_inc_ratio

      # Order cases with increasing distance from maximal information
      new_categories <- new_categories[order(new_categories$prob, decreasing = FALSE), ]

      # Select the best cases
      names_final_new_categories <- rownames(new_categories)[1L:n_cases_to_include]

      # Get the labels for these cases
      targets_pseudo_labeled <- new_categories[names_final_new_categories, 1L]
      targets_pseudo_labeled <- as.numeric(targets_pseudo_labeled)
      names(targets_pseudo_labeled) <- names_final_new_categories

      # Transform pseudo labels to a factor
      targets_pseudo_labeled <- factor(
        x = targets_pseudo_labeled,
        levels = 0L:(length(private$model_config$target_levels) - 1L),
        labels = private$model_config$target_levels
      )

      # get the corresponding input
      embeddings <-extract_column_from_py_dataset(unlabeled_data,"input")
      rownames(embeddings) <- extract_column_from_py_dataset(unlabeled_data,"id")
      embeddings <- embeddings[names_final_new_categories, , ]

      # Return results
      pseudo_data <- list(
        input = embeddings,
        labels = targets_pseudo_labeled
      )

      return(pseudo_data)
    },
    #--------------------------------------------------------------------------
    pseudo_labels_correct_prob = function(predictions, val_data) {
      # Predict on val data
      val_predictions <- self$predict(
        newdata = val_data,
        ml_trace = self$last_training$config$ml_trace,
        batch_size = self$last_training$config$batch_size
      )
      val_pred_cat <- val_predictions$expected_category
      names(val_pred_cat) <- rownames(val_predictions)
      val_pred_cat <- val_pred_cat[extract_column_from_py_dataset(val_data,"id")]

      # Calculate Assignment Error Matrix
      val_iota_object <- iotarelr::check_new_rater(
        true_values = factor(
          x = extract_column_from_py_dataset(val_data,"labels"),
          levels = 0L:(length(private$model_config$target_levels) - 1L),
          labels = private$model_config$target_levels
        ),
        assigned_values = val_pred_cat,
        free_aem = TRUE
      )

      # Estimate probability of each category
      aem <- val_iota_object$categorical_level$raw_estimates$assignment_error_matrix
      class_sizes <- val_iota_object$information$est_true_cat_sizes
      p_cat <- class_sizes %*% aem

      # Estimate probability that the category is the true category
      p_cat_true <- class_sizes * diag(aem) / p_cat
      p_cat_true <- replace(p_cat_true, list = is.nan(p_cat_true), values = 0L)

      # Correct probabilities
      number_columns <- ncol(predictions)
      n_col <- ncol(predictions) - 1L
      for (i in seq_len(nrow(predictions))) {
        predictions[i, 1L:n_col] <- predictions[i, 1L:n_col] * p_cat_true / sum(predictions[i, 1L:n_col] * p_cat_true)
        predictions[i, number_columns] <- private$model_config$target_levels[which.max(as.numeric(predictions[i, 1L:n_col]))]
      }
      return(predictions)
    },
    #--------------------------------------------------------------------------
    basic_train = function(train_data = NULL,
                           val_data = NULL,
                           test_data = NULL,
                           reset_model = FALSE,
                           use_callback = TRUE,
                           log_dir = NULL,
                           log_write_interval = 10L,
                           log_top_value = NULL,
                           log_top_total = NULL,
                           log_top_message = NULL) {
      # Clear session to provide enough resources for computations
      if (torch$cuda$is_available()) {
        torch$cuda$empty_cache()
      }


      # Generating class weights
      if (self$last_training$config$loss_balance_class_weights) {
        abs_freq_classes <- table(
          extract_column_from_py_dataset(train_data,"labels")
          )
        class_weights <- as.vector(sum(abs_freq_classes) / (length(abs_freq_classes) * abs_freq_classes))
      } else {
        class_weights <- rep(x = 1L, times = length(private$model_config$target_levels))
      }

      # Generating weights for sequence length
      if (self$last_training$config$loss_balance_sequence_length) {
        sequence_length <- extract_column_from_py_dataset(train_data,"length")
        abs_freq_length <- table(sequence_length)

        sample_weight_per_sequence_length <- as.vector(
          sum(abs_freq_length) / (length(abs_freq_length) * abs_freq_length)
        )
        sequence_order <- names(abs_freq_length)

        sample_weights <- vector(length = length(sequence_length))
        for (i in seq_along(sample_weights)) {
          idx <- which(sequence_length[i] == sequence_order)
          sample_weights[i] <- sample_weight_per_sequence_length[idx]
        }
      } else {
        sequence_length <- extract_column_from_py_dataset(train_data,"length")
        sample_weights <- rep.int(x = 1L, times = length(sequence_length))
      }

      # Reset model if requested
      if (reset_model) {
        private$create_reset_model()
      }

      # Set loss function
      loss_cls_fct_name <- "CrossEntropyLoss"

      # Check directory for checkpoints
      create_dir(
        dir_path = private$dir_checkpoint,
        trace = self$last_training$config$trace,
        msg = "Creating Checkpoint Directory"
      )

      # Set target column
      if (!private$model_config$require_one_hot) {
        target_column <- "labels"
      } else {
        target_column <- "one_hot_encoding"
      }

      data_set_weights <- datasets$Dataset$from_dict(
        reticulate::dict(list(
          sample_weights = sample_weights
        ))
      )

      dataset_train <- train_data$add_column("sample_weights",
                                             extract_column_from_py_dataset(data_set_weights,"sample_weights"))
      dataset_train <- dataset_train$select_columns(c("input", target_column, "sample_weights"))

      if (private$model_config$require_one_hot) {
        dataset_train <- dataset_train$rename_column(target_column, "labels")
      }

      pytorch_train_data <- dataset_train$with_format("torch")

      pytorch_val_data <- val_data$select_columns(c("input", target_column))
      if (private$model_config$require_one_hot) {
        pytorch_val_data <- pytorch_val_data$rename_column(target_column, "labels")
      }
      pytorch_val_data <- pytorch_val_data$with_format("torch")

      if (!is.null(test_data)) {
        pytorch_test_data <- test_data$select_columns(c("input", target_column))
        if (private$model_config$require_one_hot) {
          pytorch_test_data <- pytorch_test_data$rename_column(target_column, "labels")
        }
        pytorch_test_data <- pytorch_test_data$with_format("torch")
      } else {
        pytorch_test_data <- NULL
      }

      tmp_history <- py$TeClassifierTrain(
        model = private$model,
        loss_cls_fct_name = self$last_training$config$loss_cls_fct_name,
        optimizer_method = self$last_training$config$optimizer,
        lr_rate = self$last_training$config$lr_rate,
        lr_min=self$last_training$config$lr_min,
        scheduler_type=self$last_training$config$lr_scheduler,
        lr_warm_up_ratio = self$last_training$config$lr_warm_up_ratio,
        epochs = as.integer(self$last_training$config$epochs),
        trace = as.integer(self$last_training$config$ml_trace),
        use_callback = use_callback,
        batch_size = as.integer(self$last_training$config$batch_size),
        train_data = pytorch_train_data,
        val_data = pytorch_val_data,
        test_data = pytorch_test_data,
        filepath = file.path(private$dir_checkpoint, "best_weights.pt"),
        n_classes = as.integer(length(private$model_config$target_levels)),
        class_weights = torch$tensor(np$array(class_weights)),
        log_dir = log_dir,
        log_write_interval = log_write_interval,
        log_top_value = log_top_value,
        log_top_total = log_top_total,
        log_top_message = log_top_message
      )


      # provide rownames and replace -100
      tmp_history <- private$prepare_history_data(tmp_history)
      return(tmp_history)
    },
    #--------------------------------------------------------------------------
    set_feature_extractor = function(feature_extractor) {
      # Check
      check_class(object = feature_extractor, object_name = "feature_extractor", classes = "TEFeatureExtractor", allow_NULL = TRUE)
      if (!is.null(feature_extractor)) {
        if (!feature_extractor$is_trained()) {
          stop("The supplied feature extractor is not trained. Please
                provide train and try again.")
        }

        private$model_config$use_fe <- TRUE
        private$model_config$features <- feature_extractor$get_model_config()$features
        self$feature_extractor <- feature_extractor$clone(deep = TRUE)
      } else {
        private$model_config$use_fe <- FALSE
        private$model_config$features <- private$text_embedding_model[["features"]]
      }
    },
    #--------------------------------------------------------------------------
    check_target_levels = function(data_targets) {
      if (sum(levels(data_targets) %in% private$model_config$target_levels) != private$model_config$n_categories) {
        warning(
          "data_targets contains levels that are not defined for the classifier",
          "Defined levels are", private$model_config$target_levels, ".",
          "Please check your data or create a new classifier and pass
                all levels to the classifier's configuration."
        )
      }
    },
    #--------------------------------------------------------------------------
    do_configuration = function(args, one_hot_encoding = TRUE) {
      # Initial checks, adjustments, and preparation----------------------------
      # check if already configured
      private$check_config_for_FALSE()

      # Check arguments
      check_all_args(args = args)
      private$check_embeddings_object_type(args$text_embeddings, strict = TRUE)

      # save arguments
      private$save_all_args(args = args, group = "configure")

      # Set TextEmbeddingModel
      private$set_text_embedding_model(
        model_info = args$text_embeddings$get_model_info(),
        feature_extractor_info = args$text_embeddings$get_feature_extractor_info(),
        times = args$text_embeddings$get_times(),
        features = args$text_embeddings$get_features(),
        pad_value = args$text_embeddings$get_pad_value()
      )

      # Set Times and Features
      private$set_times_and_features(
        times = args$text_embeddings$get_times(),
        features = args$text_embeddings$get_features()
      )

      # Set target data config
      private$set_target_data(
        target_levels = args$target_levels,
        one_hot_encoding = one_hot_encoding
      )

      # Perform additional checks and adjustments
      if (is.function(private$check_param_combinations_configuration)) {
        private$check_param_combinations_configuration()
      }

      # Set ML framework
      private$ml_framework <- "pytorch"

      # Setting Label and Name
      private$set_model_info(
        model_name = private$generate_model_id(args$name),
        label = args$label,
        model_date = get_time_stamp()
      )

      # Adjust configuration
      if (is.function(private$adjust_configuration())) {
        private$adjust_configuration()
      }

      # Set FeatureExtractor and adapt config
      self$check_feature_extractor_object_type(args$feature_extractor)
      private$set_feature_extractor(args$feature_extractor)

      # Set package versions
      private$set_package_versions()

      # Finalize configuration
      private$set_configuration_to_TRUE()

      # Create_Model
      private$create_reset_model()
    },
    #---------------------------------------------------------------------------
    check_param_combinations_training = function() {
      if (self$last_training$config$use_pl) {
        if (self$last_training$config$pl_max < self$last_training$config$pl_min) {
          stop("pl_max must be at least pl_min.")
        }
        if (self$last_training$config$pl_anchor < self$last_training$config$pl_min) {
          stop("pl_anchor must be at least pl_min.")
        }
        if (self$last_training$config$pl_anchor > self$last_training$config$pl_max) {
          stop("pl_anchor must be lower or equal to pl_max.")
        }
      }

      if (self$last_training$config$use_sc) {
        if (self$last_training$config$sc_max_k < self$last_training$config$sc_min_k) {
          stop("sc_max_k must be at least sc_min_k")
        }
      }

      if (self$last_training$config$lr_rate < self$last_training$config$lr_min) {
        stop("lr_rate must be at least lr_min")
      }
    },
    #---------------------------------------------------------------------------
    prepare_data_for_training = function(data_targets, data_embeddings) {
      # Transform target data
      data_targets <- self$adjust_target_levels(data_targets)

      # Set up data
      if (inherits(data_embeddings, "EmbeddedText")) {
        tmp_data <- data_embeddings$convert_to_LargeDataSetForTextEmbeddings()
      } else {
        tmp_data <- data_embeddings
      }

      # Create DataManager------------------------------------------------------
      if (private$model_config$use_fe) {
        compressed_embeddings <- self$feature_extractor$extract_features_large(
          data_embeddings = tmp_data,
          as.integer(self$last_training$config$batch_size),
          trace = self$last_training$config$trace
        )
        data_manager <- DataManagerClassifier$new(
          data_embeddings = compressed_embeddings,
          data_targets = data_targets,
          folds = self$last_training$config$data_folds,
          val_size = self$last_training$config$data_val_size,
          class_levels = private$model_config$target_levels,
          one_hot_encoding = private$model_config$require_one_hot,
          add_matrix_map = self$last_training$config$use_sc,
          sc_method = self$last_training$config$sc_method,
          sc_min_k = self$last_training$config$sc_min_k,
          sc_max_k = self$last_training$config$sc_max_k,
          trace = self$last_training$config$trace,
          n_cores = self$last_training$config$n_cores,
          pad_value = private$text_embedding_model$pad_value
        )
      } else {
        data_manager <- DataManagerClassifier$new(
          data_embeddings = tmp_data,
          data_targets = data_targets,
          folds = self$last_training$config$data_folds,
          val_size = self$last_training$config$data_val_size,
          class_levels = private$model_config$target_levels,
          one_hot_encoding = private$model_config$require_one_hot,
          add_matrix_map = self$last_training$config$use_sc,
          sc_method = self$last_training$config$sc_method,
          sc_min_k = self$last_training$config$sc_min_k,
          sc_max_k = self$last_training$config$sc_max_k,
          trace = self$last_training$config$trace,
          n_cores = self$last_training$config$n_cores,
          pad_value = private$text_embedding_model$pad_value
        )
      }

      # Save Data Statistics
      self$last_training$data <- data_manager$get_statistics()

      # Save the number of folds
      self$last_training$config$n_folds <- data_manager$get_n_folds()

      return(data_manager)
    },
    #--------------------------------------------------------------------------
    check_data_for_pseudo_labeling = function(data_manager) {
      if (self$last_training$config$use_pl) {
        if (!data_manager$contains_unlabeled_data()) {
          warning("There are no cases without labels. Setting 'use_pl' to 'FALSE'.")
          self$last_training$config$use_pl <- FALSE
        }
      }
    },
    #--------------------------------------------------------------------------
    set_target_data = function(target_levels, one_hot_encoding = TRUE) {
      private$model_config["target_levels"] <- list(target_levels)
      private$model_config["n_categories"] <- list(length(target_levels))
      private$model_config["require_one_hot"] <- list(one_hot_encoding)
    },
    #--------------------------------------------------------------------------
    set_times_and_features = function(times, features) {
      private$model_config$features <- features
      private$model_config$times <- times
    },
    #--------------------------------------------------------------------------
    load_reload_python_scripts = function() {
      load_py_scripts(c(
        "pytorch_act_fct.py",
        "pytorch_distance_fun.py",
        "pytorch_loss_fct.py",
        "pytorch_layers.py",
        "pytorch_layers_normalization.py",
        "pytorch_stack_layers.py",
        "pytorch_autoencoder.py",
        "py_log.py",
        "py_functions.py",
        "pytorch_classifier_models.py",
        "pytorch_cls_training_loops.py",
        "pytorch_predict_batch.py",
        "pytorch_datacollators.py"
      ))
    },
    #-------------------------------------------------------------------------
    do_training = function(args) {
      # Check arguments
      check_all_args(args = args)
      private$check_target_levels(args$data_targets)
      self$check_embedding_model(args$data_embeddings, require_compressed = FALSE)

      # Save args
      private$save_all_args(args = args, group = "training")

      # Perform additional checks
      if (is.function(private$check_param_combinations_training)) {
        private$check_param_combinations_training()
      }

      # set up logger
      private$set_up_logger(log_dir = args$log_dir, log_write_interval = args$log_write_interval)

      # Prepare Data for Training
      data_manager <- private$prepare_data_for_training(
        data_targets = args$data_targets,
        data_embeddings = args$data_embeddings
      )

      # Check if data can be used for pseudo labeling
      private$check_data_for_pseudo_labeling(data_manager)

      # Check and create temporary directory for checkpoints
      private$create_checkpoint_directory()

      # Start-------------------------------------------------------------------
      if (self$last_training$config$trace) {
        message(
          get_time_stamp(),
          " Start"
        )
      }

      # Init Training------------------------------------------------------------
      private$init_train()

      # config datasets
      datasets$disable_progress_bars()
      # datasets$disable_caching()

      # Start Sustainability Tracking-------------------------------------------
      private$init_and_start_sustainability_tracking()

      # Start Training----------------------------------------------------------
      # Load Custom Model Scripts
      private$load_reload_python_scripts()

      # Start Loop inclusive final training
      for (iter in 1L:(self$last_training$config$n_folds + 1L)) {
        base::gc(verbose = FALSE, full = TRUE)

        if (!self$last_training$config$use_pl) {
          private$train_standard(
            iteration = iter,
            data_manager = data_manager,
            inc_synthetic = self$last_training$config$use_sc
          )
        } else if (self$last_training$config$use_pl) {
          private$train_with_pseudo_labels(
            init_train = TRUE,
            iteration = iter,
            data_manager = data_manager,
            inc_synthetic = self$last_training$config$use_sc
          )
        }

        # Calculate measures on categorical level
        private$calculate_measures_on_categorical_level(
          data_manager = data_manager,
          iteration = iter
        )
        gc()
      }

      # Finalize Training
      private$finalize_train()

      # Stop sustainability tracking if requested
      private$stop_sustainability_tracking()

      # Clean temporary directory
      private$clean_checkpoint_directory()

      # Set trained field
      private$trained <- TRUE

      if (self$last_training$config$trace) {
        message(
          get_time_stamp(),
          " Training Complete"
        )
      }
    },
    #-------------------------------------------------------------------------
    load_reliability_data = function(dir_path) {
      # Load R file with configuration and other data
      config_file <- load_R_config_state(dir_path)

      self$reliability <- list(
        test_metric = config_file$public$reliability$test_metric,
        test_metric_mean = config_file$public$reliability$test_metric_mean,
        raw_iota_objects = list(
          iota_objects_end = config_file$public$reliability$raw_iota_objects$iota_objects_end,
          iota_objects_end_free = config_file$public$reliability$raw_iota_objects$iota_objects_end_free
        ),
        iota_object_end = config_file$public$reliability$iota_object_end,
        iota_object_end_free = config_file$public$reliability$iota_object_end_free,
        standard_measures_end = config_file$public$reliability$standard_measures_end,
        standard_measures_mean = config_file$public$reliability$standard_measures_mean
      )
    },
    #---------------------------------------------------------------------------
    load_FeatureExtractor = function(dir_path = dir_path) {
      if (private$model_config$use_fe) {
        feature_extractor <- TEFeatureExtractor$new()
        feature_extractor$load_from_disk(file.path(dir_path, "feature_extractor"))
        self$feature_extractor <- feature_extractor
      }
    },
    #--------------------------------------------------------------------------
    generate_model_id = function(name) {
      if (is.null(name)) {
        return(paste0("cls_", generate_id(16L)))
      } else {
        return(name)
      }
    }
  )
)
