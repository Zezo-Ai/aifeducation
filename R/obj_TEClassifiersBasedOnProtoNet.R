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

#' @title Base class for classifiers relying on numerical representations of texts instead of words that use
#' the architecture of Protonets and its corresponding training techniques.
#' @description Base class for classifiers relying on [EmbeddedText] or [LargeDataSetForTextEmbeddings] as input
#' which use the architecture of Protonets and its corresponding training techniques.
#'
#' Objects of this class containing fields and methods used in several other classes in 'AI for Education'.
#'
#' This class is **not** designed for a direct application and should only be used by developers.
#'
#' @return A new object of this class.
#' @family R6 Classes for Developers
#' @export
TEClassifiersBasedOnProtoNet <- R6::R6Class(
  classname = "TEClassifiersBasedOnProtoNet",
  inherit = ClassifiersBasedOnTextEmbeddings,
  public = list(
    #-------------------------------------------------------------------------
    #' @description Method for training a neural net.
    #'
    #'   Training includes a routine for early stopping. In the case that loss<0.0001 and Accuracy=1.00 and Average
    #'   Iota=1.00 training stops. The history uses the values of the last trained epoch for the remaining epochs.
    #'
    #'   After training the model with the best values for Average Iota, Accuracy, and Loss on the validation data set
    #'   is used as the final model.
    #'
    #' @param data_embeddings `r get_param_doc_desc("data_embeddings")`
    #' @param data_targets `r get_param_doc_desc("data_targets")`.
    #' @param data_folds `r get_param_doc_desc("data_folds")`
    #' @param data_val_size `r get_param_doc_desc("data_val_size")`
    #' @param loss_balance_class_weights `r get_param_doc_desc("loss_balance_class_weights")`
    #' @param loss_balance_sequence_length `r get_param_doc_desc("loss_balance_sequence_length")`
    #' @param loss_pt_fct_name `r get_param_doc_desc("loss_pt_fct_name")`
    #' @param use_sc `r get_param_doc_desc("use_sc")`
    #' @param sc_method `r get_param_doc_desc("sc_method")`
    #' @param sc_min_k `r get_param_doc_desc("sc_min_k")`
    #' @param sc_max_k `r get_param_doc_desc("sc_max_k")`
    #' @param use_pl `r get_param_doc_desc("use_pl")`
    #' @param pl_max_steps `r get_param_doc_desc("pl_max_steps")`
    #' @param pl_anchor `r get_param_doc_desc("pl_anchor")`
    #' @param pl_max `r get_param_doc_desc("pl_max")`
    #' @param pl_min `r get_param_doc_desc("pl_min")`
    #' @param sustain_track `r get_param_doc_desc("sustain_track")`
    #' @param sustain_iso_code `r get_param_doc_desc("sustain_iso_code")`
    #' @param sustain_region `r get_param_doc_desc("sustain_region")`
    #' @param sustain_interval `r get_param_doc_desc("sustain_interval")`
    #' @param sustain_log_level `r get_param_doc_desc("sustain_log_level")`
    #' @param epochs `r get_param_doc_desc("epochs")`
    #' @param batch_size `r get_param_doc_desc("batch_size")`
    #' @param log_dir `r get_param_doc_desc("log_dir")`
    #' @param log_write_interval `r get_param_doc_desc("log_write_interval")`
    #' @param trace `r get_param_doc_desc("trace")`
    #' @param ml_trace `r get_param_doc_desc("ml_trace")`
    #' @param n_cores `r get_param_doc_desc("n_cores")`
    #' @param lr_rate `r get_param_doc_desc("lr_rate")`
    #' @param lr_min `r get_param_doc_desc("lr_min")`
    #' @param lr_warm_up_ratio `r get_param_doc_desc("lr_warm_up_ratio")`
    #' @param lr_scheduler `r get_param_doc_desc("lr_scheduler")`
    #' @param optimizer `r get_param_doc_desc("optimizer")`
    #' @param Ns `r get_param_doc_desc("Ns")`
    #' @param Nq `r get_param_doc_desc("Nq")`
    #' @param loss_alpha `r get_param_doc_desc("loss_alpha")`
    #' @param loss_margin `r get_param_doc_desc("loss_margin")`
    #' @param sampling_separate `r get_param_doc_desc("sampling_separate")`
    #' @param sampling_shuffle `r get_param_doc_desc("sampling_shuffle")`
    #' @return Function does not return a value. It changes the object into a trained classifier.
    #' @details
    #'
    #' * `sc_max_k`: All values from sc_min_k up to sc_max_k are successively used. If
    #' the number of `sc_max_k` is too high, the value is reduced to a number that allows the calculating of synthetic
    #' units.
    #' * `pl_anchor:` With the help of this value, the new cases are sorted. For
    #' this aim, the distance from the anchor is calculated and all cases are arranged into an ascending order.
    #'
    train = function(data_embeddings = NULL,
                     data_targets = NULL,
                     data_folds = 5L,
                     data_val_size = 0.25,
                     loss_pt_fct_name = "MultiWayContrastiveLoss",
                     use_sc = FALSE,
                     sc_method = "knnor",
                     sc_min_k = 1L,
                     sc_max_k = 10L,
                     use_pl = FALSE,
                     pl_max_steps = 3L,
                     pl_max = 1.00,
                     pl_anchor = 1.00,
                     pl_min = 0.00,
                     sustain_track = TRUE,
                     sustain_iso_code = NULL,
                     sustain_region = NULL,
                     sustain_interval = 15L,
                     sustain_log_level = "warning",
                     epochs = 40L,
                     batch_size = 35L,
                     Ns = 5L,
                     Nq = 3L,
                     loss_alpha = 0.5,
                     loss_margin = 0.05,
                     sampling_separate = FALSE,
                     sampling_shuffle = TRUE,
                     trace = TRUE,
                     ml_trace = 1L,
                     log_dir = NULL,
                     log_write_interval = 10L,
                     n_cores = auto_n_cores(),
                     lr_rate = 1e-3,
                     lr_min=1e-4,
                     lr_scheduler="None",
                     lr_warm_up_ratio = 0.02,
                     optimizer = "AdamW") {
      private$do_training(args = get_called_args(n = 1L))
    },
    #---------------------------------------------------------------------------
    #' @description Method for predicting the class of given data (query) based on provided examples (sample).
    #' @param newdata Object of class [EmbeddedText] or [LargeDataSetForTextEmbeddings] containing the text
    #'   embeddings for all cases which should be predicted. They form the query set.
    #' @param embeddings_s Object of class [EmbeddedText] or [LargeDataSetForTextEmbeddings] containing the text
    #'   embeddings for all reference examples. They form the sample set.
    #' @param classes_s Named `factor` containing the classes for every case within `embeddings_s`.
    #' @param batch_size `int` batch size.
    #' @param ml_trace `r get_param_doc_desc("ml_trace")`
    #' @return Returns a `data.frame` containing the predictions and the probabilities of the different labels for each
    #'   case.
    predict_with_samples = function(newdata,
                                    batch_size = 32L,
                                    ml_trace = 1L,
                                    embeddings_s = NULL,
                                    classes_s = NULL) {
      forward_results <- private$forward(
        embeddings_q = newdata,
        classes_q = NULL,
        batch_size = batch_size,
        ml_trace = ml_trace,
        embeddings_s = embeddings_s,
        classes_s = classes_s,
        prediction_mode = TRUE
      )

      # Ids of the rows of newdata
      # current_row_names <- forward_results$rownames_q

      # Possible classes
      class_labels <- forward_results$class_labels

      # Probabilities for every class
      predictions_prob <- forward_results$results

      # Index with highest probability
      predictions <- max.col(predictions_prob) - 1L

      # Transforming predictions to target levels------------------------------
      predictions <- as.character(as.vector(predictions))
      for (i in 0L:(length(class_labels) - 1L)) {
        predictions <- replace(
          x = predictions,
          predictions == as.character(i),
          values = class_labels[i + 1L]
        )
      }

      # Transforming to a factor
      predictions <- factor(predictions, levels = class_labels)

      # colnames(predictions_prob) <- class_labels
      predictions_prob <- as.data.frame(predictions_prob)
      predictions_prob$expected_category <- predictions
      # rownames(predictions_prob) <- current_row_names
      return(predictions_prob)
    },
    #---------------------------------------------------------------------------
    #' @description Method for embedding documents. Please do not confuse this type of embeddings with the embeddings of
    #'   texts created by an object of class [TextEmbeddingModel]. These embeddings embed documents according to their
    #'   similarity to specific classes.
    #' @param embeddings_q Object of class [EmbeddedText] or [LargeDataSetForTextEmbeddings] containing the text
    #'   embeddings for all cases which should be embedded into the classification space.
    #' @param embeddings_s Object of class [EmbeddedText] or [LargeDataSetForTextEmbeddings] containing the text
    #'   embeddings for all reference examples. They form the sample set. If set to `NULL` the trained prototypes are used.
    #' @param classes_s Named `factor` containing the classes for every case within `embeddings_s`.
    #'    If set to `NULL` the trained prototypes are used.
    #' @param batch_size `int` batch size.
    #' @param ml_trace `r get_param_doc_desc("ml_trace")`
    #' @return Returns a `list` containing the following elements
    #'
    #' * `embeddings_q`: embeddings for the cases (query sample).
    #' * `distances_q`: `matrix` containing the distance of every query case to every prototype.
    #' * `embeddings_prototypes`: embeddings of the prototypes which were learned during training. They represents the
    #'    center for the different classes.
    #'
    embed = function(embeddings_q = NULL, embeddings_s = NULL, classes_s = NULL, batch_size = 32L, ml_trace = 1L) {
      # Load Custom Model Scripts
      private$load_reload_python_scripts()

      # Check arguments and forward
      forward_results <- private$forward(
        embeddings_q = embeddings_q,
        classes_q = NULL,
        batch_size = batch_size,
        ml_trace = ml_trace,
        embeddings_s = embeddings_s,
        classes_s = classes_s,
        prediction_mode = FALSE
      )

      return(
        list(
          embeddings_q = forward_results$results$embeddings_query,
          distances_q = forward_results$results$embeddings_query,
          embeddings_prototypes = forward_results$results$prototype_embeddings
        )
      )
    },
    #---------------------------------------------------------------------------
    #' @description Method returns the scaling factor of the metric.
    #' @return Returns the scaling factor of the metric as `float`.
    get_metric_scale_factor = function() {
      return(tensor_to_numpy(private$model$get_metric_scale_factor()))
    },
    #---------------------------------------------------------------------------
    #' @description Method for creating a plot to visualize embeddings and their corresponding centers (prototypes).
    #' @param embeddings_q Object of class [EmbeddedText] or [LargeDataSetForTextEmbeddings] containing the text
    #'   embeddings for all cases which should be embedded into the classification space.
    #' @param classes_q Named `factor` containg the true classes for every case. Please note that the names must match
    #'   the names/ids in `embeddings_q`.
    #' @param embeddings_s Object of class [EmbeddedText] or [LargeDataSetForTextEmbeddings] containing the text
    #'   embeddings for all reference examples. They form the sample set. If set to `NULL` the trained prototypes are used.
    #' @param classes_s Named `factor` containing the classes for every case within `embeddings_s`.
    #'    If set to `NULL` the trained prototypes are used.
    #' @param inc_unlabeled `bool` If `TRUE` plot includes unlabeled cases as data points.
    #' @param size_points `int` Size of the points excluding the points for prototypes.
    #' @param size_points_prototypes `int` Size of points representing prototypes.
    #' @param alpha `float` Value indicating how transparent the points should be (important
    #'   if many points overlap). Does not apply to points representing prototypes.
    #' @param inc_margin `bool` If `TRUE` plot includes the margin around every prototype. Adding margin
    #' requires a trained model. If the model is not trained this argument is treated as set to `FALSE`.
    #' @param batch_size `int` batch size.
    #' @return Returns a plot of class `ggplot`visualizing embeddings.
    plot_embeddings = function(embeddings_q,
                               classes_q = NULL,
                               embeddings_s = NULL,
                               classes_s = NULL,
                               batch_size = 12L,
                               alpha = 0.5,
                               size_points = 3L,
                               size_points_prototypes = 8L,
                               inc_unlabeled = TRUE,
                               inc_margin = TRUE) {
      # Argument checking-------------------------------------------------------
      # Do forward
      forward_results <- private$forward(
        embeddings_q = embeddings_q,
        classes_q = NULL,
        batch_size = batch_size,
        ml_trace = 0L,
        embeddings_s = embeddings_s,
        classes_s = classes_s,
        prediction_mode = FALSE
      )

      prototypes <- as.data.frame(forward_results$results$prototype_embeddings)
      prototypes$class <- rownames(forward_results$results$prototype_embeddings)
      prototypes$type <- rep("prototype", nrow(forward_results$results$prototype_embeddings))
      colnames(prototypes) <- c("x", "y", "class", "type")

      if (!is.null(classes_q)) {
        # Check classes of q and sample
        labels_sample_and_query <- intersect(x = rownames(forward_results$results$prototype_embeddings), levels(classes_q))
        classes_q[!(classes_q %in% labels_sample_and_query)] <- NA

        true_values_names <- intersect(
          x = names(na.omit(classes_q)),
          y = private$get_rownames_from_embeddings(embeddings_q)
        )
        true_values <- as.data.frame(forward_results$results$embeddings_query[true_values_names, , drop = FALSE])
        true_values$class <- classes_q[true_values_names]
        true_values$type <- rep("labeled", length(true_values_names))
        colnames(true_values) <- c("x", "y", "class", "type")
      } else {
        true_values_names <- NULL
        true_values <- NULL
      }


      if (inc_unlabeled) {
        estimated_values_names <- setdiff(
          x = private$get_rownames_from_embeddings(embeddings_q),
          y = true_values_names
        )

        if (length(estimated_values_names) > 0L) {
          estimated_values <- as.data.frame(forward_results$results$embeddings_query[estimated_values_names, , drop = FALSE])
          # Get Classes
          class_labels <- forward_results$class_labels
          predictions_prob <- forward_results$results$predictions_prob[estimated_values_names, , drop = FALSE]
          predictions <- max.col(predictions_prob) - 1L
          # Transforming predictions to target levels------------------------------
          predictions <- as.character(as.vector(predictions))
          for (i in 0L:(length(class_labels) - 1L)) {
            predictions <- replace(
              x = predictions,
              predictions == as.character(i),
              values = class_labels[i + 1L]
            )
          }

          estimated_values$class <- predictions
          estimated_values$type <- rep("unlabeled", length(estimated_values_names))
          colnames(estimated_values) <- c("x", "y", "class", "type")
        }
      } else {
        estimated_values_names <- NULL
      }


      plot_data <- prototypes
      if (length(true_values) > 0L) {
        plot_data <- rbind(plot_data, true_values)
      }
      if (length(estimated_values_names) > 0L) {
        plot_data <- rbind(plot_data, estimated_values)
      }

      tmp_plot <- ggplot2::ggplot(data = plot_data) +
        ggplot2::geom_point(
          mapping = ggplot2::aes(
            x = x,
            y = y,
            color = class,
            shape = type,
            size = type,
            alpha = type
          )
        ) +
        ggplot2::scale_size_manual(values = c(
          prototype = size_points_prototypes,
          labeled = size_points,
          unlabeled = size_points
        )) +
        ggplot2::scale_alpha_manual(
          values = c(
            prototype = 1L,
            labeled = alpha,
            unlabeled = alpha
          )
        ) +
        ggplot2::scale_shape_manual(
          values = c(
            prototype = 17L,
            labeled = 16L,
            unlabeled = 15L
          )
        ) +
        ggplot2::theme_classic()

      if (inc_margin) {
        margin <- self$last_training$config$loss_margin
        # scaled_margin=margin*self$get_metric_scale_factor()
        if (!is.null(margin)) {
          if (private$model_config$metric_type == "Euclidean") {
            for (i in seq_len(nrow(prototypes))) {
              current_proto <- prototypes[i, ]
              tmp_plot <- tmp_plot + ggplot2::annotate(
                geom = "point",
                x = current_proto$x + margin * cos(seq(from = 0L, to = 2L * base::pi, length.out = 1000L)),
                y = current_proto$y + margin * sin(seq(from = 0L, to = 2L * base::pi, length.out = 1000L)),
              )
            }
          } else if (private$model_config$metric_type == "CosineDistance") {
            for (i in seq_len(nrow(prototypes))) {
              current_proto <- prototypes[i, ]

              # Transform to polar coordinates
              radius_r <- sqrt(current_proto$x^2 + current_proto$y^2)
              if (current_proto$y >= 0) {
                theta <- acos(current_proto$x / radius_r)
              } else {
                theta <- -acos(current_proto$x / radius_r)
              }

              # Find new theta
              # Margin of 2 correspondents to radian pi
              # Margin of 1 correspondents to radian pi/2
              # Margin of 0 correspondents to radian 0
              theta_new_upper <- theta + margin / 2 * pi
              theta_new_lower <- theta - margin / 2 * pi

              # Convert back to cartesian
              x_upper <- radius_r * cos(theta_new_upper)
              y_upper <- radius_r * sin(theta_new_upper)

              x_lower <- radius_r * cos(theta_new_lower)
              y_lower <- radius_r * sin(theta_new_lower)

              tmp_plot <- tmp_plot + ggplot2::annotate(
                geom = "segment",
                x = c(0.0, 0.0),
                y = c(0.0, 0.0),
                xend = c(x_upper, x_lower),
                yend = c(y_upper, y_lower)
              )
            }
          } else {
            warning("Margin for the metric type ", private$model_config$metric_type, " not implemented. Creating plot without margin.")
          }
        } else {
          warning("Last training has not provided a valid margin. Creating plot without margin.")
        }
      }
      return(tmp_plot)
    }
  ),
  private = list(
    #------------------------------------------------------------------------
    forward = function(embeddings_q,
                       classes_q = NULL,
                       embeddings_s = NULL,
                       classes_s = NULL,
                       batch_size = 32L,
                       ml_trace = 1L,
                       prediction_mode = TRUE) {
      # Check arguments
      check_class(object = embeddings_q, classes = c("EmbeddedText", "LargeDataSetForTextEmbeddings"), allow_NULL = FALSE)
      check_class(object = classes_q, classes = "factor", allow_NULL = TRUE)
      check_class(object = embeddings_s, classes = c("EmbeddedText", "LargeDataSetForTextEmbeddings"), allow_NULL = TRUE)
      check_class(object = classes_s, classes = "factor", allow_NULL = TRUE)
      check_type(object = batch_size, object_name = "batch_size", type = "int", FALSE)
      check_type(object = ml_trace, object_name = "ml_trace", type = "int", FALSE)
      check_type(object = prediction_mode, object_name = "prediction_mode", type = "bool")

      if ((is.null(embeddings_s) & !is.null(classes_s))) {
        warning("embeddings_s is not set but classes_s is set. Conitnue with setting both to NULL")
        classes_s <- NULL
      }
      if ((!is.null(embeddings_s) & is.null(classes_s))) {
        warning("embeddings_s is set but classes_s is not set. Conitnue with setting both to NULL")
        embeddings_s <- NULL
      }

      # Load Custom Model Scripts
      private$load_reload_python_scripts()

      # prepare embeddings
      embeddings_q <- private$prepare_embeddings_for_forward(embeddings_q, batch_size = batch_size)
      if (!is.null(embeddings_s)) {
        embeddings_s <- private$prepare_embeddings_for_forward(embeddings_s, batch_size = batch_size)
      }

      # Check number of cases in the data
      single_prediction <- private$check_single_prediction(embeddings_q)

      # Get current row names/name of the cases
      current_row_names <- private$get_rownames_from_embeddings(embeddings_q)

      # Prepare classes for sample
      if (!is.null(classes_s)) {
        class_freq_table <- table(na.omit(classes_s))
        class_freq_table <- subset(class_freq_table, class_freq_table > 0L)
        class_labels <- names(class_freq_table)

        classes_s <- as.character(classes_s)
        for (i in seq_along(class_labels)) {
          classes_s[classes_s == class_labels[i]] <- i
        }
        classes_s <- as.numeric(classes_s) - 1L
      } else {
        class_labels <- private$model_config$target_levels
        classes_s <- NULL
      }

      # Prepare classes for query
      if (!is.null(classes_q)) {
        classes_q <- as.character(classes_q)
        classes_q <- factor(x = classes_q, levels = class_labels)
        classes_q <- as.numeric(classes_q) - 1L
      } else {
        classes_q <- NULL
      }

      # Prepare data for pytorch
      if (!is.null(embeddings_s)) {
        embeddings_s <- embeddings_s$get_dataset()
        embeddings_s$set_format("torch")
      }

      if (!is.null(classes_s)) {
        classes_s <- torch$from_numpy(prepare_r_array_for_dataset(classes_s))
      }
      if (!is.null(classes_q)) {
        classes_q <- torch$from_numpy(prepare_r_array_for_dataset(classes_q))
      }

      # If at least two cases are part of the data set---------------------------
      if (!single_prediction) {
        prediction_data <- private$prepare_embeddings_as_dataset(embeddings_q)
        prediction_data$set_format("torch")

        results <- py$TeProtoNetClassifierBatchPredict(
          model = private$model,
          dataset = prediction_data,
          batch_size = as.integer(batch_size),
          embeddings_s = embeddings_s,
          classes_s = classes_s,
          prediction_mode = prediction_mode
        )

        # In the case the data has one single row-------------------------------
      } else {
        prediction_data <- torch$from_numpy(private$prepare_embeddings_as_np_array(embeddings_q))

        if (torch$cuda$is_available()) {
          device <- "cuda"
          dtype <- torch$double
        } else {
          device <- "cpu"
          dtype <- torch$float
        }

        prediction_data <- prediction_data$to(device, dtype = dtype)
        if (!is.null(classes_q)) {
          classes_q <- classes_q$to(device, dtype = dtype)
        }
        if (!is.null(classes_s)) {
          classes_s <- classes_s$to(device, dtype = dtype)
        }
        if (!is.null(embeddings_s)) {
          embeddings_s <- embeddings_s["input"][0L:embeddings_s$num_rows]
          embeddings_s=embeddings_s$to(device, dtype = dtype)
        }
        private$model$to(device, dtype = dtype)
        private$model$eval()
        results <- private$model(
          input_q = prediction_data$to(device, dtype = dtype),
          input_s = embeddings_s$to(device, dtype = dtype),
          classes_s = classes_s$to(device, dtype = dtype),
          prediction_mode = prediction_mode
        )
      }

      if (prediction_mode) {
        results <- tensor_to_numpy(results)
        rownames(results) <- current_row_names
        colnames(results) <- as.character(class_labels)
      } else {
        results <- tensor_list_to_numpy(results)

        predictions <- results[[1L]]
        rownames(predictions) <- current_row_names
        colnames(predictions) <- as.character(class_labels)

        distances <- results[[2L]]
        rownames(distances) <- current_row_names
        colnames(distances) <- as.character(class_labels)

        embeddings_query <- results[[4L]]
        rownames(embeddings_query) <- current_row_names

        prototype_embeddings <- results[[5L]]
        rownames(prototype_embeddings) <- as.character(class_labels)

        results <- list(
          predictions_prob = predictions,
          distances = distances,
          embeddings_query = embeddings_query,
          prototype_embeddings = prototype_embeddings
        )
      }

      return(list(
        results = results,
        class_labels = class_labels # ,
        # rownames_q = current_row_names
      ))
    },
    #-------------------------------------------------------------------------
    prepare_embeddings_for_forward = function(embeddings, batch_size) {
      # Check if the embeddings must be compressed before passing to the model
      requires_compression <- self$requires_compression(embeddings)

      # Check input for compatible text embedding models and feature extractors
      if (
        inherits(embeddings, "EmbeddedText") |
          inherits(embeddings, "LargeDataSetForTextEmbeddings")
      ) {
        self$check_embedding_model(text_embeddings = embeddings, require_compressed = FALSE)
      } else {
        private$check_embeddings_object_type(embeddings, strict = FALSE)
        if (requires_compression) {
          stop("Objects of class datasets.arrow_dataset.Dataset must be provided in
               compressed form.")
        }
      }

      # Convert to a LargeDataSetForTextEmbeddings
      if (inherits(embeddings, "EmbeddedText")) {
        embeddings <- embeddings$convert_to_LargeDataSetForTextEmbeddings()
      } else {
        embeddings <- embeddings
      }

      if (requires_compression) {
        # Returns a data set
        embeddings <- self$feature_extractor$extract_features_large(
          data_embeddings = embeddings,
          batch_size = as.integer(batch_size)
        )
      }

      return(embeddings)
    },
    #-------------------------------------------------------------------------
    set_random_prototypes = function() {
      n_row <- length(private$model_config$target_levels)
      n_col <- private$model_config$embedding_dim
      private$model$set_trained_prototypes(
        prototypes = torch$from_numpy(
          reticulate::np_array(
            matrix(
              nrow = n_row,
              ncol = n_col,
              data = rnorm(n = n_col * n_row, mean = 0L, sd = 1L)
            )
          )
        ),
        class_lables = reticulate::np_array(
          seq(
            from = 0L,
            to = (length(private$model_config$target_levels) - 1L)
          )
        )
      )
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

      # Reset model if requested
      if (reset_model) {
        private$create_reset_model()
      }

      # Set loss function
      loss_cls_fct_name <- "ProtoNetworkMargin"

      # Set target column
      if (!private$model_config$require_one_hot) {
        target_column <- "labels"
      } else {
        target_column <- "one_hot_encoding"
      }

      dataset_train <- train_data$select_columns(c("input", target_column))
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

       tmp_history <- py$TeClassifierTrainPrototype(
        model = private$model,
        loss_pt_fct_name = self$last_training$config$loss_pt_fct_name,
        optimizer_method = self$last_training$config$optimizer,
        lr_rate = self$last_training$config$lr_rate,
        lr_warm_up_ratio = self$last_training$config$lr_warm_up_ratio,
        lr_min=self$last_training$config$lr_min,
        scheduler_type=self$last_training$config$lr_scheduler,
        Ns = as.integer(self$last_training$config$Ns),
        Nq = as.integer(self$last_training$config$Nq),
        loss_alpha = self$last_training$config$loss_alpha,
        loss_margin = self$last_training$config$loss_margin,
        trace = as.integer(self$last_training$config$ml_trace),
        use_callback = use_callback,
        train_data = pytorch_train_data,
        val_data = pytorch_val_data,
        test_data = pytorch_test_data,
        epochs = as.integer(self$last_training$config$epochs),
        sampling_separate = self$last_training$config$sampling_separate,
        sampling_shuffle = self$last_training$config$sampling_shuffle,
        filepath = file.path(private$dir_checkpoint, "best_weights.pt"),
        n_classes = as.integer(length(private$model_config$target_levels)),
        log_dir = log_dir,
        log_write_interval = log_write_interval,
        log_top_value = log_top_value,
        log_top_total = log_top_total,
        log_top_message = log_top_message
      )

      # provide rownames and replace -100
      tmp_history <- private$prepare_history_data(tmp_history)

      return(tmp_history)
    }
  )
)
