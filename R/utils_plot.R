# This file is part of the R package "aifeducation".
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as published by
# the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>


#' @title Point of best state during training
#' @description Function searches for the best point during training according to chosen metric.
#' For selecting the best state of the model the values for the *validation* data set are used.
#' @param plot_data `matrix` containing the training history in an aggregated format. In most cases
#' the matrix is generated with the private method `prepare_history_data`. If available the function
#' expects the matrix stored in `$aggregated`.
#' @param `string` determining the measure the point should refer to.
#' @returns Returns a `list` with the epoch and the corresponding value representing
#' the best state during training.
#' @family Utils Plots Developers
#' @keywords internal
#' @noRd
get_best_state_point <- function(plot_data, measure) {
  if (measure == "loss") {
    optim <- "min"
  } else {
    optim <- "max"
  }

  if ("val_loss" %in% colnames(plot_data)) {
    selected_column <- "val_loss"
  } else {
    selected_column <- "validation_mean"
  }

  if (optim == "min") {
    best_val_measure <- min(plot_data[, selected_column])
    best_model_epoch <- which(x = (plot_data[, selected_column] == best_val_measure))[1]
  } else {
    best_val_measure <- max(plot_data[, selected_column])
    best_model_epoch <- which(x = (plot_data[, selected_column] == best_val_measure))[1]
  }

  return(
    list(
      epoch = best_model_epoch,
      value = best_val_measure
    )
  )
}

#' @title Point of final state during training
#' @description Function searches for the point during training according to chosen metric that represent
#' the final state of the model. Please note that this state is determined by several metrics. Thus, this point
#' can differ from the best point during training according to the chosen metric.
#' For selecting this state the values for the *validation* data set are used.
#' @param plot_data `matrix` containing the training history in an aggregated format. In most cases
#' the matrix is generated with the private method `prepare_history_data`. If available the function
#' expects the matrix stored in `$aggregated`.
#' @param `string` determining the measure the point should refer to.
#' @returns Returns a `list` with the epoch and the corresponding value representing
#' the final state of a model during training.
#' @note During training the model state with the best value for average iota is used.
#' In the case that there are several states with these values the state with the best balanced
#' accuracy is used. If there are still multiple values the state with the best loss is selected
#' at the highest number of epochs.
#' @family Utils Plots Developers
#' @keywords internal
#' @noRd
get_used_state_point <- function(plot_data, measure) {
  values_avg_iota <- plot_data[["avg_iota"]][, "validation_mean"]
  values_bbc <- plot_data[["balanced_accuracy"]][, "validation_mean"]
  values_loss <- plot_data[["loss"]][, "validation_mean"]
  values_epoch <- plot_data[["loss"]][, "epoch"]
  values_acc <- plot_data[["accuracy"]][, "validation_mean"]

  complete_values <- cbind(values_epoch, values_avg_iota, values_bbc, values_loss, values_acc)
  colnames(complete_values) <- c("epoch", "avg_iota", "balanced_accuracy", "loss", "accuracy")

  best_values <- subset(complete_values, subset = complete_values[, 2L] == max(complete_values[, 2L]))
  best_values <- subset(best_values, subset = (best_values[, 3L] == max(best_values[, 3L])))
  best_values <- subset(best_values, subset = (best_values[, 4L] == min(best_values[, 4L])))

  used_state_epoch <- as.numeric(best_values[nrow(best_values), 1])
  value_state_epoch <- as.numeric(complete_values[which(complete_values[, "epoch"] == used_state_epoch), measure])
  return(
    list(
      epoch = used_state_epoch,
      value = value_state_epoch
    )
  )
}


#' @title Point of best state during training
#' @description Function searches for the best point during training according to chosen metric.
#' For selecting the best state of the model the values for the *validation* data set are used.
#' In contrast to the function `get_best_state_point` this functions searches of the best point within every fold.
#' @param plot_data `matrix` containing the training history for all folds. In most cases
#' the matrix is generated with the private method `prepare_history_data`. If available the function
#' expects the matrix stored in `$folds`.
#' @param `string` determining the measure the point should refer to.
#' @returns Returns a `list` with the epochs and the corresponding values as `vector`s representing
#' the best state during training within each fold.
#' @family Utils Plots Developers
#' @keywords internal
#' @noRd
get_best_states_from_folds <- function(data_folds, measure) {
  selected_data <- data_folds[[measure]]$folds_val
  x_values <- vector(length = ncol(selected_data))
  y_values <- vector(length = ncol(selected_data))
  if (measure == "loss") {
    optim <- "min"
  } else {
    optim <- "max"
  }
  for (fold in seq_along(x_values)) {
    if (optim == "min") {
      y_value <- min(selected_data[, fold])
      x_value <- which(selected_data[, fold] == y_value)[1]
      x_values[fold] <- x_value
      y_values[fold] <- y_value
    } else {
      y_value <- max(selected_data[, fold])
      x_value <- which(selected_data[, fold] == y_value)[1]
      x_values[fold] <- x_value
      y_values[fold] <- y_value
    }
  }
  return(
    list(
      epochs = x_values,
      values = y_values
    )
  )
}

#' @title Point of final state during training
#' @description Function searches for the point during training according to chosen metric that represent
#' the final state of the model. Please note that this state is determined by several metrics. Thus, this point
#' can differ from the best point during training according to the chosen metric.
#' For selecting this state the values for the *validation* data set are used.
#' In contrast to the function `get_used_state_point` this functions searches of the point within every fold.
#' @param plot_data `matrix` containing the training history in an aggregated format. In most cases
#' the matrix is generated with the private method `prepare_history_data`. If available the function
#' expects the matrix stored in `$folds`.
#' @param `string` determining the measure the point should refer to.
#' @returns Returns a `list` with the epochs and the corresponding values as `vector`s representing
#' the final state of a model during training.
#' @note During training the model state with the best value for average iota is used.
#' In the case that there are several states with these values the state with the best balanced
#' accuracy is used. If there are still multiple values the state with the best loss is selected
#' at the highest number of epochs.
#' @family Utils Plots Developers
#' @keywords internal
#' @noRd
get_selected_states_from_folds <- function(data_folds, measure) {
  n_folds <- ncol(data_folds[[measure]]$folds_val)
  x_values <- vector(length = n_folds)
  y_values <- vector(length = n_folds)

  values_epoch <- seq.int(from = 1L, to = nrow(data_folds[["loss"]]$folds_val), by = 1L)

  for (i in seq_along(x_values)) {
    values_avg_iota <- data_folds[["avg_iota"]]$folds_val[, i]
    values_bbc <- data_folds[["balanced_accuracy"]]$folds_val[, i]
    values_loss <- data_folds[["loss"]]$folds_val[, i]
    values_acc <- data_folds[["accuracy"]]$folds_val[, i]

    complete_values <- cbind(values_epoch, values_avg_iota, values_bbc, values_loss, values_acc)
    colnames(complete_values) <- c("epoch", "avg_iota", "balanced_accuracy", "loss", "accuracy")

    best_values <- subset(complete_values, subset = complete_values[, 2L] == max(complete_values[, 2L]))
    best_values <- subset(best_values, subset = (best_values[, 3L] == max(best_values[, 3L])))
    best_values <- subset(best_values, subset = (best_values[, 4L] == min(best_values[, 4L])))

    used_state_epoch <- as.numeric(best_values[nrow(best_values), 1])
    value_state_epoch <- as.numeric(complete_values[which(complete_values[, "epoch"] == used_state_epoch), measure])

    x_values[i] <- used_state_epoch
    y_values[i] <- value_state_epoch
  }

  return(
    list(
      epochs = x_values,
      values = y_values
    )
  )
}

#' @title Add a point to a plot
#' @description Function adds a point to a plot.
#' @param plot_object A 'ggplot2' plot object.
#' @param x `float` or `vector` of number representing the x coordinates of the points.
#' @param y `float` or `vector` of number representing the y coordinates of the points.
#' @param type `string` Type of point. `type="segment"` for `ggplot2::geom_segment` or
#' `type="point"` for `ggplot2::geom_point`.
#' @param state `string` For `type="segment"` the `linetype`. For `type="point"` the `shape`.
#' Allowed values are `"Best"` for the best state according to a metric and `"Final"` for the
#' final state of the model.
#' @returns Returns the plot with the point added.
#' @note In the case of `type="segment"` `x` and `y` must be single numbers. In the case
#' of `type="point"` `x` and `y` can be `vector`s of numbers with the same length.
#' @family Utils Plots Developers
#' @keywords internal
#' @noRd
add_point <- function(plot_object, x, y, type = "segment", state = "best") {
  if (type == "segment") {
    plot_object <- plot_object + ggplot2::geom_segment(
      ggplot2::aes(
        x = .data$segment_x,
        xend = .data$segment_x_end,
        y = .data$segment_y,
        yend = .data$segment_y_end,
        linetype = state
      ),
      data = data.frame(
        segment_x = c(0L, as.numeric(x)),
        segment_x_end = c(as.numeric(x), as.numeric(x)),
        segment_y = c(as.numeric(y), 0.0),
        segment_y_end = c(as.numeric(y), as.numeric(y))
      )
    )
  } else {
    plot_object <- plot_object + ggplot2::geom_point(
      ggplot2::aes(
        x = .data$segment_x,
        y = .data$segment_y,
        shape = state
      ),
      data = data.frame(
        segment_x = as.numeric(x),
        segment_y = as.numeric(y)
      ),
      # shape = state,
      size = 3L
    )
  }

  return(plot_object)
}

#' @title Add breaks to a plot
#' @description Function adds specific breaks to a plot. The breaks always contain the
#' values passed with `special_x` and `special_y`. The breaks are even distributed but
#' values to close to `special_x` or `special_y` are removed.
#' @param plot_object A 'ggplot2' plot object.
#' @param x_min Minimal value for the x-axis.
#' @param x_max Maximal value for the x-axis.
#' @param y_min Minimal value for the y-axis.
#' @param y_max Maximal value for the y-axis.
#' @param special_x `vector` of coordinates for the x-axis that should be added to the breaks.
#' @param special_y `vector` of coordinates for the y-axis that should be added to the breaks.
#' @returns Returns the plot with the generated breaks.
#' @family Utils Plots Developers
#' @keywords internal
#' @noRd
add_breaks <- function(plot_object, x_min, x_max, y_min, y_max, special_x = NULL, special_y = NULL) {
  if ((x_max - x_min - 1) < x_max && (x_max - x_min - 1) > 0) {
    x_seq <- seq.int(from = x_min - 1, to = x_max, by = ceiling((x_max - x_min - 1) / 10))
  } else {
    x_seq <- NULL
  }
  breaks_x <- setdiff(
    x = c(
      x_min,
      x_seq,
      x_max
    ),
    y = 0L
  )

  breaks_y <- c(
    seq(from = y_min, to = y_max, by = ((y_max - y_min) / 5)),
    y_max
  )

  if (!is.null(special_x)) {
    distances_x=(max(breaks_x)-min(breaks_x))/10
    distances_x=distances_x*0.10
    for (sp_x in special_x) {
      breaks_x <- breaks_x[(breaks_x-distances_x) > sp_x & (breaks_x+ distances_x)< sp_x]
    }
    breaks_x <- c(breaks_x, special_x)
  }

  if (!is.null(special_y)) {
    distances_y=(max(breaks_y)-min(breaks_y))/5
    distances_y=distances_y*0.10
    for (sp_y in special_y) {
      breaks_y <- breaks_y[(breaks_y-distances_y) > sp_y  & (breaks_y+distances_y) < sp_y]
    }
    breaks_y <- c(breaks_y, special_y)
  }

  plot_object <- plot_object +
    ggplot2::scale_x_continuous(breaks = breaks_x) +
    ggplot2::scale_y_continuous(breaks = breaks_y)
  return(plot_object)
}
