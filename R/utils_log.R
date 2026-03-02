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


#' @title Write log
#' @description Function for writing a log file from R containing three rows and three columns. The log file can report
#'   the current status of maximal three processes. The first row describes the top process. The second row describes
#'   the status of the process within the top process. The third row can be used to describe the status of a process
#'   within the middle process.
#'
#'   The log can be read with [read_log].
#'
#' @param log_file `string` Path to the file where the log should be saved and updated.
#' @param value_top `double` Current value for the top process.
#' @param total_top `double` Maximal value for the top process.
#' @param message_top `string` Message describing the current state of the top process.
#' @param value_middle `double` Current value for the middle process.
#' @param total_middle `double` Maximal value for the middle process.
#' @param message_middle `string` Message describing the current state of the middle process.
#' @param value_bottom `double` Current value for the bottom process.
#' @param total_bottom `double` Maximal value for the bottom process.
#' @param message_bottom `string` Message describing the current state of the bottom process.
#' @param last_log `POSIXct` Time when the last log was created. If there is no log file set this value to `NULL`.
#' @param write_interval `int` Time in seconds. This time must be past before a new log is created.
#'
#' @return This function writes a log file to the given location. If `log_file` is `NULL` the function will not try to
#'   write a log file.
#' @return If `log_file` is a valid path to a file the function will write a log if the time specified by
#'   `write_interval` has passed. In addition the function will return an object of class `POSIXct` describing the time
#'   when the log file was successfully updated. If the initial attempt for writing log fails the function returns the
#'   value of `last_log` which is `NULL` by default.
#'
#' @family Utils Log Developers
#' @export
#'
write_log <- function(log_file,
                      value_top = 0L, total_top = 1L, message_top = NA,
                      value_middle = 0L, total_middle = 1L, message_middle = NA,
                      value_bottom = 0L, total_bottom = 1L, message_bottom = NA,
                      last_log = NULL, write_interval = 2L) {
  if (is.null(log_file)) {
    return(NULL)
  }

  try_write_log_data <- function() {
    log_data <- rbind(
      c(value_top, total_top, message_top),
      c(value_middle, total_middle, message_middle),
      c(value_bottom, total_bottom, message_bottom)
    )
    colnames(log_data) <- c("value", "total", "message")

    log <- try(write.csv(x = log_data, file = log_file, row.names = FALSE))

    if (!inherits(log, "try-error")) {
      return(Sys.time())
    } else {
      return(last_log)
    }
  }

  if (
    value_top %in% c(1L, total_top) ||
      value_middle %in% c(1L, total_middle) ||
      value_bottom %in% c(1L, total_bottom)
  ) {
    # if this is the first or last iteration
    return(try_write_log_data())
  } else {
    if (!is.null(last_log)) {
      diff <- difftime(Sys.time(), last_log, units = "secs")[[1L]]
      if (diff > write_interval) {
        return(try_write_log_data())
      }
    } else {
      return(try_write_log_data())
    }
  }
}

#' Function for reading a log file in R
#'
#' This function reads a log file at the given location. The log file should be
#' created with [write_log].
#'
#' @param file_path `string` Path to the log file.
#'
#' @return Returns a matrix containing the log file.
#'
#' @importFrom utils read.csv
#' @family Utils Log Developers
#' @export
#'
read_log <- function(file_path) {
  res <- NULL
  if (!is.null_or_na(file_path)) {
    log_file <- try(utils::read.csv(file_path))
    if (!inherits(log_file, "try-error")) res <- log_file
  }
  return(res)
}

#' Function that resets a log file.
#'
#' This function writes a log file with default values. The file can be read with
#' [read_log].
#'
#' @param log_path `string` Path to the log file.
#'
#' @return Function does nothing return. It is used to write an "empty" log file.
#'
#' @family Utils Log Developers
#' @export
#'
reset_log <- function(log_path) {
  if (is.null(log_path)) {
    return(invisible())
  }

  log_data <- rbind(
    c(0L, 1L, NA),
    c(0L, 1L, NA),
    c(0L, 1L, NA)
  )
  colnames(log_data) <- c("value", "total", "message")

  try(write.csv(x = log_data, file = log_path, row.names = FALSE))
}

#' Function for reading a log file containing a record of the loss during training.
#'
#' This function reads a log file that contains values for every epoch for the loss.
#' The values are grouped for training and validation data. The log contains
#' values for test data if test data was available during training.
#'
#' In general the loss is written by a python function during model's training.
#'
#' @param path_loss `string` Path to the log file.
#' @return Function returns a `matrix` that contains two or three row depending on
#' the data inside the loss log. In the case of two rows the first represents the
#' training data and the second the validation data. In the case of three rows
#' the third row represents the values for test data. All Columns represent the
#' epochs.
#'
#' @family Utils Log Developers
#' @export
#'
read_loss_log <- function(path_loss) {
  if (!file.exists(path_loss)) {
    return(NULL)
  }

  loss_data <- try(
    utils::read.table(file = path_loss, sep = ",", header = FALSE),
    silent = TRUE
  )

  if (!(inherits(loss_data, "try-error"))) {
    loss_data <- t(loss_data)
    if (ncol(loss_data) > 2L) {
      colnames(loss_data) <- c("train", "validation", "test")
    } else {
      colnames(loss_data) <- c("train", "validation")
    }

    loss_data <- as.data.frame(loss_data)
    for (i in seq_len(ncol(loss_data))) {
      loss_data[, i] <- as.numeric(loss_data[, i])
    }
    loss_data$epoch <- seq.int(
      from = 1L,
      to = nrow(loss_data),
      by = 1L
    )
  }

  return(loss_data)
}

#' Reset log for loss information
#'
#' This function writes an empty log file for loss information.
#'
#' @param log_path `string` Path to the log file.
#' @param epochs `int` Number of epochs for the complete training process.
#'
#' @return Function does nothing return. It writes a log file at the given location.
#' The file is a .csv file that contains three rows. The first row takes the
#' value for the training, the second for the validation, and the third row for the
#' test data. The columns represent epochs.
#'
#' @family Utils Log Developers
#' @export
#'
reset_loss_log <- function(log_path, epochs) {
  if (is.null(log_path)) {
    return(invisible())
  }

  log_data <- rbind(
    rep(-100L, times = epochs),
    rep(-100L, times = epochs),
    rep(-100L, times = epochs)
  )

  try(
    utils::write.table(
      x = log_data, file = log_path,
      row.names = FALSE,
      col.names = FALSE,
      sep = ","
    ),
    silent = TRUE
  )
}

#' @title Print message
#' @description Prints a message `msg` if `trace` parameter is `TRUE` with current date with `message()` or `cat()`
#'   function.
#'
#' @param msg `string` Message that should be printed.
#' @param trace `bool` Silent printing (`FALSE`) or not (`TRUE`).
#' @param msg_fun `bool` value that determines what function should be used. `TRUE` for `message()`, `FALSE` for
#'   `cat()`.
#'
#' @return This function returns nothing.
#' @family Utils Log Developers
#' @export
output_message <- function(msg, trace, msg_fun) {
  fun <- ifelse(msg_fun, message, cat)
  if (trace) fun(paste(get_time_stamp(), msg))
}

#' @title Print message (`message()`)
#' @description Prints a message `msg` if `trace` parameter is `TRUE` with current date with `message()` function.
#'
#' @param msg `string` Message that should be printed.
#' @param trace `bool` Silent printing (`FALSE`) or not (`TRUE`).
#'
#' @return This function returns nothing.
#' @family Utils Log Developers
#' @export
print_message <- function(msg, trace) {
  output_message(msg, trace, TRUE)
}

#' @title Print message  (`cat()`)
#' @description Prints a message `msg` if `trace` parameter is `TRUE` with current date with `cat()` function.
#'
#' @param msg `string` Message that should be printed.
#' @param trace `bool` Silent printing (`FALSE`) or not (`TRUE`).
#'
#' @return This function returns nothing.
#' @family Utils Log Developers
#' @export
cat_message <- function(msg, trace) {
  output_message(msg, trace, FALSE)
}

#' @title Clean pytorch log of transformers
#' @description Function for preparing and cleaning the log created by an object of class Trainer from the python
#'   library 'transformer's.
#'
#' @param log `data.frame` containing the log.
#' @return Returns a `data.frame` containing epochs, loss, and val_loss.
#' @importFrom reticulate import_builtins
#' @family Utils Log Developers
#' @export
clean_pytorch_log_transformers <- function(log) {
  if (is.data.frame(log)){
    history_data <- log
  } else if (inherits(x=log,what = "pandas.DataFrame")){
    python_bulit_ins <- reticulate::import_builtins()
    data_colnames <- python_bulit_ins$list(log$columns)
    history_data <- log$to_numpy()
    colnames(history_data) <- data_colnames
    history_data <- as.data.frame(history_data)
  } else {
    stop("Class of log is not supported.")
  }

  max_epochs <- max(history_data$epoch)

  cols <- c("epoch", "loss", "val_loss")

  cleaned_log <- matrix(
    data = NA,
    nrow = max_epochs,
    ncol = length(cols)
  )
  colnames(cleaned_log) <- cols
  for (i in 1L:max_epochs) {
    cleaned_log[i, "epoch"] <- i

    tmp_loss <- subset(history_data, history_data$epoch == i & !is.na(history_data$loss))
    tmp_loss <- tmp_loss[1L, "loss"]
    cleaned_log[i, "loss"] <- tmp_loss

    tmp_val_loss <- subset(history_data, history_data$epoch == i & !is.na(history_data$eval_loss))
    tmp_val_loss <- tmp_val_loss[1L, "eval_loss"]
    cleaned_log[i, "val_loss"] <- tmp_val_loss
  }
  return(as.data.frame(cleaned_log))
}
