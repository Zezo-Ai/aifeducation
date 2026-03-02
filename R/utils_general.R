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

#' Time stamp
#'
#' Function returns the time on the machine at the moment of calling.
#'
#' @return Returns a `string` with date and time in format "%y-%m-%d %H:%M:%S".
#'
#' @family Utils Developers
#' @export
get_time_stamp <- function() {
  return(
    as.character(format(Sys.time()), "%y-%m-%d %H:%M:%S")
  )
}

#' Transforming classes to one-hot encoding
#'
#' Function transforming a vector of classes (int) into
#' a binary class matrix.
#'
#' @param class_vector `vector` containing integers for every class. The
#' integers must range from 0 to n_classes-1.
#' @param n_classes `int` Total number of classes.
#' @return Returns a `matrix` containing the binary representation for
#' every class.
#'
#' @family Utils Developers
#' @export
to_categorical_c <- function(class_vector, n_classes) {
  binary_class_rep <- matrix(
    data = 0L,
    nrow = length(class_vector),
    ncol = n_classes
  )

  for (i in seq_along(class_vector)) {
    binary_class_rep[i, class_vector[i] + 1L] <- 1L
  }

  return(binary_class_rep)
}

#' @title Check if NULL or NA
#' @description Function for checking if an object is `NULL` or `NA`.
#'
#' @param object An object to test.
#' @return Returns `FALSE` if the object is not `NULL` and not `NA`. Returns `TRUE` in all other cases.
#'
#' @family Utils Developers
#' @keywords internal
#' @noRd
is.null_or_na <- function(object) {
  if (is.null(object)) {
    return(TRUE)
  } else {
    if (sum(is.na(object)) == length(object)) {
      return(TRUE)
    } else {
      return(FALSE)
    }
  }
}

#' @title Combine embeddings in array form
#' @description Function combines two array by adding an array to another array along the
#' first dimension.
#'
#' @param ... `array` or `list` of `array`s. Arrays to be combined. All array must have row names.
#'
#' @return Returns the combined array
#'
#' @family Utils Developers
#' @keywords internal
#' @noRd
array_form_bind <- function(...) {
  objects <- list(...)
  arrays <- NULL

  for (object in objects) {
    if (is.list(object)) {
      for (j in seq_len(length(object))) {
        arrays[length(arrays) + 1L] <- list(object[[j]])
      }
    } else {
      arrays[length(arrays) + 1L] <- list(object)
    }
  }

  # arrays <- list(...)
  if (length(arrays) > 1L) {
    total_rows <- 0L

    dimension <- dim(arrays[[1L]])

    for (i in seq_len(length(arrays))) {
      total_rows <- total_rows + dim(arrays[[i]])[1L]

      # Check number of dimensions
      if (sum(dim(arrays[[i]])[-1L] != dimension[-1L])) {
        stop("The dimensions of the array differ.")
      }
    }

    combined_array <- array(
      data = NA,
      dim = c(total_rows, dimension[2L], dimension[3L])
    )

    intercept <- 0L
    row_names <- NULL


    for (i in seq_len(length(arrays))) {
      index <- seq.int(
        from = 1L,
        to = nrow(arrays[[i]]),
        by = 1L
      ) + intercept

      combined_array[index, , ] <- arrays[[i]]

      intercept <- intercept + length(index)

      row_names <- append(x = row_names, rownames(arrays[[i]]))
    }

    rownames(combined_array) <- row_names
    return(combined_array)
  } else {
    return(arrays[[1L]])
  }
}


#' @title Generate ID suffix for objects
#' @description Function for generating an ID suffix for objects of class [TextEmbeddingModel],
#'   [TEClassifierRegular], and [TEClassifierProtoNet].
#'
#' @param length `int` determining the length of the id suffix.
#' @return Returns a `string` of the requested length.
#' @family Utils Developers
#' @export
generate_id <- function(length = 16L) {
  id_suffix <- NULL
  sample_values <- c(
    "a", "A",
    "b", "B",
    "c", "C",
    "d", "D",
    "e", "E",
    "f", "F",
    "g", "G",
    "h", "H",
    "i", "I",
    "j", "J",
    "k", "K",
    "l", "L",
    "m", "M",
    "n", "N",
    "o", "O",
    "p", "P",
    "q", "Q",
    "r", "R",
    "s", "S",
    "t", "T",
    "u", "U",
    "v", "V",
    "w", "W",
    "x", "X",
    "y", "Y",
    "z", "Z",
    seq(from = 0L, to = 9L, by = 1L)
  )


  id_suffix <- sample(
    x = sample_values,
    size = length,
    replace = TRUE
  )
  id_suffix <- paste(id_suffix, collapse = "")
  return(id_suffix)
}

#' @title Number of cores for multiple tasks
#' @description Function for getting the number of cores that should be used
#' for parallel processing of tasks. The number of cores is set to 75 % of the
#' available cores. If the environment variable `CI` is set to `"true"` or if the
#' process is running on cran `2` is returned.
#'
#' @importFrom parallel detectCores
#'
#' @return Returns `int` as the number of cores.
#'
#' @family Utils Developers
#' @export
auto_n_cores <- function() {
  if (
    Sys.getenv("CI") == "true" ||
      Sys.getenv("NOT_CRAN") == "true" ||
      Sys.getenv("_R_CHECK_LIMIT_CORES_") == "true"
  ) {
    n_cores <- min(2L, parallel::detectCores())
  } else {
    n_cores <- floor(parallel::detectCores() * 0.75)
  }
  return(n_cores = max(1L, n_cores))
}

#' @title Detect base model's architecture
#' @description Function for detecting the base type of a base model.
#'
#' @param model A model of the transformer library.
#'
#' @return Returns a `string` describing base model's architecture.
#'
#' @family Utils Developers
#' @keywords internal
#' @noRd
detect_base_model_type <- function(model) {
  if (inherits(
    x=model, what=c(
      "transformers.configuration_utils.PretrainedConfig",
      "transformers.configuration_utils.PreTrainedConfig"
      )
    )
    ) {
    type_string <- model$architectures
  } else {
    type_string <- model$config
  }

  if (stringi::stri_detect(str = tolower(type_string), regex = "^funnel([:alnum:]*)")) {
    return("funnel")
  } else if (stringi::stri_detect(str = tolower(type_string), regex = "^bert([:alnum:]*)")) {
    return("bert")
  } else if (stringi::stri_detect(str = tolower(type_string), regex = "^debertav2([:alnum:]*)")) {
    return("deberta_v2")
  } else if (stringi::stri_detect(str = tolower(type_string), regex = "^mpnet([:alnum:]*)")) {
    return("mpnet")
  } else if (stringi::stri_detect(str = tolower(type_string), regex = "^longformer([:alnum:]*)")) {
    return("longformer")
  } else if (stringi::stri_detect(str = tolower(type_string), regex = "^roberta([:alnum:]*)")) {
    return("roberta")
  } else if (stringi::stri_detect(str = tolower(type_string), regex = "^modernbert([:alnum:]*)")) {
    return("modernbert")
  } else {
    stop("Architecture for the model could not be detected.")
  }
}
