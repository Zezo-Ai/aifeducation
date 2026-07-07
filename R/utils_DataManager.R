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


#' @title Create tasks for generating synthetic cases
#' @description This function creates a valid list of tasks for generating synthetic cases. The
#' result of this function should be used within the function `get_synthetic_cases_from_matrix`.
#' @param target Named `factor` containing the labels of the corresponding embeddings.
#' @param sequence_length `int` Length of the text embedding sequences.
#' @param method `vector` containing strings of the requested methods for generating new cases. Currently
#'   "knnor" from this package is available.
#' @param min_k `int` The minimal number of nearest neighbors during sampling process.
#' @param max_k `int` The maximum number of nearest neighbors during sampling process.
#' @return `list` with the following components:
#'   * `cat`: `string` Category/class to generate synthetic cases for.
#'   * `required_cases`: `int` Number of synthetic cases to generate.
#'   * `k`: `int` Number of neighbors used for generating synthetic cases.
#'   * `selected_cases`: `vector` of `int` representing the indices of cases that should be used.
#'   * `chunks`: `int` Sequence length for which the synthetic cases are generated.
#'
#' @family Utils Developers
#' @keywords internal
#' @noRd
create_sc_tasks_and_config <- function(sequence_length, target, max_k, min_k) {
  input <- list()
  # get possible seq lengths in order to group the cases by sequence length
  seq_length_categories <- as.numeric(names(table(sequence_length)))

  # Create tasks for every group of sequence lengths
  for (current_seq_length in seq_length_categories) {
    condition <- (sequence_length == current_seq_length)
    idx <- which(condition)
    cat_freq <- table(target[idx])
    categories <- names(cat_freq)
    max_freq <- max(cat_freq)

    for (cat in categories) {
      if (cat_freq[cat] > 4L && cat_freq[cat] < max_freq) {
        # Check k and adjust if necessary
        n_neighbors <- cat_freq[cat] - 2L

        if (n_neighbors <= max_k) {
          max_k_final <- n_neighbors
          if (min_k > max_k_final) {
            min_k_final <- max_k_final
          } else {
            min_k_final <- min_k
          }
        } else {
          max_k_final <- max_k
          min_k_final <- min_k
        }

        max_k_final <- as.numeric(max_k_final)
        min_k_final <- as.numeric(min_k_final)

        # calculate required cases
        n_k <- max_k_final - min_k_final + 1L
        required_cases_vector <- vector(length = n_k)
        required_cases_vector[] <- 0L
        required_cases_total <- max_freq - cat_freq[cat]
        required_cases_per_n_k <- floor(required_cases_total / n_k)
        residual <- required_cases_total - required_cases_per_n_k * n_k
        for (i in seq_len(n_k)) {
          if (residual > 0L) {
            required_cases_vector[i] <- required_cases_per_n_k + 1L
            residual <- residual - 1L
          } else {
            required_cases_vector[i] <- required_cases_per_n_k
          }
        }
        if (sum(required_cases_vector) != required_cases_total) {
          stop("Error in required_cases_vector.")
        }

        ids_to_small <- which(required_cases_vector <= 1L)
        sum_to_small <- sum(required_cases_vector[ids_to_small])
        if (sum(required_cases_vector > 1L) == 0L) {
          valid_ids <- seq.int(from = 1L, to = max(1L, floor(sum_to_small / 2L)))
        } else {
          valid_ids <- which(required_cases_vector > 1L)
        }
        ids_to_small <- setdiff(x = ids_to_small, y = valid_ids)
        sum_to_small <- sum(required_cases_vector[ids_to_small])

        cases_per_valid <- floor(sum_to_small / length(valid_ids))
        residual <- sum_to_small - cases_per_valid * length(valid_ids)
        for (vid in valid_ids) {
          if (residual > 0L) {
            required_cases_vector[vid] <- required_cases_vector[vid] + cases_per_valid + 1L
            residual <- residual - 1L
          } else {
            required_cases_vector[vid] <- required_cases_vector[vid] + cases_per_valid
          }
        }
        required_cases_vector[ids_to_small] <- 0L

        if (sum(required_cases_vector) != required_cases_total) {
          stop("Error in required_cases_vector.")
        }

        ids <- which(required_cases_vector > 1L)
        for (id in ids) {
          input[[length(input) + 1L]] <- list(
            cat = as.character(cat),
            required_cases = required_cases_vector[id],
            k = min_k_final + id - 1L,
            selected_cases = idx,
            chunks = current_seq_length
          )
        }
      }
    }
  }
  return(input)
}


#-----------------------------------------------------------------------------
#' @title Create synthetic cases for balancing training data
#' @description This function creates synthetic cases for balancing the training with classifier models.
#' @param matrix_form Named `matrix` containing the text embeddings in a matrix form.
#' @param target Named `factor` containing the labels of the corresponding embeddings.
#' @param times `int` for the number of sequences/times.
#' @param features `int` for the number of features within each sequence.
#' @param sequence_length `int` Length of the text embedding sequences.
#' @param method `vector` containing strings of the requested methods for generating new cases. Currently
#'   "knnor" from this package is available.
#' @param min_k `int` The minimal number of nearest neighbors during sampling process.
#' @param max_k `int` The maximum number of nearest neighbors during sampling process.
#' @param pad_value `int` Value for indicating padding.
#' @return `list` with the following components:
#'   * `syntetic_embeddings`: Named `data.frame` containing the text embeddings of the synthetic cases.
#'   * `syntetic_targets`: Named `factor` containing the labels of the corresponding synthetic cases.
#'   * `n_syntetic_units`: `table` showing the number of synthetic cases for every label/category.
#'
#' @family Utils Developers
#'
#' @export
#' @import foreach
#' @import doParallel
get_synthetic_cases_from_matrix <- function(matrix_form,
                                            times,
                                            features,
                                            target,
                                            sequence_length,
                                            method = "knnor",
                                            min_k = 1L,
                                            max_k = 6L,
                                            pad_value = -100L) {
  input <- create_sc_tasks_and_config(
    sequence_length = sequence_length,
    target = target,
    min_k = min_k,
    max_k = max_k
  )

  index <- 1
  result_list <- foreach::foreach(
    index = seq_len(length(input)),
    .export = "create_synthetic_units_from_matrix",
    .errorhandling = "pass"
  ) %dopar% {
    tmp_results <- create_synthetic_units_from_matrix(
      matrix_form = matrix_form[
        input[[index]]$selected_cases,
        c(1L:(input[[index]]$chunks * features))
      ],
      target = target[input[[index]]$selected_cases],
      required_cases = input[[index]]$required_cases,
      k = input[[index]]$k,
      method = method,
      cat = input[[index]]$cat
    )
  }

  # get number of synthetic cases
  n_syn_cases <- 0L
  for (i in seq_len(length(result_list))) {
    if (!is.null(result_list[[i]]$syntetic_embeddings)) {
      n_syn_cases <- n_syn_cases + nrow(result_list[[i]]$syntetic_embeddings)
    }
  }

  syntetic_embeddings <- matrix(
    data = pad_value,
    nrow = n_syn_cases,
    ncol = times * features
  )
  colnames(syntetic_embeddings) <- colnames(matrix_form)
  syntetic_embeddings <- as.data.frame(syntetic_embeddings)
  syntetic_targets <- NULL

  n_row <- 0L
  names_vector <- NULL
  for (i in seq_len(length(result_list))) {
    if (!is.null(result_list[[i]]$syntetic_embeddings)) {
      syntetic_embeddings[
        (n_row + 1L):(n_row + nrow(result_list[[i]]$syntetic_embeddings)),
        c(seq_len(ncol(result_list[[i]]$syntetic_embeddings)))
      ] <- result_list[[i]]$syntetic_embeddings[, c(seq_len(ncol(result_list[[i]]$syntetic_embeddings)))]
      syntetic_targets <- append(syntetic_targets, values = result_list[[i]]$syntetic_targets)
      n_row <- n_row + nrow(result_list[[i]]$syntetic_embeddings)
      names_vector <- append(
        x = names_vector,
        values = rownames(result_list[[i]]$syntetic_embeddings)
      )
    }
  }

  # Transform matrix back to array
  syntetic_embeddings <- matrix_to_array_c(
    matrix = as.matrix(syntetic_embeddings),
    times = times,
    features = features
  )
  rownames(syntetic_embeddings) <- names_vector

  n_syntetic_units <- table(syntetic_targets)

  results <- NULL
  results["syntetic_embeddings"] <- list(syntetic_embeddings)
  results["syntetic_targets"] <- list(syntetic_targets)
  results["n_syntetic_units"] <- list(n_syntetic_units)
  return(results)
}

#---------------------------------------------
#' @title Create synthetic units
#' @description Function for creating synthetic cases in order to balance the data for training with
#'   [TEClassifierRegular] or [TEClassifierProtoNet]]. This is an auxiliary function for use with
#'   [get_synthetic_cases_from_matrix] to allow parallel computations.
#'
#' @param matrix_form Named `matrix` containing the text embeddings in matrix form. In most cases this object is taken
#'   from [EmbeddedText]$embeddings.
#' @param target Named `factor` containing the labels/categories of the corresponding cases.
#' @param required_cases `int` Number of cases necessary to fill the gab between the frequency of the class under
#'   investigation and the major class.
#' @param k `int` The number of nearest neighbors during sampling process.
#' @param method `vector` containing strings of the requested methods for generating new cases. Currently
#'   "knnor" from this package is available.
#' @param cat `string` The category for which new cases should be created.
#' @return Returns a `list` which contains the text embeddings of the new synthetic cases as a named `data.frame` and
#'   their labels as a named `factor`.
#'
#' @family Utils Developers
#'
#' @export
create_synthetic_units_from_matrix <- function(matrix_form,
                                               target,
                                               required_cases,
                                               k,
                                               method,
                                               cat) {
  # Transform to a binary problem
  tmp_target <- as.numeric((target == cat))
  if (length(tmp_target) != nrow(matrix_form)) {
    stop("Number of labels and number of embeddings do not match.")
  }
  if (anyNA(tmp_target)) {
    stop("Labels contain NA.")
  }
  if (anyNA(matrix_form)) {
    stop("Labels contain NA.")
  }
  if (!is.numeric(matrix_form)) {
    stop("matrix_form must be numeric")
  }
  if (!is.character(cat)) {
    stop("cat must be of type character")
  }


  syn_data <- NULL
  if (method == "knnor") {
    syn_data <- try(
      knnor(
        dataset = list(
          embeddings = matrix_form,
          labels = tmp_target
        ),
        k = as.integer(k),
        aug_num = as.integer(required_cases),
        cycles_number_limit = 5000L
      ),
      silent = TRUE
    )
  }

  if (
    !inherits(x = syn_data, what = "try-error") &&
      (!is.null(syn_data) || nrow(syn_data$syn_data) > 0L)
  ) {
    if (nrow(syn_data) != required_cases) {
      stop("Number or requestes cases could not be generated.")
    }

    n_cols_embedding <- ncol(matrix_form)
    syn_data <- syn_data
    rownames(syn_data) <- paste0(
      method, "_", cat, "_", k, "_", n_cols_embedding, "_",
      seq(from = 1L, to = nrow(syn_data), by = 1L)
    )
    syn_data <- as.data.frame(syn_data)
    tmp_target <- rep(cat, times = nrow(syn_data))
    names(tmp_target) <- rownames(syn_data)

    results <- list(
      syntetic_embeddings = syn_data,
      syntetic_targets = tmp_target
    )
  } else {
    results <- list(
      syntetic_embeddings = NULL,
      syntetic_targets = NULL
    )
  }
  return(results)
}

#------------------------------------------------------------------------------
#' @title Function for splitting data into a train and validation sample
#' @description This function creates a train and validation sample based on stratified random sampling. The relative
#'   frequencies of each category in the train and validation sample equal the relative frequencies of the initial data
#'   (proportional stratified sampling).
#'
#' @param embedding Object of class [EmbeddedText].
#' @param target Named `factor` containing the labels of every case.
#' @param val_size `double` Ratio between 0 and 1 indicating the relative frequency of cases which should be used as
#'   validation sample.
#' @return Returns a `list` with the following components:
#'   * `target_train`: Named `factor` containing the labels of the training sample.
#'   * `embeddings_train`: Object of class [EmbeddedText] containing the text embeddings for the training sample.
#'   * `target_test`: Named `factor` containing the labels of the validation sample.
#'   * `embeddings_test`: Object of class [EmbeddedText] containing the text embeddings for the validation sample.
#'
#' @family Utils Developers
#' @keywords internal
#' @noRd
get_train_test_split <- function(embedding = NULL,
                                 target,
                                 val_size) {
  categories <- names(table(target))
  val_sampe <- NULL
  for (cat in categories) {
    tmp <- subset(target, target == cat)
    val_sampe[cat] <- list(
      sample(names(tmp), size = max(1L, length(tmp) * val_size))
    )
  }
  val_data <- target[unlist(val_sampe)]
  train_data <- target[setdiff(names(target), names(val_data))]

  if (!is.null(embedding)) {
    val_embeddings <- embedding$clone(deep = TRUE)
    val_embeddings$embeddings <- val_embeddings$embeddings[names(val_data), ]
    val_embeddings$embeddings <- na.omit(val_embeddings$embeddings)
    train_embeddings <- embedding$clone(deep = TRUE)
    train_embeddings$embeddings <- train_embeddings$embeddings[names(train_data), ]
    train_embeddings$embeddings <- na.omit(train_embeddings$embeddings)

    results <- list(
      target_train = train_data,
      embeddings_train = train_embeddings,
      target_test = val_data,
      embeddings_test = val_embeddings
    )
  } else {
    results <- list(
      target_train = train_data,
      embeddings_train = NA,
      target_test = val_data,
      embeddings_test = NA
    )
  }

  return(results)
}

#-------------------------------------------------------------------------------
#' @title Create a stratified random sample
#' @description This function creates a stratified random sample.The difference to `get_train_test_split` is that this
#'   function does not require text embeddings and does not split the text embeddings into a train and validation
#'   sample.
#'
#' @param targets Named `vector` containing the labels/categories for each case.
#' @param val_size `double` Value between 0 and 1 indicating how many cases of each label/category should be part of the
#'   validation sample.
#' @return `list` which contains the names of the cases belonging to the train sample and to the validation sample.
#' @family Utils Developers
#' @keywords internal
#' @noRd
get_stratified_train_test_split <- function(targets, val_size = 0.25) {
  test_sample <- NULL
  categories <- names(table(targets))

  for (cat in categories) {
    condition <- (targets == cat)
    tmp <- names(subset(
      x = targets,
      subset = condition
    ))
    test_sample[cat] <- list(
      sample(tmp, size = max(1L, length(tmp) * val_size))
    )
  }
  test_sample <- unlist(test_sample, use.names = FALSE)
  train_sample <- setdiff(names(targets), test_sample)

  results <- list(
    test_sample = test_sample,
    train_sample = train_sample
  )
  return(results)
}

#------------------------------------------------------------------------------
#' @title Get the number of chunks/sequences for each case
#' @description Function for calculating the number of chunks/sequences for every case.
#'
#' @param text_embeddings `data.frame` or `array` containing the text embeddings.
#' @param features `int` Number of features within each sequence.
#' @param times `int` Number of sequences.
#' @param pad_value `r get_param_doc_desc("pad_value")`
#' @return Named`vector` of integers representing the number of chunks/sequences for every case.
#'
#' @family Utils Developers
#'
#' @export
get_n_chunks <- function(text_embeddings, features, times, pad_value = -100L) {
  n_chunks <- vector(length = nrow(text_embeddings))
  n_chunks[] <- 0L

  if (length(dim(text_embeddings)) == 2L) {
    for (i in 1L:times) {
      window <- c(1L:features) + (i - 1L) * features
      sub_matrix <- text_embeddings[, window, drop = FALSE]
      tmp_sums <- rowSums(sub_matrix)
      n_chunks <- n_chunks + as.numeric(!tmp_sums == times * pad_value)
    }
  } else if (length(dim(text_embeddings)) == 3L) {
    for (i in 1L:times) {
      sub_matrix <- text_embeddings[, i, , drop = FALSE]
      tmp_sums <- rowSums(sub_matrix)
      n_chunks <- n_chunks + as.numeric(!tmp_sums == features * pad_value)
    }
  } else {
    stop("Dimensionality of text_embeddings must be 2 (matrix) or 3 (array).")
  }
  names(n_chunks) <- rownames(text_embeddings)
  return(n_chunks)
}
