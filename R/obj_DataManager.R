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

#' @title Data manager for classification tasks
#'
#' @description Abstract class for managing the data and samples during training a classifier. DataManagerClassifier is
#'   used with all classifiers based on text embeddings.
#'
#' @return Objects of this class are used for ensuring the correct data management for training different types of
#'   classifiers. They are also used for data augmentation by creating synthetic cases with different
#'   techniques.
#' @family R6 Classes for Developers
#' @export
DataManagerClassifier <- R6::R6Class(
  classname = "DataManagerClassifier",
  public = list(
    #' @field config ('list')\cr
    #'   Field for storing configuration of the [DataManagerClassifier].
    config = NULL,

    #' @field state ('list')\cr
    #'   Field for storing the current state of the [DataManagerClassifier].
    state = list(
      iteration = NULL,
      step = NULL
    ),

    #' @field datasets ('list')\cr
    #'   Field for storing the data sets used during training. All elements of the list are data sets of class
    #'   `datasets.arrow_dataset.Dataset`. The following data sets are available:
    #'
    #' * data_labeled: all cases which have a label.
    #' * data_unlabeled: all cases which have no label.
    #' * data_labeled_synthetic: all synthetic cases with their corresponding labels.
    #' * data_labeled_pseudo: subset of data_unlabeled if pseudo labels were estimated by a classifier.
    #'
    datasets = NULL,

    #' @field name_idx ('named vector')\cr
    #'   Field for storing the pairs of indexes and names of every case. The pairs for labeled and unlabeled data are
    #'   separated.
    name_idx = NULL,

    #' @field samples ('list')\cr
    #'   Field for storing the assignment of every cases to a train, validation or test data set depending on the
    #'   concrete fold. Only the indexes and not the names are stored. In addition, the list contains the assignment for
    #'   the final training which excludes a test data set. If the [DataManagerClassifier] uses `i` folds the sample for
    #'   the final training can be requested with `i+1`.
    samples = NULL,

    #' @description Creating a new instance of this class.
    #' @param data_embeddings `r get_param_doc_desc("data_embeddings")`
    #' @param data_targets `r get_param_doc_desc("data_targets")`
    #' @param folds `r get_param_doc_desc("folds")`
    #' @param val_size `r get_param_doc_desc("val_size")`
    #' @param class_levels `r get_param_doc_desc("class_levels")`
    #' @param one_hot_encoding `r get_param_doc_desc("one_hot_encoding")`
    #' @param add_matrix_map `r get_param_doc_desc("add_matrix_map")`
    #' @param sc_methods `r get_param_doc_desc("sc_methods")`
    #' @param sc_min_k `r get_param_doc_desc("sc_min_k")`
    #' @param sc_max_k `r get_param_doc_desc("sc_max_k")`
    #' @param trace `r get_param_doc_desc("trace")`
    #' @param n_cores `r get_param_doc_desc("n_cores")`
    #' @param pad_value `r get_param_doc_desc("pad_value")`
    #' @return Method returns an initialized object of class [DataManagerClassifier].
    initialize = function(data_embeddings,
                          data_targets,
                          class_levels,
                          folds = 5L,
                          val_size = 0.25,
                          pad_value = -100L,
                          one_hot_encoding = TRUE,
                          add_matrix_map = TRUE,
                          sc_methods = "knnor",
                          sc_min_k = 1L,
                          sc_max_k = 10L,
                          trace = TRUE,
                          n_cores = auto_n_cores()) {
      # Checking Prerequisites---------------------------------------------------
      check_all_args(get_called_args(n = 1L))

      # Set cache dir
      private$cache_dir <- file.path(create_and_get_tmp_dir(), paste0("dm_", generate_id(5L)))

      # Create Dataset-------------------------------------------------------
      private$prepare_datasets(
        data_embeddings = data_embeddings,
        data_targets = data_targets,
        trace = trace
      )

      # Create indices name mapping for labeled and unlabeled data--------------
      private$create_indices_name_map()

      # Check for valid number of folds------------------------------------------
      fin_k_folds <- private$check_and_calculate_number_folds(folds)

      # Saving Configuration-----------------------------------------------------
      self$config$n_folds <- fin_k_folds
      self$config$val_size <- val_size
      self$config$one_hot_encoding <- one_hot_encoding
      self$config$add_matrix_map <- add_matrix_map
      self$config$class_levels <- class_levels
      self$config$n_classes <- length(class_levels)
      self$config$features <- data_embeddings$get_features()
      self$config$times <- data_embeddings$get_times()
      self$config$sc$methods <- sc_methods
      self$config$sc$max_k <- sc_max_k
      self$config$sc$min_k <- sc_min_k
      self$config$n_cores <- n_cores
      self$config$pad_value <- pad_value

      # Add one hot encoding if necessary
      if (self$config$one_hot_encoding) {
        self$datasets$data_labeled <- private$add_one_hot_encoding(self$datasets$data_labeled)
      }

      # Add matrix map if necessary
      if (self$config$add_matrix_map) {
        self$datasets$data_labeled <- private$add_matrix_form(self$datasets$data_labeled)
        self$datasets$data_unlabeled <- private$add_matrix_form(self$datasets$data_unlabeled)
      }

      # create folds
      private$create_folds()

      # create sample for final training
      private$create_final_sample()
    },

    #' @description Method for requesting the configuration of the [DataManagerClassifier].
    #' @return Returns a `list` storing the configuration of the [DataManagerClassifier].
    get_config = function() {
      return(self$config)
    },

    #' @description Method for requesting the complete labeled data set.
    #' @return Returns an object of class `datasets.arrow_dataset.Dataset` containing all cases with labels.
    get_labeled_data = function() {
      return(self$datasets$data_labeled)
    },

    #' @description Method for requesting the complete unlabeled data set.
    #' @return Returns an object of class `datasets.arrow_dataset.Dataset` containing all cases without labels.
    get_unlabeled_data = function() {
      return(self$datasets$data_unlabeled)
    },

    #' @description Method for requesting the assignments to train, validation, and test data sets for every fold and
    #'   the final training.
    #' @return Returns a `list` storing the assignments to a train, validation, and test data set for every fold. In the
    #'   case of the sample for the final training the test data set is always empty (`NULL`).
    get_samples = function() {
      return(self$samples)
    },

    #' @description Method for setting the current state of the [DataManagerClassifier].
    #' @param iteration `int` determining the current iteration of the training. That is iteration determines the fold
    #'   to use for training, validation, and testing. If *i* is the number of fold *i+1* request the sample for the
    #'   final training. For requesting the sample for the final training iteration can take a string `"final"`.
    #' @param step `int` determining the step for estimating and using pseudo labels during training. Only relevant if
    #'   training is requested with pseudo labels.
    #' @return Method does not return anything. It is used for setting the internal state of the DataManager.
    set_state = function(iteration, step = NULL) {
      check_type(object = step, object_name = "step", type = "int", min = 1L, max = Inf, allow_NULL = TRUE, allowed_values = NULL)

      if (!is.numeric(iteration)) {
        if (iteration == "final") {
          iteration <- self$config$n_folds + 1L
        } else {
          stop(
            iteration,
            " is not a valid input for iteration"
          )
        }
      }

      self$state$iteration <- iteration
      self$state$step <- step
    },

    #' @description Method for requesting the number of folds the [DataManagerClassifier] can use with the current data.
    #' @return Returns the number of folds the [DataManagerClassifier] uses.
    get_n_folds = function() {
      return(self$config$n_folds)
    },

    #' @description Method for requesting the number of classes.
    #' @return Returns the number classes.
    get_n_classes = function() {
      return(self$config$n_classes)
    },

    #' @description Method for requesting descriptive sample statistics.
    #' @return Returns a table describing the absolute frequencies of the labeled and unlabeled data. The rows contain
    #'   the length of the sequences while the columns contain the labels.
    get_statistics = function() {
      if (!is.null(self$datasets)) {
        if (!is.null(self$datasets$data_labeled)) {
          self$datasets$data_labeled$set_format("np")
          length_labeled <- extract_column_from_py_dataset(self$datasets$data_labeled,"length")
          labels_labeled <- extract_column_from_py_dataset(self$datasets$data_labeled,"labels")

          if (!is.null(self$datasets$data_unlabeled)) {
            self$datasets$data_unlabeled$set_format("np")
            length_unlabeled <- extract_column_from_py_dataset(self$datasets$data_unlabeled,"length")
            labels_unlabeled <- rep(NA, times = length(length_unlabeled))
          } else {
            length_unlabeled <- NULL
            labels_unlabeled <- NULL
          }

          statistics <- table(c(length_labeled, length_unlabeled),
            c(labels_labeled, labels_unlabeled),
            useNA = "ifany"
          )
          return(statistics)
        } else {
          stop("Labeled data not set.")
        }
      } else {
        stop("Data not set.")
      }
    },

    #' @description Method for checking if the dataset contains cases without labels.
    #' @return Returns `TRUE` if the dataset contains cases without labels. Returns `FALSE`
    #' if all cases have labels.
    contains_unlabeled_data = function() {
      if (is.null(self$datasets$data_unlabeled)) {
        return(FALSE)
      } else {
        if (self$datasets$data_unlabeled$num_rows > 0L) {
          return(TRUE)
        } else {
          return(FALSE)
        }
      }
    },

    #' @description Method for requesting a data set for training depending in the current state of the
    #'   DataManagerClassifier.
    #' @param inc_labeled `bool` If `TRUE` the data set includes all cases which have labels.
    #' @param inc_unlabeled `bool` If `TRUE` the data set includes all cases which have no labels.
    #' @param inc_synthetic `bool` If `TRUE` the data set includes all synthetic cases with their corresponding labels.
    #' @param inc_pseudo_data `bool` If `TRUE` the data set includes all cases which have pseudo labels.
    #' @return Returns an object of class `datasets.arrow_dataset.Dataset` containing the requested kind of data along
    #'   with all requested transformations for training. Please note that this method returns a data sets that is
    #'   designed for training only. The corresponding validation data set is requested with `get_val_dataset` and the
    #'   corresponding test data set with `get_test_dataset`.
    get_dataset = function(inc_labeled = TRUE,
                           inc_unlabeled = FALSE,
                           inc_synthetic = FALSE,
                           inc_pseudo_data = FALSE) {
      # Checks
      check_type(object = inc_labeled, type = "bool", FALSE)
      check_type(object = inc_unlabeled, type = "bool", FALSE)
      check_type(object = inc_synthetic, type = "bool", FALSE)
      check_type(object = inc_pseudo_data, type = "bool", FALSE)

      tmp_data <- NULL

      requested_datasets <- list()

      if (inc_labeled) {
        self$datasets$data_labeled$set_format("np")
        requested_datasets[length(requested_datasets) + 1L] <- list(self$datasets$data_labeled$select(
          as.integer(self$samples[[self$state$iteration]]$train)
        ))
      }

      if (inc_synthetic) {
        if (!is.null(self$datasets$data_labeled_synthetic)) {
          self$datasets$data_labeled_synthetic$set_format("np")
          requested_datasets[length(requested_datasets) + 1L] <- list(self$datasets$data_labeled_synthetic)
        }
      }

      if (inc_pseudo_data) {
        if (!is.null(self$datasets$data_labeled_pseudo)) {
          self$datasets$data_labeled_pseudo$set_format("np")
          requested_datasets[length(requested_datasets) + 1L] <- list(self$datasets$data_labeled_pseudo)
        }
      }
      if (inc_unlabeled) {
        if (!is.null(self$datasets$data_unlabeled)) {
          self$datasets$data_unlabeled$set_format("np")
          requested_datasets[length(requested_datasets) + 1L] <- list(self$datasets$data_unlabeled)
        }
      }


      if (length(requested_datasets) < 1L) {
        tmp_data <- NULL
      } else if (length(requested_datasets) == 1L) {
        tmp_data <- requested_datasets[[1L]]
      } else {
        tmp_data <- datasets$concatenate_datasets(dsets = requested_datasets, axis = 0L)
      }
      return(tmp_data)
    },

    #' @description Method for requesting a data set for validation depending in the current state of the
    #'   [DataManagerClassifier].
    #' @return Returns an object of class `datasets.arrow_dataset.Dataset` containing the requested kind of data along
    #'   with all requested transformations for validation. The corresponding data set for training can be requested
    #'   with `get_dataset` and the corresponding data set for testing with `get_test_dataset`.
    get_val_dataset = function() {
      return(self$datasets$data_labeled$select(as.integer(self$samples[[self$state$iteration]]$val)))
    },

    #' @description Method for requesting a data set for testing depending in the current state of the
    #'   DataManagerClassifier.
    #' @return Returns an object of class `datasets.arrow_dataset.Dataset` containing the requested kind of data along
    #'   with all requested transformations for validation. The corresponding data set for training can be requested
    #'   with `get_dataset` and the corresponding data set for validation with `get_val_dataset`.
    get_test_dataset = function() {
      if (self$state$iteration <= self$get_n_folds()) {
        return(self$datasets$data_labeled$select(as.integer(self$samples[[self$state$iteration]]$test)))
      } else {
        return(NULL)
      }
    },

    #' @description Method for generating synthetic data used during training. The process uses all labeled data
    #'   belonging to the current state of the DataManagerClassifier.
    #' @param  trace `bool` If `TRUE` information on the process are printed to the console.
    #' @param inc_pseudo_data `bool` If `TRUE` data with pseudo labels are used in addition to the labeled data for
    #'   generating synthetic cases.
    #' @return This method does nothing return. It generates a new data set for synthetic cases which are stored as an
    #'   object of class `datasets.arrow_dataset.Dataset` in the field `datasets$data_labeled_synthetic`. Please note
    #'   that a call of this method will override an existing data set in the corresponding field.
    create_synthetic = function(trace = TRUE,
                                inc_pseudo_data = FALSE) {
      # checks
      check_type(object = trace, type = "bool", FALSE)
      check_type(object = inc_pseudo_data, type = "bool", FALSE)

      # Print status message to console
      if (trace) {
        if (self$state$iteration <= self$config$n_folds) {
          message(
            get_time_stamp(),
            " | ", "Iteration ", self$state$iteration, " from ", self$config$n_folds,
            " | ", "Generating synthetic cases"
          )
        } else {
          message(
            get_time_stamp(),
            " | ", "Final training",
            " | ", "Generating synthetic cases"
          )
        }
      }

      tmp_data <- self$get_dataset(
        inc_labeled = TRUE,
        inc_unlabeled = FALSE,
        inc_synthetic = FALSE,
        inc_pseudo_data = inc_pseudo_data
      )

      # Set up parallel processing
      requireNamespace(package = "foreach", quietly = TRUE)
      # if(is_on_CI()==FALSE){
      cl <- parallel::makeCluster(self$config$n_cores)
      doParallel::registerDoParallel(cl)
      # } else {
      #  foreach::registerDoSEQ()
      #  cl=NULL
      # }


      # Create Synthetic Cases
      tmp_data$set_format("np")
      syn_cases <- get_synthetic_cases_from_matrix(
        matrix_form = extract_column_from_py_dataset(tmp_data,"matrix_form"),
        target = extract_column_from_py_dataset(tmp_data,"labels"),
        sequence_length = extract_column_from_py_dataset(tmp_data,"length"),
        method = self$config$sc$methods,
        min_k = self$config$sc$min_k,
        max_k = self$config$sc$max_k,
        times = self$config$times,
        features = self$config$features
      )

      # Unload cluster for parallel processing
      if (!is.null(cl)) {
        parallel::stopCluster(cl)
      }

      # Post-process synthetic cases
      embeddings_syntehtic <- syn_cases$syntetic_embeddings
      targets_synthetic <- as.numeric(as.character(syn_cases$syntetic_targets))

      # Check if cases with zero length are included
      seq_length <- get_n_chunks(
        text_embeddings = embeddings_syntehtic,
        features = self$config$features,
        times = self$config$times,
        pad_value = self$config$pad_value
      )
      idx_non_zero_length <- which(x = (seq_length > 0L))

      # exclude cases with zero length
      embeddings_syntehtic <- embeddings_syntehtic[idx_non_zero_length, , , drop = FALSE]
      targets_synthetic <- targets_synthetic[idx_non_zero_length]

      if (dim(embeddings_syntehtic)[1L] > 1L) {
        # Add or replace current dataset for synthetic cases
        self$datasets$data_labeled_synthetic <- datasets$Dataset$from_dict(
          reticulate::dict(
            list(
              id = rownames(embeddings_syntehtic),
              input = prepare_r_array_for_dataset(embeddings_syntehtic),
              labels = targets_synthetic,
              length = get_n_chunks(
                text_embeddings = embeddings_syntehtic,
                features = self$config$features,
                times = self$config$times,
                pad_value = self$config$pad_value
              )
            ),
            convert = FALSE
          )
        )


        if (self$config$one_hot_encoding) {
          self$datasets$data_labeled_synthetic <- private$add_one_hot_encoding(self$datasets$data_labeled_synthetic)
        }

        if (self$config$add_matrix_map) {
          self$datasets$data_labeled_synthetic <- private$add_matrix_form(self$datasets$data_labeled_synthetic)
        }
      } else {
        self$datasets$data_labeled_synthetic <- NULL
      }

      if (trace) {
        if (self$state$iteration <= self$config$n_folds) {
          message(
            get_time_stamp(),
            " | ", "Iteration ", self$state$iteration, " from ", self$config$n_folds,
            " | ", "Generating synthetic cases done"
          )
        } else {
          message(
            get_time_stamp(),
            " | ", "Final training",
            " | ", "Generating synthetic cases done"
          )
        }
      }
    },

    #' @description Method for adding data with pseudo labels generated by a classifier
    #' @param inputs `array` or `matrix` representing the input data.
    #' @param labels `factor` containing the corresponding pseudo labels.
    #' @return This method does nothing return. It generates a new data set for synthetic cases which are stored as an
    #'   object of class `datasets.arrow_dataset.Dataset` in the field `datasets$data_labeled_pseudo`. Please note that
    #'   a call of this method will override an existing data set in the corresponding field.
    add_replace_pseudo_data = function(inputs,
                                       labels) {
      # checks
      check_class(object = inputs, classes = c("array", "matrix"), allow_NULL = FALSE)
      check_class(object = labels, classes = "factor", allow_NULL = FALSE)

      private$check_labels(labels)
      # Add or replace current dataset for pseudo data
      self$datasets$data_labeled_pseudo <- datasets$Dataset$from_dict(
        reticulate::dict(
          list(
            id = rownames(inputs),
            input = prepare_r_array_for_dataset(inputs),
            labels = as.numeric(labels) - 1L,
            length = get_n_chunks(
              text_embeddings = inputs,
              features = self$config$features,
              times = self$config$times,
              pad_value = self$config$pad_value
            )
          ),
          convert = FALSE
        )
      )

      if (self$config$one_hot_encoding) {
        self$datasets$data_labeled_pseudo <- private$add_one_hot_encoding(self$datasets$data_labeled_pseudo)
      }

      if (self$config$add_matrix_map) {
        self$datasets$data_labeled_pseudo <- private$add_matrix_form(self$datasets$data_labeled_pseudo)
      }
    }
    #-----------------------------------------------------------------------------
  ),
  private = list(
    cache_dir = NULL,
    finalize = function() {
      unlink(x = private$cache_dir, recursive = TRUE)
      gc()
    },
    prepare_datasets = function(data_embeddings, data_targets, trace) {
      if (inherits(data_embeddings, "EmbeddedText")) {
        data_set_embeddings <- data_embeddings$convert_to_LargeDataSetForTextEmbeddings()
        data_set_embeddings <- data_set_embeddings$get_dataset()
      } else {
        data_set_embeddings <- data_embeddings$get_dataset()
      }

      # Reduce embeddings to unique ids
      n_init_cases <- data_set_embeddings$num_rows
      data_set_embeddings <- reduce_to_unique(data_set_embeddings, "id")
      n_final_cases <- data_set_embeddings$num_rows

      # Prepare classes for join
      class_vector <- vector(length = n_final_cases)
      class_vector[] <- NA
      names(class_vector) <- extract_column_from_py_dataset(data_set_embeddings,"id")

      # Convert labels
      data_targets <- na.omit(data_targets)
      data_targets_names <- names(data_targets)
      data_targets <- as.numeric(data_targets) - 1L
      names(data_targets) <- data_targets_names

      # Estimate all targets which have an embedding input
      relevant_target_names <- intersect(names(class_vector), y = names(data_targets))

      # Add relevant data to the class vector
      class_vector[relevant_target_names] <- data_targets[relevant_target_names]

      # Sort class vector according to the input ids
      class_vector <- class_vector[extract_column_from_py_dataset(data_set_embeddings,"id")]

      # Get indices for labeled-unlabeled split
      indices_unlabeled <- which(is.na(class_vector)) - 1L
      indices_labeled <- which(!is.na(class_vector)) - 1L

      # Add labels to data set
      data_set_embeddings <- data_set_embeddings$add_column("labels", np$array(class_vector))

      # Split data
      self$datasets$data_labeled <- data_set_embeddings$select(as.integer(indices_labeled))
      if (length(indices_unlabeled) != 0L) {
        self$datasets$data_unlabeled <- data_set_embeddings$select(as.integer(indices_unlabeled))
      } else {
        self$datasets$data_unlabeled <- NULL
      }


      # Report numbers
      if (trace) {
        message(
          get_time_stamp(),
          " Total Cases: ", n_init_cases,
          " Unique Cases: ", n_final_cases,
          " Labeled Cases: ", length(indices_labeled)
        )
      }
    },
    #--------------------------------------------------------------------------
    get_all_labels = function() {
      self$datasets$data_labeled$set_format("np")
      return(extract_column_from_py_dataset(self$datasets$data_labeled,"labels"))
    },
    #-------------------------------------------------------------------------
    check_and_calculate_number_folds = function(folds) {
      sample_target <- private$get_all_labels()
      freq_cat <- table(sample_target)
      min_freq <- min(freq_cat)
      if (min_freq < 6L) {
        stop("Frequency of the smallest category respective class is", min_freq, ". At least
                   6 cases are necessary. Consider to remove this category/class.")
      } else {
        if (min_freq / folds < 3L) {
          fin_k_folds <- floor(min_freq / 3L)
          warning("Frequency of the smallest category respective class is not sufficent to ensure
                    at least 3 cases per fold. Adjusting number of folds from ", folds, "to", fin_k_folds, ".")
        } else {
          fin_k_folds <- folds
        }
      }
      return(fin_k_folds)
    },
    add_matrix_form = function(dataset) {
      if (!is.null(dataset)) {
        private$load_reload_python_scripts()
        dataset <- dataset$map(py$map_input_to_matrix_form,
          fn_kwargs = list(
            times = as.integer(self$config$times),
            features = as.integer(self$config$features)
          ),
          load_from_cache_file = FALSE,
          keep_in_memory = FALSE,
          cache_file_name = create_py_dataset_cache_file_path(file.path(private$cache_dir, generate_id(15L)))
        )
        return(dataset)
      } else {
        return(NULL)
      }
    },
    add_one_hot_encoding = function(dataset) {
      if (!is.null(dataset)) {
        private$load_reload_python_scripts()
        dataset <- dataset$map(py$map_labels_to_one_hot,
          fn_kwargs = reticulate::dict(list(num_classes = as.integer(self$config$n_classes))),
          load_from_cache_file = FALSE,
          keep_in_memory = FALSE,
          cache_file_name = create_py_dataset_cache_file_path(file.path(private$cache_dir, generate_id(15L)))
        )
        return(dataset)
      } else {
        return(NULL)
      }
    },
    load_reload_python_scripts = function() {
      reticulate::py_run_file(system.file("python/py_functions.py",
        package = "aifeducation"
      ))
    },
    create_indices_name_map = function() {
      self$name_idx$labeled_data <- seq.int(from = 0L, to = (length(extract_column_from_py_dataset(self$datasets$data_labeled,"id"))) - 1L)
      names(self$name_idx$labeled_data) <- extract_column_from_py_dataset(self$datasets$data_labeled,"id")

      if (!is.null(self$datasets$data_unlabeled)) {
        self$name_idx$unlabeled_data <- seq.int(from = 0L, to = (length(extract_column_from_py_dataset(self$datasets$data_unlabeled,"id"))) - 1L)
        names(self$name_idx$unlabeled_data) <- extract_column_from_py_dataset(self$datasets$data_unlabeled,"id")
      } else {
        self$name_idx$unlabeled_data <- NULL
      }
    },
    create_folds = function() {
      # Create Train, Test, and Validation Samples--------------------------------
      # Check maximal number of folds, adjust, and create folds
      data_targets <- factor(x = extract_column_from_py_dataset(self$datasets$data_labeled,"labels"))
      names(data_targets) <- extract_column_from_py_dataset(self$datasets$data_labeled,"id")

      folds <- get_folds(
        target = data_targets,
        k_folds = self$config$n_folds
      )

      self$samples <- NULL
      for (i in 1L:self$config$n_folds) {
        train_val_split <- get_train_test_split(
          embedding = NULL,
          target = data_targets[folds$train_sample[[i]]],
          val_size = self$config$val_size
        )

        fold <- list(
          train = self$name_idx$labeled_data[names(train_val_split$target_train)],
          val = self$name_idx$labeled_data[names(train_val_split$target_test)],
          test = self$name_idx$labeled_data[folds$val_sample[[i]]]
        )

        self$samples[i] <- list(fold)
      }
    },
    create_final_sample = function(data_targets) {
      data_targets <- factor(x = extract_column_from_py_dataset(self$datasets$data_labeled,"labels"))
      names(data_targets) <- extract_column_from_py_dataset(self$datasets$data_labeled,"id")

      names_final_split <- get_stratified_train_test_split(
        targets = data_targets,
        val_size = self$config$val_size
      )
      final_split <- list(
        train = self$name_idx$labeled_data[names_final_split$train_sample],
        val = self$name_idx$labeled_data[names_final_split$test_sample],
        test = NULL
      )
      self$samples[length(self$samples) + 1L] <- list(final_split)
    },
    check_labels = function(labels) {
      if (!is.factor(labels)) {
        stop("labels must be an object of class factor.")
      } else {
        levels_identical <- sum(levels(labels) == self$config$class_levels)
        if (levels_identical != self$config$n_classes) {
          stop(
            "Levels of the labels are not identical with the levels of the classifier.",
            "Necessary levels:", self$config$class_levels,
            "Provided levels":levels(labels)
          )
        }
      }
    }
  )
)
