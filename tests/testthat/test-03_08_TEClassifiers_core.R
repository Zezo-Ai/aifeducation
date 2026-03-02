testthat::skip_on_cran()
testthat::skip_if_not(
  condition = check_aif_py_modules(trace = FALSE),
  message = "Necessary python modules not available"
)

# Start time
test_time_start <- Sys.time()

# config------------------------------------------------------------------------
load_all_py_scripts()
object_class_names <- get_TEClassifiers_class_names(super_class = "ClassifiersBasedOnTextEmbeddings")

max_samples <- 20
max_samples_CI <- 2

max_samples_training <- 2
max_samples_training_CI <- 1

class_range <- c(2, 3)

prob_precision <- 1e-6

# Skip Tests-------------------------------------------------------------------
skip_creation_test <- TRUE
skip_function_save_load <- TRUE
skip_training_test <- FALSE
skip_documentation <- TRUE

# SetUp-------------------------------------------------------------------------
# Set paths
root_path_general_data <- testthat::test_path("test_data/Embeddings")

create_dir(testthat::test_path("test_artefacts"), FALSE)
root_path_results <- testthat::test_path("test_artefacts/TEClassifiers_core")
create_dir(root_path_results, FALSE)
root_path_feature_extractor <- testthat::test_path("test_data_tmp/classifier/feature_extractor_pytorch")

# SetUp datasets
# Disable tqdm progressbar
transformers$logging$disable_progress_bar()
datasets$disable_progress_bars()

# Load test data-----------------------------------------------------------------
test_data <- get_test_data_for_classifiers(
  class_range = class_range,
  path_test_embeddings = paste0(root_path_general_data, "/imdb_embeddings")
)

target_data <- test_data$target_data
target_levels <- test_data$target_levels
test_embeddings_large <- test_data$test_embeddings_large
test_embeddings <- test_data$test_embeddings
test_embeddings_reduced <- test_data$test_embeddings_reduced
test_embeddings_reduced_LD <- test_data$test_embeddings_reduced_LD
test_embeddings_single_case <- test_data$test_embeddings_single_case
test_embeddings_single_case_LD <- test_data$test_embeddings_single_case_LD

# Load feature extractors-------------------------------------------------------
# devtools::load_all()
feature_extractor <- NULL
if (file.exists(root_path_feature_extractor)) {
  feature_extractor <- load_from_disk(root_path_feature_extractor)
} else {
  feature_extractor <- NULL
}
# feature_extractor$extract_features_large(data_embeddings = test_embeddings$convert_to_LargeDataSetForTextEmbeddings(),batch_size = 2,trace=TRUE)
# feature_extractor$extract_features_large()

# Tests------------------------------------------------------------------------
for (object_class_name in object_class_names) {
  if (!skip_creation_test) {
    for (i in 1:check_adjust_n_samples_on_CI(
      n_samples_requested = max_samples,
      n_CI = max_samples_CI
    )) {
      # Test for different number of classes
      n_classes <- sample(class_range, size = 1L)

      # Create a List of all relevant combinations of arguments and reduce the number
      # to the desired sample size.
      # These are available for all tests.
      test_combination <- generate_args_for_tests(
        object_name = object_class_name,
        method = "configure",
        var_objects = list(
          feature_extractor = feature_extractor
        ),
        necessary_objects = list(
          text_embeddings = test_embeddings,
          target_levels = target_levels[[n_classes]]
        ),
        var_override = list(
          name = NULL,
          label = "Classifier for Estimating a Postive or Negative Rating of Movie Reviews",
          trace = random_bool_on_CI()
        )
      )

      classifier <- NULL
      gc()

      # Create test object with a given combination of args
      classifier <- create_object(object_class_name)
      # print(test_combination)
      suppressMessages(
        do.call(
          what = classifier$configure,
          args = test_combination
        )
      )

      test_that(paste("count parameter", object_class_name, get_current_args_for_print(test_combination)), {
        expect_gte(object = classifier$count_parameter(), expected = 0)
      })

      test_that(paste("Predictions", object_class_name, get_current_args_for_print(test_combination)), {
        # Number of predictions
        suppressMessages(
          reference_predictions <- classifier$predict(
            newdata = test_embeddings_reduced,
            batch_size = 2,
            ml_trace = 0
          )
        )
        expect_equal(
          object = length(reference_predictions$expected_category),
          expected = nrow(test_embeddings_reduced$embeddings)
        )

        # Data Source Invariance
        suppressMessages(
          predictions_LD <- classifier$predict(
            newdata = test_embeddings_reduced_LD,
            batch_size = 2,
            ml_trace = 0
          )
        )
        expect_equal(
          reference_predictions[, 1:(ncol(reference_predictions) - 1)],
          predictions_LD[, 1:(ncol(predictions_LD) - 1)],
          tolerance = 1e-6
        )

        # predict - randomness
        # Embedded Text
        suppressMessages(
          predictions_2 <- classifier$predict(
            newdata = test_embeddings_reduced,
            batch_size = 2,
            ml_trace = 0
          )
        )
        expect_equal(
          reference_predictions[, 1:(ncol(reference_predictions) - 1)],
          predictions_2[, 1:(ncol(predictions_2) - 1)],
          tolerance = 1e-6
        )
        # LargeDataSet
        suppressMessages(
          predictions_2 <- classifier$predict(
            newdata = test_embeddings_reduced_LD,
            batch_size = 2,
            ml_trace = 0
          )
        )
        expect_equal(
          reference_predictions[, 1:(ncol(reference_predictions) - 1)],
          predictions_2[, 1:(ncol(predictions_2) - 1)],
          tolerance = 1e-6
        )

        "predict - order invariance"
        if (!is.null(test_combination$attention)) {
          if (!(test_combination$attention == "Fourier" & (object_class_name %in% c("TEClassifierRegular", "TEClassifierProtoNet")))) {
            embeddings_ET_perm <- test_embeddings_reduced$clone(deep = TRUE)
            perm <- sample(x = seq.int(from = 1, to = nrow(embeddings_ET_perm$embeddings)), replace = FALSE)
            embeddings_ET_perm$embeddings <- embeddings_ET_perm$embeddings[perm, , , drop = FALSE]
            ids <- rownames(test_embeddings_reduced$embeddings)
            # Embedded Text
            suppressMessages(
              predictions_Perm <- classifier$predict(
                newdata = embeddings_ET_perm,
                batch_size = 50,
                ml_trace = 0
              )
            )
            expect_equal(
              reference_predictions[ids, 1:(ncol(reference_predictions) - 1)],
              predictions_Perm[ids, 1:(ncol(predictions_Perm) - 1)],
              tolerance = prob_precision
            )
            # Large Dataset
            predictions_Perm <- NULL
            suppressMessages(
              predictions <- classifier$predict(
                newdata = test_embeddings_reduced_LD,
                batch_size = 50,
                ml_trace = 0
              )
            )
            suppressMessages(
              predictions_Perm <- classifier$predict(
                newdata = embeddings_ET_perm$convert_to_LargeDataSetForTextEmbeddings(),
                batch_size = 50,
                ml_trace = 0
              )
            )
            expect_equal(
              reference_predictions[ids, 1:(ncol(reference_predictions) - 1)],
              predictions_Perm[ids, 1:(ncol(predictions_Perm) - 1)],
              tolerance = prob_precision
            )
          }
        }

        # Singe Case Prediction
        suppressMessages(
          prediction <- classifier$predict(
            newdata = test_embeddings_single_case,
            batch_size = 2,
            ml_trace = 0
          )
        )
        expect_equal(
          object = nrow(prediction),
          expected = 1
        )

        suppressMessages(
          prediction_LD <- classifier$predict(
            newdata = test_embeddings_single_case_LD,
            batch_size = 2,
            ml_trace = 0
          )
        )
        expect_equal(
          object = nrow(prediction_LD),
          expected = 1
        )
      })
      gc()
    }
  }

  # Function for loading and saving models-----------------------------------

  if (!skip_function_save_load) {
    test_that(paste("function save and load", object_class_name), {
      # Test for different number of classes
      n_classes <- sample(class_range, size = 1L)

      # Randomly select a configuration for training
      test_combination <- generate_args_for_tests(
        object_name = object_class_name,
        method = "configure",
        var_objects = list(
          feature_extractor = feature_extractor
        ),
        necessary_objects = list(
          text_embeddings = test_embeddings,
          target_levels = target_levels[[n_classes]]
        ),
        var_override = list(
          name = NULL,
          label = "Classifier for Estimating a Postive or Negative Rating of Movie Reviews",
          trace = random_bool_on_CI()
        )
      )
      classifier <- NULL
      gc()


      # Create test object with a given combination of args
      classifier <- create_object(object_class_name)
      suppressMessages(
        do.call(
          what = classifier$configure,
          args = test_combination
        )
      )

      # Predictions before saving and loading
      suppressMessages(
        predictions <- classifier$predict(
          newdata = test_embeddings_reduced,
          batch_size = 2,
          ml_trace = 0
        )
      )

      # Save and load
      folder_name <- paste0("function_save_load_", generate_id())
      dir_path <- paste0(root_path_results, "/", folder_name)
      save_to_disk(
        object = classifier,
        dir_path = root_path_results,
        folder_name = folder_name
      )
      classifier2 <- NULL
      classifier2 <- load_from_disk(dir_path = dir_path)

      # Is config equal after loading
      expect_equal(
        classifier$get_model_config(),
        classifier2$get_model_config()
      )

      # Predict after loading
      suppressMessages(
        predictions_2 <- classifier2$predict(
          newdata = test_embeddings_reduced,
          batch_size = 2,
          ml_trace = 0
        )
      )

      # Compare predictions
      i <- sample(x = seq.int(from = 1, to = nrow(predictions)), size = 1)
      expect_equal(predictions[i, , drop = FALSE],
        predictions_2[i, , drop = FALSE],
        tolerance = 1e-6
      )

      # Clean Directory
      unlink(
        x = dir_path,
        recursive = TRUE
      )
    })
    gc()
  }

  # Documentation--------------------------------------------------------------
  if (!skip_documentation) {
    test_that(paste("Documentation", object_class_name), {
      # Test for different number of classes
      n_classes <- sample(class_range, size = 1L)

      # Randomly select a configuration
      test_combination <- generate_args_for_tests(
        object_name = object_class_name,
        method = "configure",
        var_objects = list(
          feature_extractor = feature_extractor
        ),
        necessary_objects = list(
          text_embeddings = test_embeddings,
          target_levels = target_levels[[n_classes]]
        ),
        var_override = list(
          name = NULL,
          label = "Classifier for Estimating a Postive or Negative Rating of Movie Reviews",
          trace = random_bool_on_CI()
        )
      )
      classifier <- NULL
      gc()

      # Create test object with a given combination of args
      classifier <- create_object(object_class_name)
      suppressMessages(
        do.call(
          what = classifier$configure,
          args = test_combination
        )
      )

      classifier$set_model_description(
        eng = "Description",
        native = "Beschreibung",
        abstract_eng = "Abstract",
        abstract_native = "Zusammenfassung",
        keywords_eng = c("Test", "Neural Net"),
        keywords_native = c("Test", "Neuronales Netz")
      )
      desc <- classifier$get_model_description()
      expect_equal(
        object = desc$eng,
        expected = "Description"
      )
      expect_equal(
        object = desc$native,
        expected = "Beschreibung"
      )
      expect_equal(
        object = desc$abstract_eng,
        expected = "Abstract"
      )
      expect_equal(
        object = desc$abstract_native,
        expected = "Zusammenfassung"
      )
      expect_equal(
        object = desc$keywords_eng,
        expected = c("Test", "Neural Net")
      )
      expect_equal(
        object = desc$keywords_native,
        expected = c("Test", "Neuronales Netz")
      )


      classifier$set_model_license("test_license")
      expect_equal(
        object = classifier$get_model_license(),
        expected = "test_license"
      )


      classifier$set_documentation_license("test_license")
      expect_equal(
        object = classifier$get_documentation_license(),
        expected = "test_license"
      )


      classifier$set_publication_info(
        authors = personList(
          person(given = "Max", family = "Mustermann")
        ),
        citation = "Test Classifier",
        url = "https://Test.html"
      )
      pub_info <- classifier$get_publication_info()
      expect_equal(
        object = pub_info$developed_by$authors,
        expected = personList(
          person(given = "Max", family = "Mustermann")
        )
      )

      expect_equal(
        object = pub_info$developed_by$citation,
        expected = "Test Classifier"
      )

      expect_equal(
        object = pub_info$developed_by$url,
        expected = "https://Test.html"
      )
    })
  }

  # Test training of the classifier-------------------------------------------
  if (!skip_training_test) {
    # Create combinations for the training configuration
    log_dir <- paste0(root_path_results, "/", generate_id(5))
    create_dir(log_dir, trace = FALSE)

    for (j in 1:check_adjust_n_samples_on_CI(n_samples_requested = max_samples_training, n_CI = max_samples_training_CI)) {
      # Test for different number of classes
      n_classes <- sample(class_range, size = 1L)

      if (object_class_name == "TEClassifierSequential" && j <= 1) {
        use_pl <- TRUE
        tf_n_layers=0L
        ng_conv_n_layers=0L
        rec_n_layers=0L
        dense_n_layers=1L
      } else {
        use_pl <- FALSE
        tf_n_layers=NULL
        ng_conv_n_layers=NULL
        rec_n_layers=NULL
        dense_n_layers=NULL
      }

      if (object_class_name == "TEClassifierParallel" && j <= 1) {
        use_sc <- TRUE
      } else {
        use_sc <- FALSE
      }

      # Config sample
      test_combination <- generate_args_for_tests(
        object_name = object_class_name,
        method = "configure",
        var_objects = list(
          feature_extractor = feature_extractor
        ),
        necessary_objects = list(
          text_embeddings = test_embeddings,
          target_levels = target_levels[[n_classes]]
        ),
        var_override = list(
          name = NULL,
          label = "Classifier for Estimating a Postive or Negative Rating of Movie Reviews",
          trace = random_bool_on_CI(),
          tf_n_layers=tf_n_layers,
          ng_conv_n_layers=ng_conv_n_layers,
          rec_n_layers=rec_n_layers,
          dense_n_layers=dense_n_layers
        )
      )

      # traing config sample
      train_args_combinations <- generate_args_for_tests(
        object_name = object_class_name,
        method = "train",
        var_objects = list(),
        necessary_objects = list(
          data_embeddings = test_embeddings,
          data_targets = target_data[[n_classes]]
        ),
        var_override = list(
          name = NULL,
          label = "Classifier for Estimating a Postive or Negative Rating of Movie Reviews",
          trace = random_bool_on_CI(),
          log_dir = log_dir
        )
      )

      # Create test object with a given combination of args
      classifier <- NULL
      gc()
      classifier <- create_object(object_class_name)
      suppressMessages(
        do.call(
          what = classifier$configure,
          args = test_combination
        )
      )

      test_that(paste(
        "training", object_class_name,
        get_current_args_for_print(test_combination),
        get_current_args_for_print(train_args_combinations)
      ), {
        expect_no_error(
          suppressMessages(
            do.call(
              what = classifier$train,
              args = train_args_combinations
            )
          )
        )

        expect_true(classifier$get_sustainability_data()$sustainability_tracked)

        state_log_exists <- file.exists(paste0(log_dir, "/aifeducation_state.log"))
        expect_true(state_log_exists)

        if (state_log_exists) {
          log_state <- read.csv(paste0(log_dir, "/aifeducation_state.log"))
          expect_equal(nrow(log_state), 3)
          expect_equal(ncol(log_state), 3)
          expect_equal(colnames(log_state), c("value", "total", "message"))
          unlink(paste0(log_dir, "/aifeducation_state.log"))
        }

        loss_log_exists <- file.exists(paste0(log_dir, "/aifeducation_loss.log"))
        expect_true(loss_log_exists)
        if (loss_log_exists == TRUE) {
          log_loss <- read.csv(paste0(log_dir, "/aifeducation_loss.log"), header = FALSE)
          expect_gte(ncol(log_loss), 2)
          expect_gte(nrow(log_loss), 2)
          unlink(paste0(log_dir, "/aifeducation_loss.log"))
        }
      })

      # Plot training history
      test_that(paste(
        "plot_training_history", object_class_name,
        get_current_args_for_print(test_combination),
        get_current_args_for_print(train_args_combinations)
      ), {
        if (train_args_combinations$use_pl == TRUE) {
          pl_step <- 1
        } else {
          pl_step <- NULL
        }
        expect_s3_class(object = classifier$plot_training_history(pl_step = pl_step, measure = "loss", final_training = FALSE, add_min_max = TRUE, ind_best_model = TRUE, ind_selected_model = TRUE), class = "ggplot")
        expect_s3_class(object = classifier$plot_training_history(pl_step = pl_step, measure = "avg_iota", final_training = FALSE, add_min_max = TRUE, ind_best_model = FALSE, ind_selected_model = TRUE), class = "ggplot")
        expect_s3_class(object = classifier$plot_training_history(pl_step = pl_step, measure = "accuracy", final_training = FALSE, add_min_max = TRUE, ind_best_model = FALSE, ind_selected_model = FALSE), class = "ggplot")
        expect_s3_class(object = classifier$plot_training_history(pl_step = pl_step, measure = "balanced_accuracy", final_training = FALSE, add_min_max = TRUE, ind_best_model = TRUE, ind_selected_model = FALSE), class = "ggplot")

        expect_s3_class(object = classifier$plot_training_history(pl_step = pl_step, measure = "loss", final_training = FALSE, add_min_max = TRUE, y_min = 0, y_max = 2, ind_best_model = TRUE, ind_selected_model = FALSE), class = "ggplot")
        expect_s3_class(object = classifier$plot_training_history(pl_step = pl_step, measure = "avg_iota", final_training = FALSE, add_min_max = TRUE, y_min = 0, y_max = 1, ind_best_model = FALSE, ind_selected_model = FALSE), class = "ggplot")
        expect_s3_class(object = classifier$plot_training_history(pl_step = pl_step, measure = "accuracy", final_training = FALSE, add_min_max = TRUE, y_min = 0, y_max = 1, ind_best_model = FALSE, ind_selected_model = TRUE), class = "ggplot")
        expect_s3_class(object = classifier$plot_training_history(pl_step = pl_step, measure = "balanced_accuracy", final_training = FALSE, add_min_max = TRUE, y_min = 0, y_max = 1, ind_best_model = TRUE, ind_selected_model = TRUE), class = "ggplot")

        expect_s3_class(object = classifier$plot_training_history(pl_step = pl_step, measure = "loss", final_training = FALSE, add_min_max = FALSE, ind_best_model = TRUE, ind_selected_model = TRUE), class = "ggplot")
        expect_s3_class(object = classifier$plot_training_history(pl_step = pl_step, measure = "avg_iota", final_training = FALSE, add_min_max = FALSE, ind_best_model = TRUE, ind_selected_model = FALSE), class = "ggplot")
        expect_s3_class(object = classifier$plot_training_history(pl_step = pl_step, measure = "accuracy", final_training = FALSE, add_min_max = FALSE, ind_best_model = FALSE, ind_selected_model = FALSE), class = "ggplot")
        expect_s3_class(object = classifier$plot_training_history(pl_step = pl_step, measure = "balanced_accuracy", final_training = FALSE, add_min_max = FALSE, ind_best_model = FALSE, ind_selected_model = TRUE), class = "ggplot")

        expect_s3_class(object = classifier$plot_training_history(pl_step = pl_step, measure = "loss", final_training = TRUE, ind_best_model = TRUE, ind_selected_model = FALSE), class = "ggplot")
        expect_s3_class(object = classifier$plot_training_history(pl_step = pl_step, measure = "avg_iota", final_training = TRUE, ind_best_model = TRUE, ind_selected_model = TRUE), class = "ggplot")
        expect_s3_class(object = classifier$plot_training_history(pl_step = pl_step, measure = "accuracy", final_training = TRUE, ind_best_model = FALSE, ind_selected_model = TRUE), class = "ggplot")
        expect_s3_class(object = classifier$plot_training_history(pl_step = pl_step, measure = "balanced_accuracy", final_training = TRUE, ind_best_model = FALSE, ind_selected_model = FALSE), class = "ggplot")
      })

      test_that(paste(
        "plot_coding_stream", object_class_name,
        get_current_args_for_print(test_combination),
        get_current_args_for_print(train_args_combinations)
      ), {
        expect_s3_class(object = classifier$plot_coding_stream(), class = "ggplot")
      })
      gc()
    }
  }

  # Clean Directory--------------------------------------------------------------
  if (dir.exists(root_path_results)) {
    unlink(
      x = root_path_results,
      recursive = TRUE
    )
  }
}

# Monitor test time
monitor_test_time_on_CI(
  start_time = test_time_start,
  test_name = "03_08_TEClassifiers_core"
)
