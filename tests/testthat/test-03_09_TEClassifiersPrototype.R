testthat::skip_on_cran()
testthat::skip_if_not(
  condition = check_aif_py_modules(trace = FALSE),
  message = "Necessary python modules not available"
)

# Start time
test_time_start <- Sys.time()

# config------------------------------------------------------------------------
load_all_py_scripts()
object_class_names <- get_TEClassifiers_class_names(super_class = "TEClassifiersBasedOnProtoNet")
# Do not use these test for the old ProtoNet Classifier
object_class_names <- setdiff(x = object_class_names, y = "TEClassifierProtoNet")
#Select on class randomly
object_class_names=sample(object_class_names,size=1L)
max_samples <- 10
max_samples_CI <- 1

class_range <- c(2, 3)

# SetUp-------------------------------------------------------------------------
# Set paths
root_path_general_data <- testthat::test_path("test_data/Embeddings")
create_dir(testthat::test_path("test_artefacts"), FALSE)
root_path_results <- testthat::test_path("test_artefacts/TeClassifierProtoNet")
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
feature_extractor <- NULL

if (file.exists(root_path_feature_extractor)) {
  feature_extractor <- load_from_disk(root_path_feature_extractor)
} else {
  feature_extractor <- NULL
}

for (object_class_name in object_class_names) {
  for (i in 1:check_adjust_n_samples_on_CI(
    n_samples_requested = max_samples,
    n_CI = max_samples_CI
  )) {
    # Test for different number of classes
    n_classes <- sample(class_range, size = 1L)

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
        tf_n_layers=0L,
        ng_conv_n_layers=0L
      )
    )


    # Create test object with a given combination of args
    classifier <- create_object(object_class_name)
    suppressMessages(
      do.call(
        what = classifier$configure,
        args = test_combination
      )
    )
    # Predict with sample cases-------------------------------------------------
    test_that(paste("Predictions with Sample Cases", object_class_name, get_current_args_for_print(test_combination)), {
      # Number of Predictions
      reference_predictions <- classifier$predict_with_samples(
        newdata = test_embeddings,
        embeddings_s = test_embeddings_reduced,
        classes_s = target_data[[n_classes]][rownames(test_embeddings_reduced$embeddings)],
        batch_size = 2,
        ml_trace = 0
      )

      expect_equal(
        object = length(reference_predictions$expected_category),
        expected = nrow(test_embeddings$embeddings)
      )

      # Randomness
      # Embedded Text
      predictions_2 <- classifier$predict_with_samples(
        newdata = test_embeddings,
        embeddings_s = test_embeddings_reduced,
        classes_s = target_data[[n_classes]][rownames(test_embeddings_reduced$embeddings)],
        batch_size = 2,
        ml_trace = 0
      )
      expect_equal(reference_predictions[, 1:(ncol(reference_predictions) - 1)], predictions_2[, 1:(ncol(predictions_2) - 1)],
        tolerance = 1e-6
      )
      # LargeDataSet
      predictions_2 <- NULL
      predictions_2 <- classifier$predict_with_samples(
        newdata = test_embeddings_large,
        embeddings_s = test_embeddings_reduced_LD,
        classes_s = target_data[[n_classes]][rownames(test_embeddings_reduced$embeddings)],
        batch_size = 2,
        ml_trace = 0
      )
      expect_equal(reference_predictions[, 1:(ncol(reference_predictions) - 1)], predictions_2[, 1:(ncol(predictions_2) - 1)],
        tolerance = 1e-6
      )

      # Order Invariance
      if (!is.null(test_combination$attention)) {
        if (test_combination$attention != "fourier") {
          embeddings_ET_perm <- test_embeddings$clone(deep = TRUE)
          perm <- sample(x = seq.int(from = 1, to = nrow(embeddings_ET_perm$embeddings)), replace = FALSE)
          embeddings_ET_perm$embeddings <- embeddings_ET_perm$embeddings[perm, , , drop = FALSE]
          ids <- rownames(test_embeddings$embeddings)
          # Embedded Text
          predictions_Perm <- classifier$predict_with_samples(
            newdata = embeddings_ET_perm,
            embeddings_s = test_embeddings_reduced,
            classes_s = target_data[[n_classes]][rownames(test_embeddings_reduced$embeddings)],
            batch_size = 50,
            ml_trace = 0
          )
          expect_equal(
            reference_predictions[ids, 1:(ncol(reference_predictions) - 1)],
            predictions_Perm[ids, 1:(ncol(predictions_Perm) - 1)],
            tolerance = 1e-6
          )

          # LargeDataSet
          predictions_Perm <- NULL
          predictions_Perm <- classifier$predict_with_samples(
            newdata = embeddings_ET_perm$convert_to_LargeDataSetForTextEmbeddings(),
            embeddings_s = test_embeddings_reduced,
            classes_s = target_data[[n_classes]][rownames(test_embeddings_reduced$embeddings)],
            batch_size = 50,
            ml_trace = 0
          )
          expect_equal(
            reference_predictions[ids, 1:(ncol(reference_predictions) - 1)],
            predictions_Perm[ids, 1:(ncol(predictions_Perm) - 1)],
            tolerance = 1e-6
          )
        }
      }

      # Single Case
      # Embedded Text
      prediction <- classifier$predict_with_samples(
        newdata = test_embeddings_single_case,
        embeddings_s = test_embeddings_reduced,
        classes_s = target_data[[n_classes]][rownames(test_embeddings_reduced$embeddings)],
        batch_size = 2,
        ml_trace = 0
      )
      expect_equal(
        object = nrow(prediction),
        expected = 1
      )
      # LargeDataSet
      prediction_LD <- classifier$predict_with_samples(
        newdata = test_embeddings_single_case_LD,
        embeddings_s = test_embeddings_reduced_LD,
        classes_s = target_data[[n_classes]][rownames(test_embeddings_reduced$embeddings)],
        batch_size = 2,
        ml_trace = 0
      )
      expect_equal(
        object = nrow(prediction_LD),
        expected = 1
      )
    })


    # Embed----------------------------------------------------------------------
    test_that(paste("embed without sample cases", object_class_name, get_current_args_for_print(test_combination)), {
      # Predictions
      embeddings <- classifier$embed(
        embeddings_q = test_embeddings_reduced,
        embeddings_s = NULL,
        classes_s = NULL,
        batch_size = 50
      )

      # check case order invariance
      perm <- sample(x = seq.int(from = 1, to = nrow(test_embeddings_reduced$embeddings)))
      test_embeddings_reduced_perm <- test_embeddings_reduced$clone(deep = TRUE)
      test_embeddings_reduced_perm$embeddings <- test_embeddings_reduced_perm$embeddings[perm, , ]
      embeddings_perm <- classifier$embed(
        embeddings_q = test_embeddings_reduced_perm,
        batch_size = 50
      )
      for (j in seq_len(nrow(embeddings$embeddings_q))) {
        expect_equal(embeddings$embeddings_q[j, ],
          embeddings_perm$embeddings_q[which(perm == j), ],
          tolerance = 1e-5
        )
      }
    })
    gc()

    test_that(paste("embed with sample cases", object_class_name, get_current_args_for_print(test_combination)), {
      # Predictions
      embeddings <- classifier$embed(
        embeddings_q = test_embeddings,
        embeddings_s = test_embeddings_reduced,
        classes_s = target_data[[n_classes]][rownames(test_embeddings_reduced$embeddings)],
        batch_size = 50
      )

      # check case order invariance
      perm <- sample(x = seq.int(from = 1, to = nrow(test_embeddings$embeddings)))
      test_embeddings_perm <- test_embeddings$clone(deep = TRUE)
      test_embeddings_perm$embeddings <- test_embeddings_perm$embeddings[perm, , ]
      embeddings_perm <- classifier$embed(
        embeddings_q = test_embeddings_perm,
        embeddings_s = test_embeddings_reduced,
        classes_s = target_data[[n_classes]][rownames(test_embeddings_reduced$embeddings)],
        batch_size = 50
      )
      for (j in seq_len(nrow(embeddings$embeddings_q))) {
        expect_equal(embeddings$embeddings_q[j, ],
          embeddings_perm$embeddings_q[which(perm == j), ],
          tolerance = 1e-5
        )
      }
    })

    test_that(paste("plot without sample cases", object_class_name, get_current_args_for_print(test_combination)), {
      # plot
      classifier <- create_object(object_class_name)
      suppressMessages(
        do.call(
          what = classifier$configure,
          args = test_combination
        )
      )
      plot <- classifier$plot_embeddings(
        embeddings_q = test_embeddings_reduced,
        classes_q = target_data[[n_classes]],
        embeddings_s = NULL,
        classes_s = NULL,
        batch_size = 50,
        inc_margin = FALSE
      )
      expect_s3_class(plot, "ggplot")
    })

    test_that(paste("plot with sample cases", object_class_name, get_current_args_for_print(test_combination)), {
      # plot
      plot <- classifier$plot_embeddings(
        embeddings_q = test_embeddings,
        classes_q = target_data[[n_classes]],
        embeddings_s = test_embeddings_reduced,
        classes_s = target_data[[n_classes]][rownames(test_embeddings_reduced$embeddings)],
        batch_size = 50,
        inc_margin = FALSE
      )
      expect_s3_class(plot, "ggplot")
    })
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
  test_name = "03_09_TEClassifierPrototype"
)
