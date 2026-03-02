testthat::skip_on_cran()
testthat::skip_if_not(
  condition = check_aif_py_modules(trace = FALSE),
  message = "Necessary python modules not available"
)

# Start time
test_time_start <- Sys.time()

# config------------------------------------------------------------------------
load_all_py_scripts()
# object_class_names <- get_TEClassifiers_class_names(super_class = "TEClassifiersBasedOnProtoNet")
object_class_names <- "TEClassifierProtoNet"
max_samples <- 20
max_samples_CI <- 2

max_samples_training <- 2
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
  for (n_classes in class_range) {
    # Embed----------------------------------------------------------------------
    for (i in 1:check_adjust_n_samples_on_CI(max_samples, max_samples_CI)) {
      # Create test object with a given combination of args
      test_combinations <- generate_args_for_tests(
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

      classifier <- create_object(object_class_name)
      suppressMessages(
        do.call(
          what = classifier$configure,
          args = test_combinations
        )
      )

      if (test_combinations$attention_type != "Fourier") {
        test_that(paste("embed", object_class_name, get_current_args_for_print(test_combinations)), {
          # Predictions
          embeddings <- classifier$embed(
            embeddings_q = test_embeddings_reduced,
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
      }
      gc()

      test_that(paste("plot", object_class_name, get_current_args_for_print(test_combinations)), {
        # plot
        plot <- classifier$plot_embeddings(
          embeddings_q = test_embeddings_reduced,
          classes_q = target_data[[n_classes]],
          batch_size = 50
        )
        expect_s3_class(plot, "ggplot")
      })
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
  test_name = "03_09_TEClassifierProtoNet_old"
)
