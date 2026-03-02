testthat::skip_on_cran()
testthat::skip_if_not(
  condition = check_aif_py_modules(trace = FALSE),
  message = "Necessary python modules not available"
)
load_all_py_scripts()

# Start time
test_time_start <- Sys.time()

# SetUp-------------------------------------------------------------------------
# Set paths
root_path_general_data <- testthat::test_path("test_data/OldModels")
testthat::skip_if_not(
  condition = dir.exists(root_path_general_data),
  message = "Folder with old models is not available"
)

versions <- c("1.0.1", "1.1.1")

for (version in versions) {
  test_that(paste("TEM", version), {
    tem_path <- file.path(root_path_general_data, paste("Version", version), "tem_model")
    if (dir.exists(tem_path)) {
      tem <- load_from_disk(tem_path)

      example_data <- LargeDataSetForText$new()
      example_data$add_from_data.frame(imdb_movie_reviews)

      test_embeddings <- tem$embed_large(text_dataset = example_data, trace = FALSE)

      expect_equal(test_embeddings$n_rows(), nrow(imdb_movie_reviews))
      expect_s3_class(test_embeddings, "LargeDataSetForTextEmbeddings")
    }
  })

  test_that(paste("CLS", version), {
    tem_path <- file.path(root_path_general_data, paste("Version", version), "tem_model")
    cls_path <- file.path(root_path_general_data, paste("Version", version), "cls_model")
    if(dir.exists(cls_path)){
      cls_model <- load_from_disk(cls_path)
    }
    if (dir.exists(tem_path) && dir.exists(cls_path)) {
      tem <- load_from_disk(tem_path)

      example_data <- LargeDataSetForText$new()
      example_data$add_from_data.frame(imdb_movie_reviews)

      test_embeddings <- tem$embed_large(text_dataset = example_data, trace = FALSE)

      predictions <- cls_model$predict(newdata = test_embeddings)

      expect_equal(nrow(predictions), nrow(imdb_movie_reviews))
      expect_s3_class(predictions, "data.frame")
      expect_true("expected_category" %in% colnames(predictions))
    }
  })
}

# Monitor test time
monitor_test_time_on_CI(
  start_time = test_time_start,
  test_name = "04_01_LoadOldModels"
)
