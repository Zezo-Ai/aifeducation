testthat::skip_on_cran()
testthat::skip_if_not(
  condition = check_aif_py_modules(trace = FALSE),
  message = "Necessary python modules not available"
)
load_all_py_scripts()

# Start time
test_time_start <- Sys.time()


# SetUp Test---------------------------------------------------------------------
root_path_general_data <- testthat::test_path("test_data/Embeddings")
create_dir(testthat::test_path("test_artefacts"), FALSE)
root_path_results <- testthat::test_path("test_artefacts/LargeDataSetForTextEmbeddings")
create_dir(root_path_results, FALSE)

# object is imdb_embeddings
imdb_embeddings <- load_from_disk(paste0(root_path_general_data, "/imdb_embeddings"))

imdb_embeddings <- imdb_embeddings$convert_to_LargeDataSetForTextEmbeddings()
imdb_embeddings <- imdb_embeddings$convert_to_EmbeddedText()

# Start test---------------------------------------------------------------------
test_that("LargeDataSetForTextEmbeddings - Create", {
  expect_no_error(LargeDataSetForTextEmbeddings$new())

  data_set <- LargeDataSetForTextEmbeddings$new()

  expect_no_error(data_set$configure(
    model_name = imdb_embeddings$get_model_info()$model_name,
    model_label = imdb_embeddings$get_model_info()$model_label,
    model_date = imdb_embeddings$get_model_info()$model_date,
    model_method = imdb_embeddings$get_model_info()$model_method,
    model_version = imdb_embeddings$get_model_info()$model_version,
    model_language = imdb_embeddings$get_model_info()$model_language,
    param_seq_length = imdb_embeddings$get_model_info()$param_seq_length,
    param_chunks = imdb_embeddings$get_model_info()$param_chunks,
    param_features = imdb_embeddings$get_model_info()$param_features,
    param_overlap = imdb_embeddings$get_model_info()$param_overlap,
    param_emb_layer_min = imdb_embeddings$get_model_info()$param_emb_layer_min,
    param_emb_layer_max = imdb_embeddings$get_model_info()$param_emb_layer_max,
    param_emb_pool_type = imdb_embeddings$get_model_info()$param_emb_pool_type,
    param_aggregation = imdb_embeddings$get_model_info()$param_aggregation,
    param_pad_value = -100
  ))
})

# Test basic parameters--------------------------------------------------------
test_that("LargeDataSetForTextEmbeddings - No FeatureExtractor", {
  new_dataset <- LargeDataSetForTextEmbeddings$new()
  new_dataset$configure(
    model_name = imdb_embeddings$get_model_info()$model_name,
    model_label = imdb_embeddings$get_model_info()$model_label,
    model_date = imdb_embeddings$get_model_info()$model_date,
    model_method = imdb_embeddings$get_model_info()$model_method,
    model_version = imdb_embeddings$get_model_info()$model_version,
    model_language = imdb_embeddings$get_model_info()$model_language,
    param_seq_length = imdb_embeddings$get_model_info()$param_seq_length,
    param_chunks = imdb_embeddings$get_model_info()$param_chunks,
    param_features = imdb_embeddings$get_features(),
    param_overlap = imdb_embeddings$get_model_info()$param_overlap,
    param_emb_layer_min = imdb_embeddings$get_model_info()$param_emb_layer_min,
    param_emb_layer_max = imdb_embeddings$get_model_info()$param_emb_layer_max,
    param_emb_pool_type = imdb_embeddings$get_model_info()$param_emb_pool_type,
    param_aggregation = imdb_embeddings$get_model_info()$param_aggregation,
    param_pad_value = -100
  )

  # Correct Features
  expect_equal(new_dataset$get_features(), imdb_embeddings$get_features())

  # Correct original features
  expect_equal(new_dataset$get_original_features(), imdb_embeddings$get_features())

  # Correct Times
  expect_equal(new_dataset$get_times(), imdb_embeddings$get_times())

  # Check model information
  for (entry in names(new_dataset$get_model_info())) {
    expect_equal(
      new_dataset$get_model_info()[entry],
      imdb_embeddings$get_model_info()[entry]
    )
  }

  # Correct padding value
  expect_equal(new_dataset$get_pad_value(), -100)

  # Add embeddings from array
  expect_no_error(
    new_dataset$add_embeddings_from_array(imdb_embeddings$embeddings)
  )
  expect_equal(new_dataset$n_rows(), nrow(imdb_embeddings$embeddings))

  # add from EbbeddedText
  expect_no_error(
    new_dataset$add_embeddings_from_EmbeddedText(imdb_embeddings)
  )
  expect_equal(new_dataset$n_rows(), 2 * nrow(imdb_embeddings$embeddings))

  # Compression test
  expect_false(new_dataset$is_compressed())

  # Conversation
  new_dataset$reduce_to_unique_ids()
  new_data_set_converted <- new_dataset$convert_to_EmbeddedText()
  expect_equal(nrow(new_data_set_converted$embeddings), nrow(imdb_embeddings$embeddings))
  for (entry in names(new_data_set_converted$get_model_info())) {
    expect_equal(
      new_data_set_converted$get_model_info()[entry],
      new_dataset$get_model_info()[entry]
    )
  }

  # Correct padding value
  expect_equal(new_data_set_converted$get_pad_value(), -100)

  # Selection test
  one_case <- new_dataset$select(3)
  expect_equal(one_case$num_rows, 1)
  multiple_cases <- new_dataset$select(c(0, 1, 2, 3))
  expect_equal(multiple_cases$num_rows, 4)

  #Column extraction test
  expect_all_true(
    inherits(
      new_dataset$extract_column("input",format="R"),
      "array"
    )
  )
  expect_all_true(
    inherits(
      new_dataset$extract_column("input",format="torch"),
      "torch.Tensor"
    )
  )
  expect_all_true(
    inherits(
      new_dataset$extract_column("input",format="numpy"),
      "numpy.ndarray"
    )
  )
})

#-----------------------------------------------------------------------------
test_that("LargeDataSetForTextEmbeddings - Method Save and Load", {
  save_path <- paste0(root_path_results, "/dataset")
  folder_name <- generate_id()
  load_path <- paste0(save_path, "/", folder_name)

  new_dataset <- LargeDataSetForTextEmbeddings$new()
  new_dataset$configure(
    model_name = imdb_embeddings$get_model_info()$model_name,
    model_label = imdb_embeddings$get_model_info()$model_label,
    model_date = imdb_embeddings$get_model_info()$model_date,
    model_method = imdb_embeddings$get_model_info()$model_method,
    model_version = imdb_embeddings$get_model_info()$model_version,
    model_language = imdb_embeddings$get_model_info()$model_language,
    param_seq_length = imdb_embeddings$get_model_info()$param_seq_length,
    param_chunks = imdb_embeddings$get_model_info()$param_chunks,
    param_features = imdb_embeddings$get_features(),
    param_overlap = imdb_embeddings$get_model_info()$param_overlap,
    param_emb_layer_min = imdb_embeddings$get_model_info()$param_emb_layer_min,
    param_emb_layer_max = imdb_embeddings$get_model_info()$param_emb_layer_max,
    param_emb_pool_type = imdb_embeddings$get_model_info()$param_emb_pool_type,
    param_aggregation = imdb_embeddings$get_model_info()$param_aggregation,
    param_pad_value = -100
  )

  new_dataset$add_embeddings_from_array(imdb_embeddings$embeddings)
  expect_no_error(new_dataset$save(save_path, folder_name = folder_name))

  new_dataset$load(load_path)

  # Number of cases
  expect_equal(new_dataset$n_rows(), nrow(imdb_embeddings$embeddings))
})
#-----------------------------------------------------------------------------
test_that("LargeDataSetForTextEmbeddings - Function Save and Load", {
  save_path <- paste0(root_path_results, "/dataset")
  folder_name <- generate_id()
  load_path <- paste0(save_path, "/", folder_name)

  new_dataset <- LargeDataSetForTextEmbeddings$new()
  new_dataset$configure(
    model_name = imdb_embeddings$get_model_info()$model_name,
    model_label = imdb_embeddings$get_model_info()$model_label,
    model_date = imdb_embeddings$get_model_info()$model_date,
    model_method = imdb_embeddings$get_model_info()$model_method,
    model_version = imdb_embeddings$get_model_info()$model_version,
    model_language = imdb_embeddings$get_model_info()$model_language,
    param_seq_length = imdb_embeddings$get_model_info()$param_seq_length,
    param_chunks = imdb_embeddings$get_model_info()$param_chunks,
    param_features = imdb_embeddings$get_features(),
    param_overlap = imdb_embeddings$get_model_info()$param_overlap,
    param_emb_layer_min = imdb_embeddings$get_model_info()$param_emb_layer_min,
    param_emb_layer_max = imdb_embeddings$get_model_info()$param_emb_layer_max,
    param_emb_pool_type = imdb_embeddings$get_model_info()$param_emb_pool_type,
    param_aggregation = imdb_embeddings$get_model_info()$param_aggregation,
    param_pad_value = -100
  )

  new_dataset$add_embeddings_from_array(imdb_embeddings$embeddings)

  # Save
  expect_no_error(save_to_disk(
    object = new_dataset,
    dir_path = save_path,
    folder_name = folder_name
  ))
  # Load
  new_dataset <- NULL
  new_dataset <- load_from_disk(dir_path = load_path)

  # Number of cases
  expect_equal(new_dataset$n_rows(), nrow(imdb_embeddings$embeddings))

  # Correct Features
  expect_equal(new_dataset$get_features(), imdb_embeddings$get_features())

  # Correct original features
  expect_equal(new_dataset$get_original_features(), imdb_embeddings$get_features())

  # Correct Times
  expect_equal(new_dataset$get_times(), imdb_embeddings$get_times())

  # Check model information
  for (entry in names(new_dataset$get_model_info())) {
    expect_equal(
      new_dataset$get_model_info()[entry],
      imdb_embeddings$get_model_info()[entry]
    )
  }
})

# Clean Directory
if (dir.exists(root_path_results)) {
  unlink(
    x = root_path_results,
    recursive = TRUE
  )
}

# Monitor test time
monitor_test_time_on_CI(
  start_time = test_time_start,
  test_name = "03_02_LargeDataSetForTextEmbeddings"
)
