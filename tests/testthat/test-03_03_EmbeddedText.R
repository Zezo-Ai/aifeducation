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
root_path_results <- testthat::test_path("test_artefacts/EmbeddedTexts")
create_dir(root_path_results, FALSE)

# SetUp datasets
# Disable tqdm progressbar
transformers$logging$disable_progress_bar()
datasets$disable_progress_bars()

# object is imdb_embeddings
imdb_embeddings <- load_from_disk(paste0(root_path_general_data, "/imdb_embeddings"))

# Start test---------------------------------------------------------------------
test_that("EmbeddedText - Create", {
  expect_no_error(EmbeddedText$new())

  new_data_set <- EmbeddedText$new()
  expect_no_error(new_data_set$configure(
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
    param_pad_value = -100,
    embeddings = imdb_embeddings$embeddings
  ))
})

# Test basic parameters--------------------------------------------------------
test_that("EmbeddedText - No FeatureExtractor", {
  new_embedded_text <- EmbeddedText$new()
  new_embedded_text$configure(
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
    param_pad_value = -100,
    embeddings = imdb_embeddings$embeddings
  )

  # Correct Features
  expect_equal(new_embedded_text$get_features(), imdb_embeddings$get_features())

  # Correct original features
  expect_equal(new_embedded_text$get_original_features(), imdb_embeddings$get_features())

  # Correct Times
  expect_equal(new_embedded_text$get_times(), imdb_embeddings$get_times())

  # Check model information
  for (entry in names(new_embedded_text$get_model_info())) {
    expect_equal(
      new_embedded_text$get_model_info()[entry],
      imdb_embeddings$get_model_info()[entry]
    )
  }

  # Correct padding value
  expect_equal(new_embedded_text$get_pad_value(), -100)

  # Compression test
  expect_false(new_embedded_text$is_compressed())

  # Conversation
  new_data_set_converted <- new_embedded_text$convert_to_LargeDataSetForTextEmbeddings()
  expect_equal(new_data_set_converted$n_rows(), nrow(imdb_embeddings$embeddings))
  for (entry in names(new_data_set_converted$get_model_info())) {
    expect_equal(
      new_data_set_converted$get_model_info()[entry],
      new_embedded_text$get_model_info()[entry]
    )
  }

  # Correct padding value
  expect_equal(new_data_set_converted$get_pad_value(), -100)
})

# Print Method-------------------------------------------------------------------
test_that("LargeDataSetForTexts - print method", {
  new_embedded_text <- EmbeddedText$new()
  new_embedded_text$configure(
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
    param_pad_value = -100,
    embeddings = imdb_embeddings$embeddings
  )
  suppressMessages(
    expect_no_error(new_embedded_text$print())
  )
  suppressMessages(
    expect_no_error(print(new_embedded_text))
  )
})

# Test basic parameters--------------------------------------------------------
test_that("EmbeddedText - Save and Load", {
  new_embedded_text <- EmbeddedText$new()
  new_embedded_text$configure(
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
    param_pad_value = -100,
    embeddings = imdb_embeddings$embeddings
  )

  folder_name <- "embedded_text_test"
  save_to_disk(
    object = new_embedded_text,
    dir_path = root_path_results,
    folder_name = folder_name
  )

  loaded_embeddings <- load_from_disk(
    dir_path = file.path(root_path_results, folder_name)
  )
  expect_equal(
    loaded_embeddings$get_model_info(),
    new_embedded_text$get_model_info()
  )
  expect_equal(
    loaded_embeddings$get_times(),
    new_embedded_text$get_times()
  )
  expect_equal(
    loaded_embeddings$get_model_label(),
    new_embedded_text$get_model_label()
  )
  expect_equal(
    loaded_embeddings$is_compressed(),
    new_embedded_text$is_compressed()
  )
  expect_equal(
    loaded_embeddings$is_configured(),
    new_embedded_text$is_configured()
  )
  expect_equal(
    loaded_embeddings$embeddings,
    new_embedded_text$embeddings
  )
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
  test_name = "03_03_EmbeddedText"
)
