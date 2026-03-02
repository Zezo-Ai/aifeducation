testthat::skip_on_cran()

testthat::skip_if_not(
  condition = check_aif_py_modules(trace = FALSE),
  message = "Necessary python modules not available"
)
load_all_py_scripts()

# Start time
test_time_start <- Sys.time()

# Config transformer library
transformers$utils$logging$set_verbosity_error()
os$environ$setdefault("TOKENIZERS_PARALLELISM", "false")

# Disable tqdm progressbar
transformers$logging$disable_progress_bar()
datasets$disable_progress_bars()

# Path Management
test_art_path <- testthat::test_path("test_artefacts")
test_art_tmp_path <- testthat::test_path("test_artefacts/tokenizers")
tmp_full_models_pt_path <- paste0(test_art_tmp_path, "/pytorch")
create_dir(test_art_path, FALSE)
create_dir(test_art_tmp_path, FALSE)
create_dir(tmp_full_models_pt_path, FALSE)

# Data Mangement
example_data <- imdb_movie_reviews
raw_texts <- LargeDataSetForText$new(example_data)

# Test Configuration
object_class_names <- setdiff(x = TokenizerIndex, y = "HuggingFaceTokenizer")
samples_per_object <- 5

# object_class_names=c("WordPieceTokenizer")
# object_class_names=c("BPETokenizer")

# Tests
for (object_class_name in object_class_names) {
  for (i in 1:samples_per_object) {
    config_args <- generate_args_for_tests(
      object_name = object_class_name,
      method = "configure",
      var_objects = list(),
      necessary_objects = list(
        text_dataset = raw_texts
      )
    )
    training_args <- generate_args_for_tests(
      object_name = object_class_name,
      method = "train",
      var_objects = list(),
      necessary_objects = list(
        text_dataset = raw_texts
      )
    )


    test_that(paste(object_class_name, get_current_args_for_print(config_args), get_current_args_for_print(training_args)), {
      tokenizer <- create_object(object_class_name)
      suppressMessages(
        do.call(
          what = tokenizer$configure,
          args = config_args
        )
      )

      suppressMessages(
        do.call(
          what = tokenizer$train,
          args = training_args
        )
      )

      # tokenizer statistics
      expect_equal(
        colnames(tokenizer$get_tokenizer_statistics()),
        c("step", "date", "max_tokens_length", "n_sequences", "n_words", "n_tokens", "mu_t", "mu_w", "mu_g")
      )
      expect_equal(nrow(tokenizer$get_tokenizer_statistics()), 1)

      # special tokens
      expect_equal(
        colnames(tokenizer$get_special_tokens()),
        c("type", "token", "id")
      )
      expect_equal(ncol(tokenizer$get_tokenizer_statistics()), 9)
      expect_equal(nrow(tokenizer$get_tokenizer_statistics()), 1)

      # Sustainability
      if (training_args$sustain_track == TRUE) {
        expect_equal(nrow(tokenizer$get_sustainability_data()), 1)
      } else {
        expect_equal(nrow(tokenizer$get_sustainability_data()), 0)
      }

      # Save and Load
      save_to_disk(
        object = tokenizer,
        dir_path = test_art_tmp_path,
        folder_name = "WordPieceTokenizer"
      )

      # Load from disk
      tokenizer2 <- load_from_disk(
        dir_path = paste0(test_art_tmp_path, "/", "WordPieceTokenizer")
      )

      expect_equal(tokenizer$get_special_tokens(), tokenizer2$get_special_tokens())
      expect_equal(tokenizer$get_sustainability_data(), tokenizer2$get_sustainability_data())
      expect_equal(tokenizer$get_tokenizer_statistics(), tokenizer2$get_tokenizer_statistics())

      expect_equal(
        tokenizer$encode(raw_text = "This is a test.", token_encodings_only = TRUE),
        tokenizer2$encode(raw_text = "This is a test.", token_encodings_only = TRUE)
      )

      # Clear directory for next test
      unlink(paste0(test_art_tmp_path, "/", "WordPieceTokenizer"), recursive = TRUE)
    })
  }
}


# Clean Directory
if (dir.exists(test_art_tmp_path)) {
  unlink(
    x = test_art_tmp_path,
    recursive = TRUE
  )
}

# Monitor test time
monitor_test_time_on_CI(
  start_time = test_time_start,
  test_name = "02_05_tokenizer"
)
