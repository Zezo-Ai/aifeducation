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

# Load python scripts
load_all_py_scripts()
run_py_file("data_collator.py")

# Path Management
test_art_path <- testthat::test_path("test_artefacts")
test_art_tmp_path <- testthat::test_path("test_artefacts/collators")
create_dir(test_art_path, FALSE)
create_dir(test_art_tmp_path, FALSE)

test_tmp_data_path <- testthat::test_path("test_data_tmp")
create_dir(test_tmp_data_path, FALSE)
test_tmp_data_base_model_path <- paste0(test_tmp_data_path, "/", "TEM")

# Data Management
example_data <- imdb_movie_reviews
raw_texts <- LargeDataSetForText$new(example_data)

# Genereate Tokenizer
raw_texts_training <- LargeDataSetForText$new(example_data[1:50, ])

tok_type <- "WordPieceTokenizer"
Tokenizer <- create_object(tok_type)
Tokenizer$configure(
  vocab_size = 2000,
  vocab_do_lower_case = sample(x = c(TRUE, FALSE), size = 1)
)
Tokenizer$train(
  text_dataset = raw_texts,
  statistics_max_tokens_length = 256,
  sustain_track = TRUE,
  sustain_iso_code = "DEU",
  sustain_region = NULL,
  sustain_interval = 15,
  sustain_log_level = "error",
  trace = FALSE
)

# Config Test
mlm_prob <- 0.5

tokenizer <- Tokenizer$get_tokenizer()
collator <- py$DataCollatorForWholeWordMask(tokenizer, mlm_probability = mlm_prob)

lines <- list(
  "Whole word masking is fun",
  "Short",
  "This is a much longer sentence to test padding"
)

max_length <- 20L
tokenized_lines <- lapply(lines, function(line) {
  tokenizer(line, truncation = TRUE, max_length = as.integer(max_length), return_special_tokens_mask = TRUE, return_attention_mask = TRUE)
})

# --- Applying DataCollator ---
batch <- collator(tokenized_lines)
input_ids <- batch$input_ids
labels <- batch$labels
attention_mask <- batch$attention_mask

pad_id <- tokenizer$pad_token_id
mask_id <- tokenizer$mask_token_id
cls_id <- tokenizer$cls_token_id
sep_id <- tokenizer$sep_token_id

input_ids_np <- input_ids$numpy()
labels_np <- labels$numpy()
mask_np <- attention_mask$numpy()


# --- Testing ---
test_that("Batch dims are correct", {
  expect_equal(input_ids$shape[0], length(lines))
  expect_true(all(input_ids$shape == labels$shape))
  expect_true(all(input_ids$shape == attention_mask$shape))
})

test_that("Padding and attention mask are correct", {
  for (i in seq_along(lines)) {
    seq_len <- sum(mask_np[i, ] != 0)
    if (seq_len < ncol(input_ids_np)) {
      expect_true(all(input_ids_np[i, (seq_len + 1):ncol(input_ids_np)] == pad_id))
    }
  }
})

test_that("Whole Word Masking is correct", {
  for (i in seq_along(lines)) {
    masked_positions <- which(labels_np[i, ] != -100)
    if (length(masked_positions) > 0) {
      expect_true(all(input_ids_np[i, masked_positions] == mask_id))
    }
  }
})

test_that("Number of words to mask is about mlm_probability", {
  for (i in seq_along(lines)) {
    # Get word_ids
    word_ids <- tokenized_lines[[i]]$word_ids()

    # Calculate number of words
    unique_words <- unique(Filter(Negate(is.null), word_ids))
    total_words <- length(unique_words)

    # Get masked positions and words
    masked_positions <- which(labels_np[i, ] != -100)
    masked_word_ids <- unique(word_ids[masked_positions])
    masked_word_ids <- Filter(Negate(is.null), masked_word_ids)
    masked_words <- length(unique(masked_word_ids))

    # Expected number of masked words
    expected_words <- max(1, round(total_words * mlm_prob))

    expect_true(abs(masked_words - expected_words) <= 1,
      info = paste(
        "Found", masked_words,
        "expected about", expected_words
      )
    )
  }
})

test_that("Special tokens [CLS] and [SEP] are not masked", {
  for (i in seq_along(lines)) {
    expect_equal(input_ids_np[i, 1], cls_id)
    last_real <- sum(mask_np[i, ] != 0)
    expect_equal(input_ids_np[i, last_real], sep_id)
  }
})

# test_that("Compatibility with a model BERT", {
#  model <- transformers$BertForMaskedLM$from_pretrained("bert-base-uncased")
#  expect_silent({
#    out <- model(input_ids = input_ids, labels = labels)
#  })
#  expect_true(!is.null(out$loss))
# })

# Monitor test time
monitor_test_time_on_CI(
  start_time = test_time_start,
  test_name = "01_07_DataCollator"
)
