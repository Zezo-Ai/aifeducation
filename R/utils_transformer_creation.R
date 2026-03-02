#' @title Dataset tokenization
#' @description A given dataset must contain a column 'text' storing raw texts.
#'
#' @param dataset `datasets.arrow_dataset.Dataset` Dataset that contains a column 'text' storing the raw texts.
#' @param tokenizer `transformers.Tokenizer()` Tokenizer.
#' @param max_length `integer` Max length for a given tokenizer.
#'
#' @return Tokenized dataset with a given tokenizer.
#'
#' @family Utils Transformers Creation Developers
#' @keywords internal
#' @noRd
tokenize_dataset <- function(dataset, tokenizer, max_length, add_special_tokens = TRUE,
                             log_file = NULL, write_interval = 2,
                             value_top = 0, total_top = 1, message_top = "NA") {
  run_py_file("datasets_transformer_prepare_data.py")

  batch_size <- 2L

  id <- as.character(generate_id(16))

  tokenized_texts_raw <- dataset$map(
    py$tokenize_raw_text,
    batched = TRUE,
    batch_size = batch_size,
    load_from_cache_file = FALSE,
    keep_in_memory = FALSE,
    cache_file_name = create_py_dataset_cache_file_path(paste0(create_and_get_tmp_dir(), "/", id)),
    new_fingerprint = id,
    fn_kwargs = reticulate::dict(
      list(
        tokenizer = tokenizer,
        truncation = TRUE,
        padding = FALSE,
        max_length = as.integer(max_length),
        return_overflowing_tokens = TRUE,
        return_length = TRUE,
        return_special_tokens_mask = TRUE,
        return_offsets_mapping = FALSE,
        return_attention_mask = TRUE,
        return_tensors = "np",
        request_word_ids = TRUE,
        log_file = log_file,
        write_interval = write_interval,
        value_top = value_top, total_top = total_top, message_top = message_top,
        total_middle = floor(dataset$num_rows / batch_size),
        add_special_tokens = add_special_tokens
      )
    ),
    remove_columns = dataset$column_names
  )
  return(tokenized_texts_raw)
}
