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

#' @title Base class for tokenizers
#' @description Base class for tokenizers containing all methods shared by the sub-classes.
#' @return `r get_description("return_object")`
#' @family R6 Classes for Developers
#' @export
TokenizerBase <- R6::R6Class(
  classname = "TokenizerBase",
  inherit = AIFEMaster,
  private = list(
    tokenizer_statistics = data.frame(),
    configured = FALSE,
    #------------------------------------------------------------------------
    # Method for loading tokenizer statistics
    load_tokenizer_statistics = function(model_dir) {
      path <- file.path(model_dir, "tokenizer_statistics.csv")
      if (file.exists(path)) {
        private$tokenizer_statistics <- utils::read.csv(file = path)
      } else {
        private$tokenizer_statistics <- NA
      }
    },
    #------------------------------------------------------------------------
    # Method for saving tokenizer statistics
    save_tokenizer_statistics = function(dir_path, folder_name) {
      if (!is.null_or_na(private$tokenizer_statistics)) {
        save_location <- file.path(dir_path, folder_name)
        create_dir(dir_path, trace = TRUE, msg_fun = FALSE)
        write.csv(
          x = private$tokenizer_statistics,
          file = file.path(save_location, "tokenizer_statistics.csv"),
          row.names = FALSE,
          quote = FALSE
        )
      }
    }
  ),
  public = list(
    #--------------------------------------------------------------------------
    #' @description Method for saving a model on disk.
    #' @param dir_path `r get_description("save_dir")`
    #' @param folder_name `r get_param_doc_desc("folder_name")`
    #' @return `r get_description("return_save_on_disk")`
    #' @importFrom utils write.csv
    save = function(dir_path, folder_name) {
      check_type(object = dir_path, type = "string", FALSE)
      check_type(object = folder_name, type = "string", FALSE)

      # Create Directory and Folder
      save_location <- file.path(dir_path, folder_name)
      create_dir(dir_path, trace = TRUE, msg_fun = FALSE)
      create_dir(save_location, trace = TRUE, msg_fun = FALSE)

      # Save tokenizer statistics
      private$save_tokenizer_statistics(
        dir_path = dir_path,
        folder_name = folder_name
      )

      # Save Sustainability Data
      private$save_sustainability_data(dir_path = dir_path, folder_name = folder_name)

      # Save Tokenizer
      file_paths <- private$model$save_pretrained(save_location)
    },
    #--------------------------------------------------------------------------
    #' @description Loads an object from disk
    #' and updates the object to the current version of the package.
    #' @param dir_path `r get_description("load_dir")`
    #' @return `r get_description("return_load_on_disk")`
    load_from_disk = function(dir_path) {
      # Load private and public config files
      private$load_config_file(dir_path)

      # Load the tokenizer model
      private$model <- try(transformers$PreTrainedTokenizerFast$from_pretrained(dir_path), silent = TRUE)
      if (inherits(private$model, "try-error")) {
        private$model <- transformers$PreTrainedTokenizer$from_pretrained(dir_path)
      }

      # Load Tokenizer Statistics
      private$load_tokenizer_statistics(dir_path)

      # Load Sustainability Data
      private$load_sustainability_data(model_dir = dir_path)

      # Prevent modification
      private$set_configuration_to_TRUE()
      private$trained <- TRUE
    },
    # ------------------------------------------------------------------------
    #' @description Tokenizer statistics
    #' @return Returns a `data.frame` containing the tokenizer's statistics.
    get_tokenizer_statistics = function() {
      return(private$tokenizer_statistics)
    },
    # ------------------------------------------------------------------------
    #' @description Vocabulary. size
    #' @return Returns a `int` representing the size of the vocabulary.
    get_vocab_size = function() {
      return(private$model_config$vocab_size)
    },
    #--------------------------------------------------------------------------
    #' @description Python tokenizer
    #' @return Returns the python tokenizer within the model.
    get_tokenizer = function() {
      return(private$model)
    },
    #-------------------------------------------------------------------------
    #' @description Method for encoding words of raw texts into integers.
    #' @param raw_text `r get_param_doc_desc("raw_text")`
    #' @param token_overlap `r get_param_doc_desc("token_overlap")`
    #' @param max_token_sequence_length `r get_param_doc_desc("max_token_sequence_length")`
    #' @param n_chunks `r get_param_doc_desc("n_chunks")`
    #' @param token_encodings_only `r get_param_doc_desc("token_encodings_only")`
    #' @param return_token_type_ids `r get_param_doc_desc("return_token_type_ids")`
    #' @param token_to_int `r get_param_doc_desc("token_to_int")`
    #' @param trace `r get_param_doc_desc("trace")`
    #' @return `list` containing the integer or token sequences of the raw texts with
    #' special tokens.
    encode = function(raw_text,
                      token_overlap = 0L,
                      max_token_sequence_length = 512L,
                      n_chunks = 1L,
                      token_encodings_only = FALSE,
                      token_to_int = TRUE,
                      return_token_type_ids = TRUE,
                      trace = FALSE) {
      # Checking
      check_type(object = raw_text, type = "vector", FALSE)
      check_type(object = token_encodings_only, type = "bool", FALSE)
      check_type(object = token_to_int, type = "bool", FALSE)
      check_type(object = trace, type = "bool", FALSE)

      # Start
      n_units <- length(raw_text)
      #---------------------------------------------------------------------
      if (token_encodings_only) {
        encodings <- NULL
        encodings_only <- NULL
        for (i in 1L:n_units) {
          tokens_unit <- NULL

          tokens <- private$model(
            raw_text[i],
            stride = as.integer(token_overlap),
            padding = "max_length",
            truncation = TRUE,
            return_overflowing_tokens = TRUE,
            return_length = FALSE,
            return_offsets_mapping = FALSE,
            return_attention_mask = FALSE,
            max_length = as.integer(max_token_sequence_length),
            return_tensors = "np"
          )

          tmp_seq_len <- nrow(tokens[["input_ids"]])

          chunks <- min(tmp_seq_len, n_chunks)

          for (j in 1L:chunks) {
            tokens_unit[j] <- list(tokens["input_ids"][j, ])
            if (trace) {
              cat(paste(get_time_stamp(), i, "/", n_units, "block", j, "/", chunks, "\n"))
            }
          }
          encodings_only[i] <- list(tokens_unit)
        }
        if (token_to_int) {
          return(encodings_only)
        } else {
          # Convert ids to tokens

          token_seq_list <- NULL
          for (i in seq_len(length(encodings_only))) {
            tmp_sequence <- encodings_only[[i]]
            tmp_seqeunce_tok <- NULL
            for (j in seq_len(length(tmp_sequence))) {
              tmp_seqeunce_tok[length(tmp_seqeunce_tok) + 1L] <- list(
                private$model$convert_ids_to_tokens(
                  ids = as.integer(tmp_sequence[[j]]), skip_special_tokens = FALSE
                )
              )
            }
            token_seq_list[length(token_seq_list) + 1L] <- list(tmp_seqeunce_tok)
          }
          return(token_seq_list)
        }

        #--------------------------------------------------------------------
      } else {
        encodings <- NULL
        chunk_list <- vector(length = n_units)
        total_chunk_list <- vector(length = n_units)
        for (i in 1L:n_units) {
          tokens <- private$model(
            raw_text[i],
            stride = as.integer(token_overlap),
            padding = "max_length",
            truncation = TRUE,
            max_length = as.integer(max_token_sequence_length),
            return_overflowing_tokens = TRUE,
            return_length = FALSE,
            return_offsets_mapping = FALSE,
            return_attention_mask = TRUE,
            return_token_type_ids = return_token_type_ids,
            return_tensors = "pt"
          )

          tmp_dataset <- datasets$Dataset$from_dict(tokens)

          tmp_seq_len <- tmp_dataset$num_rows
          chunk_list[i] <- min(tmp_seq_len, n_chunks)
          total_chunk_list[i] <- tmp_seq_len
          if (chunk_list[i] == 1L) {
            tmp_dataset <- tmp_dataset$select(list(as.integer((1L:chunk_list[[i]]) - 1L)))
          } else {
            tmp_dataset <- tmp_dataset$select(as.integer((1L:chunk_list[[i]]) - 1L))
          }
          encodings <- datasets$concatenate_datasets(c(encodings, tmp_dataset))
        }
        return(encodings_list = list(
          encodings = encodings,
          chunks = chunk_list,
          total_chunks = total_chunk_list
        ))
      }
    },
    #--------------------------------------------------------------------------
    #' @description Method for decoding a sequence of integers into tokens
    #' @param int_seqence `r get_param_doc_desc("int_seqence")`
    #' @param to_token `r get_param_doc_desc("to_token")`
    #' @return `list` of token sequences
    decode = function(int_seqence, to_token = FALSE) {
      # Check
      check_type(object = int_seqence, type = "list", FALSE)
      check_type(object = to_token, type = "bool", FALSE)

      # Start
      tmp_token_list <- NULL
      for (i in seq_len(length(int_seqence))) {
        tmp_seq_token_list <- NULL
        for (j in seq_len(length(int_seqence[[i]]))) {
          tmp_vector <- int_seqence[[i]][[j]]
          mode(tmp_vector) <- "integer"
          if (!to_token) {
            tmp_seq_token_list[j] <- list(private$model$decode(
              token_ids = tmp_vector,
              skip_special_tokens = TRUE
            ))
          } else {
            tmp_seq_token_list[j] <- list(paste(private$model$convert_ids_to_tokens(tmp_vector), collapse = " "))
          }
        }
        tmp_token_list[i] <- list(tmp_seq_token_list)
      }
      return(tmp_token_list)
    },
    #---------------------------------------------------------------------------
    #' @description Method for receiving the special tokens of the model
    #' @return Returns a `matrix` containing the special tokens in the rows
    #' and their type, token, and id in the columns.
    get_special_tokens = function() {
      special_tokens <- c(
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token"
      )
      tokens_map <- matrix(
        nrow = length(special_tokens),
        ncol = 3L,
        data = NA
      )
      colnames(tokens_map) <- c("type", "token", "id")
      rownames(tokens_map) <- special_tokens

      for (i in seq_len(length(special_tokens))) {
        tokens_map[i, 1L] <- special_tokens[i]
        tokens_map[i, 2L] <- replace_null_with_na(private$model[special_tokens[i]])
        tokens_map[i, 3L] <- replace_null_with_na(
          private$model[paste0(special_tokens[i], "_id")]
        )
      }
      return(tokens_map)
    },
    #--------------------------------------------------------------------------
    #' @description Method for receiving the special tokens of the model
    #' @return Returns an 'int' counting the number of special tokens.
    n_special_tokens = function() {
      special_tokens <- self$get_special_tokens()
      return(
        length(
          unique(
            special_tokens[, "token"]
          )
        )
      )
    },
    #--------------------------------------------------------------------------
    #' @description Method for calculating tokenizer statistics as suggested by
    #' Kaya and Tantuğ (2024).
    #'
    #' Kaya, Y. B., & Tantuğ, A. C. (2024). Effect of tokenization granularity
    #' for Turkish large language models. Intelligent Systems with
    #' Applications, 21, 200335.
    #' \ifelse{text}{\doi{doi:10.1016/j.iswa.2024.200335}}{<https://doi.org/10.1016/j.iswa.2024.200335>}
    #'
    #' @param text_dataset `r get_param_doc_desc("text_dataset")`
    #' @param statistics_max_tokens_length `r get_param_doc_desc("statistics_max_tokens_length")`
    #' @param step `string` describing the context of the estimation.
    #' @returns Returns a `data.frame` containing the estimates.
    calculate_statistics = function(text_dataset,
                                    statistics_max_tokens_length,
                                    step = "creation") {
      # Calculate tokenizer statistics
      tokenized_texts_raw <- tokenize_dataset(
        dataset = text_dataset$get_dataset(),
        tokenizer = private$model,
        max_length = statistics_max_tokens_length,
        add_special_tokens = FALSE,
        log_file = NULL,
        write_interval = 2L,
        value_top = 1L,
        total_top = 1L,
        message_top = "NA"
      )

      statistics <- as.data.frame(
        calc_tokenizer_statistics(
          dataset = tokenized_texts_raw,
          statistics_max_tokens_length = statistics_max_tokens_length,
          step = step
        )
      )
      return(statistics)
    },
    #--------------------------------------------------------------------------
    #' @description Method for calculating the necessary number of chunks to cover
    #' a specific quantile of documents completely.
    #' @description Tokenizer split raw texts into a sequences of tokens. Most neural nets can only
    #' process a given sequence length. Thus, long texts must be split into a
    #' sequence of sequences in order to represent the whole text. This method calculates
    #' how many sequences are necessary. The number of sequences of sequences is called chunks.
    #' @description You can think of these chunks as dividing a long texts into several paragraphs.
    #' @param text_dataset `r get_param_doc_desc("text_dataset")`
    #' @param batch_size `r get_param_doc_desc("batch_size")`
    #' @param seq_len_tokens `r get_param_doc_desc("seq_len_tokens")`
    #' @param token_overlap `r get_param_doc_desc("token_overlap")`
    #' @param trace `r get_param_doc_desc("trace")`
    #' @returns Returns a `vector` containing the number of chunks to cover a given
    #' quantile of documents completely.
    calc_quantiles = function(text_dataset, batch_size = 32L, seq_len_tokens = 512L, token_overlap = 0L, trace = TRUE) {
      check_class_and_type(
        object = text_dataset,
        object_name = "text_dataset",
        type_classes = "LargeDataSetForText",
        allow_NULL = FALSE
      )
      check_class_and_type(
        object = seq_len_tokens,
        object_name = "seq_len_tokens",
        type_classes = "int",
        min = 2L,
        max = NULL,
        allow_NULL = FALSE
      )
      check_class_and_type(
        object = token_overlap,
        object_name = "token_overlap",
        type_classes = "int",
        min = 0L,
        max = NULL,
        allow_NULL = FALSE
      )
      check_class_and_type(
        object = batch_size,
        object_name = "batch_size",
        type_classes = "int",
        min = 1L,
        max = NULL,
        allow_NULL = FALSE
      )
      n_texts <- text_dataset$n_rows()
      results <- vector(length = n_texts)

      # Get total number of batches for the loop
      total_number_of_bachtes <- ceiling(text_dataset$n_rows() / batch_size)

      # Get indices for every batch
      batches_index <- get_batches_index(
        number_rows = text_dataset$n_rows(),
        batch_size = batch_size,
        zero_based = TRUE
      )

      PgrInd <- ProgressIndicator$new(
        max_iter = total_number_of_bachtes,
        inc_absolute = TRUE,
        calc_eta = TRUE
      )

      for (i in seq(total_number_of_bachtes)) {
        tmp_subset <- text_dataset$select(as.integer(batches_index[[i]]))
        tmp_encodings <- self$encode(
          raw_text = extract_column_from_py_dataset(
            py_dataset = tmp_subset,
            column_name = "text",
            format = "R"
          ),
          token_overlap = token_overlap,
          max_token_sequence_length = seq_len_tokens,
          n_chunks = 1L,
          token_encodings_only = FALSE,
          token_to_int = TRUE,
          return_token_type_ids = FALSE,
          trace = FALSE
        )
        results[batches_index[[i]] + 1] <- tmp_encodings$total_chunks
        if (trace) {
          PgrInd$print_step(
            iter = i,
            text_pre = "Calculating Chunks",
            text_post = "done"
          )
        }
      }
      return(
        ceiling(
          quantile(
            x = results,
            probs = c(.10, .20, .30, .40, .50, .60, .70, .75, .80, .85, .90, .95, .99, .999,.9999,.99999, 1.0)
          )
        )
      )
    },
    #--------------------------------------------------------------------------
    #' @description Print method for classifiers.
    #' @return Prints a short description of the object.
    print = function() {
      rows <- c("Object", "Configured", "Trained", "Vocab Size", "Tokens/Word", "Mask Token", "Pad Token", "Unk Token")
      padded_rows <- pad_str(rows, width = NULL, pad = " ", end = ": ")
      statistics <- self$get_tokenizer_statistics()
      special_tokens <- self$get_special_tokens()
      message(
        appendLF = FALSE,
        padded_rows[1L], class(self)[1L], "\n",
        padded_rows[2L], private$configured, "\n",
        padded_rows[3L], private$trained[1L], "\n",
        padded_rows[4L], self$get_vocab_size(), "\n",
        padded_rows[5L], statistics[1L, "mu_g"], "\n",
        padded_rows[6L], special_tokens["mask_token", "token"], "\n",
        padded_rows[7L], special_tokens["pad_token", "token"], "\n",
        padded_rows[8L], special_tokens["unk_token", "token"], "\n"
      )
    }
  )
)


#' @title WordPieceTokenizer
#' @description Tokenizer based on the WordPiece model (Wu et al. 2016).
#' @return `r get_description("return_object")`
#' @references Wu, Y., Schuster, M., Chen, Z., Le, Q. V., Norouzi, M., Macherey, W.,
#' Krikun, M., Cao, Y., Gao, Q., Macherey, K., Klingner, J., Shah, A.,
#' Johnson, M., Liu, X., Kaiser, Ł., Gouws, S., Kato, Y., Kudo, T., Kazawa,
#' H., . . . Dean, J. (2016). Google's Neural Machine Translation System:
#' Bridging the Gap between Human and Machine Translation.
#' \ifelse{text}{\doi{doi:10.48550/arXiv.1609.08144}}{<https://doi.org/10.48550/arXiv.1609.08144>}
#'
#' @family Tokenizer
#' @export
WordPieceTokenizer <- R6::R6Class(
  classname = "WordPieceTokenizer",
  inherit = TokenizerBase,
  private = list(),
  public = list(
    #--------------------------------------------------------------------------
    #' @description Configures a new object of this class.
    #' @param vocab_size `r get_param_doc_desc("vocab_size")`
    #' @param vocab_do_lower_case `r get_param_doc_desc("vocab_do_lower_case")`
    #' @return `r get_description("return_nothing")`
    configure = function(vocab_size = 10000L,
                         vocab_do_lower_case = FALSE) {
      private$check_config_for_FALSE()

      private$save_all_args(
        args = get_called_args(n = 1L),
        group = "configure"
      )

      # Set package versions
      private$set_package_versions()

      # Set configured to TRUE to avoid changes in the model
      private$set_configuration_to_TRUE()
    },
    #--------------------------------------------------------------------------
    #' @description Trains a new object of this class
    #' @param text_dataset `r get_param_doc_desc("text_dataset")`
    #' @param statistics_max_tokens_length `r get_param_doc_desc("statistics_max_tokens_length")`
    #' @param sustain_track `r get_param_doc_desc("sustain_track")`
    #' @param sustain_iso_code `r get_param_doc_desc("sustain_iso_code")`
    #' @param sustain_region `r get_param_doc_desc("sustain_region")`
    #' @param sustain_interval `r get_param_doc_desc("sustain_interval")`
    #' @param sustain_log_level `r get_param_doc_desc("sustain_log_level")`
    #' @param trace `r get_param_doc_desc("trace")`
    #' @return `r get_description("return_nothing")`
    train = function(text_dataset,
                     statistics_max_tokens_length = 512L,
                     sustain_track = FALSE,
                     sustain_iso_code = NULL,
                     sustain_region = NULL,
                     sustain_interval = 15L,
                     sustain_log_level = "warning",
                     trace = FALSE) {
      private$check_config_for_TRUE()
      private$check_for_untrained()

      private$save_all_args(
        args = get_called_args(n = 1L),
        group = "training"
      )

      private$init_and_start_sustainability_tracking()

      # Define tokens
      sep_token <- "[SEP]"
      sep_id <- 1L
      cls_token <- "[CLS]"
      cls_id <- 0L
      unk_token <- "[UNK]"
      pad_token <- "[PAD]"
      mask_token <- "[MASK]"
      bos_token <- "[CLS]"
      eos_token <- "[SEP]"

      special_tokens <- c(
        cls_token,
        sep_token,
        unk_token,
        pad_token,
        mask_token,
        bos_token,
        eos_token
      )

      # Create model
      tok_new <- tok$Tokenizer(tok$models$WordPiece(unk_token = unk_token))
      tok_new$normalizer <- tok$normalizers$BertNormalizer(
        lowercase = private$model_config$vocab_do_lower_case,
        clean_text = TRUE,
        handle_chinese_chars = TRUE,
        strip_accents = private$model_config$vocab_do_lower_case
      )
      tok_new$pre_tokenizer <- tok$pre_tokenizers$BertPreTokenizer()
      tok_new$post_processor <- tok$processors$BertProcessing(
        sep = reticulate::tuple(list(sep_token, as.integer(sep_id))),
        cls = reticulate::tuple(list(cls_token, as.integer(cls_id)))
      )

      tok_new$decoder <- tok$decoders$WordPiece()

      # configurate training
      trainer <- tok$trainers$WordPieceTrainer(
        vocab_size = as.integer(private$model_config$vocab_size),
        special_tokens = special_tokens,
        show_progress = trace
      )

      # calculate the model
      run_py_file("datasets_transformer_compute_vocabulary.py")

      tok_new$train_from_iterator(
        iterator = py$batch_iterator(
          batch_size = 200L,
          dataset = text_dataset$get_dataset(),
          log_file = NULL,
          write_interval = 2L,
          value_top = 0L,
          total_top = 1L,
          message_top = "NA"
        ),
        trainer = trainer,
        length = as.integer(text_dataset$n_rows())
      )

      # Create the complete and final model
      private$model <- transformers$PreTrainedTokenizerFast(
        tokenizer_object = tok_new,
        unk_token = unk_token,
        sep_token = sep_token,
        pad_token = pad_token,
        cls_token = cls_token,
        mask_token = mask_token,
        bos_token = bos_token,
        eos_token = eos_token,
        padding_side = "right",
        truncation_side = "right"
      )

      # Calculate tokenizer statistics
      private$tokenizer_statistics <- self$calculate_statistics(
        text_dataset = text_dataset,
        statistics_max_tokens_length = statistics_max_tokens_length,
        step = "creation"
      )

      # Update
      private$model_config$vocab_size <- length(private$model$get_vocab())

      # Set trained field
      private$trained <- TRUE

      private$stop_sustainability_tracking("Create tokenizer")
    }
  )
)
# Add the model to the user list
TokenizerIndex$WordPieceTokenizer <- ("WordPieceTokenizer")


# ===============================================================================

#' @title HuggingFaceTokenizer
#' @description Abstract class for all tokenizers used with the 'transformers' library.
#' @return `r get_description("return_object")`
#' @family Tokenizer
#' @export
HuggingFaceTokenizer <- R6::R6Class(
  classname = "HuggingFaceTokenizer",
  inherit = TokenizerBase,
  private = list(
    load_config_file = function(dir_path) {}
  ),
  public = list(
    #--------------------------------------------------------------------------
    #' @description Creates a tokenizer from a pretrained model
    #' @param model_dir `r get_description("model_dir")`
    #' @param tokenizer_type `string` Name of the tokenizer belonging to a base
    #' model (e.g. BertTokenizer). This parameter is only used if an error occurs during
    #' loading as fall back option.
    #' @return `r get_description("return_object")`
    create_from_hf = function(model_dir, tokenizer_type = NULL) {
      # Load the model
      private$model <- try(transformers$AutoTokenizer$from_pretrained(model_dir), silent = TRUE)
      if (inherits(private$model, "try-error")) {
        private$model <- transformers[tokenizer_type]$from_pretrained(model_dir)
      }

      # Set configured to TRUE to avoid changes in the model
      private$set_configuration_to_TRUE()

      # Set package versions
      private$set_package_versions()

      # Set trained field
      private$trained <- TRUE
    },
    # ------------------------------------------------------------------------
    #' @description Vocabulary. size
    #' @return Returns a `int` representing the size of the vocabulary.
    get_vocab_size = function() {
      return(private$model$vocab_size)
    }
  )
)
# Add the model to the user list
TokenizerIndex$HuggingFaceTokenizer <- ("HuggingFaceTokenizer")
