# This file is part of the R package "aifeducation".
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as published by
# the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>

#' @title Get dictionary of all parameters
#' @description Function provides a `list` containing important characteristics
#' of the parameter used in the models. The `list` does contain only the definition of
#' arguments for transformer models and all classifiers. The arguments of other functions
#' in this package are documented separately.
#'
#' The aim of this list is to automatize argument checking and widget generation for
#' *AI for Education - Studio*.
#'
#' @return Returns a  named `list`. The names correspond to specific arguments.
#' The `list` contains a `list` for every argument with the following components:
#'
#' * type: The type of allowed values.
#' * allow_null: A `bool` indicating if the argument can be set to `NULL`.
#' * min: The minimal value the argument can be. Set to `NULL` if not relevant. Set to `-Inf` if there is no minimum.
#' * max: The maximal value the argument can be. Set to `NULL` if not relevant. Set to `Inf` if there is no Minimum.
#' * desc: A `string` which includes the description of the argument written in markdown. This string is for the documentation the parameter.
#' * values_desc: A named `list` containing a description of every possible value. The names must exactly match the strings in allowed_values. Descriptions should be written in markdown.
#' * allowed_values: `vector` of allowed values. This is only relevant if the argument is not numeric. During the checking of the arguments
#'   it is checked if the provided values can be found in this vector. If all values are allowed set to `NULL`.
#' * default_value: The default value of the argument. If there is no default set to `NULL`.
#' * default_historic: Historic default value. This can be necessary for backward compatibility.
#' * gui_box: `string` Name of the box in AI for Education - Studio where the argument appears. If it should not appear set to `NULL`.
#' * gui_label: `string` Label of the controlling widget in AI for Education - Studio.
#'
#' @family Parameter Dictionary
#' @export
#'
get_param_dict <- function() {
  param <- list()
  # General---------------------------------------------------------------------
  param$name <- list(
    type = "string",
    allow_null = TRUE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "Name of the new model. Please refer to common name conventions.
    Free text can be used with parameter `label`. If set to `NULL` a unique ID
    is generated automatically.",
    default_value = NULL,
    test_values = NULL
  )
  param$model_name <- param$name

  param$label <- list(
    type = "string",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "Label for the new model. Here you can use free text.",
    default_value = NULL,
    test_values = NULL
  )
  param$model_label <- param$label


  param$model_language <- list(
    type = "string",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "Languages that the models can work with.",
    default_value = NULL,
    test_values = NULL
  )

  param$track_mode <- list(
    type = "string",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = c("training", "inference"),
    desc = "Determines the stept to which the data refer.",
    default_value = NULL,
    test_values = NULL
  )

  param$token_overlap <- list(
    type = "int",
    min = 0L,
    max = Inf,
    allow_null = FALSE,
    allowed_values = NULL,
    desc = "Number of tokens from the previous chunk that should be added at the beginng of the next chunk.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = 0L,
    default_historic = 0L,
    test_values = c(0L, 10L)
  )
  param$overlap <- param$token_overlap

  param$max_token_sequence_length <- list(
    type = "int",
    min = 20L,
    max = Inf,
    allow_null = FALSE,
    allowed_values = NULL,
    desc = "Maximal number of tokens per chunk.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = 20L,
    default_historic = 20L,
    test_values = c(20L, 30L)
  )

  param$max_length <- param$max_token_sequence_length
  param$max_length$test_values=400L
  param$max_length$desc <- "Maximal number of token per chunks. Must be equal or lower
  as the maximal postional embeddings for the model."

  param$max_sequence_length <- param$max_token_sequence_length
  param$max_sequence_length$desc <- "Maximal number of tokens for every sequence."
  param$max_sequence_length$gui_box <- "Sequence Modeling"
  param$max_sequence_length$gui_label <- "Max Sequence Length"
  param$max_sequence_length$default_value <- 250L
  param$max_sequence_length$default_historic <- 250L
  param$max_sequence_length$test_values <- c(150L, 250L)

  param$n_chunks <- list(
    type = "int",
    min = 2L,
    max = Inf,
    allow_null = FALSE,
    allowed_values = NULL,
    desc = "Maximal number chunks.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = 1L,
    default_historic = 1L,
    test_values = c(4L, 7L)
  )
  param$chunks <- param$n_chunks
  param$chunks$min <- 2L

  param$emb_layer_min <- list(
    type = "int",
    min = 1L,
    max = Inf,
    allow_null = FALSE,
    allowed_values = NULL,
    desc = "Minimal layer from which the embeddings should be calculated.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = 1L,
    default_historic = 1L,
    test_values = 1L
  )

  param$emb_layer_max <- list(
    type = "int",
    min = 1L,
    max = Inf,
    allow_null = FALSE,
    allowed_values = NULL,
    desc = "Maximal layer from which the embeddings should be calculated.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = 1L,
    default_historic = 1L,
    test_values =c(1L,2L)
  )

  param$emb_pool_type <- list(
    type = "string",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = c("CLS", "Average"),
    desc = "Method to summarize the embedding of single tokens into a text embedding.
    In the case of `'CLS'` all cls-tokens between `emb_layer_min` and `emb_layer_max` are averaged.
    In the case of `'Average'` the embeddings of all tokens are averaged.
    Please note that BaseModelFunnel allows only 'CLS'.",
    default_value = NULL,
    test_values = NULL
  )

  param$funnel_pooling_type <- list(
    type = "string",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = c("Mean", "Max"),
    desc = "Method for pooling over the seqence length.",
    default_value = "Mean",
    default_historic = "Mean",
    gui_box = "Transformer Encoder Layers",
    gui_label = "Sequence Pooling",
    test_values = NULL
  )

  param$global_attn_every_n_layers <- list(
    type = "int",
    min = 2L,
    max = 36L,
    allow_null = FALSE,
    allowed_values = NULL,
    desc = "Number determining to use a global attention every x-th layer.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = 3L,
    default_historic = 3L,
    test_values = 2L
  )

  param$statistics_max_tokens_length <- list(
    type = "int",
    min = 20L,
    max = 8192L,
    allow_null = FALSE,
    allowed_values = NULL,
    desc = "Maximum sequence length for calculating the statistics.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = 20L,
    default_historic = 20L,
    test_values = c(100L, 512L)
  )

  param$token_encodings_only <- list(
    type = "bool",
    allow_null = FALSE,
    min = 0L,
    max = 1L,
    allowed_values = NULL,
    desc = "
  * `TRUE`: Returns a `list` containg only the tokens.
  * `FALSE`: Returns a `list` containg a list for the tokens, the number of chunks, and
    the number potential number of chunks for each document/text.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = TRUE,
    test_values = NULL
  )

  param$token_to_int <- list(
    type = "bool",
    allow_null = FALSE,
    min = 0L,
    max = 1L,
    allowed_values = NULL,
    desc = "
  * `TRUE`: Returns the tokens as `int` index.
  * `FALSE`: Returns the tokens as `string`s.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = TRUE,
    test_values = NULL
  )

  param$to_token <- list(
    type = "bool",
    allow_null = FALSE,
    min = 0L,
    max = 1L,
    allowed_values = NULL,
    desc = "
  * `FALSE`: Transforms the integers to plain text.
  * `TRUE`: Transforms the integers to a sequence of tokens.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = TRUE,
    test_values = NULL
  )

  param$return_token_type_ids <- list(
    type = "bool",
    allow_null = FALSE,
    min = 0L,
    max = 1L,
    allowed_values = NULL,
    desc = "If `TRUE` additionally returns the return_token_type_ids.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = TRUE,
    test_values = NULL
  )

  param$int_seqence <- list(
    type = "list",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "`list` of integer sequence that should be converted to tokens.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = TRUE,
    test_values = NULL
  )


  # Transformer related---------------------------------------------------------

  param$tokenizer <- list(
    type = "TokenizerBase",
    min = NULL,
    max = NULL,
    allow_null = FALSE,
    allowed_values = NULL,
    desc = paste(
      "Tokenizer for the model."
    ),
    gui_box = NULL,
    gui_label = NULL,
    default_value = NULL,
    default_historic = NULL,
    test_values = NULL
  )

  param$base_model <- list(
    type = "BaseModelCore",
    min = NULL,
    max = NULL,
    allow_null = FALSE,
    allowed_values = NULL,
    desc = paste(
      "BaseModels for processing raw texts."
    ),
    gui_box = NULL,
    gui_label = NULL,
    default_value = NULL,
    default_historic = NULL,
    test_values = NULL
  )

  param$pad_value <- list(
    type = "int",
    min = -Inf,
    max = -1L,
    allow_null = FALSE,
    allowed_values = NULL,
    desc = "Value indicating padding. This value should no be in the range of
      regluar values for computations. Thus it is not recommended to chance this value.
      Default is `-100`.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = -100L,
    default_historic = 0L,
    test_values = NULL
  )
  param$param_pad_value <- param$pad_value

  # Transformer related parameters
  param$pytorch_safetensors <- list(
    type = "bool",
    allow_null = FALSE,
    min = 0L,
    max = 1L,
    allowed_values = NULL,
    desc = "
  * `TRUE`: a 'pytorch' model is saved in safetensors format.
  * `FALSE` (or 'safetensors' is not available): model is saved in the standard pytorch format (.bin). ",
    gui_box = NULL,
    gui_label = NULL,
    default_value = TRUE,
    test_values = NULL
  )

  param$model_dir <- list(
    type = "string",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "Path to the directory where the model should be saved.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = NULL,
    test_values = NULL
  )
  param$output_dir <- param$model_dir
  param$tokenizer_dir <- param$model_dir
  param$tokenizer_dir$desc <- "Path to the directory where the tokenizer is saved."

  param$folder_name <- list(
    type = "string",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "Name of the folder where the model should be saved.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = NULL,
    test_values = NULL
  )

  param$model_dir_path <- list(
    type = "string",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "Path to the directory where the original model is stored.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = NULL,
    test_values = NULL
  )

  param$text_dataset <- list(
    type = "LargeDataSetForText",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "[LargeDataSetForText] Object storing textual data.",
    gui_box = NULL,
    gui_label = "Text Collection",
    default_value = NULL,
    test_values = NULL
  )

  param$raw_text <- list(
    type = "vector",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "Raw text.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = NULL,
    test_values = NULL
  )

  param$doc_id <- list(
    type = "vector",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "Id for every text.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = NULL,
    test_values = NULL
  )

  param$return_large_dataset <- list(
    type = "bool",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "If `TRUE` a [LargeDataSetForTextEmbeddings] is returned. If `FALSE` an object if class [EmbeddedText] is returned.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = FALSE,
    test_values = NULL
  )

  param$vocab_size <- list(
    type = "int",
    allow_null = FALSE,
    min = 1000L,
    max = 500000L,
    allowed_values = NULL,
    desc = "Size of the vocabulary.",
    gui_box = "Vocabulary",
    gui_label = "Vocabulary Size",
    default_value = 30000L,
    test_values = c(500L, 10000L)
  )

  param$vocab_do_lower_case <- list(
    type = "bool",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "`TRUE` if all tokens should be lower case.",
    gui_box = "Vocabulary",
    gui_label = "Lower Case",
    default_value = FALSE,
    test_values = NULL
  )

  param$add_prefix_space <- list(
    type = "bool",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "`TRUE` if an additional space should be inserted to the leading words.",
    gui_box = "Vocabulary",
    gui_label = "Add Prefix Space",
    default_value = FALSE,
    test_values = NULL
  )

  param$trim_offsets <- list(
    type = "bool",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "`TRUE` trims the whitespaces from the produced offsets.",
    gui_box = "Vocabulary",
    gui_label = "Trim Offsets",
    default_value = FALSE,
    test_values = NULL
  )

  param$whole_word <- list(
    type = "bool",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "* `TRUE`: whole word masking should be applied. Only relevant if a `WordPieceTokenizer` is used.
  * `FALSE`: token masking is used. ",
    gui_box = "Training Settings",
    gui_label = "Whole Word Masking",
    default_value = TRUE,
    test_values = NULL
  )

  param$full_sequences_only <- list(
    type = "bool",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "`TRUE` for using only chunks with a sequence length equal to `chunk_size`.",
    gui_box = "Sequence Modeling",
    gui_label = "Full Sequences Only",
    default_value = FALSE,
    test_values = NULL
  )

  param$max_position_embeddings <- list(
    type = "int",
    allow_null = FALSE,
    min = 10L,
    max = 4048L,
    allowed_values = NULL,
    desc = "Number of maximum position embeddings. This parameter also determines the maximum length of a sequence which
  can be processed with the model.",
    gui_box = "Sequence Modeling",
    gui_label = "Max Sequence Length",
    default_value = 512L,
    test_values = 512L
  )

  param$attention_window <- list(
    type = "int",
    allow_null = FALSE,
    min = 2L,
    max = Inf,
    allowed_values = NULL,
    desc = "Size of the window around each token for attention mechanism in every layer.",
    gui_box = "Sequence Modeling",
    gui_label = "Attention Window",
    default_value = 512L,
    test_values = c(4L, 50L)
  )

  param$hidden_size <- list(
    type = "int",
    allow_null = FALSE,
    min = 1L,
    max = 2048L,
    allowed_values = NULL,
    desc = "Number of neurons in each layer. This parameter determines the dimensionality of the resulting text
  embedding.",
    gui_box = "Transformer Encoder Layers",
    gui_label = "Hidden Size",
    default_value = 768L,
    test_values = c(12L, 20L)
  )
  param$d_model <- param$hidden_size

  param$hidden_act <- list(
    type = "string",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = c("GELU", "relu", "silu", "gelu_new"),
    desc = "Name of the activation function.",
    gui_box = "Transformer Encoder Layers",
    gui_label = "Hidden Activation Function",
    default_value = "gelu",
    test_values = NULL
  )
  param$hidden_activation <- param$hidden_act

  param$num_hidden_layers <- list(
    type = "int",
    allow_null = FALSE,
    min = 1L,
    max = Inf,
    allowed_values = NULL,
    desc = "Number of hidden layers.",
    gui_box = "Transformer Encoder Layers",
    gui_label = "Number of Hidden Layers",
    default_value = 7L,
    test_values = c(2L, 4L)
  )
  param$num_hidden_layer <- param$num_hidden_layers

  param$num_decoder_layers <- list(
    type = "int",
    allow_null = FALSE,
    min = 1L,
    max = Inf,
    allowed_values = NULL,
    desc = "Number of decoding layers.",
    gui_box = "Decoder Layers",
    gui_label = "Number of Decoding Layers",
    default_value = 7L,
    test_values = c(1L, 2L)
  )

  param$target_hidden_size <- list(
    type = "int",
    allow_null = FALSE,
    min = 1L,
    max = Inf,
    allowed_values = NULL,
    desc = "Number of neurons of the final layer. This parameter determines the dimensionality of the resulting text
  embedding.",
    gui_box = "Transformer Encoder Layers",
    gui_label = "Target Hidden Size",
    default_value = 768L,
    test_values = c(64L)
  )

  param$d_head <- param$target_hidden_size
  param$d_head$desc <- "Number of neurons of the final layer."
  param$d_head$default_value <- 64L

  param$hidden_dropout_prob <- list(
    type = "double",
    allow_null = FALSE,
    min = 0L,
    max = 0.6,
    allowed_values = NULL,
    desc = "Ratio of dropout.",
    gui_box = "Transformer Encoder Layers",
    gui_label = "Hidden Dropout",
    default_value = 0.5,
    test_values = NULL
  )
  param$embedding_dropout <- param$hidden_dropout_prob
  param$embedding_dropout$desc <- "Dropout chance for the embeddings."

  param$mlp_dropout <- param$hidden_dropout_prob
  param$mlp_dropout$desc <- "Dropout rate for the mlp layer."

  param$activation_dropout <- list(
    type = "double",
    allow_null = FALSE,
    min = 0L,
    max = 0.6,
    allowed_values = NULL,
    desc = "Dropout probability between the layers of the feed-forward blocks.",
    gui_box = "Transformer Encoder Layers",
    gui_label = "Dropout between the layers of the feed-forward blocks",
    default_value = 0.5,
    test_values = NULL
  )

  param$attention_probs_dropout_prob <- list(
    type = "double",
    allow_null = FALSE,
    min = 0L,
    max = 0.6,
    allowed_values = NULL,
    desc = "Ratio of dropout for attention probabilities.",
    gui_box = "Transformer Encoder Layers",
    gui_label = "Dropout for Attention Probabilities",
    default_value = 0.1,
    test_values = NULL
  )
  param$attention_dropout <- param$attention_probs_dropout_prob

  param$p_mask <- list(
    type = "double",
    allow_null = FALSE,
    min = 0.05,
    max = 0.60,
    allowed_values = NULL,
    desc = "Ratio that determines the number of tokens used for masking.",
    gui_box = "Sequence Modeling",
    gui_label = "Masking Probability",
    default_value = 0.4,
    test_values = NULL
  )

  param$p_perm <- list(
    type = "double",
    allow_null = FALSE,
    min = 0.05,
    max = 0.60,
    allowed_values = NULL,
    desc = "Ratio that determines the number of tokens used for permutation.",
    gui_box = "Sequence Modeling",
    gui_label = "Permutation Probability",
    default_value = 0.4,
    test_values = NULL
  )

  param$block_sizes <- list(
    type = "vector",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "`vector` of `int` determining the number and sizes of each block.",
    gui_box = "Transformer Encoder Layers",
    gui_label = "Block Size",
    default_value = c(4L, 4L),
    test_values = list(c(4, 4))
  )

  param$pooling_type <- list(
    type = "string",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = c("Mean", "Max"),
    desc = "Type of pooling.
  * `\"mean\"` for pooling with mean.
  * `\"max\"` for pooling with maximum values. ",
    gui_box = "Transformer Encoder Layers",
    gui_label = "Pooling Type",
    default_value = "Max",
    test_values = NULL
  )

  param$chunk_size <- list(
    type = "int",
    allow_null = FALSE,
    min = 100L,
    max = Inf,
    allowed_values = NULL,
    desc = "Maximum length of every sequence. Must be equal or less the global maximum size
    allowed by the model.",
    gui_box = "Sequence Modeling",
    gui_label = "Max Sequence Length",
    default_value = 256L,
    test_values = c(50L, 100L)
  )

  param$min_seq_len <- list(
    type = "int",
    allow_null = FALSE,
    min = 10L,
    max = Inf,
    allowed_values = NULL,
    desc = "Only relevant if `full_sequences_only = FALSE`. Value determines the minimal sequence length included in
  training process. ",
    gui_box = "Sequence Modeling",
    gui_label = "Min Sequence Length",
    default_value = 10L,
    test_values = c(10L)
  )

  # Data related-----------------------------------------------------------------
  param$data_embeddings <- list(
    type = c("EmbeddedText", "LargeDataSetForTextEmbeddings"),
    allow_null = FALSE,
    desc = "Object of class [EmbeddedText] or [LargeDataSetForTextEmbeddings].",
    test_values = NULL
  )
  param$text_embeddings <- param$data_embeddings

  param$data_targets <- list(
    type = "factor",
    allow_null = FALSE,
    desc = "containing the labels for cases stored in embeddings. Factor must be
      named and has to use the same names as used in in the embeddings.",
    test_values = NULL
  )
  param$target_levels <- list(
    type = "vector",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "containing the levels (categories or classes) within the target data. Please
    note that order matters. For ordinal data please ensure that the levels are sorted correctly with later levels
    indicating a higher category/class. For nominal data the order does not matter.",
    gui_label = "Target Levels",
    default_value = NULL,
    test_values = NULL
  )
  param$class_levels <- param$target_levels

  param$data_folds <- list(
    type = "int",
    allow_null = FALSE,
    min = 1L,
    max = Inf,
    desc = "determining the number of cross-fold samples.",
    gui_box = "General Settings",
    gui_label = "Number of Folds",
    default_value = 5L,
    test_values = 2L
  )
  param$folds <- param$data_folds

  param$data_val_size <- list(
    type = "(double)",
    allow_null = FALSE,
    min = 0L,
    max = 1L,
    desc = "between 0 and 1, indicating the proportion of cases which should be
      used for the validation sample during the estimation of the model.
      The remaining cases are part of the training data.",
    gui_box = "General Settings",
    gui_label = "Size of Validation Data Set",
    default_value = 0.1,
    test_values = 0.33
  )
  param$val_size <- param$data_val_size

  param$loss_balance_class_weights <- list(
    type = "bool",
    allow_null = FALSE,
    desc = "If `TRUE` class weights are generated based on the frequencies of the
      training data with the method Inverse Class Frequency. If `FALSE` each class has the weight 1.",
    gui_box = "Loss",
    gui_label = "Balance Class Weights",
    default_value = TRUE,
    test_values = NULL
  )
  param$loss_balance_sequence_length <- list(
    type = "bool",
    allow_null = FALSE,
    desc = "If `TRUE` sample weights are generated for the length of sequences based on
      the frequencies of the training data with the method Inverse Class Frequency.
      If `FALSE` each sequences length has the weight 1.",
    gui_box = "Loss",
    gui_label = "Balance Sequence Length",
    default_value = TRUE,
    test_values = NULL
  )
  param$one_hot_encoding <- list(
    type = "bool",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "If `TRUE` all labels are converted to one hot encoding.",
    default_value = NULL,
    test_values = NULL
  )
  param$add_matrix_map <- list(
    type = "bool",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "If `TRUE` all embeddings are transformed into a two dimensional matrix.
      The number of rows equals the number of cases. The number of columns equals `times*features`.",
    default_value = NULL,
    test_values = NULL
  )

  # Synthetic cases-------------------------------------------------------------
  param$use_sc <- list(
    type = "bool",
    allow_null = FALSE,
    desc = "`TRUE` if the estimation should integrate synthetic cases. `FALSE` if not.",
    gui_box = "Synthetic Cases",
    gui_label = "Use Synthetic Cases",
    default_value = FALSE,
    test_values = NULL
  )
  param$sc_method <- list(
    type = "string",
    allow_null = FALSE,
    allowed_values = "knnor",
    desc = "containing the method for generating synthetic cases.",
    gui_box = "Synthetic Cases",
    gui_label = "Method for Creating Synthetic Cases",
    default_value = "knnor",
    test_values = NULL
  )
  param$sc_methods <- param$sc_method

  param$sc_min_k <- list(
    type = "int",
    allow_null = FALSE,
    min = 1L,
    max = Inf,
    desc = "determining the minimal number of k which is used for creating synthetic units.",
    gui_box = "Synthetic Cases",
    gui_label = "Min k",
    default_value = 1L,
    test_values = 1L
  )
  param$sc_max_k <- list(
    type = "int",
    allow_null = FALSE,
    min = 1L,
    max = Inf,
    desc = "determining the maximal number of k which is used for creating synthetic units.",
    gui_box = "Synthetic Cases",
    gui_label = "Max k",
    default_value = 1L,
    test_values = 3L
  )

  param$n_cores <- list(
    type = "int",
    allow_null = FALSE,
    min = 1L,
    max = Inf,
    desc = "Number of cores which should be used during the calculation of synthetic cases. Only relevant if `use_sc=TRUE`.",
    default_value = 1L,
    test_values = 2L
  )

  # Pseudo labeling------------------------------------------------------------
  param$use_pl <- list(
    type = "bool",
    allow_null = FALSE,
    desc = "`TRUE` if the estimation should integrate pseudo-labeling. `FALSE` if not.",
    gui_box = "Pseudo Labeling",
    gui_label = "Use Pseudo Labeling",
    default_value = FALSE,
    test_values = NULL
  )
  param$pl_max_steps <- list(
    type = "int",
    allow_null = FALSE,
    min = 1L,
    max = Inf,
    desc = "determining the maximum number of steps during pseudo-labeling.",
    gui_box = "Pseudo Labeling",
    gui_label = "Number of Steps",
    default_value = 5L,
    test_values = 2L
  )
  param$pl_anchor <- list(
    type = "double",
    min = 0L,
    max = 1L,
    allow_null = FALSE,
    allowed_values = NULL,
    desc = "indicating the reference point for sorting the new cases of every label.",
    gui_box = "Pseudo Labeling",
    gui_label = "Certainty Anchor Value",
    default_value = 1L,
    test_values = 1L
  )
  param$pl_max <- list(
    type = "(double",
    allow_null = FALSE,
    min = 0L,
    max = 1L,
    desc = "setting the maximal level of confidence for considering a case for pseudo-labeling.",
    gui_box = "Pseudo Labeling",
    gui_label = "Max Certainty",
    default_value = 1L,
    test_values = 1L
  )
  param$pl_min <- list(
    type = "double)",
    allow_null = FALSE,
    min = 0L,
    max = 1L,
    desc = "setting the mnimal level of confidence for considering a case for pseudo-labeling.",
    gui_box = "Pseudo Labeling",
    gui_label = "Min Certainty",
    default_value = 0L,
    test_values = 0L
  )

  # Sustainability--------------------------------------------------------------
  param$sustain_track <- list(
    type = "bool",
    allow_null = FALSE,
    desc = "If `TRUE` energy consumption is tracked during training via the python library 'codecarbon'.",
    gui_box = NULL,
    default_value = TRUE,
    test_values = TRUE
  )
  param$sustain_iso_code <- list(
    type = "string",
    allow_null = FALSE,
    allowed_values = NULL,
    desc = "ISO code (Alpha-3-Code) for the country. This variable must be set if
      sustainability should be tracked. A list can be found on Wikipedia:
      <https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes>.",
    gui_box = "Sustainability",
    gui_label = "Alpha-3-Code",
    default_value = "DEU",
    test_values = "DEU"
  )
  param$sustain_region <- list(
    type = "string",
    allow_null = TRUE,
    allowed_values = NULL,
    desc = "Region within a country. Only available for USA and Canada See the documentation of
      codecarbon for more information. <https://docs.codecarbon.io/latest/getting-started/parameters/>",
    gui_box = NULL,
    default_value = NULL,
    test_values = NULL
  )
  param$sustain_interval <- list(
    type = "int",
    allow_null = FALSE,
    min = 1L,
    max = Inf,
    desc = "Interval in seconds for measuring power usage.",
    gui_box = NULL,
    default_value = 15L,
    test_values = 15L
  )
  param$sustain_log_level <- list(
    type = "string",
    allow_null = FALSE,
    allowed_values = c("debug", "info", "warning", "error", "critical"),
    min = NULL,
    max = NULL,
    desc = "Level for printing information to the console.",
    gui_box = NULL,
    default_value = "warning",
    test_values = "error"
  )

  # Training related------------------------------------------------------------
  param$loss_cls_fct_name <- list(
    type = "string",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = c("FocalLoss", "CrossEntropyLoss"),
    desc = "Name of the loss function to use during training.",
    values_desc = list(
      CrossEntropyLoss = "Applies the a cross cross entropy loss.",
      FocalLoss = "Applies the focal loss described by [Lin et al. 2017](https://doi.org/10.48550/arXiv.1708.02002)."
    ),
    gui_box = "General Settings",
    gui_label = "Loss Function",
    default_value = "FocalLoss",
    test_values = NULL
  )
  param$loss_pt_fct_name <- param$loss_cls_fct_name
  param$loss_pt_fct_name$allowed_values <- "MultiWayContrastiveLoss"
  param$loss_pt_fct_name$values_desc <- list(
    MultiWayContrastiveLoss = "Applies the loss described by [Zhang et al. 2019](https://doi.org/10.1007/978-3-030-16145-3_24)."
  )
  param$loss_pt_fct_name$default_value <- "MultiWayContrastiveLoss"
  param$loss_pt_fct_name$gui_box <- "General Settings"

  param$optimizer <- list(
    type = "string",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = c("Adam", "RMSprop", "AdamW", "SGD"),
    desc = "determining the optimizer used for training.",
    gui_box = "General Settings",
    gui_label = "Optimizer",
    default_value = "AdamW",
    test_values = NULL
  )

  param$epochs <- list(
    type = "int",
    allow_null = FALSE,
    min = 1L,
    max = Inf,
    desc = "Number of training epochs.",
    gui_box = "General Settings",
    gui_label = "Epochs",
    default_value = 100L,
    test_values = 2L
  )
  param$n_epoch <- param$epochs

  param$batch_size <- list(
    type = "int",
    allow_null = FALSE,
    min = 1L,
    max = Inf,
    desc = "Size of the batches for training.",
    gui_box = "General Settings",
    gui_label = "Batch Size",
    default_value = 32L,
    test_values = 2L
  )

  param$n_batches <- list(
    type = "int",
    allow_null = FALSE,
    min = 1L,
    max = Inf,
    desc = "Number of batches.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = 2L,
    test_values = 2L
  )

  param$lr_rate <- list(
    type = "(double",
    allow_null = FALSE,
    min = 0L,
    max = 1L,
    desc = "Initial learning rate for the training. Sets the maximal learning rate.",
    magnitude = 0.1,
    gui_box = "Learning Rate",
    gui_label = "Learning Rate",
    default_value = 1e-3,
    test_values = 1e-3
  )
  param$learning_rate <- param$lr_rate
  param$lr_min <- list(
    type = "(double",
    allow_null = FALSE,
    min = 0L,
    max = 1L,
    desc = "Minimal learning rate during training.",
    magnitude = 0.1,
    gui_box = "Learning Rate",
    gui_label = "Minimal Learning Rate",
    default_value = 1e-4,
    test_values = 1e-4
  )
  param$lr_scheduler <- list(
    type = "string",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = c("None", "Linear", "Cyclic"),
    values_desc = list(
      None = "Uses no learning rate scheduler and no warm up phase.",
      Linear = "Decreases the learning rate linear from the maximal value (lr_rate) to its minimum (lr_min).",
      Cyclic = "Decreases the learning rate for 5 epochs to lr_min. Thereafter the learning rate increases for 5 epochs to lr_rate."
    ),
    desc = "Learning rate scheduler. To use a constant learning rate for the whole training set this parameter to 'None'.",
    gui_box = "Learning Rate",
    gui_label = "Scheduler",
    default_value = "None",
    test_values = NULL
  )

  param$lr_warm_up_ratio <- list(
    type = "(double)",
    allow_null = FALSE,
    min = 0L,
    max = 0.50,
    desc = "Number of epochs used for warm up. To disable warm up set this value to 0.0.",
    gui_box = "Learning Rate",
    gui_label = "Warm Up Ratio",
    default_value = 0.01,
    test_values = 0.01
  )
  param$dir_checkpoint <- list(
    type = "string",
    allow_null = FALSE,
    allowed_values = NULL,
    desc = "Path to the directory where the checkpoint during training should be saved.
      If the directory does not exist, it is created.",
    test_values = NULL
  )

  # Logging related-------------------------------------------------------------
  param$log_dir <- list(
    type = "string",
    allow_null = TRUE,
    allowed_values = NULL,
    desc = "Path to the directory where the log files should be saved.
      If no logging is desired set this argument to `NULL`.",
    default_value = NULL,
    test_values = NULL
  )
  param$log_file <- list(
    type = "string",
    allow_null = TRUE,
    allowed_values = NULL,
    desc = "Path to the file where the log files should be saved.
      If no logging is desired set this argument to `NULL`.",
    default_value = NULL,
    test_values = NULL
  )
  param$log_write_interval <- list(
    type = "int",
    allow_null = FALSE,
    min = 1L,
    max = Inf,
    desc = "Time in seconds determining the interval in which the logger should try to update
      the log files. Only relevant if `log_dir` is not `NULL`.",
    default_value = 60L,
    test_values = 60L
  )
  param$trace <- list(
    type = "bool",
    allow_null = FALSE,
    desc = "`TRUE` if information about the estimation phase should be printed to the console.",
    default_value = FALSE,
    test_values = NULL
  )
  param$ml_trace <- list(
    type = "int",
    min = 0L,
    max = 1L,
    allow_null = FALSE,
    allowed_values = NULL,
    desc = "`ml_trace=0` does not print any information about the training process from pytorch on the console.",
    default_value = 0L,
    test_values = 0L
  )
  param$pytorch_trace <- param$ml_trace

  # Meta Learning related--------------------------------------------------------
  param$embedding_dim <- list(
    desc = " determining the number of dimensions for the embedding.",
    type = "int",
    max = Inf,
    min = 2L,
    allow_null = FALSE,
    gui_box = "General Settings",
    gui_label = "Number of Dimensions for Embeddings",
    default_value = 2L,
    test_values = 2L
  )
  param$Ns <- list(
    type = "int",
    allow_null = FALSE,
    min = 1L,
    max = Inf,
    allowed_values = NULL,
    desc = "Number of cases for every class in the sample.",
    gui_label = "Number of Cases in the Sample",
    gui_box = "Sampling",
    default_value = 5L,
    test_values = 5L
  )
  param$Nq <- list(
    type = "int",
    allow_null = FALSE,
    min = 1L,
    max = Inf,
    allowed_values = NULL,
    desc = "Number of cases for every class in the query.",
    gui_label = "Number of Cases in the Query",
    gui_box = "Sampling",
    default_value = 3L,
    test_values = 3L
  )
  param$loss_alpha <- list(
    type = "double",
    allow_null = FALSE,
    min = 0L,
    max = 1L,
    allowed_values = NULL,
    desc = "Value between 0 and 1 indicating how strong the loss should focus on pulling cases to
      its corresponding prototypes or pushing cases away from other prototypes. The higher the value the more the
      loss concentrates on pulling cases to its corresponding prototypes.",
    gui_box = "Loss",
    gui_label = "Alpha",
    default_value = 0.5,
    test_values = NULL
  )
  param$loss_margin <- list(
    type = "double",
    allow_null = FALSE,
    min = 0L,
    max = 1L,
    allowed_values = NULL,
    desc = "Value greater 0 indicating the minimal distance of every case from prototypes of other classes. Please note that
    in contrast to the orginal work by Zhang et al. (2019) this implementation
    reaches better performance if the margin is a magnitude lower (e.g. 0.05 instead of 0.5).",
    gui_box = "Loss",
    gui_label = "Margin",
    default_value = 0.05,
    test_values = NULL
  )
  param$sampling_separate <- list(
    type = "bool",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "If `TRUE` the cases for every class are divided into a data set for sample and
      for query. These are never mixed. If `TRUE` sample and query cases are drawn from the same data pool. That is,
      a case can be part of sample in one epoch and in another epoch it can be part of query. It is ensured that a
      case is never part of sample and query at the same time. In addition, it is ensured that every cases exists
      only once during a training step.",
    gui_box = "Sampling",
    gui_label = "Strictly Separte Sample and Query",
    default_value = FALSE,
    test_values = NULL
  )
  param$sampling_shuffle <- list(
    type = "bool",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "if `TRUE` cases a randomly drawn from the data during every step. If `FALSE` the
      cases are not shuffled.",
    gui_box = "Sampling",
    gui_label = "Shuffle Order of Cases",
    default_value = TRUE,
    test_values = NULL
  )

  # FeatureExtractor related-----------------------------------------------------
  param$features <- list(
    type = "int",
    allow_null = FALSE,
    min = 1L,
    max = Inf,
    allowed_values = NULL,
    desc = "Number of features the model should use.",
    gui_box = "General Settings",
    gui_label = "Number of Features",
    default_value = 64L,
    test_values = 12L
  )
  param$method <- list(
    type = "string",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = c("Dense", "LSTM"),
    desc = "Method to use for the feature extraction. `'lstm'` for an extractor based on LSTM-layers or `'Dense'` for dense layers.",
    default_value = "Dense",
    gui_box = "General Settings",
    gui_label = "Method",
    test_values = NULL
  )

  param$orthogonal_method <- list(
    type = "string",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = c("matrix_exp", "cayley", "householder","None"),
    desc = "Method for ensuring orthogonality of weights.",
    default_historic = "householder",
    default_value = " matrix_exp",
    gui_box = "General Settings",
    gui_label = "Orthogonal Method",
    test_values = NULL
  )


  param$noise_factor <- list(
    type = "double",
    allow_null = FALSE,
    min = 0L,
    max = 1L,
    allowed_values = NULL,
    desc = "Value between 0 and a value lower 1 indicating how much noise should
      be added to the input during training.",
    default_value = 1e-2,
    gui_box = "General Settings",
    gui_label = "Noise Factor",
    test_values = NULL
  )
  param$feature_extractor <- list(
    type = "TEFeatureExtractor",
    allow_null = TRUE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "Object of class [TEFeatureExtractor] which should be used in order to reduce
    the number of dimensions of the text embeddings. If no feature extractor should be applied set `NULL`.",
    gui_label = "Feature Extractor",
    default_value = NULL,
    test_values = NULL
  )

  # Layer configuration=========================================================
  # Global settings------------------------------------------------------------
  param$residual_type <- list(
    type = "string",
    min = NULL,
    max = NULL,
    allow_null = FALSE,
    allowed_values = c("ResidualGate", "Addition", "None"),
    values_desc = list(
      None = "Add no residual connection.",
      Addition = "Adds a residual connection by adding the original input to the output.",
      ResidualGate = "Adds a residucal connection by creating a weightes sum from the original input and the output.
                      The weight is a learnable parameter. This type of residual connection is described by [Savarese and Figueiredo (2017)](https://home.ttic.edu/~savarese/savarese_files/Residual_Gates.pdf)."
    ),
    desc = "Type of residual connenction for all layers and stack of layers.",
    gui_box = "General Settings",
    gui_label = "Residual Connection",
    default_value = "ResidualGate",
    default_historic = NULL,
    test_values = NULL
  )

  param$skip_connection_type <- list(
    type = "string",
    min = NULL,
    max = NULL,
    allow_null = FALSE,
    allowed_values = c("ResidualGate", "Addition", "None"),
    values_desc = list(
      None = "Add no residual connection.",
      Addition = "Adds a residual connection by adding the original input to the output.",
      ResidualGate = "Adds a residucal connection by creating a weightes sum from the original input and the output.
                      The weight is a learnable parameter. This type of residual connection is described by [Savarese and Figueiredo (2017)](https://home.ttic.edu/~savarese/savarese_files/Residual_Gates.pdf)."
    ),
    desc = "Type of residual connenction for the complete model.",
    gui_box = "General Settings",
    gui_label = "Residual Connection",
    default_value = "ResidualGate",
    default_historic = NULL,
    test_values = NULL
  )

  param$skip_connection_type <- param$residual_type
  param$skip_connection_type$gui_box <- "General Settings"

  param$tf_residual_type <- param$residual_type
  param$tf_residual_type$gui_box <- "Transformer Encoder Layers"

  param$rec_residual_type <- param$residual_type
  param$rec_residual_type$gui_box <- "Recurrent Layers"

  param$dense_residual_type <- param$residual_type
  param$dense_residual_type$gui_box <- "Dense Layers"

  param$ng_conv_residual_type <- param$residual_type
  param$ng_conv_residual_type$gui_box <- "Multiple N-Gram Layers"

  param$normalization_type <- list(
    type = "string",
    min = NULL,
    max = NULL,
    allow_null = FALSE,
    allowed_values = c(
      "LayerNorm",
      "BatchNorm",
      "PowerNorm",
      "RMSNorm",
      "None"),
    values_desc = list(
      LayerNorm = "Applies normalization as described by [Ba, Kiros, and Hinton (2016)](https://doi.org/10.48550/arXiv.1607.06450). Implementation supports masking of sequences.",
      BatchNorm = "Applies normalization as described by [Loffe and Szegedy](https://doi.org/10.48550/arXiv.1502.03167). Implementation supports masking of sequences.",
      RMSNorm = "Applies normalization as described by  [Zhang and Sennrich](https://doi.org/10.48550/arXiv.1910.07467). Implementation supports masking of sequences.",
      PowerNorm= "Applies normalization as described by [Shen et al.](https://doi.org/10.48550/arXiv.2003.07845). Implementation supports masking of sequences.",
      None = "Applies no normalization."
    ),
    desc = "Type of normalization applied to all layers and stack layers.",
    gui_box = "General Settings",
    gui_label = "Normalization",
    default_value = "LayerNorm",
    default_historic = NULL,
    test_values = NULL
  )

  param$feat_normalization_type <- param$normalization_type
  param$feat_normalization_type$gui_box <- "Feature Layer"

  param$tf_normalization_type <- param$normalization_type
  param$tf_normalization_type$gui_box <- "Transformer Encoder Layers"

  param$rec_normalization_type <- param$normalization_type
  param$rec_normalization_type$gui_box <- "Recurrent Layers"

  param$dense_normalization_type <- param$normalization_type
  param$dense_normalization_type$gui_box <- "Dense Layers"

  param$ng_conv_normalization_type <- param$normalization_type
  param$ng_conv_normalization_type$gui_box <- "Multiple N-Gram Layers"

  param$merge_normalization_type <- param$normalization_type
  param$merge_normalization_type$gui_box <- "Merge Layer"

  param$tf_normalization_position <- list(
    type = "string",
    min = NULL,
    max = NULL,
    allow_null = FALSE,
    allowed_values = c("Pre", "Post"),
    values_desc = list(
      pre = "Applies normalization before the layers as described by [Xiong et al. (2020)](https://doi.org/10.48550/arXiv.2002.04745).",
      post = "Applies normalization after the layers as described in the original transformer model."
    ),
    desc = "Position where the normalization should be applied.",
    gui_box = "General Settings",
    gui_label = "Normalization Position",
    default_value = "Pre",
    default_historic = "Post",
    test_values = NULL
  )

  # Intermediate Feature--------------------------------------------------------
  param$cls_pooling_features <- list(
    type = "int",
    allow_null = FALSE,
    min = 1L,
    max = Inf,
    allowed_values = NULL,
    desc = "Number of features to be extracted at the end of the model.",
    gui_box = "Classifiction Pooling Layer",
    gui_label = "Size",
    default_value = 32L,
    test_values = 4L
  )
  param$merge_pooling_features <- param$cls_pooling_features
  param$merge_pooling_features$gui_box <- "Merge Layer"

  param$cls_head_type <- list(
    type = "string",
    min = NULL,
    max = NULL,
    allow_null = FALSE,
    allowed_values = c("Regular", "PairwiseOrthogonal","PairwiseOrthogonalDense"),
    values_desc = list(
      Regular = "Applies a fully connected dense layer between the network and the
      final layer that calculates the classes.",
      PairwiseOrthogonal = "Applies a pairwise orthogonal layer as classification head as described by [Li et al. 2020](https://doi.org/10.1109/TIP.2020.2990277).",
      PairwiseOrthogonalDense="Same as 'PairwiseOrthogonal' but with an additional dense layer before the head."
    ),
    desc = "Type of classification head.",
    gui_box = "Classifiction Pooling Layer",
    gui_label = "Classification Head",
    default_value = "Regular",
    default_historic = "Regular",
    test_values = NULL
  )

  param$projection_type <- list(
    type = "string",
    min = NULL,
    max = NULL,
    allow_null = FALSE,
    allowed_values = c("Regular", "PairwiseOrthogonal","PairwiseOrthogonalDense"),
    values_desc = list(
      Regular = "Applies a fully connected dense layer for calculating the position of a point on all axes. That is,
      all neurons can contribute to every axes.",
      PairwiseOrthogonal = "Applies a pairwise orthogonal layer without activation for calculating the position of a point. That is,
      a neuron contributes only to one specific axis and an axis is influenced only by a disjoint Subsample of all neurons. The layer is described by
       [Li et al. 2020](https://doi.org/10.1109/TIP.2020.2990277) in the context of classification heads.",
      PairwiseOrthogonalDense="Same as 'PairwiseOrthogonal' but with an additional dense layer before the head."
    ),
    desc = "Type of projection.",
    gui_box = "General Settings",
    gui_label = "Projection Type",
    default_value = "Regular",
    default_historic = "Regular",
    test_values = NULL
  )

  param$cls_pooling_type <- list(
    type = "string",
    min = NULL,
    max = NULL,
    allow_null = FALSE,
    allowed_values = c("Max", "Min", "MinMax"),
    desc = "Type of extracting intermediate features.",
    gui_box = "Classifiction Pooling Layer",
    gui_label = "Feature Extraction Method",
    default_value = "MinMax",
    default_historic = NULL,
    test_values = NULL
  )
  param$merge_pooling_type <- param$cls_pooling_type
  param$merge_pooling_type$gui_box <- "Merge Layer"


  # Parametrizations------------------------------------------------------------
  param$parametrizations <- list(
    type = "string",
    min = NULL,
    max = NULL,
    allow_null = FALSE,
    allowed_values = c("None", "OrthogonalWeights", "WeightNorm", "SpectralNorm"),
    desc = "Re-Parametrizations of the weights of layers.",
    values_desc = list(
      None = "Does not apply any re-parametrizations.",
      OrthogonalWeights = "Applies an orthogonal re-parametrizations of the weights with PyTorchs implemented function using orthogonal_map='matrix_exp'.",
      WeightNorm = "Applies a weight norm with the default settings of PyTorch's corresponding function. Weight norm is described by [Salimans and Kingma 2016](https://doi.org/10.48550/arXiv.1602.07868).",
      SpectralNorm = "Applies a spectral norm with the default settings of PyTorch's corresponding function. The norm is described by [Miyato et al. 2018](https://doi.org/10.48550/arXiv.1802.05957)."
    ),
    gui_box = "General Settings",
    gui_label = "Re-Parametrization",
    default_value = "None",
    default_historic = "None",
    test_values = NULL
  )
  param$rec_parametrizations <- param$parametrizations
  param$rec_parametrizations$allowed_values <- "None"
  param$rec_parametrizations$gui_box <- "Recurrent Layers"

  param$tf_parametrizations <- param$parametrizations
  param$tf_parametrizations$gui_box <- "Transformer Encoder Layers"

  param$dense_parametrizations <- param$parametrizations
  param$dense_parametrizations$gui_box <- "Dense Layers"

  param$ng_conv_parametrizations <- param$parametrizations
  param$ng_conv_parametrizations$gui_box <- "Multiple N-Gram Layers"

  param$feat_parametrizations <- param$parametrizations
  param$feat_parametrizations$gui_box <- "Feature Layer"

  # Bias------------------------------------------------------------------------
  param$bias <- list(
    type = "bool",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "If `TRUE` a bias term is added to all layers. If `FALSE` no bias term is added to the layers.",
    gui_box = "General Settings",
    gui_label = "Add Bias",
    default_value = FALSE,
    default_historic = TRUE,
    test_values = NULL
  )
  param$rec_bias <- param$bias
  param$rec_bias$gui_box <- "Recurrent Layers"
  param$tf_bias <- param$bias
  param$tf_bias$gui_box <- "Transformer Encoder Layers"
  param$dense_bias <- param$bias
  param$dense_bias$gui_box <- "Dense Layers"
  param$ng_conv_bias <- param$bias
  param$ng_conv_bias$gui_box <- "Multiple N-Gram Layers"
  param$feat_bias <- param$bias
  param$feat_bias$gui_box <- "Feature Layer"

  # Activation functions---------------------------------------------------------
  param$act_fct <- list(
    type = "string",
    min = NULL,
    max = NULL,
    allow_null = FALSE,
    allowed_values = c("ELU", "LeakyReLU", "ReLU", "GELU", "Sigmoid", "Tanh", "PReLU"),
    desc = "Activation function for all layers.",
    gui_box = "General Settings",
    gui_label = "Activation Function",
    default_value = "ELU",
    default_historic = "GELU",
    test_values = NULL
  )
  param$feat_act_fct <- param$act_fct
  param$feat_act_fct$gui_box <- "Feature Layer"
  param$ng_conv_act_fct <- param$act_fct
  param$ng_conv_act_fct$gui_box <- "Multiple N-Gram Layers"
  param$dense_act_fct <- param$act_fct
  param$dense_act_fct$gui_box <- "Dense Layers"
  param$rec_act_fct <- param$act_fct
  param$rec_act_fct$allowed_values <- "Tanh"
  param$rec_act_fct$gui_box <- "Recurrent Layers"
  param$tf_act_fct <- param$act_fct
  param$tf_act_fct$gui_box <- "Transformer Encoder Layers"

  # Recurrent Layer--------------------------------------------------------------
  param$rec_dropout <- list(
    type = "double",
    allow_null = FALSE,
    min = 0L,
    max = 0.6,
    allowed_values = NULL,
    desc = "determining the dropout between recurrent layers.",
    gui_box = "Recurrent Layers",
    gui_label = "Dropout",
    default_value = 0.5,
    test_values = NULL
  )

  param$rec_type <- list(
    type = "string",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = c("GRU", "LSTM"),
    desc = "Type of the recurrent layers. `rec_type='GRU'` for Gated Recurrent Unit and `rec_type='LSTM'` for Long Short-Term Memory.",
    gui_box = "Recurrent Layers",
    gui_label = "Type",
    default_value = "GRU",
    test_values = NULL
  )

  param$rec_bidirectional <- list(
    type = "bool",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "If `TRUE` a bidirectional version of the recurrent layers is used.",
    gui_box = "Recurrent Layers",
    gui_label = "Bidirectional",
    default_value = FALSE,
    test_values = NULL
  )

  param$rec_size <- list(
    type = "int",
    allow_null = FALSE,
    min = 1L,
    max = Inf,
    allowed_values = NULL,
    desc = "Number of neurons for each recurrent layer.",
    gui_box = "Recurrent Layers",
    gui_label = "Size",
    default_value = 32L,
    test_values = c(2L, 5L)
  )

  param$rec_layers <- list(
    type = "int",
    allow_null = FALSE,
    min = 0L,
    max = Inf,
    allowed_values = NULL,
    desc = "Number of recurrent layers.",
    gui_box = "Recurrent Layers",
    gui_label = "Number of Layers",
    default_value = 1L,
    test_values = c(0L, 1L, 2L)
  )
  param$rec_n_layers <- param$rec_layers

  # Dense Layer------------------------------------------------------------------
  param$dense_dropout <- list(
    type = "double",
    allow_null = FALSE,
    min = 0L,
    max = 0.6,
    allowed_values = NULL,
    desc = "determining the dropout between dense layers.",
    gui_box = "Dense Layers",
    gui_label = "Dropout",
    default_value = 0.5,
    test_values = NULL
  )
  param$dense_size <- list(
    type = "int",
    allow_null = FALSE,
    min = 1L,
    max = Inf,
    allowed_values = NULL,
    desc = "Number of neurons for each dense layer.",
    gui_box = "Dense Layers",
    gui_label = "Size",
    default_value = 32L,
    test_values = c(10L, 16L)
  )

  param$dense_layers <- list(
    type = "int",
    allow_null = FALSE,
    min = 0L,
    max = Inf,
    allowed_values = NULL,
    desc = "Number of dense layers.",
    gui_box = "Dense Layers",
    gui_label = "Number of Layers",
    default_value = 0L,
    test_values = c(0L, 1L, 2L)
  )
  param$dense_n_layers <- param$dense_layers

  # Feature Layer----------------------------------------------------------------
  param$shared_feat_layer <- list(
    type = "bool",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "If `TRUE` all streams use the same feature layer. If `FALSE` all streams use their own feature layer.",
    gui_box = "Feature Layer",
    gui_label = "Shared Layer",
    default_value = TRUE,
    test_values = NULL
  )

  param$feat_dropout <- list(
    type = "double",
    allow_null = FALSE,
    min = 0L,
    max = 0.6,
    allowed_values = NULL,
    desc = "determining the dropout for the dense projection of the feature layer.",
    gui_box = "Feature Layer",
    gui_label = "Dropout",
    default_value = 0.1,
    test_values = NULL
  )

  param$feat_size <- param$dense_size
  param$feat_size$gui_box <- "Feature Layer"
  param$feat_size$min <- 2L
  param$feat_size$test_values = c(4L,6L)

  # Transformer Layer------------------------------------------------------------
  param$encoder_dropout <- list(
    type = "double",
    allow_null = FALSE,
    min = 0L,
    max = 0.6,
    allowed_values = NULL,
    desc = "determining the dropout for the dense projection within the transformer encoder layers.",
    gui_box = "Transformer Encoder Layers",
    gui_label = "Dense Dropout",
    default_value = 0.1,
    test_values = NULL
  )
  param$tf_dropout_rate_1 <- param$encoder_dropout
  param$tf_dropout_rate_1$desc <- "determining the dropout after the attention mechanism within the transformer encoder layers."
  param$tf_dropout_rate_1$gui_label <- "Attention Dropout"
  param$tf_dropout_rate_2 <- param$encoder_dropout

  param$self_attention_heads <- list(
    type = "int",
    allow_null = FALSE,
    min = 0L,
    max = Inf,
    allowed_values = NULL,
    desc = "determining the number of attention heads for a self-attention layer. Only relevant if `attention_type='multihead'`",
    gui_box = "Transformer Encoder Layers",
    gui_label = "Number of Attention Heads",
    default_value = 2L,
    test_values = 2L
  )
  param$num_attention_heads <- param$self_attention_head
  param$n_head <- param$self_attention_head
  param$tf_num_heads <- param$self_attention_head

  param$intermediate_size <- list(
    type = "int",
    allow_null = TRUE,
    min = 1L,
    max = Inf,
    allowed_values = NULL,
    desc = "determining the size of the projection layer within a each transformer encoder.",
    gui_box = "Transformer Encoder Layers",
    gui_label = "Intermediate Size",
    default_value = 128L,
    test_values = c(20L)
  )
  param$tf_dense_dim <- param$intermediate_size
  param$d_inner <- param$intermediate_size

  param$attention_type <- list(
    type = "string",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = c("Fourier", "MultiHead"),
    values_desc = list(
      multihead = "The original multi-head attention as described by [Vaswani et al. (2017)](https://doi.org/10.48550/arXiv.1706.03762).",
      fourier = "Attention with fourier transformation as described by [Lee-Thorp et al. (2021)](https://doi.org/10.48550/arXiv.2105.03824)."
    ),
    desc = "Choose the attention type.",
    gui_box = "Transformer Encoder Layers",
    gui_label = "Attention Type",
    default_value = "Fourier",
    test_values = NULL
  )
  param$tf_attention_type <- param$attention_type

  param$add_pos_embedding <- list(
    type = "bool",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "`TRUE` if positional embedding should be used.",
    gui_box = "Transformer Encoder Layers",
    gui_label = "Add Positional Embedding",
    default_value = FALSE,
    test_values = NULL
  )

  param$tf_positional_type <- list(
    type = "string",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = c("None", "absolute"),
    values_desc = list(
      absolute = "Adds positional information by using an embedding matrix as described by Chollet, Kalinowski, and Allaire (2022, pp. 378-379).
                  This implementation is different to the original work by [Vaswani et al. (2017)](https://doi.org/10.48550/arXiv.1706.03762).",
      None = "No absolute positional information is added."
    ),
    desc = "Type of processing positional information.",
    gui_box = "Transformer Encoder Layers",
    gui_label = "Positional Information",
    default_value = "absolute",
    default_historic = "None",
    test_values = NULL
  )

  param$repeat_encoder <- list(
    type = "int",
    allow_null = FALSE,
    min = 0L,
    max = Inf,
    allowed_values = NULL,
    desc = "determining how many times the encoder should be added to the network.",
    gui_box = "Transformer Encoder Layers",
    gui_label = "Number of Layers",
    default_value = 0L,
    test_values = c(0L, 1L, 2L)
  )
  param$tf_n_layers <- param$repeat_encoder

  # Conv Layer-------------------------------------------------------------------
  param$ng_conv_dropout <- list(
    type = "double",
    allow_null = FALSE,
    min = 0L,
    max = 0.6,
    allowed_values = NULL,
    desc = "determining the dropout for n-gram convolution layers.",
    gui_box = "Multiple N-Gram Layers",
    gui_label = "Dropout",
    default_value = 0.1,
    test_values = NULL
  )

  param$ng_conv_n_layers <- list(
    type = "int",
    allow_null = FALSE,
    min = 0L,
    max = Inf,
    allowed_values = NULL,
    desc = "determining how many times the n-gram layers should be added to the network.",
    gui_box = "Multiple N-Gram Layers",
    gui_label = "Number of Layers",
    default_value = 0L,
    test_values = c(0L, 1L, 2L)
  )
  param$ng_conv_ks_min <- list(
    type = "int",
    allow_null = FALSE,
    min = 2L,
    max = Inf,
    allowed_values = NULL,
    desc = "determining the minimal window size for n-grams.",
    gui_box = "Multiple N-Gram Layers",
    gui_label = "Smallest N-Gram",
    default_value = 2L,
    test_values = 2L
  )
  param$ng_conv_ks_max <- list(
    type = "int",
    allow_null = FALSE,
    min = 2L,
    max = Inf,
    allowed_values = NULL,
    desc = "determining the maximal window size for n-grams.",
    gui_box = "Multiple N-Gram Layers",
    gui_label = "Biggest N-Gram",
    default_value = 2L,
    test_values = 3L
  )

  # Parallel specific-----------------------------------------------------------
  param$merge_attention_type <- param$attention_type
  param$merge_attention_type$allowed_values <- c("Fourier", "MultiHead")
  param$merge_attention_type$gui_box <- "Merge Layer"

  param$merge_num_heads <- param$self_attention_head
  param$merge_num_heads$gui_box <- "Merge Layer"

  # Prototpye specific----------------------------------------------------------
  param$metric_type <- list(
    type = "string",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = c("Euclidean", "CosineDistance"),
    desc = "Type of metric used for calculating the distance.",
    gui_box = "Loss",
    gui_label = "Metric Type",
    default_value = "Euclidean",
    test_values = NULL
  )

  # Relevant for plot functions and methods
  param$y_min <- list(
    type = "int",
    min = -Inf,
    max = Inf,
    allow_null = TRUE,
    allowed_values = NULL,
    desc = "Minimal value for y-axis. Set to `NULL` for an automatic adjustment.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = NULL,
    default_historic = NULL,
    test_values = NULL
  )
  param$y_max <- list(
    type = "int",
    min = -Inf,
    max = Inf,
    allow_null = TRUE,
    allowed_values = NULL,
    desc = "Maximal value for y-axis. Set to `NULL` for an automatic adjustment.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = NULL,
    default_historic = NULL,
    test_values = NULL
  )
  param$x_min <- list(
    type = "int",
    min = -Inf,
    max = Inf,
    allow_null = TRUE,
    allowed_values = NULL,
    desc = "Minimal value for x-axis. Set to `NULL` for an automatic adjustment.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = NULL,
    default_historic = NULL,
    test_values = NULL
  )
  param$x_max <- list(
    type = "int",
    min = -Inf,
    max = Inf,
    allow_null = TRUE,
    allowed_values = NULL,
    desc = "Maximal value for x-axis. Set to `NULL` for an automatic adjustment.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = NULL,
    default_historic = NULL,
    test_values = NULL
  )
  param$text_size <- list(
    type = "int",
    min = 1,
    max = Inf,
    allow_null = FALSE,
    allowed_values = NULL,
    desc = "Size of text elements.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = NULL,
    default_historic = NULL,
    test_values = NULL
  )
  param$add_min_max <- list(
    type = "bool",
    min = NULL,
    max = NULL,
    allow_null = FALSE,
    allowed_values = NULL,
    desc = "If `TRUE` the minimal and maximal values during performance estimation are port of the plot. If `FALSE` only the mean values are shown. Parameter is ignored if `final_training=TRUE`.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = NULL,
    default_historic = NULL,
    test_values = NULL
  )
  param$ind_selected_model <- list(
    type = "bool",
    min = NULL,
    max = NULL,
    allow_null = FALSE,
    allowed_values = NULL,
    desc = "If `TRUE` the plot indicates the states of the model which are used after training. These are the final states of the fold or the final state of the last training loop.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = NULL,
    default_historic = NULL,
    test_values = NULL
  )
  param$ind_best_model <- list(
    type = "bool",
    min = NULL,
    max = NULL,
    allow_null = FALSE,
    allowed_values = NULL,
    desc = "If `TRUE` the plot indicates the best states of the model according to the chosen measure.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = NULL,
    default_historic = NULL,
    test_values = NULL
  )

  param$final_training <- list(
    type = "bool",
    min = NULL,
    max = NULL,
    allow_null = FALSE,
    allowed_values = NULL,
    desc = "If `FALSE` the values of the performance estimation are used. If `TRUE` only the epochs of the final training are used.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = NULL,
    default_historic = NULL,
    test_values = NULL
  )
  param$masked_text <- list(
    type = "string",
    allow_null = FALSE,
    min = NULL,
    max = NULL,
    allowed_values = NULL,
    desc = "Text with mask tokens.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = NULL,
    test_values = NULL
  )
  param$n_solutions <- list(
    type = "int",
    min = 1,
    max = Inf,
    allow_null = FALSE,
    allowed_values = NULL,
    desc = "Number of solutions the model should predict.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = NULL,
    default_historic = NULL,
    test_values = NULL
  )

  param$n_samples <- list(
    type = "int",
    min = 1,
    max = Inf,
    allow_null = FALSE,
    allowed_values = NULL,
    desc = "Number of samples.",
    gui_box = NULL,
    gui_label = NULL,
    default_value = NULL,
    default_historic = NULL,
    test_values = NULL
  )


  # ============================================================================
  return(param)
}

#' @title Definition of an argument
#' @description Function returns the definition of an argument. Please note that
#' only definitions of arguments can be requested which are used for transformers or
#' classifier models.
#' @param param_name `string` Name of the parameter to request its definition.
#' @returns Returns a `list` with the definition of the argument. See [get_param_dict]
#' for more details.
#' @family Parameter Dictionary
#' @export
get_param_def <- function(param_name) {
  return(get_param_dict()[[param_name]])
}

#' @title Create rd formula
#' @description Function returns the syntax for displaying a formula depending on
#' the output format.
#' @param string_formula `string` Formula expression without specific format.
#' @returns Returns a `string` which can be used in rd files to display a formula for
#' different output types.
#' @family Parameter Dictionary
#' @keywords internal
doc_formula <- function(string_formula) {
  return(
    paste0(
      "\\ifelse{latex}{$", string_formula, "$}{\\ifelse{html}{\\eqn{", string_formula, "}}{`", string_formula, "`}}"
    )
  )
}

#' @title Description of an argument
#' @description Function provides the description of an argument in markdown.
#' Its aim is to be used for documenting the parameter of functions.
#' @param param_name `string` Name of the parameter to request its definition.
#' @return Returns a string which contains the description of the argument in markdown.
#' The concrete format depends on the type of the argument.
#' @family Parameter Dictionary
#' @export
get_param_doc_desc <- function(param_name) {
  param_def <- get_param_def(param_name = param_name)

  if (length(param_def$type) == sum(param_def$type %in% c("bool", "int", "double", "(double", "double)", "(double)", "string", "vector", "list"))) {
    is_class <- TRUE
  } else {
    is_class <- FALSE
  }

  desc <- param_def$desc

  if (is_class) {
    type <- paste0("`", param_def$type, "`")
    type <- stringi::stri_replace_all(
      str = type, replacement = "",
      regex = "\\(|\\)"
    )
    if (param_def$type == "bool") {
      allowed_values <- NULL
    } else if (param_def$type == "string") {
      if (is.null(param_def$allowed_values)) {
        allowed_values <- "any"
      } else {
        allowed_values <- paste(paste0("'", param_def$allowed_values, "'"), collapse = ", ")
      }
    } else if (param_def$type %in% c("double", "(double", "double)", "(double)")) {
      if (param_def$min != -Inf) {
        if (param_def$type == "(double" || param_def$type == "(double)") {
          border_min <- paste(param_def$min, "<")
        } else {
          border_min <- paste(param_def$min, "<=")
        }
      } else {
        border_min <- NULL
      }
      if (param_def$max != Inf) {
        if (param_def$type == "double)" || param_def$type == "(double)") {
          border_max <- paste("<", param_def$max)
        } else {
          border_max <- paste("<=", param_def$max)
        }
      } else {
        border_max <- NULL
      }
      allowed_values <- doc_formula(paste0(border_min, " x ", border_max))
    } else if (param_def$type == "int") {
      if (param_def$min != -Inf) {
        border_min <- paste(param_def$min, "<=")
      } else {
        border_min <- NULL
      }
      if (param_def$max != Inf) {
        border_max <- paste("<=", param_def$max)
      } else {
        border_max <- NULL
      }
      allowed_values <- doc_formula(paste0(border_min, " x ", border_max))
    } else {
      allowed_values <- NULL
    }
  } else {
    type <- paste0("`", paste0(param_def$type, collapse = ", "), "`")
    allowed_values <- NULL
  }

  if (!is.null(allowed_values)) {
    allowed_values <- paste("Allowed values:", allowed_values)
  } else {
    allowed_values <- ""
  }

  desc_string <- paste0(
    type, " ",
    desc, " ",
    allowed_values
  )
  return(desc_string)
}

#' @title Called arguments
#' @description Function for receiving all arguments that were called by a method or function.
#'
#' @param n `int` level of the nested environments where to extract the arguments.
#'
#' @importFrom rlang caller_fn
#' @importFrom rlang fn_fmls
#'
#' @return Returns a named `list` of all arguments and their values.
#'
#' @family Parameter Dictionary
#' @export
get_called_args <- function(n = 1L) {
  fn <- rlang::caller_fn(n)
  formal_args <- rlang::fn_fmls(fn)
  final_args <- formal_args
  for (arguments in names(formal_args)) {
    final_args[arguments] <- list(get(x = arguments, envir = rlang::caller_env(n)))
  }
  return(final_args)
}

#' @title Magnitudes of an argument
#' @description Function calculates different magnitude for a numeric argument.
#' @param max `double` The maximal value.
#' @param min `double` The minimal value.
#' @param magnitude `double` Factor using for creating the magnitude.
#' @param n_elements `int` Number of values to return.
#' @return Returns a numeric `vector` with the generated values.
#' The values are calculated with the following formula:
#' max * magnitude^i for i=1,...,n_elements.
#' Only values equal or greater `min` are returned.
#' @family Parameter Dictionary
get_magnitude_values <- function(magnitude, n_elements = 9L, max = NULL, min = NULL) {
  value_vector <- vector(length = n_elements)
  for (i in seq_along(value_vector)) {
    value_vector[i] <- max(min, max * magnitude^i)
  }
  return(unique(value_vector))
}
