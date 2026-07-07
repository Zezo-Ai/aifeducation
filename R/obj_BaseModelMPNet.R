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

#' @title MPNet
#' @description Represents models based on MPNet.
#' @references Song,K., Tan, X., Qin, T., Lu, J. & Liu, T.-Y. (2020). MPNet: Masked and Permuted Pre-training for
#'   Language Understanding. \doi{10.48550/arXiv.2004.09297}
#' @return `r get_description("return_object")`
#' @family Base Model
#' @export
BaseModelMPNet <- R6::R6Class(
  classname = "BaseModelMPNet",
  inherit = BaseModelCore,
  private = list(
    model_type = "mpnet",
    slow_tokenizer = "MPNetTokenizer",
    adjust_max_sequence_length = 2L,
    return_token_type_ids = FALSE,
    create_model = function(args) {
      configuration <- transformers$MPNetConfig(
        vocab_size = as.integer(length(args$tokenizer$get_tokenizer()$get_vocab())),
        hidden_size = as.integer(args$hidden_size),
        num_hidden_layers = as.integer(args$num_hidden_layers),
        num_attention_heads = as.integer(args$num_attention_heads),
        intermediate_size = as.integer(args$intermediate_size),
        hidden_act = tolower(args$hidden_act),
        hidden_dropout_prob = args$hidden_dropout_prob,
        attention_probs_dropout_prob = args$attention_probs_dropout_prob,
        max_position_embeddings = as.integer(args$max_position_embeddings),
        initializer_range = 0.02,
        layer_norm_eps = 1e-12
      )

      device <- ifelse(torch$cuda$is_available(), "cuda", "cpu")
      private$model <- py$MPNetForMPLM_PT(configuration)$to(device)
    },
    #--------------------------------------------------------------------------
    create_data_collator = function() {
      # collator_maker <- NULL
      # if (
      #   check_versions(a = get_py_package_version("transformers"), operator = "<", b = "4.49.0")
      # ) {
      #   transformers_version_controll=0L
      # }
      # else if (check_versions(a = get_py_package_version("transformers"), operator = ">=", b = "4.49.0") &&
      #          check_versions(a = get_py_package_version("transformers"), operator = "<", b = "5.0.0")
      # ) {
      #   transformers_version_controll=1L
      # } else if (
      #   check_versions(a = get_py_package_version("transformers"), operator = ">=", b = "5.0.0")
      # ) {
      #   transformers_version_controll=2L
      #   stop(
      #     "MPNet with transformers of version 5.0.0 and higher is temporarily not available. ",
      #     "This will change in future versions of aifeducation.")
      # } else {
      #   stop("Version not implemented. Version of transformers is ",get_py_package_version("transformers"))
      # }

      # collator_maker <- py$CollatorMaker_PT(
      #   tokenizer = self$Tokenizer$get_tokenizer(),
      #   mlm = TRUE,
      #   transformers_version_controll=transformers_version_controll,
      #   max_length = as.integer(private$model$config$max_position_embeddings - private$adjust_max_sequence_length),
      #   mlm_probability = self$last_training$config$p_mask,
      #   plm_probability = self$last_training$config$p_perm,
      #   mask_whole_words = self$last_training$config$whole_word
      # )

      if (self$last_training$config$whole_word) {
        data_collator_type <- "WordMLMAndTokenPLM"
      } else {
        data_collator_type <- "TokenMPLM"
      }

      tmp_data_collator <- py$make_collator(
        data_collator_type,
        tokenizer = self$Tokenizer$get_tokenizer(),
        mlm = TRUE,
        max_length = as.integer(private$model$config$max_position_embeddings - private$adjust_max_sequence_length),
        mlm_probability = self$last_training$config$p_mask,
        plm_probability = self$last_training$config$p_perm,
        masking_strategy = "bert",
        return_explicit_mlm_labels = TRUE
      )

      return(tmp_data_collator)
    },
    #--------------------------------------------------------------------------
    load_BaseModel = function(dir_path) {
      run_py_file("MPNetForMPLM_PT.py")
      private$model <- py$MPNetForMPLM_PT$from_pretrained(
        dir_path,
        from_tf = FALSE,
        use_safetensors = TRUE
      )
    },
    #---------------------------------------------------------------------------
    check_arg_combinations = function(args) {
      if (args$hidden_size %% args$num_attention_heads != 0L) {
        stop("hidden_size must be a multiple auf num_attention_heads.")
      }
    }
  ),
  public = list(
    #---------------------------------------------------------------------------
    #' @description Configures a new object of this class.
    #' Please ensure that your chosen configuration comply with the following
    #' guidelines:
    #' * hidden_size is a multiple of num_attention_heads.
    #'
    #' @param tokenizer `r get_param_doc_desc("tokenizer")`
    #' @param max_position_embeddings `r get_param_doc_desc("max_position_embeddings")`
    #' @param hidden_size `r get_param_doc_desc("hidden_size")`
    #' @param num_hidden_layers `r get_param_doc_desc("num_hidden_layers")`
    #' @param num_attention_heads `r get_param_doc_desc("num_attention_heads")`
    #' @param intermediate_size `r get_param_doc_desc("intermediate_size")`
    #' @param hidden_act `r get_param_doc_desc("hidden_act")`
    #' @param hidden_dropout_prob `r get_param_doc_desc("hidden_dropout_prob")`
    #' @param attention_probs_dropout_prob `r get_param_doc_desc("attention_probs_dropout_prob")`
    #' @return `r get_description("return_nothing")`
    configure = function(tokenizer,
                         max_position_embeddings = 512L,
                         hidden_size = 768L,
                         num_hidden_layers = 12L,
                         num_attention_heads = 12L,
                         intermediate_size = 3072L,
                         hidden_act = "GELU",
                         hidden_dropout_prob = 0.1,
                         attention_probs_dropout_prob = 0.1) {
      arguments <- get_called_args(n = 1L)
      private$do_configuration(args = arguments)
    },
    #--------------------------------------------------------------------------
    #' @description Traines a BaseModel
    #' @param text_dataset `r get_param_doc_desc("text_dataset")`
    #' @param p_mask `r get_param_doc_desc("p_mask")`
    #' @param p_perm `r get_param_doc_desc("p_perm")`
    #' @param whole_word `r get_param_doc_desc("whole_word")`
    #' @param val_size `r get_param_doc_desc("val_size")`
    #' @param n_epoch `r get_param_doc_desc("n_epoch")`
    #' @param batch_size `r get_param_doc_desc("batch_size")`
    #' @param max_sequence_length `r get_param_doc_desc("max_sequence_length")`
    #' @param full_sequences_only `r get_param_doc_desc("full_sequences_only")`
    #' @param min_seq_len `r get_param_doc_desc("min_seq_len")`
    #' @param learning_rate `r get_param_doc_desc("learning_rate")`
    #' @param sustain_track `r get_param_doc_desc("sustain_track")`
    #' @param sustain_iso_code `r get_param_doc_desc("sustain_iso_code")`
    #' @param sustain_region `r get_param_doc_desc("sustain_region")`
    #' @param sustain_interval `r get_param_doc_desc("sustain_interval")`
    #' @param sustain_log_level `r get_param_doc_desc("sustain_log_level")`
    #' @param trace `r get_param_doc_desc("trace")`
    #' @param pytorch_trace `r get_param_doc_desc("pytorch_trace")`
    #' @param log_dir `r get_param_doc_desc("log_dir")`
    #' @param log_write_interval `r get_param_doc_desc("log_write_interval")`
    #' @return `r get_description("return_nothing")`
    train = function(text_dataset,
                     p_mask = 0.15,
                     p_perm = 0.15,
                     whole_word = TRUE,
                     val_size = 0.1,
                     n_epoch = 1L,
                     batch_size = 12L,
                     max_sequence_length = 250L,
                     full_sequences_only = FALSE,
                     min_seq_len = 50L,
                     learning_rate = 3e-3,
                     sustain_track = FALSE,
                     sustain_iso_code = NULL,
                     sustain_region = NULL,
                     sustain_interval = 15L,
                     sustain_log_level = "warning",
                     trace = TRUE,
                     pytorch_trace = 1L,
                     log_dir = NULL,
                     log_write_interval = 2L) {
      #run_py_file("data_collator.py")
      private$do_training(args = get_called_args(n = 1L))
    }
  )
)

# Add the model to the user list
BaseModelsIndex$MPNet <- list(
  class_name = "BaseModelMPNet",
  model_type = "mpnet",
  reference = "Song,K., Tan, X., Qin, T., Lu, J. & Liu, T.-Y. (2020). MPNet:
  Masked and Permuted Pre-training for Language Understanding. doi: [10.48550/arXiv.2004.09297](https://doi.org/10.48550/arXiv.2004.09297)",
  req_sentencepiece = FALSE
)
