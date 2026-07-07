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

#' @title RoBERTa-XML
#' @description Represents models based on RoBERTa-XML.
#' @references Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G.,
#' Guzman, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2019).
#' Unsupervised Cross-lingual Representation Learning at Scale. \doi{10.48550/arXiv.1911.02116}
#' @return `r get_description("return_object")`
#' @family Base Model
#' @export
BaseModelRobertaXML <- R6::R6Class(
  classname = "BaseModelRobertaXML",
  inherit = BaseModelCore,
  private = list(
    model_type = "robertaxml",
    slow_tokenizer = "XLMRobertaTokenizer",
    adjust_max_sequence_length = 4L,
    return_token_type_ids = FALSE,
    create_model = function(args) {
      configuration <- transformers$XLMRobertaConfig(
        vocab_size = as.integer(length(args$tokenizer$get_tokenizer()$get_vocab())),
        hidden_size = as.integer(args$hidden_size),
        num_hidden_layers = as.integer(args$num_hidden_layers),
        num_attention_heads = as.integer(args$num_attention_heads),
        intermediate_size = as.integer(args$intermediate_size),
        hidden_act = tolower(args$hidden_act),
        hidden_dropout_prob = args$hidden_dropout_prob,
        attention_probs_dropout_prob = args$attention_probs_dropout_prob,
        max_position_embeddings = as.integer(args$max_position_embeddings),
        type_vocab_size = 2L,
        initializer_range = 0.02,
        layer_norm_eps = 1e-12,
        pad_token_id = as.integer(args$tokenizer$get_tokenizer()["pad_token_id"]),
        bos_token_id = as.integer(args$tokenizer$get_tokenizer()["bos_token_id"]),
        eos_token_id = as.integer(args$tokenizer$get_tokenizer()["eos_token_id"]),
        classifier_dropout = 0.0,
        is_decoder = FALSE,
        use_cache = TRUE,
        add_cross_attention = FALSE,
        tie_word_embeddings = TRUE,
      )
      private$model <- transformers$XLMRobertaForMaskedLM(configuration)
    },
    load_BaseModel = function(dir_path) {
      private$model <- transformers$XLMRobertaForMaskedLM$from_pretrained(dir_path)
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
    }
  )
)

# Add the model to the user list
BaseModelsIndex$robertaxml <- list(
  class_name = "BaseModelRobertaXML",
  model_type = "robertaxml",
  reference = "Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G.,
  Guzman, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2019).
  Unsupervised Cross-lingual Representation Learning at Scale.
  doi: [10.48550/arXiv.1911.02116](https://doi.org/10.48550/arXiv.1911.02116)",
  req_sentencepiece = TRUE
)
