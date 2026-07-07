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

#' @title RoBERTa
#' @description Represents models based on RoBERTa.
#' @references Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., &
#'   Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. \doi{10.48550/arXiv.1907.11692}
#' @return `r get_description("return_object")`
#' @family Base Model
#' @export
BaseModelRoberta <- R6::R6Class(
  classname = "BaseModelRoberta",
  inherit = BaseModelCore,
  private = list(
    model_type = "roberta",
    slow_tokenizer = "RobertaTokenizer",
    adjust_max_sequence_length = 4L,
    return_token_type_ids = FALSE,
    create_model = function(args) {
      configuration <- transformers$RobertaConfig(
        vocab_size = as.integer(length(args$tokenizer$get_tokenizer()$get_vocab())),
        max_position_embeddings = as.integer(args$max_position_embeddings),
        hidden_size = as.integer(args$hidden_size),
        num_hidden_layers = as.integer(args$num_hidden_layers),
        num_attention_heads = as.integer(args$num_attention_heads),
        intermediate_size = as.integer(args$intermediate_size),
        hidden_act = tolower(args$hidden_act),
        hidden_dropout_prob = args$hidden_dropout_prob,
        attention_probs_dropout_prob = args$attention_probs_dropout_prob,
        type_vocab_size = 2L,
        initializer_range = 0.02,
        layer_norm_eps = 1e-12,
        position_embedding_type = "absolute",
        is_decoder = FALSE,
        use_cache = TRUE
      )
      private$model <- transformers$RobertaForMaskedLM(configuration)
    },
    load_BaseModel = function(dir_path) {
      private$model <- transformers$RobertaForMaskedLM$from_pretrained(dir_path)
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
BaseModelsIndex$roberta <- list(
  class_name = "BaseModelRoberta",
  model_type = "roberta",
  reference = "Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O.,
  Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A Robustly Optimized
  BERT Pretraining Approach. doi: [10.48550/arXiv.1907.11692](https://doi.org/10.48550/arXiv.1907.11692)",
  req_sentencepiece = FALSE
)
