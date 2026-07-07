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

#' @title ALBERT-Transformer
#' @description Represents models based on ALBERT
#' @references  Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., &
#' Soricut, R. (2019). ALBERT; A Lite BERT for Self-supervised Learning of
#' Language Representations. arXiv. \doi{10.48550/ARXIV.1909.11942}
#' @return `r get_description("return_object")`
#' @family Base Model
#' @export
BaseModelAlbert <- R6::R6Class(
  classname = "BaseModelAlbert",
  inherit = BaseModelCore,
  private = list(
    model_type = "albert",
    slow_tokenizer = "AlbertTokenizer",
    create_model = function(args) {
      configuration <- transformers$AlbertConfig(
        vocab_size = as.integer(length(args$tokenizer$get_tokenizer()$get_vocab())),
        embedding_size = as.integer(args$embedding_size),
        hidden_size = as.integer(args$hidden_size),
        num_hidden_layers = as.integer(args$num_hidden_layers),
        num_hidden_groups = as.integer(args$num_hidden_groups),
        num_attention_heads = as.integer(args$num_attention_heads),
        intermediate_size = as.integer(args$intermediate_size),
        inner_group_num = 1L,
        hidden_act = tolower(args$hidden_act),
        hidden_dropout_prob = args$hidden_dropout_prob,
        attention_probs_dropout_prob = args$attention_probs_dropout_prob,
        max_position_embeddings = as.integer(args$max_position_embeddings),
        type_vocab_size = 2L,
        initializer_range = 0.02,
        layer_norm_eps = 1e-12,
        classifier_dropout_prob = 0.1,
        pad_token_id = as.integer(args$tokenizer$get_tokenizer()["pad_token_id"]),
        bos_token_id = as.integer(args$tokenizer$get_tokenizer()["bos_token_id"]),
        eos_token_id = as.integer(args$tokenizer$get_tokenizer()["eos_token_id"]),
        tie_word_embeddings = TRUE
      )
      private$model <- transformers$AlbertForMaskedLM(configuration)
    },
    load_BaseModel = function(dir_path) {
      private$model <- transformers$AlbertForMaskedLM$from_pretrained(dir_path)
    },
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
    #' @param num_hidden_groups `r get_param_doc_desc("num_hidden_groups")`
    #' @param embedding_size `r get_param_doc_desc("embedding_size")`
    #' @param num_attention_heads `r get_param_doc_desc("num_attention_heads")`
    #' @param intermediate_size `r get_param_doc_desc("intermediate_size")`
    #' @param hidden_act `r get_param_doc_desc("hidden_act")`
    #' @param hidden_dropout_prob `r get_param_doc_desc("hidden_dropout_prob")`
    #' @param attention_probs_dropout_prob `r get_param_doc_desc("attention_probs_dropout_prob")`
    #' @return `r get_description("return_nothing")`
    configure = function(tokenizer,
                         max_position_embeddings = 512L,
                         hidden_size = 768L,
                         embedding_size = 128L,
                         num_hidden_layers = 12L,
                         num_hidden_groups = 1L,
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
BaseModelsIndex$Albert <- list(
  class_name = "BaseModelAlbert",
  model_type = "albert",
  reference = "Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., &
  Soricut, R. (2019). ALBERT; A Lite BERT for Self-supervised Learning of
  Language Representations. arXiv. doi: [10.48550/ARXIV.1909.11942](https://doi.org/10.48550/ARXIV.1909.11942)",
  req_sentencepiece = TRUE
)
