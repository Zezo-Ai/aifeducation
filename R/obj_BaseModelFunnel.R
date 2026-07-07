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

#' @title Funnel transformer
#' @description Represents models based on the Funnel-Transformer.
#' @references Dai, Z., Lai, G., Yang, Y. & Le, Q. V. (2020). Funnel-Transformer: Filtering out Sequential Redundancy
#'   for Efficient Language Processing. \doi{10.48550/arXiv.2006.03236}
#' @return `r get_description("return_object")`
#' @family Base Model
#' @export
BaseModelFunnel <- R6::R6Class(
  classname = "BaseModelFunnel",
  inherit = BaseModelCore,
  private = list(
    model_type = "funnel",
    slow_tokenizer = "FunnelTokenizer",
    adjust_max_sequence_length = 1,
    sequence_mode = "vary",
    create_model = function(args) {
      configuration <- transformers$FunnelConfig(
        vocab_size = as.integer(length(args$tokenizer$get_tokenizer()$get_vocab())),
        block_sizes = as.integer(args$block_sizes),
        block_repeats = NULL,
        num_decoder_layers = as.integer(args$num_decoder_layers),
        d_model = as.integer(args$hidden_size),
        n_head = as.integer(args$num_attention_heads),
        d_head = as.integer(args$d_head),
        d_inner = as.integer(args$intermediate_size),
        hidden_act = tolower(args$hidden_act),
        hidden_dropout_prob = args$hidden_dropout_prob,
        attention_probs_dropout_prob = args$attention_probs_dropout_prob,
        activation_dropout = as.integer(args$activation_dropout),
        initializer_range = 0.02,
        layer_norm_eps = 1e-12,
        pooling_type = tolower(args$funnel_pooling_type),
        attention_type = "relative_shift",
        separate_cls = TRUE,
        truncate_seq = TRUE,
        pool_q_only = TRUE,
        max_position_embeddings = as.integer(args$max_position_embeddings),
      )
      private$model <- transformers$FunnelForMaskedLM(configuration)
    },
    load_BaseModel = function(dir_path) {
      private$model <- transformers$FunnelForMaskedLM$from_pretrained(dir_path)
    },
    set_model_config_from_hf = function() {
      super$set_model_config_from_hf()
      private$model_config["num_attention_heads"] <- list(private$model$config["n_head"])
      private$model_config["hidden_size"] <- list(private$model$config["d_model"])
      private$model_config["intermediate_size"] <- list(private$model$config["d_inner"])
      private$model_config["funnel_pooling_type"] <- list(private$model$config["pooling_type"])
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
    #' @param block_sizes `r get_param_doc_desc("block_sizes")`
    #' @param num_hidden_layers `r get_param_doc_desc("num_hidden_layers")`
    #' @param num_attention_heads `r get_param_doc_desc("num_attention_heads")`
    #' @param num_decoder_layers `r get_param_doc_desc("num_decoder_layers")`
    #' @param d_head `r get_param_doc_desc("d_head")`
    #' @param funnel_pooling_type `r get_param_doc_desc("funnel_pooling_type")`
    #' @param intermediate_size `r get_param_doc_desc("intermediate_size")`
    #' @param hidden_act `r get_param_doc_desc("hidden_act")`
    #' @param hidden_dropout_prob `r get_param_doc_desc("hidden_dropout_prob")`
    #' @param attention_probs_dropout_prob `r get_param_doc_desc("attention_probs_dropout_prob")`
    #' @param activation_dropout `r get_param_doc_desc("activation_dropout")`
    #' @return `r get_description("return_nothing")`
    configure = function(tokenizer,
                         max_position_embeddings = 512L,
                         hidden_size = 768L,
                         block_sizes = c(4L, 4L, 4L),
                         num_attention_heads = 12L,
                         intermediate_size = 3072L,
                         num_decoder_layers = 2L,
                         d_head = 64L,
                         funnel_pooling_type = "Mean",
                         hidden_act = "GELU",
                         hidden_dropout_prob = 0.1,
                         attention_probs_dropout_prob = 0.1,
                         activation_dropout = 0.0) {
      arguments <- get_called_args(n = 1L)
      private$do_configuration(args = arguments)
    },
    #--------------------------------------------------------------------------
    #' @description Number of layers.
    #' @return Returns an `int` describing the number of layers available for
    #' embedding.
    get_n_layers = function() {
      return(sum(private$model$config$block_repeats * private$model$config$block_sizes))
    }
  )
)

# Add the model to the user list
BaseModelsIndex$Funnel <- list(
  class_name = "BaseModelFunnel",
  model_type = "funnel",
  reference = "Dai, Z., Lai, G., Yang, Y. & Le, Q. V. (2020). Funnel-Transformer:
  Filtering out Sequential Redundancy for Efficient Language Processing. doi: [10.48550/arXiv.2006.03236](https://doi.org/10.48550/arXiv.2006.03236)",
  req_sentencepiece = FALSE
)
