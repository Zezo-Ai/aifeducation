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

#' @title ModernBert
#' @description Represents models based on Modern Bert.
#' @references Warner, B., Chaffin, A., Clavie, B., Weller, O., Hallstroem, O., Taghadouini, S.,
#' Gallagher, A., Biswas, R., Ladhak, F., Aarsen, T., Cooper, N., Adams, G.,
#' Howard, J. & Poli, I. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional
#' Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference.
#' \doi{10.48550/arXiv.2412.13663}
#' @return `r get_description("return_object")`
#' @family Base Model
#' @export
BaseModelModernBert <- R6::R6Class(
  classname = "BaseModelModernBert",
  inherit = BaseModelCore,
  private = list(
    model_type = "modernbert",
    slow_tokenizer = NULL,
    create_model = function(args) {
      configuration <- transformers$ModernBertConfig(
        vocab_size = as.integer(length(args$tokenizer$get_tokenizer()$get_vocab())),
        hidden_size = as.integer(args$hidden_size),
        intermediate_size = as.integer(args$intermediate_size),
        num_hidden_layers = as.integer(args$num_hidden_layers),
        num_attention_heads = as.integer(args$num_attention_heads),
        hidden_activation = tolower(args$hidden_activation),
        max_position_embeddings = as.integer(args$max_position_embeddings),
        initializer_range = 0.02,
        norm_eps = 1e-12,
        global_attn_every_n_layers = as.integer(args$global_attn_every_n_layers),
        pad_token_id = args$tokenizer$get_tokenizer()$pad_token_id,
        eos_token_id = args$tokenizer$get_tokenizer()$eos_token_id,
        bos_token_id = args$tokenizer$get_tokenizer()$bos_token_id,
        cls_token_id = args$tokenizer$get_tokenizer()$cls_token_id,
        sep_token_id = args$tokenizer$get_tokenizer()$sep_token_id,
        embedding_dropout = args$embedding_dropout,
        mlp_dropout = args$mlp_dropout,
        attention_dropout = args$attention_dropout,
        deterministic_flash_attn = TRUE
      )
      private$model <- transformers$ModernBertForMaskedLM(configuration)
    },
    load_BaseModel = function(dir_path) {
      private$model <- transformers$ModernBertForMaskedLM$from_pretrained(dir_path)
    },
    check_arg_combinations = function(args) {
      if (args$num_hidden_layers %% 2 != 0L) {
        stop("num_hidden_layers must be a multiple of 2.")
      }

      if (args$hidden_size %% args$num_attention_heads != 0L) {
        stop("hidden_size must be a multiple auf num_attention_heads.")
      }

      if ((args$hidden_size / args$num_attention_heads) %% 2 != 0) {
        stop("(hidden_size/num_attention_heads) must be a multiple of 2.")
      }

      if (args$global_attn_every_n_layers > args$num_hidden_layers) {
        print("args$global_attn_every_n_layers")
        print(args$global_attn_every_n_layers)
        print("args$num_hidden_layers")
        print(args$num_hidden_layers)
        stop("global_attn_every_n_layers must be equal or smaller num_hidden_layers.")
      }
    }
  ),
  public = list(
    #---------------------------------------------------------------------------
    #' @description Configures a new object of this class.
    #' Please ensure that your chosen configuration comply with the following
    #' guidelines:
    #' * hidden_size is a multiple of num_attention_heads.
    #' * hidden_size/num_attention_heads must be a multiple of 2.
    #' * global_attn_every_n_layers is equal or smaller as num_hidden_layers.
    #'
    #' @param tokenizer `r get_param_doc_desc("tokenizer")`
    #' @param max_position_embeddings `r get_param_doc_desc("max_position_embeddings")`
    #' @param hidden_size `r get_param_doc_desc("hidden_size")`
    #' @param num_hidden_layers `r get_param_doc_desc("num_hidden_layers")`
    #' @param num_attention_heads `r get_param_doc_desc("num_attention_heads")`
    #' @param global_attn_every_n_layers `r get_param_doc_desc("global_attn_every_n_layers")`
    #' @param intermediate_size `r get_param_doc_desc("intermediate_size")`
    #' @param hidden_activation `r get_param_doc_desc("hidden_activation")`
    #' @param embedding_dropout `r get_param_doc_desc("embedding_dropout")`
    #' @param mlp_dropout `r get_param_doc_desc("mlp_dropout")`
    #' @param attention_dropout `r get_param_doc_desc("attention_dropout")`
    #' @return `r get_description("return_nothing")`
    configure = function(tokenizer,
                         max_position_embeddings = 512L,
                         hidden_size = 768L,
                         num_hidden_layers = 12L,
                         num_attention_heads = 12L,
                         global_attn_every_n_layers = 3L,
                         intermediate_size = 3072L,
                         hidden_activation = "GELU",
                         embedding_dropout = 0.1,
                         mlp_dropout = 0.1,
                         attention_dropout = 0.1) {
      arguments <- get_called_args(n = 1L)
      private$do_configuration(args = arguments)
    }
  )
)

# Add the model to the user list
BaseModelsIndex$ModernBert <- list(
  class_name = "BaseModelModernBert",
  model_type = "modernbert",
  reference = "Warner, B., Chaffin, A., Clavie, B., Weller, O., Hallstroem, O., Taghadouini, S.,
  Gallagher, A., Biswas, R., Ladhak, F., Aarsen, T., Cooper, N., Adams, G.,
  Howard, J. & Poli, I. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional
  Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference.
  doi: [10.48550/arXiv.2412.13663](https://doi.org/10.48550/arXiv.2412.13663)",
  req_sentencepiece = FALSE
)
