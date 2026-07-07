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

#' @title EuroBERT
#' @description Represents models based on EuroBERT.
#' @references Boizard, N., Gisserot-Boukhlef, H., Alves, D. M., Martins, A.,
#' Hammal, A., Corro, C., Hudelot, C., Malherbe, E., Malaboeuf, E.,
#' Jourdan, F., Hautreux, G., Alves, J., El-Haddad, K., Faysse, M.,
#' Peyrard, M., Guerreiro, N. M., Fernandes, P.,
#' Rei, R. & Colombo, P. (2025). EuroBERT: Scaling Multilingual Encoders for
#' European Languages. \doi{10.48550/arXiv.2503.05500}
#' @return `r get_description("return_object")`
#' @family Base Model
#' @export
BaseModelEuroBert <- R6::R6Class(
  classname = "BaseModelEuroBert",
  inherit = BaseModelCore,
  private = list(
    model_type = "eurobert",
    slow_tokenizer = NULL,
    adjust_max_sequence_length = 0L,
    return_token_type_ids = FALSE,
    create_model = function(args) {
      configuration <- transformers$EuroBertConfig(
        vocab_size = as.integer(length(args$tokenizer$get_tokenizer()$get_vocab())),
        hidden_size = as.integer(args$hidden_size),
        intermediate_size = as.integer(args$intermediate_size),
        num_hidden_layers = as.integer(args$num_hidden_layers),
        num_attention_heads = as.integer(args$num_attention_heads),
        num_key_value_heads = as.integer(args$num_attention_heads),
        hidden_act = tolower(args$hidden_act),
        max_position_embeddings = as.integer(args$max_position_embeddings),
        initializer_range = 0.02,
        rms_norm_eps = 1e-05,
        use_cache = TRUE,
        pad_token_id = as.integer(args$tokenizer$get_tokenizer()["pad_token_id"]),
        bos_token_id = as.integer(args$tokenizer$get_tokenizer()["bos_token_id"]),
        eos_token_id = as.integer(args$tokenizer$get_tokenizer()["eos_token_id"]),
        pretraining_tp = 1L,
        tie_word_embeddings = FALSE,
        attention_bias = FALSE,
        attention_dropout = args$attention_dropout,
        mlp_bias = FALSE,
        head_dim = NULL,
        mask_token_id = as.integer(args$tokenizer$get_tokenizer()["mask_token_id"]),
        classifier_pooling = "late"
      )
      private$model <- transformers$EuroBertForMaskedLM(configuration)
    },
    load_BaseModel = function(dir_path) {
      private$model <- transformers$EuroBertForMaskedLM$from_pretrained(dir_path)
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
    #' @param attention_dropout `r get_param_doc_desc("attention_dropout")`
    #' @param attention_probs_dropout_prob `r get_param_doc_desc("attention_probs_dropout_prob")`
    #' @return `r get_description("return_nothing")`
    configure = function(tokenizer,
                         max_position_embeddings = 512L,
                         hidden_size = 768L,
                         num_hidden_layers = 12L,
                         num_attention_heads = 12L,
                         intermediate_size = 3072L,
                         hidden_act = "GELU",
                         attention_dropout = 0.1) {
      arguments <- get_called_args(n = 1L)
      private$do_configuration(args = arguments)
    }
  )
)

# Add the model to the user list
BaseModelsIndex$eurobert <- list(
  class_name = "BaseModelEuroBert",
  model_type = "eurobert",
  reference = "Boizard, N., Gisserot-Boukhlef, H., Alves, D. M., Martins, A.,
  Hammal, A., Corro, C., Hudelot, C., Malherbe, E., Malaboeuf, E.,
  Jourdan, F., Hautreux, G., Alves, J., El-Haddad, K., Faysse, M.,
  Peyrard, M., Guerreiro, N. M., Fernandes, P.,
  Rei, R. & Colombo, P. (2025). EuroBERT: Scaling Multilingual Encoders for
  European Languages. doi: [10.48550/arXiv.2503.05500](https://doi.org/10.48550/arXiv.2503.05500)",
  req_sentencepiece = FALSE
)
