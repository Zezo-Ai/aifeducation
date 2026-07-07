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

#' @title Xmod-Transformer
#' @description Represents models based on Xmod
#' @description Xmod models support different languages. Before starting to
#' work with this model set the correct language with the method
#' `set_default_language`. You receive a list with all supported languages
#' by calling `get_supported_languages`.
#'
#' @references Pfeiffer, J., Goyal, N., Lin, X. V., Li, X., Cross, J.,
#' Riedel, S., & Artetxe, M. (2022). Lifting the Curse of Multilinguality
#' by Pre-training Modular Transformers. arXiv. \doi{10.48550/ARXIV.2205.06266}
#' @return `r get_description("return_object")`
#' @family Base Model
#' @export
BaseModelXmod <- R6::R6Class(
  classname = "BaseModelXmod",
  inherit = BaseModelCore,
  private = list(
    model_type = "xmod",
    slow_tokenizer = "XLMRobertaTokenizer",
    adjust_max_sequence_length = 4L,
    create_model = function(args) {
      configuration <- transformers$XmodConfig(
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
        use_cache = TRUE,
        classifier_dropout = 0.1,
        pre_norm = FALSE,
        adapter_reduction_factor = 2L,
        adapter_layer_norm = FALSE,
        adapter_reuse_layer_norm = TRUE,
        ln_before_adapter = TRUE,
        languages = reticulate::tuple(args$languages, convert = FALSE),
        default_language = if (is.null(args$default_language)) {
          args$languages[1L]
        } else {
          args$default_language
        },
        is_decoder = FALSE,
        add_cross_attention = FALSE,
        tie_word_embeddings = TRUE
      )
      private$model <- transformers$XmodForMaskedLM(configuration)
    },
    load_BaseModel = function(dir_path) {
      tmp_model <- transformers$XmodForMaskedLM$from_pretrained(dir_path)
      private$model <- tmp_model
      supported_languages <- tmp_model$config$languages
      tmp_model$set_default_language(supported_languages[1L])
      message("Load model with ", supported_languages[1L], " as default language.")
      return(private$model)
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
    #' @param languages `r get_param_doc_desc("languages")`
    #' @param default_language `r get_param_doc_desc("default_language")`
    #' @param hidden_size `r get_param_doc_desc("hidden_size")`
    #' @param num_hidden_layers `r get_param_doc_desc("num_hidden_layers")`
    #' @param num_attention_heads `r get_param_doc_desc("num_attention_heads")`
    #' @param intermediate_size `r get_param_doc_desc("intermediate_size")`
    #' @param hidden_act `r get_param_doc_desc("hidden_act")`
    #' @param hidden_dropout_prob `r get_param_doc_desc("hidden_dropout_prob")`
    #' @param attention_probs_dropout_prob `r get_param_doc_desc("attention_probs_dropout_prob")`
    #' @return `r get_description("return_nothing")`
    configure = function(tokenizer,
                         languages = c("eng", "deu"),
                         default_language = "deu",
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
    #---------------------------------------------------------------------------
    #' @description Get the supported languages of the model.
    #' @param language_code `string` Language code to use as default.
    #' @return Returns a `vector` of `string`s that represent the supported
    #' language codes.
    get_supported_languages = function() {
      languages <- reticulate::py_to_r(private$model$config$languages)
      return(languages)
    },
    #---------------------------------------------------------------------------
    #' @description Set the default language of the model.
    #' @param language_code `string` Language code to use as default.
    #' @return `r get_description("return_nothing")`
    set_default_language = function(language_code) {
      private$model$set_default_language(language_code)
    },
    #---------------------------------------------------------------------------
    #' @description Get the default language of the model.
    #' @return Returns a `string` representing the default language.
    get_default_language = function() {
      return(private$model$config$default_language)
    },
    #---------------------------------------------------------------------------
    #' @description Print method for classifiers.
    #' @return Prints a short description of the object.
    print = function() {
      rows <- c(
        "Object", "Configured", "Trained", "Parameter", "Seq. Len.", "Features",
        "N Layer", "Vocab Size", "Tokens/Word", "Mask Token", "Pad Token",
        "Unk token", "languages", "default_language"
      )
      padded_rows <- pad_str(rows, width = NULL, pad = " ", end = ": ")
      statistics <- self$Tokenizer$get_tokenizer_statistics()
      special_tokens <- self$Tokenizer$get_special_tokens()

      message(
        appendLF = FALSE,
        padded_rows[1L], class(self)[1L], "\n",
        padded_rows[2L], self$is_configured(), "\n",
        padded_rows[3L], self$is_trained(), "\n",
        padded_rows[4L], self$count_parameter(), "\n",
        padded_rows[5L], self$get_model_config()$max_position_embeddings, "\n",
        padded_rows[6L], self$get_final_size(), "\n",
        padded_rows[7L], self$get_n_layers(), "\n",
        padded_rows[8L], self$Tokenizer$get_vocab_size(), "\n",
        padded_rows[9L], statistics[1L, "mu_g"], "\n",
        padded_rows[10L], special_tokens["mask_token", "token"], "\n",
        padded_rows[11L], special_tokens["pad_token", "token"], "\n",
        padded_rows[12L], special_tokens["unk_token", "token"], "\n",
        padded_rows[13L], toString(self$get_supported_languages()), "\n",
        padded_rows[14L], self$get_default_language(), "\n"
      )
    }
  )
)

# Add the model to the user list
BaseModelsIndex$Xmod <- list(
  class_name = "BaseModelXmod",
  model_type = "xmod",
  reference = "Pfeiffer, J., Goyal, N., Lin, X. V., Li, X., Cross, J.,
  Riedel, S., & Artetxe, M. (2022). Lifting the Curse of Multilinguality
  by Pre-training Modular Transformers. arXiv. doi: [10.48550/arXiv.2205.06266](https://doi.org/10.48550/arXiv.2205.06266)",
  req_sentencepiece = TRUE
)
