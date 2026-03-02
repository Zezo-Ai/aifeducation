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

#' @title Text embedding model
#' @description This `R6` class stores a text embedding model which can be used to tokenize, encode, decode, and embed
#'   raw texts. The object provides a unique interface for different text processing methods.
#' @return Objects of class [TextEmbeddingModel] transform raw texts into numerical representations which can be used
#'   for downstream tasks. For this aim objects of this class allow to tokenize raw texts, to encode tokens to sequences
#'   of integers, and to decode sequences of integers back to tokens.
#' @family Text Embedding
#' @export
TextEmbeddingModel <- R6::R6Class(
  classname = "TextEmbeddingModel",
  inherit = AIFEBaseModel,
  private = list(
    publication_info = list(
      developed_by = list(
        authors = NULL,
        citation = NULL,
        url = NULL
      ),
      modified_by = list(
        authors = NULL,
        citation = NULL,
        url = NULL
      )
    ),

    #--------------------------------------------------------------------------
    # Method for generating a model id
    generate_model_id = function(name) {
      if (is.null(name)) {
        return(paste0("tem_", generate_id(16L)))
      } else {
        return(name)
      }
    },
    #--------------------------------------------------------------------------
    # Method for setting the model info
    set_model_info = function(model_name, label, model_date, model_language) {
      private$model_info$model_name <- model_name
      private$model_info$model_label <- label
      private$model_info$model_date <- model_date
      private$model_info$model_language <- model_language
    },
    #-------------------------------------------------------------------------
    load_reload_python_scripts = function() {
      load_py_scripts(
        files = c(
          "pytorch_layers.py",
          "MPNetForMPLM_PT.py",
          "pytorch_text_embedding_model.py"
        )
      )
    },
    #-------------------------------------------------------------------------
    # Method for checking and setting the embedding configuration
    check_and_set_embedding_layers = function(emb_layer_min,
                                              emb_layer_max) {
      if (self$BaseModel$get_model_type() == "funnel") {
        max_layers_funnel <- sum(
          self$BaseModel$get_model()$config$block_repeats *
            self$BaseModel$get_model()$config$block_sizes
        )

        if (emb_layer_min == "First") {
          emb_layer_min <- 1L
        } else if (emb_layer_min == "Middle") {
          emb_layer_min <- floor(0.5 * max_layers_funnel)
        } else if (emb_layer_min == "2_3_layer") {
          emb_layer_min <- floor(2L / 3L * max_layers_funnel)
        } else if (emb_layer_min == "Last") {
          emb_layer_min <- max_layers_funnel
        }

        if (emb_layer_max == "First") {
          emb_layer_max <- 1L
        } else if (emb_layer_max == "Middle") {
          emb_layer_max <- floor(0.5 * max_layers_funnel)
        } else if (emb_layer_max == "2_3_layer") {
          emb_layer_max <- floor(2L / 3L * max_layers_funnel)
        } else if (emb_layer_max == "Last") {
          emb_layer_max <- max_layers_funnel
        }
      } else {
        if (emb_layer_min == "First") {
          emb_layer_min <- 1L
        } else if (emb_layer_min == "Middle") {
          emb_layer_min <- floor(0.5 * self$BaseModel$get_model()$config$num_hidden_layers)
        } else if (emb_layer_min == "2_3_layer") {
          emb_layer_min <- floor(2L / 3L * self$BaseModel$get_model()$config$num_hidden_layers)
        } else if (emb_layer_min == "Last") {
          emb_layer_min <- self$BaseModel$get_model()$config$num_hidden_layers
        }

        if (emb_layer_max == "First") {
          emb_layer_max <- 1L
        } else if (emb_layer_max == "Middle") {
          emb_layer_max <- floor(0.5 * self$BaseModel$get_model()$config$num_hidden_layers)
        } else if (emb_layer_max == "2_3_layer") {
          emb_layer_max <- floor(2L / 3L * self$BaseModel$get_model()$config$num_hidden_layers)
        } else if (emb_layer_max == "Last") {
          emb_layer_max <- self$BaseModel$get_model()$config$num_hidden_layers
        }
      }

      # Check requested configuration
      if (emb_layer_min > emb_layer_max) {
        stop("emb_layer_min layer must be smaller or equal emb_layer_max.")
      }
      if (emb_layer_min < 1L) {
        stop("emb_laser_min must be at least 1.")
      }
      if (self$BaseModel$get_model_type() == "funnel") {
        if (emb_layer_max > self$BaseModel$get_model()$config$num_hidden_layers) {
          stop("emb_layer_max can not exceed the number of layers. The transformer has", max_layers_funnel, "layers.")
        }
      } else {
        if (emb_layer_max > self$BaseModel$get_model()$config$num_hidden_layers) {
          stop("emb_layer_max can not exceed the number of layers. The transformer has", self$BaseModel$get_model()$config$num_hidden_layers, "layers.")
        }
      }

      if (!is.integer(as.integer(emb_layer_min)) | !is.integer(as.integer(emb_layer_max))) {
        stop("emb_layer_min and emb_layer_max must be integers or the following string:
               'first','last','middle','2_3_layer'")
      }

      private$model_config$emb_layer_min <- emb_layer_min
      private$model_config$emb_layer_max <- emb_layer_max
    },
    #-------------------------------------------------------------------------
    # Method for checking and setting pooling type
    check_and_set_pooling_type = function(emb_pool_type) {
      if (self$BaseModel$get_model_type() == "funnel" & emb_pool_type != "CLS") {
        message(
          "Text embedding based on a funnel transformer allows only 'CLS' as pooling type and not ",
          emb_pool_type, ". Changing method to 'CLS'"
        )
        emb_pool_type <- "CLS"
      }
      private$model_config$emb_pool_type <- emb_pool_type
    },
    #-------------------------------------------------------------------------
    # Method for checking and setting max_length
    check_and_set_max_length = function(max_length) {
      if (max_length > (self$BaseModel$get_model()$config$max_position_embeddings)) {
        message(
          "max_length is ", max_length, ". This value is not allowed to exceed ",
          self$BaseModel$get_model()$config$max_position_embeddings,
          "Set value to ", self$BaseModel$get_model()$config$max_position_embeddings, "."
        )
        private$model_config$max_length <- as.integer(self$BaseModel$get_model()$config$max_position_embeddings)
      } else {
        private$model_config$max_length <- as.integer(max_length)
      }
    }
  ),
  public = list(
    #' @field BaseModel ('BaseModelCore')\cr
    #' Object of class `BaseModelCore`.
    BaseModel = NULL,

    #--------------------------------------------------------------------------
    #' @description Method for creating a new text embedding model
    #' @param model_name `r get_param_doc_desc("model_name")`
    #' @param model_label `r get_param_doc_desc("model_label")`
    #' @param model_language `r get_param_doc_desc("model_language")`
    #' @param base_model `r get_param_doc_desc("base_model")`
    #' @param max_length `r get_param_doc_desc("max_length")`
    #' @param chunks `r get_param_doc_desc("chunks")`
    #' @param overlap `r get_param_doc_desc("overlap")`
    #' @param emb_layer_min `r get_param_doc_desc("emb_layer_min")`
    #' @param emb_layer_max `r get_param_doc_desc("emb_layer_max")`
    #' @param emb_pool_type `r get_param_doc_desc("emb_pool_type")`
    #' @param pad_value `r get_param_doc_desc("pad_value")`
    #' @param trace `r get_param_doc_desc("trace")`
    #' @return `r get_description("return_nothing")`
    #'
    #' @import reticulate
    #' @import stats
    #' @import reshape2
    configure = function(model_name = NULL,
                         model_label = NULL,
                         model_language = NULL,
                         max_length = 512L,
                         chunks = 2L,
                         overlap = 0L,
                         emb_layer_min = 1L,
                         emb_layer_max = 2L,
                         emb_pool_type = "Average",
                         pad_value = -100L,
                         base_model = NULL) {
      # Load or reload python scripts
      private$load_reload_python_scripts()

      # Check if the object is not configured
      private$check_config_for_FALSE()

      # Load BaseModel
      self$BaseModel <- base_model$clone(deep = TRUE)

      # Save Embedding Config
      private$save_all_args(args = get_called_args(n = 1L), group = "configure")

      # Set Model info
      private$set_model_info(
        model_name = private$generate_model_id(model_name),
        label = model_label,
        model_date = get_time_stamp(),
        model_language = model_language
      )

      # Check and Set Embedding Configuration
      private$check_and_set_embedding_layers(
        emb_layer_min = emb_layer_min,
        emb_layer_max = emb_layer_max
      )

      # Check and set max length
      private$check_and_set_max_length(max_length)

      # Check and set pooling type
      private$check_and_set_pooling_type(emb_pool_type)

      # Set package versions
      private$set_package_versions()

      # Close config
      private$set_configuration_to_TRUE()
    },
    #--------------------------------------------------------------------------
    #' @description Loads an object from disk
    #' and updates the object to the current version of the package.
    #' @param dir_path `r get_description("load_dir")`
    #' @return `r get_description("return_load_on_disk")`
    load_from_disk = function(dir_path) {
      # Load private and public config files
      private$load_config_file(dir_path)

      # Load or reload python scripts
      private$load_reload_python_scripts()

      # Load Base model
      version_lower <- check_versions(
        a = "1.1.2",
        operator = ">",
        b = self$get_package_versions()$r_package_versions$aifeducation
      )
      if (version_lower) {
        # Old version that does not use BaseModel and Tokenizer
        path_to_files <- file.path(dir_path, "model_data")
        tmp_pytorch_model <- transformers$AutoModelForMaskedLM$from_pretrained(path_to_files)
        tmp_type <- detect_base_model_type(tmp_pytorch_model$config)
        tmp_BaseModel <- create_object(tmp_type)
        tmp_BaseModel$create_from_hf(
          model_dir = path_to_files,
          tokenizer_dir = path_to_files
        )
        self$BaseModel <- tmp_BaseModel
      } else {
        # Regular case
        self$BaseModel <- load_from_disk(dir_path = file.path(dir_path, "base_model"))
      }

      # Load Sustainability Data Inference
      private$load_sustainability_data_inference(model_dir = dir_path)

      # Finalize config
      private$set_configuration_to_TRUE()
    },
    #--------------------------------------------------------------------------
    #' @description Method for saving a model on disk.
    #' @param dir_path `r get_description("save_dir")`
    #' @param folder_name `r get_param_doc_desc("folder_name")`
    #' @return `r get_description("return_save_on_disk")`
    #'
    #' @importFrom utils write.csv
    save = function(dir_path, folder_name) {
      save_location <- file.path(dir_path, folder_name)
      create_dir(dir_path = save_location, trace = FALSE)

      # Save BaseModel
      save_to_disk(
        object = self$BaseModel,
        dir_path = save_location,
        folder_name = "base_model"
      )

      # Save Sustainability Data Inference
      private$save_sustainability_data_inference(dir_path = dir_path, folder_name = folder_name)
    },
    #-------------------------------------------------------------------------
    #' @description Method for encoding words of raw texts into integers.
    #' @param raw_text `r get_param_doc_desc("raw_text")`
    #' @param token_encodings_only `r get_param_doc_desc("token_encodings_only")`
    #' @param token_to_int `r get_param_doc_desc("token_to_int")`
    #' @param trace `r get_param_doc_desc("trace")`
    #' @return `list` containing the integer or token sequences of the raw texts with
    #' special tokens.
    encode = function(raw_text,
                      token_encodings_only = FALSE,
                      token_to_int = TRUE,
                      trace = FALSE) {
      return(
        self$BaseModel$Tokenizer$encode(
          raw_text = raw_text,
          token_overlap = private$model_config$overlap,
          max_token_sequence_length = private$model_config$max_length,
          n_chunks = private$model_config$chunks,
          token_encodings_only = token_encodings_only,
          token_to_int = token_to_int,
          trace = trace
        )
      )
    },
    #--------------------------------------------------------------------------
    #' @description Method for decoding a sequence of integers into tokens
    #' @param int_seqence `r get_param_doc_desc("int_seqence")`
    #' @param to_token `r get_param_doc_desc("to_token")`
    #' @return `list` of token sequences
    decode = function(int_seqence, to_token = FALSE) {
      return(
        self$BaseModel$Tokenizer$decode(
          int_seqence = int_seqence,
          to_token = to_token
        )
      )
    },
    # Embedding------------------------------------------------------------------
    #' @description Method for creating text embeddings from raw texts.
    #' This method should only be used if a small number of texts should be transformed
    #' into text embeddings. For a large number of texts please use the method `embed_large`.
    #' @param raw_text `r get_param_doc_desc("raw_text")`
    #' @param doc_id `r get_param_doc_desc("doc_id")`
    #' @param batch_size `r get_param_doc_desc("batch_size")`
    #' @param trace `r get_param_doc_desc("trace")`
    #' @param return_large_dataset `r get_param_doc_desc("return_large_dataset")`
    #' @return Method returns an object of class [EmbeddedText] or [LargeDataSetForTextEmbeddings]. This object
    #' contains the embeddings as a [data.frame] and information about the
    #' model creating the embeddings.
    embed = function(raw_text = NULL, doc_id = NULL, batch_size = 8L, trace = FALSE, return_large_dataset = FALSE) {
      # check arguments
      check_type(object = raw_text, type = "vector", FALSE)
      check_type(object = doc_id, type = "vector", FALSE)
      check_type(object = batch_size, type = "int", FALSE)
      check_type(object = trace, type = "bool", FALSE)
      check_type(object = return_large_dataset, type = "bool", FALSE)

      # Load python scripts
      private$load_reload_python_scripts()

      # Object for storing embeddings
      batch_results <- list()

      # Get tokenizer
      tokenizer <- self$BaseModel$Tokenizer$get_tokenizer()

      # get device and data type
      if (torch$cuda$is_available()) {
        pytorch_device <- "cuda"
        pytorch_dtype <- torch$float
      } else {
        pytorch_device <- "cpu"
        pytorch_dtype <- torch$double
      }

      require_token_type_ids <- self$BaseModel$get_private()$return_token_type_ids
      sequence_mode <- self$BaseModel$get_private()$sequence_mode

      # Create a model for embedding
      pytorch_embedding_model <- py$TextEmbeddingModel(
        base_model = self$BaseModel$get_model(),
        chunks = as.integer(private$model_config$chunks),
        emb_layer_min = as.integer(private$model_config$emb_layer_min),
        emb_layer_max = as.integer(private$model_config$emb_layer_max),
        pad_value = private$model_config$pad_value,
        emb_pool_type = private$model_config$emb_pool_type,
        sequence_mode = sequence_mode
      )

      pytorch_embedding_model$to(device = pytorch_device, dtype = pytorch_dtype)
      pytorch_embedding_model$eval()

      n_documents <- length(raw_text)
      for (i in seq_along(raw_text)) {
        tokens <- tokenizer(
          raw_text[i],
          stride = as.integer(private$model_config$overlap),
          padding = "max_length",
          truncation = TRUE,
          max_length = as.integer(private$model_config$max_length),
          return_overflowing_tokens = TRUE,
          return_length = FALSE,
          return_offsets_mapping = FALSE,
          return_attention_mask = TRUE,
          return_token_type_ids = require_token_type_ids,
          return_tensors = "pt"
        )

        if (require_token_type_ids) {
          tmp_embeddings <- pytorch_embedding_model(
            input_ids = tokens["input_ids"]$to(device = pytorch_device),
            attention_mask = tokens["attention_mask"]$to(device = pytorch_device),
            token_type_ids = tokens["token_type_ids"]$to(device = pytorch_device)
          )
        } else {
          tmp_embeddings <- pytorch_embedding_model(
            input_ids = tokens["input_ids"]$to(device = pytorch_device),
            attention_mask = tokens["attention_mask"]$to(device = pytorch_device)
          )
        }


        tmp_embeddings <- tensor_to_numpy(tmp_embeddings)
        rownames(tmp_embeddings) <- doc_id[i]

        batch_results[length(batch_results) + 1] <- list(tmp_embeddings)

        if (trace == TRUE) {
          cat(paste(
            date(),
            "Document", i, "/", n_documents, "Done", "\n"
          ))
        }
      }

      # Summarizing the results
      text_embedding <- array_form_bind(batch_results)

      embeddings <- EmbeddedText$new()
      embeddings$configure(
        model_name = private$model_info$model_name,
        model_label = private$model_info$model_label,
        model_date = private$model_info$model_date,
        model_method = self$BaseModel$get_model_type(),
        model_language = private$model_info$model_language,
        param_seq_length = private$model_config$max_length,
        param_features = dim(text_embedding)[3L],
        param_chunks = private$model_config$chunks,
        param_overlap = private$model_config$overlap,
        param_emb_layer_min = private$model_config$emb_layer_min,
        param_emb_layer_max = private$model_config$emb_layer_max,
        param_emb_pool_type = private$model_config$emb_pool_type,
        param_pad_value = private$model_config$pad_value,
        param_aggregation = NA,
        embeddings = text_embedding
      )

      if (!return_large_dataset) {
        return(embeddings)
      } else {
        embedded_texts_large <- LargeDataSetForTextEmbeddings$new()
        embedded_texts_large$configure(
          model_name = private$model_info$model_name,
          model_label = private$model_info$model_label,
          model_date = private$model_info$model_date,
          model_method = self$BaseModel$get_model_type(),
          model_language = private$model_info$model_language,
          param_seq_length = private$model_config$max_length,
          param_features = dim(embeddings$embeddings)[length(dim(embeddings$embeddings))],
          param_chunks = private$model_config$chunks,
          param_overlap = private$model_config$overlap,
          param_emb_layer_min = private$model_config$emb_layer_min,
          param_emb_layer_max = private$model_config$emb_layer_max,
          param_emb_pool_type = private$model_config$emb_pool_type,
          param_pad_value = private$model_config$pad_value,
          param_aggregation = NA
        )
        # Add new data
        embedded_texts_large$add_embeddings_from_EmbeddedText(embeddings)
        return(embedded_texts_large)
      }
    },
    #--------------------------------------------------------------------------
    #' @description Method for creating text embeddings from raw texts.
    #'
    #' @param text_dataset `r get_param_doc_desc("text_dataset")`
    #' @param batch_size `r get_param_doc_desc("batch_size")`
    #' @param trace `r get_param_doc_desc("trace")`
    #' @param log_file `r get_param_doc_desc("log_file")`
    #' @param log_write_interval `r get_param_doc_desc("log_write_interval")`
    #' @return Method returns an object of class [LargeDataSetForTextEmbeddings].
    embed_large = function(text_dataset,
                           batch_size = 32L,
                           trace = FALSE,
                           log_file = NULL,
                           log_write_interval = 2L) {
      # Check arguments
      check_class(object = text_dataset, classes = c("LargeDataSetForText", allow_NULL = FALSE))
      check_type(object = batch_size, type = "int", FALSE)
      check_type(object = trace, type = "bool", FALSE)

      # Get total number of batches for the loop
      total_number_of_bachtes <- ceiling(text_dataset$n_rows() / batch_size)

      # Get indices for every batch
      batches_index <- get_batches_index(
        number_rows = text_dataset$n_rows(),
        batch_size = batch_size,
        zero_based = TRUE
      )
      # Set up log
      last_log <- NULL

      # Process every batch
      for (i in 1L:total_number_of_bachtes) {
        tmp_subset <- text_dataset$select(as.integer(batches_index[[i]]))
        embeddings <- self$embed(
          raw_text = extract_column_from_py_dataset(tmp_subset,"text"),
          doc_id = extract_column_from_py_dataset(tmp_subset,"id"),
          batch_size = batch_size,
          trace = FALSE
        )
        if (i == 1L) {
          # Create Large Dataset
          embedded_texts_large <- LargeDataSetForTextEmbeddings$new()
          embedded_texts_large$configure(
            model_name = private$model_info$model_name,
            model_label = private$model_info$model_label,
            model_date = private$model_info$model_date,
            model_method = self$BaseModel$get_model_type(),
            model_language = private$model_info$model_language,
            param_seq_length = private$model_config$max_length,
            param_features = dim(embeddings$embeddings)[3L],
            param_chunks = private$model_config$chunks,
            param_overlap = private$model_config$overlap,
            param_emb_layer_min = private$model_config$emb_layer_min,
            param_emb_layer_max = private$model_config$emb_layer_max,
            param_emb_pool_type = private$model_config$emb_pool_type,
            param_aggregation = NA,
            param_pad_value = private$model_config$pad_value
          )
          # Add new data
          embedded_texts_large$add_embeddings_from_EmbeddedText(embeddings)
        } else {
          # Add new data
          embedded_texts_large$add_embeddings_from_EmbeddedText(embeddings)
        }
        if (trace) {
          cat(paste(
            get_time_stamp(),
            "Batch", i, "/", total_number_of_bachtes, "done", "\n"
          ))
        }

        # Update log
        last_log <- write_log(
          log_file = log_file,
          last_log = last_log,
          write_interval = log_write_interval,
          value_top = i,
          value_middle = 0L,
          value_bottom = 0L,
          total_top = total_number_of_bachtes,
          total_middle = 1L,
          total_bottom = 1L,
          message_top = "Batches",
          message_middle = NA,
          message_bottom = NA
        )
        gc()
      }
      return(embedded_texts_large)
    },
    #---------------------------------------------------------------------------
    #' @description Method for requesting the number of features.
    #' @return Returns a `double` which represents the number of features. This number represents the
    #' hidden size of the embeddings for every chunk or time.
    get_n_features = function() {
      return(self$BaseModel$get_final_size())
    },
    #---------------------------------------------------------------------------
    #' @description Value for indicating padding.
    #' @return Returns an `int` describing the value used for padding.
    get_pad_value = function() {
      return(private$model_config$pad_value)
    },
    #--------------------------------------------------------------------------
    #' @description Method for setting the bibliographic information of the model.
    #' @param type `string` Type of information which should be changed/added.
    #' `developer`, and `modifier` are possible.
    #' @param authors List of people.
    #' @param citation `string` Citation in free text.
    #' @param url `string` Corresponding URL if applicable.
    #' @return Function does not return a value. It is used to set the private
    #' members for publication information of the model.
    set_publication_info = function(type,
                                    authors,
                                    citation,
                                    url = NULL) {
      if (type == "developer") {
        private$publication_info$developed_by$authors <- authors
        private$publication_info$developed_by$citation <- citation
        private$publication_info$developed_by$url <- url
      } else if (type == "modifier") {
        private$publication_info$modified_by$authors <- authors
        private$publication_info$modified_by$citation <- citation
        private$publication_info$modified_by$url <- url
      }
    },
    #---------------------------------------------------------------------------
    #' @description Method for requesting a summary of tracked energy consumption during training and an estimate of the
    #'   resulting CO2 equivalents in kg.
    #' @param track_mode `r get_param_doc_desc("track_mode")`
    #' @return Returns a `list` containing the tracked energy consumption, CO2 equivalents in kg, information on the
    #'   tracker used, and technical information on the training infrastructure.
    get_sustainability_data = function(track_mode = "training") {
      if (track_mode == "training") {
        return(self$BaseModel$get_sustainability_data("training"))
      } else if (track_mode == "inference") {
        return(private$sustainability_inference)
      }
    },
    #--------------------------------------------------------------------------
    #' @description Calculates the energy consumption for inference of the given task.
    #' @param text_dataset `r get_param_doc_desc("text_dataset")`
    #' @param batch_size `r get_param_doc_desc("batch_size")`
    #' @param sustain_iso_code `r get_param_doc_desc("sustain_iso_code")`
    #' @param sustain_region `r get_param_doc_desc("sustain_region")`
    #' @param sustain_interval `r get_param_doc_desc("sustain_interval")`
    #' @param sustain_log_level `r get_param_doc_desc("sustain_log_level")`
    #' @param trace `r get_param_doc_desc("trace")`
    #' @return Returns nothing. Method saves the statistics internally.
    #' The statistics can be accessed with the method `get_sustainability_data("inference")`
    estimate_sustainability_inference_embed = function(text_dataset = NULL,
                                                       batch_size = 32L,
                                                       sustain_iso_code = NULL,
                                                       sustain_region = NULL,
                                                       sustain_interval = 10L,
                                                       sustain_log_level = "warning",
                                                       trace = TRUE) {
      # Prepare Data
      print_message(
        msg = "Prepare Data",
        trace = trace
      )

      n_cases <- text_dataset$n_rows()

      # Gather information on data
      print_message(
        msg = "Gather Information",
        trace = trace
      )

      emp_seq_length <- vector(length = n_cases)
      arrow_dataset <- text_dataset$get_dataset()
      for (i in 1L:n_cases) {
        tmp_encode <- self$encode(arrow_dataset[i - 1L]$text,
          token_encodings_only = FALSE,
          token_to_int = TRUE,
          trace = FALSE
        )
        emp_seq_length[i] <- tmp_encode$chunks * private$model_config$max_length - (tmp_encode$chunks - 1L) * private$model_config$overlap
      }

      # Start Tracking
      private$init_and_start_sustainability_tracker(
        trace = trace,
        country_iso_code = sustain_iso_code,
        region = sustain_region,
        measure_power_secs = sustain_interval,
        sustain_log_level = sustain_log_level
      )

      # Start Task
      result_embeddings <- self$embed_large(
        text_dataset = text_dataset,
        batch_size = batch_size,
        trace = FALSE,
        log_file = NULL,
        log_write_interval = 2L
      )

      # Stop Tracking
      results <- private$stop_sustainability_tracker(
        trace = trace,
        task = "Embed"
      )

      # Add additional information
      results$data <- "empirical data"
      results$n <- n_cases
      results$batch <- batch_size
      results$min_seq_len <- min(emp_seq_length)
      results$mean_seq_len <- mean(emp_seq_length)
      results$sd_seq_len <- sd(emp_seq_length)
      results$max_seq_len <- max(emp_seq_length)

      if (is.null_or_na(private$sustainability_inference)) {
        private$sustainability_inference <- results
      } else {
        private$sustainability_inference <- rbind(
          private$sustainability_inference,
          results
        )
      }
    }
  )
)

# Add Object to index
TextEmbeddingObjectsIndex$TextEmbeddingModel <- "TextEmbeddingModel"
