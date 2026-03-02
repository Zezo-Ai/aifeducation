testthat::skip_on_cran()
testthat::skip_if_not(
  condition = check_aif_py_modules(trace = FALSE),
  message = "Necessary python modules not available"
)
load_all_py_scripts()

testthat::skip_if_not(
  condition = dir.exists(testthat::test_path("test_data_tmp/TEM")),
  message = "Base models for tests not available"
)


max_samples <- 4
max_samples_CI <- 1

# Start time
test_time_start <- Sys.time()

# SetUp-------------------------------------------------------------------------
# Set paths
root_path_data <- testthat::test_path("test_data_tmp/TEM")
create_dir(testthat::test_path("test_artefacts"), FALSE)

root_path_results <- testthat::test_path("test_artefacts/TEM")
create_dir(root_path_results, FALSE)

# SetUp datasets
# Disable tqdm progressbar
transformers$logging$disable_progress_bar()
datasets$disable_progress_bars()

# load data for test
# Use internal sample data
example_data <- imdb_movie_reviews

# Create LargeDataSet
example_data_for_large <- example_data
empty_vector <- vector(length = nrow(example_data))
empty_vector[] <- NA
example_data_for_large$citation <- empty_vector
example_data_for_large$bib_entry <- empty_vector
example_data_for_large$license <- empty_vector
example_data_for_large$url_license <- empty_vector
example_data_for_large$text_license <- empty_vector
example_data_for_large$url_source <- empty_vector

example_data_large <- LargeDataSetForText$new()
example_data_large$add_from_data.frame(example_data_for_large)

example_data_large_single <- LargeDataSetForText$new()
example_data_large_single$add_from_data.frame(example_data_for_large[1, ])


# config
# Set Chunks
base_model_type_list <- BaseModelsIndex

# Start tests--------------------------------------------------------------------
for (base_model_type in base_model_type_list) {
  # Set path to the base model
  model_path <- paste0(
    root_path_data, "/",
    base_model_type
  )

  # Error Checking: Max layer greater as the number of layers
  # Load a BaseModel
  base_model <- load_from_disk(model_path)

  config <- generate_args_for_tests(
    object_name = "TextEmbeddingModel",
    method = "configure",
    var_objects = list(),
    necessary_objects = list(
      base_model = base_model
    ),
    var_override = list(
      model_name = paste0(base_model_type, "_embedding"),
      model_label = paste0("Text Embedding via", base_model_type),
      model_language = "english",
      trace = random_bool_on_CI(),
      emb_layer_max = 50000L
    )
  )
  test_that(paste(base_model_type, get_current_args_for_print(config), "Max layer greater as the number of layers"), {
    text_embedding_model <- TextEmbeddingModel$new()
    expect_error(
      suppressMessages(
        do.call(what = text_embedding_model$configure,
        args = config
      )
    )
  )
  })

  # Error Checking: min layer is smaller 1
  # Load a BaseModel
  base_model <- load_from_disk(model_path)

  config <- generate_args_for_tests(
    object_name = "TextEmbeddingModel",
    method = "configure",
    var_objects = list(),
    necessary_objects = list(
      base_model = base_model
    ),
    var_override = list(
      model_name = paste0(base_model_type, "_embedding"),
      model_label = paste0("Text Embedding via", base_model_type),
      model_language = "english",
      trace = random_bool_on_CI(),
      emb_layer_min = 0L
    )
  )
  test_that(paste(base_model_type, get_current_args_for_print(config), "Error Checking: min layer is smaller 1"), {
    text_embedding_model <- TextEmbeddingModel$new()
    expect_error(
      suppressMessages(
        do.call(what = text_embedding_model$configure,
        args = config
      )
    )
    )
  })

  # Error Checking: Configuration already set
  # Load a BaseModel
  base_model <- load_from_disk(model_path)

  config <- generate_args_for_tests(
    object_name = "TextEmbeddingModel",
    method = "configure",
    var_objects = list(),
    necessary_objects = list(
      base_model = base_model
    ),
    var_override = list(
      model_name = paste0(base_model_type, "_embedding"),
      model_label = paste0("Text Embedding via", base_model_type),
      model_language = "english",
      trace = random_bool_on_CI(),
      min_layer = 0L
    )
  )
  test_that(paste(base_model_type, get_current_args_for_print(config), "Error Checking: Configuration already set"), {
    text_embedding_model <- TextEmbeddingModel$new()
    suppressMessages(
      do.call(what = text_embedding_model$configure,
      args = config
    )
    )
    expect_error(
      suppressMessages(
        do.call(what = text_embedding_model$configure,
        args = config
      )
    )
    )
  })


  # Test of core features of the model
  for (i in 1:check_adjust_n_samples_on_CI(
    n_samples_requested = max_samples,
    n_CI = max_samples_CI
  )) {
    # Load a BaseModel
    base_model <- load_from_disk(model_path)

    config <- generate_args_for_tests(
      object_name = "TextEmbeddingModel",
      method = "configure",
      var_objects = list(),
      necessary_objects = list(
        base_model = base_model
      ),
      var_override = list(
        model_name = paste0(base_model_type, "_embedding"),
        model_label = paste0("Text Embedding via", base_model_type),
        model_language = "english",
        trace = random_bool_on_CI()
      )
    )


    # Central methods--------------------------------------------------------
    # Create Model
    text_embedding_model <- TextEmbeddingModel$new()
    suppressMessages(
      do.call(what = text_embedding_model$configure,
              args = config
      )
    )
    # Check history
    test_that(paste(base_model_type, get_current_args_for_print(config), "history"), {
      history <- text_embedding_model$BaseModel$last_training$history
      expect_equal(nrow(history), 2)
      expect_equal(ncol(history), 3)
      expect_true("epoch" %in% colnames(history))
      expect_true("loss" %in% colnames(history))
      expect_true("val_loss" %in% colnames(history))
    })

    # Check Configuration of the model
    test_that(paste(base_model_type, get_current_args_for_print(config), "configuration"), {
      tr_comp <- text_embedding_model$get_model_config()
      expect_equal(tr_comp$emb_layer_min, config$emb_layer_min)
      expect_equal(tr_comp$emb_layer_max, config$emb_layer_max)
      if (base_model_type == "BaseModelFunnel") {
        expect_equal(tr_comp$emb_pool_type, "CLS")
      } else {
        expect_equal(tr_comp$emb_pool_type, config$emb_pool_type)
      }
    })

    # Method embed--------------------------------------------------------
    test_that(paste(base_model_type, get_current_args_for_print(config), "embed"), {
      # general
      embeddings <- text_embedding_model$embed(
        raw_text = example_data$text[1:10],
        doc_id = example_data$id[1:10],
        batch_size = 5
      )
      expect_s3_class(embeddings, class = "EmbeddedText")
      expect_false(embeddings$is_compressed())
      expect_equal(embeddings$n_rows(), 10)
      expect_equal(embeddings$get_pad_value(), config$pad_value)

      # Check if embeddings are array with 3 dimensions
      expect_equal(length(dim(embeddings$embeddings)), 3)

      # Check if data is valid
      expect_false(anyNA(embeddings$embeddings), FALSE)
      expect_false(0 %in% get_n_chunks(
        text_embeddings = embeddings$embeddings,
        features = text_embedding_model$get_n_features(),
        times = config$chunks,
        pad_value = config$pad_value
      ))

      # check case order invariance
      perm <- sample(x = 1:10, size = 10, replace = FALSE)
      embeddings_perm <- text_embedding_model$embed(
        raw_text = example_data$text[perm],
        doc_id = example_data$id[perm],
        batch_size = 5
      )
      for (i in 1:10) {
        expect_equal(embeddings$embeddings[i, , , drop = FALSE],
          embeddings_perm$embeddings[rownames(embeddings$embeddings)[i], , , drop = FALSE],
          tolerance = 1e-6
        )
      }

      # Check embedding in LargeDataSetForTextEmbeddings
      embeddings_large <- text_embedding_model$embed(
        raw_text = example_data$text[1:10],
        doc_id = example_data$id[1:10],
        batch_size = 5,
        return_large_dataset = TRUE
      )
      expect_s3_class(embeddings_large, class = "LargeDataSetForTextEmbeddings")
      expect_equal(embeddings$embeddings,
        embeddings_large$convert_to_EmbeddedText()$embeddings,
        tolerance = 1e-6
      )
      expect_equal(embeddings_large$get_pad_value(), config$pad_value)

      # Check absence of random variation
      embeddings_2 <- text_embedding_model$embed(
        raw_text = example_data$text[1:10],
        doc_id = example_data$id[1:10],
        batch_size = 5
      )
      for (i in 1:10) {
        expect_equal(embeddings$embeddings[i, , , drop = FALSE],
          embeddings_2$embeddings[i, , , drop = FALSE],
          tolerance = 1e-6
        )
      }
    })

    test_that(paste(base_model_type, get_current_args_for_print(config), "embed single case"), {
      embeddings <- text_embedding_model$embed(
        raw_text = example_data$text[1:1],
        doc_id = example_data$id[1:1]
      )
      expect_s3_class(embeddings, class = "EmbeddedText")
      expect_false(embeddings$is_compressed())
      expect_equal(embeddings$n_rows(), 1)
    })

    # Method embed_large--------------------------------------------------
    test_that(paste(base_model_type, get_current_args_for_print(config), "embed_large"), {
      # general
      embeddings <- text_embedding_model$embed_large(example_data_large)
      expect_s3_class(embeddings, class = "LargeDataSetForTextEmbeddings")
      expect_false(embeddings$is_compressed())
      expect_equal(embeddings$n_rows(), nrow(example_data))
      expect_equal(embeddings$get_pad_value(), config$pad_value)
    })

    test_that(paste(base_model_type, get_current_args_for_print(config), "embed_large with log"), {
      # general
      log_dir <- paste0(root_path_results, "/", generate_id())
      create_dir(log_dir, FALSE)
      log_file <- paste0(log_dir, "aifeducation_state.log")
      embeddings <- text_embedding_model$embed_large(example_data_large,
        log_file = log_file,
        log_write_interval = 2
      )
      expect_s3_class(embeddings, class = "LargeDataSetForTextEmbeddings")
      expect_false(embeddings$is_compressed())
      expect_equal(embeddings$n_rows(), nrow(example_data))

      state_log_exists <- file.exists(log_file)
      expect_true(state_log_exists)
      if (state_log_exists) {
        log_state <- read.csv(log_file)
        expect_equal(nrow(log_state), 3)
        expect_equal(ncol(log_state), 3)
        expect_equal(colnames(log_state), c("value", "total", "message"))
      }
    })

    test_that(paste(base_model_type, get_current_args_for_print(config), "embed_large single case"), {
      embeddings <- text_embedding_model$embed_large(example_data_large_single)
      expect_s3_class(embeddings, class = "LargeDataSetForTextEmbeddings")
      expect_false(embeddings$is_compressed())
      expect_equal(embeddings$n_rows(), 1)
    })

    # encoding------------------------------------------------------------
    test_that(paste(base_model_type, get_current_args_for_print(config), "encoding"), {
      # Request for tokens only
      for (token_to_int in c(TRUE, FALSE)) {
        encodings <- text_embedding_model$encode(
          raw_text = example_data$text[1:10],
          token_encodings_only = TRUE,
          token_to_int = token_to_int
        )
        expect_length(encodings, 10)
        expect_type(encodings, type = "list")

        # Check order invariance
        perm <- sample(x = 1:10, size = 10, replace = FALSE)
        encodings_perm <- text_embedding_model$encode(
          raw_text = example_data$text[perm],
          token_encodings_only = TRUE,
          token_to_int = token_to_int
        )
        for (i in 1:10) {
          expect_equal(
            encodings[[i]],
            encodings_perm[[which(x = (perm == i))]]
          )
        }
      }

      # Request for all tokens types
      for (token_to_int in c(TRUE, FALSE)) {
        encodings <- text_embedding_model$encode(
          raw_text = example_data$text[1:10],
          token_encodings_only = FALSE
        )
        expect_type(encodings, type = "list")
        expect_equal(encodings$encodings$num_rows, sum(encodings$chunks))
      }
    })

    # Decoding-----------------------------------------------------------
    test_that(paste(base_model_type, get_current_args_for_print(config), "decoding"), {
      encodings <- text_embedding_model$encode(
        raw_text = example_data$text[1:10],
        token_encodings_only = TRUE,
        token_to_int = TRUE
      )
      for (to_token in c(TRUE, FALSE)) {
        decodings <- text_embedding_model$decode(
          encodings,
          to_token = to_token
        )
        expect_length(decodings, 10)
        expect_type(decodings, type = "list")
      }
    })

    # Method get_special_tokens
    test_that(paste(base_model_type, get_current_args_for_print(config), "get_special_tokens"), {
      tokens <- text_embedding_model$BaseModel$Tokenizer$get_special_tokens()
      expect_equal(nrow(tokens), 7)
      expect_equal(ncol(tokens), 3)
    })

    # Method fill_mask
    test_that(paste(base_model_type, get_current_args_for_print(config), "fill_mask"), {
      tokens <- text_embedding_model$BaseModel$get_special_tokens()
      mask_token <- tokens[which(tokens[, 1] == "mask_token"), 2]

      first_solution <- text_embedding_model$BaseModel$fill_mask(
        masked_text = paste("This is a", mask_token, "."),
        n_solutions = 5
      )

      expect_equal(length(first_solution), 1)
      expect_true(is.data.frame(first_solution[[1]]))
      expect_equal(nrow(first_solution[[1]]), 5)
      expect_equal(ncol(first_solution[[1]]), 3)

      second_solution <- text_embedding_model$BaseModel$fill_mask(
        masked_text = paste("This is a", mask_token, "."),
        n_solutions = 1
      )
      expect_equal(length(second_solution), 1)
      expect_true(is.data.frame(second_solution[[1]]))
      expect_equal(nrow(second_solution[[1]]), 1)
      expect_equal(ncol(second_solution[[1]]), 3)

      third_solution <- text_embedding_model$BaseModel$fill_mask(
        masked_text = paste(
          "This is a", mask_token, ".",
          "The weather is", mask_token, "."
        ),
        n_solutions = 5
      )
      expect_equal(length(third_solution), 2)
      for (i in 1:2) {
        expect_true(is.data.frame(third_solution[[i]]))
        expect_equal(nrow(third_solution[[i]]), 5)
        expect_equal(ncol(third_solution[[i]]), 3)
      }
    })

    # Estimate sustainability inference embed
    test_that(paste(base_model_type, get_current_args_for_print(config), "decoding"), {
      suppressMessages(
        text_embedding_model$estimate_sustainability_inference_embed(
          text_dataset = example_data_large,
          batch_size = 4,
          sustain_iso_code = "DEU",
          sustain_interval = 2,
          sustain_log_level = "error",
          trace = TRUE
        )
      )

      expect_equal(nrow(text_embedding_model$get_sustainability_data("inference")), 1)
    })

    # Function Saving and Loading-----------------------------------------
    test_that(paste(base_model_type, get_current_args_for_print(config), "function_save_load"), {
      folder_name <- paste0(
        "function_save_load_",
        "_",
        base_model_type, "_",
        config$emb_pool_type, "_",
        config$max_layer, "_",
        config$min_layer
      )
      save_location <- paste0(root_path_results, "/", folder_name)
      create_dir(save_location, FALSE)

      # embeddings for saving
      embeddings <- text_embedding_model$embed(
        raw_text = example_data$text[1:10],
        doc_id = example_data$id[1:10]
      )

      # Save Model
      expect_no_error(save_to_disk(
        object = text_embedding_model,
        dir_path = root_path_results,
        folder_name = folder_name
      ))
      # Load Model
      text_embedding_model_reloaded <- load_from_disk(dir_path = save_location)

      # embeddings after loading saving
      embeddings_2 <- text_embedding_model_reloaded$embed(
        raw_text = example_data$text[1:10],
        doc_id = example_data$id[1:10]
      )
      # compare embeddings
      i <- sample(x = seq.int(from = 1, to = embeddings$n_rows()), size = 1)
      expect_equal(embeddings$embeddings[i, , , drop = FALSE],
        embeddings_2$embeddings[i, , , drop = FALSE],
        tolerance = 1e-6
      )

      # Check loaded history
      expect_s3_class(text_embedding_model_reloaded,
        class = "TextEmbeddingModel"
      )
      history <- text_embedding_model_reloaded$BaseModel$last_training$history
      expect_equal(nrow(history), 2)
      expect_equal(ncol(history), 3)
      expect_true("epoch" %in% colnames(history))
      expect_true("loss" %in% colnames(history))
      expect_true("val_loss" %in% colnames(history))

      # Check loaded sustainability data
      sustain_data <- text_embedding_model_reloaded$BaseModel$get_sustainability_data()

      # One row for creation and one row for training
      expect_equal(
        text_embedding_model$BaseModel$get_sustainability_data(),
        text_embedding_model_reloaded$BaseModel$get_sustainability_data()
      )

      # Check tokenizer statistics
      expect_equal(
        text_embedding_model$BaseModel$Tokenizer$get_tokenizer_statistics(),
        text_embedding_model_reloaded$BaseModel$Tokenizer$get_tokenizer_statistics()
      )

      # Clean Directory
      unlink(
        x = save_location,
        recursive = TRUE
      )
    })

    # Documentation----------------------------------------------------------
    # Description
    test_that(paste(base_model_type, get_current_args_for_print(config), "description"), {
      text_embedding_model$set_model_description(
        eng = "Description",
        native = "Beschreibung",
        abstract_eng = "Abstract",
        abstract_native = "Zusammenfassung",
        keywords_eng = c("Test", "Neural Net"),
        keywords_native = c("Test", "Neuronales Netz")
      )
      desc <- text_embedding_model$get_model_description()
      expect_equal(
        object = desc$eng,
        expected = "Description"
      )
      expect_equal(
        object = desc$native,
        expected = "Beschreibung"
      )
      expect_equal(
        object = desc$abstract_eng,
        expected = "Abstract"
      )
      expect_equal(
        object = desc$abstract_native,
        expected = "Zusammenfassung"
      )
      expect_equal(
        object = desc$keywords_eng,
        expected = c("Test", "Neural Net")
      )
      expect_equal(
        object = desc$keywords_native,
        expected = c("Test", "Neuronales Netz")
      )
    })

    # Model License
    test_that(paste(base_model_type, get_current_args_for_print(config), "software license"), {
      text_embedding_model$set_model_license("test_license")
      expect_equal(
        object = text_embedding_model$get_model_license(),
        expected = "test_license"
      )
    })

    # Documentation License
    test_that(paste(base_model_type, get_current_args_for_print(config), "documentation license"), {
      text_embedding_model$set_documentation_license("test_license")
      expect_equal(
        object = text_embedding_model$get_documentation_license(),
        expected = "test_license"
      )
    })

    # Publication information
    test_that(paste(base_model_type, get_current_args_for_print(config), "Publication information"), {
      text_embedding_model$set_publication_info(
        type = "developer",
        authors = personList(
          person(given = "Max", family = "Mustermann")
        ),
        citation = "Test Classifier",
        url = "https://Test.html"
      )

      text_embedding_model$set_publication_info(
        type = "modifier",
        authors = personList(
          person(given = "Nico", family = "Meyer")
        ),
        citation = "Test Classifier Revisited",
        url = "https://Test_revisited.html"
      )


      pub_info <- text_embedding_model$get_publication_info()

      expect_equal(
        object = pub_info$developed_by$authors,
        expected = personList(
          person(given = "Max", family = "Mustermann")
        )
      )
      expect_equal(
        object = pub_info$developed_by$citation,
        expected = "Test Classifier"
      )
      expect_equal(
        object = pub_info$developed_by$url,
        expected = "https://Test.html"
      )

      expect_equal(
        object = pub_info$modified_by$authors,
        expected = personList(
          person(given = "Nico", family = "Meyer")
        )
      )
      expect_equal(
        object = pub_info$modified_by$citation,
        expected = "Test Classifier Revisited"
      )
      expect_equal(
        object = pub_info$modified_by$url,
        expected = "https://Test_revisited.html"
      )
    })

    test_that(paste(base_model_type, get_current_args_for_print(config), "Model information"), {
      # Method get_model_info
      model_info <- text_embedding_model$get_model_info()
      expect_equal(model_info$model_license, "test_license")
      expect_equal(model_info$model_name, paste0(base_model_type, "_embedding"))
      expect_equal(model_info$model_label, paste0("Text Embedding via", base_model_type))
      expect_equal(model_info$model_language, "english")
    })
  }
}

# Monitor test time
monitor_test_time_on_CI(
  start_time = test_time_start,
  test_name = "03_06_TextEmbeddingModels"
)
