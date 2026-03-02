testthat::skip_on_cran()
testthat::skip_if_not(
  condition = check_aif_py_modules(trace = FALSE),
  message = "Necessary python modules not available"
)
load_all_py_scripts()

# Start time
test_time_start <- Sys.time()

# SetUp-------------------------------------------------------------------------
# Set paths
root_path_data <- testthat::test_path("test_data/FeatureExtractor")
root_path_general_data <- testthat::test_path("test_data/Embeddings")
create_dir(testthat::test_path("test_artefacts"), FALSE)
root_path_results <- testthat::test_path("test_artefacts/FeatureExtractor")
create_dir(root_path_results, FALSE)

# SetUp datasets
# Disable tqdm progressbar
transformers$logging$disable_progress_bar()
datasets$disable_progress_bars()

# load data for test
# object is imdb_embeddings
imdb_embeddings <- load_from_disk(paste0(root_path_general_data, "/imdb_embeddings"))

dataset_list <- list(
  "EmbeddedText" = imdb_embeddings,
  "LargeDataSetForTextEmbeddings" = imdb_embeddings$convert_to_LargeDataSetForTextEmbeddings()
)

# config
ml_frameworks <- c("pytorch")

method_list <- list(
  "pytorch" = c("LSTM", "Dense")
)

# Start tests--------------------------------------------------------------------
for (framework in ml_frameworks) {
  for (data_type in names(dataset_list)) {
    for (method in method_list[[framework]]) {
      # Create----------------------------------------------------------------
      extractor <- TEFeatureExtractor$new()
      extractor$configure(
        name = "Test_extractor",
        label = "Test Extractor",
        text_embeddings = dataset_list[[data_type]],
        features = 128,
        method = method,
        orthogonal_method = "matrix_exp",
        noise_factor = 0.2
      )

      # Train-----------------------------------------------------------------
      test_that(paste(framework, method, data_type, "train without log"), {
        expect_no_error(
          suppressMessages(
            extractor$train(
              data_embeddings = dataset_list[[data_type]],
              data_val_size = 0.25,
              sustain_track = TRUE,
              sustain_iso_code = "DEU",
              sustain_region = NULL,
              sustain_interval = 15,
              sustain_log_level = "error",
              epochs = 2,
              batch_size = 100,
              optimizer = "Adam",
              trace = random_bool_on_CI(),
              ml_trace = 0
            )
          )
        )

        # Check if sustainability data has been tracked
        expect_true(extractor$get_sustainability_data()$sustainability_tracked)
      })
      gc()

      test_that(paste(framework, method, data_type, "train with log"), {
        train_path <- paste0(root_path_results, "/", "train_", generate_id())
        create_dir(train_path, FALSE)
        expect_no_error(
          suppressMessages(
            extractor$train(
              data_embeddings = dataset_list[[data_type]],
              data_val_size = 0.25,
              sustain_track = TRUE,
              sustain_iso_code = "DEU",
              sustain_region = NULL,
              sustain_interval = 15,
              sustain_log_level = "error",
              epochs = 2,
              batch_size = 100,
              log_dir = train_path,
              trace = random_bool_on_CI(),
              ml_trace = 0
            )
          )
        )

        state_log_exists <- file.exists(paste0(train_path, "/aifeducation_state.log"))
        expect_true(state_log_exists)
        if (state_log_exists) {
          log_state <- read.csv(paste0(train_path, "/aifeducation_state.log"))
          expect_equal(nrow(log_state), 3)
          expect_equal(ncol(log_state), 3)
          expect_equal(colnames(log_state), c("value", "total", "message"))
        }

        loss_log_exists <- file.exists(paste0(train_path, "/aifeducation_loss.log"))
        expect_true(loss_log_exists)
        if (loss_log_exists == TRUE) {
          log_loss <- read.csv(paste0(train_path, "/aifeducation_loss.log"), header = FALSE)
          expect_gte(ncol(log_loss), 2)
          expect_gte(nrow(log_loss), 2)
        }

        # Clean Directory
        unlink(
          x = train_path,
          recursive = TRUE
        )
      })

      # Predict---------------------------------------------------------------
      test_that(paste(framework, method, data_type, "predict - basic"), {
        if (data_type == "EmbeddedText") {
          predictions <- extractor$extract_features(
            data_embeddings = dataset_list[[data_type]],
            batch_size = 50
          )
        } else {
          predictions <- extractor$extract_features_large(
            data_embeddings = dataset_list[[data_type]],
            batch_size = 50
          )
        }
        expect_equal(predictions$get_features(), 128)
        expect_equal(dataset_list[[data_type]]$n_rows(), predictions$n_rows())
        expect_true(predictions$is_compressed())
      })

      test_that(paste(framework, method, data_type, "predict - randomness"), {
        if (data_type == "EmbeddedText") {
          predictions <- extractor$extract_features(
            data_embeddings = dataset_list[[data_type]],
            batch_size = 50
          )
          predictions_2 <- extractor$extract_features(
            data_embeddings = dataset_list[[data_type]],
            batch_size = 50
          )
          expect_equal(predictions, predictions_2)
        } else {
          predictions <- extractor$extract_features_large(
            data_embeddings = dataset_list[[data_type]],
            batch_size = 50
          )
          predictions_2 <- extractor$extract_features_large(
            data_embeddings = dataset_list[[data_type]],
            batch_size = 50
          )

          i <- sample(seq.int(from = 1, to = predictions$n_rows()))
          expect_equal(
            predictions$extract_column("input")[i,,,drop=FALSE],
            predictions_2$extract_column("input")[i,,,drop=FALSE]
          )
        }
      })

      test_that(paste(framework, method, data_type, "predict - order invariance"), {
        embeddings_ET_perm <- dataset_list[["EmbeddedText"]]$clone(deep = TRUE)
        perm <- sample(x = seq.int(from = 1, to = nrow(embeddings_ET_perm$embeddings)), replace = FALSE)
        embeddings_ET_perm$embeddings <- embeddings_ET_perm$embeddings[perm, , , drop = FALSE]

        if (data_type == "EmbeddedText") {
          predictions <- extractor$extract_features(
            data_embeddings = dataset_list[[data_type]],
            batch_size = 50
          )
          predictions_Perm <- extractor$extract_features(
            data_embeddings = embeddings_ET_perm,
            batch_size = 50
          )
          i <- sample(seq.int(from = 1, to = predictions$n_rows()), size = 1)

          expect_equal(
            predictions$embeddings[i, , , drop = FALSE],
            predictions_Perm$embeddings[which(perm == i), , , drop = FALSE]
          )
        } else {
          predictions <- extractor$extract_features_large(
            data_embeddings = dataset_list[[data_type]],
            batch_size = 50
          )
          predictions_Perm <- extractor$extract_features_large(
            data_embeddings = embeddings_ET_perm$convert_to_LargeDataSetForTextEmbeddings(),
            batch_size = 50
          )
          i <- sample(seq.int(from = 1, to = predictions$n_rows()), size = 1)
          expect_equal(
            predictions$extract_column("input")[i,,,drop=FALSE],
            predictions_Perm$extract_column("input")[which(perm == i),,,drop=FALSE]
          )
        }
      })

      if (data_type == "EmbeddedText") {
        test_that(paste(framework, method, "predict - data source invariance"), {
          predictions_ET <- extractor$extract_features(
            data_embeddings = dataset_list[["EmbeddedText"]],
            batch_size = 50
          )
          predictions_LD <- extractor$extract_features_large(
            data_embeddings = dataset_list[["LargeDataSetForTextEmbeddings"]],
            batch_size = 50
          )
          i <- sample(seq.int(from = 1, to = predictions_ET$n_rows()), size = 1)
          expect_equal(
            unname(predictions_ET$embeddings[i, , , drop = FALSE]),
            predictions_LD$extract_column("input")[i,,,drop=FALSE],
            tolerance = 1e-7
          )
        })
      }
      gc()

      # Function for loading and saving models-----------------------------------
      test_that(paste(framework, method, data_type, "function save and load"), {
        # Predictions before saving and loading
        if (data_type == "EmbeddedText") {
          predictions <- extractor$extract_features(
            data_embeddings = dataset_list[[data_type]],
            batch_size = 50
          )
        } else {
          predictions <- extractor$extract_features_large(
            data_embeddings = dataset_list[[data_type]],
            batch_size = 50,
            trace = FALSE
          )
        }

        # Save and load
        folder_name <- paste0("function_save_load_", generate_id())
        dir_path <- paste0(root_path_results, "/", folder_name)
        save_to_disk(
          object = extractor,
          dir_path = root_path_results,
          folder_name = folder_name
        )
        extractor2 <- NULL
        extractor2 <- load_from_disk(dir_path = dir_path)

        # Is config equal after loading
        expect_equal(
          extractor$get_model_config(),
          extractor2$get_model_config()
        )

        # Predict after loading
        if (data_type == "EmbeddedText") {
          predictions_2 <- extractor2$extract_features(
            data_embeddings = dataset_list[[data_type]],
            batch_size = 50
          )
        } else {
          predictions_2 <- extractor2$extract_features_large(
            data_embeddings = dataset_list[[data_type]],
            batch_size = 50,
            trace = FALSE
          )
        }

        # Compare predictions
        i <- sample(x = seq.int(from = 1, to = predictions$n_rows()), size = 1)
        expect_equal(predictions$embeddings[i, , , drop = FALSE],
          predictions_2$embeddings[i, , , drop = FALSE],
          tolerance = 1e-6
        )

        # Clean Directory
        unlink(
          x = dir_path,
          recursive = TRUE
        )
      })
      gc()
    }
  }
}

# Clean Directory
if (dir.exists(root_path_results)) {
  unlink(
    x = root_path_results,
    recursive = TRUE
  )
}

# Monitor test time
monitor_test_time_on_CI(
  start_time = test_time_start,
  test_name = "03_05_FeatureExtractor"
)
