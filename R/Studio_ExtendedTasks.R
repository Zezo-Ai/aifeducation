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

# LargeDataSetForText ------------------------------------------------------------
long_add_texts_to_dataset <- function(source_path,
                                      destination_path,
                                      destination_folder,
                                      log_path,
                                      clean_text,
                                      include_txt,
                                      include_pdf,
                                      include_xlsx,
                                      excel_id_column,
                                      excel_text_column,
                                      excel_license_column,
                                      excel_bib_entry_column,
                                      excel_url_license_column,
                                      excel_text_license_column,
                                      excel_url_source_column,
                                      log_write_interval = 2,
                                      py_environment_type,
                                      py_env_name) {
  promises::future_promise({
    requireNamespace("aifeducation")
    # Set up py env
    prepare_session(
      env_type = py_environment_type,
      envname = py_env_name,
      check_session = FALSE
    )

    # Set up top level progress monitoring
    top_total <- include_txt + include_pdf + include_xlsx
    top_value <- 0
    total_message <- "File types"

    # Create new data set
    new_dataset <- LargeDataSetForText$new()

    # Start processing different file types
    if (include_txt) {
      top_value <- top_value + 1

      new_dataset$add_from_files_txt(
        dir_path = source_path,
        batch_size = 2,
        log_file = log_path,
        log_top_value = top_value,
        log_top_total = top_total,
        log_top_message = total_message,
        log_write_interval = log_write_interval,
        clean_text = clean_text,
        trace = FALSE
      )
    }

    if (include_pdf) {
      top_value <- top_value + 1

      new_dataset$add_from_files_pdf(
        dir_path = source_path,
        batch_size = 2,
        log_file = log_path,
        log_top_value = top_value,
        log_top_total = top_total,
        log_top_message = total_message,
        log_write_interval = log_write_interval,
        clean_text = clean_text,
        trace = FALSE
      )
    }

    if (include_xlsx) {
      top_value <- top_value + 1

      new_dataset$add_from_files_xlsx(
        dir_path = source_path,
        trace = FALSE,
        id_column = excel_id_column,
        text_column = excel_text_column,
        license_column = excel_license_column,
        bib_entry_column = excel_bib_entry_column,
        url_license_column = excel_url_license_column,
        text_license_column = excel_text_license_column,
        url_source_column = excel_url_source_column,
        log_file = log_path,
        log_top_value = top_value,
        log_top_total = top_total,
        log_top_message = total_message,
        log_write_interval = log_write_interval
      )
    }

    # Save
    save_to_disk(
      object = new_dataset,
      dir_path = destination_path,
      folder_name = destination_folder
    )

    # Returns number of documents added to the data set
    return(new_dataset$n_rows())
  })
}

# TextEmbeddingModel --------------------------------------------------------------
long_transform_text_to_embeddings <- function(source_path,
                                              destination_path,
                                              destination_folder,
                                              log_path,
                                              batch_size,
                                              model_path,
                                              log_write_interval = 2,
                                              current_conda_env,
                                              py_environment_type,
                                              py_env_name) {
  promises::future_promise({
    # Set up py env
    prepare_session(
      env_type = py_environment_type,
      envname = py_env_name,
      check_session = FALSE
    )

    # Read the large data set for raw texts
    raw_texts <- load_from_disk(source_path)

    # Set up top level progress monitoring
    # top_total <- raw_texts$n_rows()
    # top_value <- 0
    # total_message <- "Documents"

    # Load the model
    model <- load_from_disk(model_path)

    # Start embedding
    embeddings <- model$embed_large(
      large_datas_set = raw_texts,
      batch_size = batch_size,
      trace = FALSE,
      log_file = log_path,
      log_write_interval = log_write_interval
    )

    # Save
    save_to_disk(
      object = embeddings,
      dir_path = destination_path,
      folder_name = destination_folder
    )

    # Returns number of documents that are embedded
    return(embeddings$n_rows())
  })
}

# Classifiers ==================================================================

long_models <- function(args) {
  promises::future_promise({
    # Set up py env
    prepare_session(
      env_type = args[[1]]$meta_args$py_environment_type,
      envname = args[[1]]$meta_args$py_env_name,
      check_session = FALSE
    )

    # Create object or load object
    if (args[[1]]$meta_args$object_class %in% BaseModelsIndex) {
      object <- load_from_disk(args[[1]]$path_args$path_to_base_model)
    } else {
      object <- create_object(args[[1]]$meta_args$object_class)
    }

    requested_methods <- names(args)
    for (method in requested_methods) {
      # add missing objects to arguments by loading them
      args[[method]]$args <- add_missing_args(
        args = args[[method]]$args,
        path_args = args[[method]]$path_args,
        meta_args = args[[method]]$meta_args
      )
      # Call Method
      do.call(what = object[[method]], args = args[[method]]$args)
    }

    # Create dir for saving the object
    dir_destination <- paste0(
      args$configure$path_args$destination_path, "/",
      args$configure$path_args$folder_name
    )
    create_dir(dir_destination, FALSE)

    # Save
    save_to_disk(
      object = object,
      dir_path = args[[1]]$path_args$destination_path,
      folder_name = args[[1]]$path_args$folder_name
    )

    # Returns message
    return("Model trained.")
  })
}

# Transformers =====================================================================

long_transformers <- function(args) {
  promises::future_promise({
    # Set up py env
    prepare_session(
      env_type = args[[1]]$meta_args$py_environment_type,
      envname = args[[1]]$meta_args$py_env_name,
      check_session = FALSE
    )

    object_list <- list()

    for (task in names(args)) {
      current_task <- args[[task]]

      # Create or select the relevant object
      if (current_task$meta_args$object_class %in% names(object_list)) {
        tmp_object <- object_list[[current_task$meta_args$object_class]]
      } else {
        tmp_object <- create_object(current_task$meta_args$object_class)
        object_list[current_task$meta_args$object_class] <- list(tmp_object)
      }

      # Request the method to apply
      method <- current_task$meta_args$method

      if (inherits(tmp_object, "BaseModelCore")) {
        current_task$args$tokenizer <- object_list[[1]]
      }

      # add missing objects to arguments by loading them
      current_task$args <- add_missing_args(
        args = current_task$args,
        path_args = current_task$path_args,
        meta_args = current_task$meta_args
      )

      # Call Method
      do.call(what = tmp_object[[method]], args = current_task$args)
    }

    # Create dir for saving the object
    dir_destination <- paste0(
      args$configure$path_args$destination_path, "/",
      args$configure$path_args$folder_name
    )
    create_dir(dir_destination, FALSE)

    # Save the last used object
    save_to_disk(
      object = tmp_object,
      dir_path = args[[1]]$path_args$destination_path,
      folder_name = args[[1]]$path_args$folder_name
    )

    # Returns message
    return("Transformer created.")
  })
}
