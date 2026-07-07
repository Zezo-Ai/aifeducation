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

#' @title Saving objects created with 'aifeducation'
#' @description Function for saving objects created with 'aifeducation'.
#'
#' @param object Object of class [TEClassifierRegular], [TEClassifierProtoNet],  [TEFeatureExtractor],
#'   [TextEmbeddingModel], [LargeDataSetForTextEmbeddings], [LargeDataSetForText] or [EmbeddedText] which should be
#'   saved.
#' @param dir_path `string` Path to the directory where the should model is stored.
#' @param folder_name `string` Name of the folder where the files should be stored.
#' @return Function does not return a value. It saves the model to disk.
#' @return No return value, called for side effects.
#'
#' @family Saving and Loading
#'
#' @export
save_to_disk <- function(object,
                         dir_path,
                         folder_name) {
  # Check class of object
  check_class(
    object = object,
    object_name = "object",
    classes = c(
      TEClassifiers_class_names,
      "TEFeatureExtractor",
      "TextEmbeddingModel",
      "LargeDataSetForTextEmbeddings",
      "LargeDataSetForText",
      "EmbeddedText",
      ".AIFEBaseTransformer",
      "BaseModelCore",
      "TokenizerBase"
    ),
    FALSE
  )
  check_type(object = dir_path, object_name = "dir_path", type = "string", FALSE)
  check_type(object = folder_name, object_name = "folder_name", type = "string", FALSE)

  # Create path to save location
  save_location <- file.path(dir_path, folder_name)

  # Create path to r_interface
  path_r_config_state <- file.path(save_location, "r_config_state.rda")

  # Check directory
  create_dir(dir_path, FALSE)
  create_dir(save_location, FALSE)

  if (!inherits(object, ".AIFEBaseTransformer")) {
    # Create config and save to disk
    config_file <- create_config_state(object)
    save(config_file, file = path_r_config_state)

    # Save Python objects and additional files
    object$save(
      dir_path = dir_path,
      folder_name = folder_name
    )
  } else {}
}


#' @title Loading objects created with 'aifeducation'
#' @description Function for loading objects created with 'aifeducation'.
#'
#' @param dir_path `string` Path to the directory where the model is stored.
#' @return Returns an object of class [TEClassifierRegular], [TEClassifierProtoNet],  [TEFeatureExtractor],
#'   [TextEmbeddingModel], [LargeDataSetForTextEmbeddings], [LargeDataSetForText] or [EmbeddedText].
#'
#' @family Saving and Loading
#'
#' @export
load_from_disk <- function(dir_path) {
  # Case for all native ai for education models
  if (file.exists(file.path(dir_path, "r_config_state.rda"))) {
    # load config
    loaded_config <- load_R_config_state(dir_path)

    # Create object
    model <- create_object(loaded_config$class)

    # load and update model
    model$load_from_disk(dir_path = dir_path)
    return(model)
  } else {
    stop("There is no compatible model at", dir_path)
  }
}


load_R_config_state <- function(dir_path) {
  # Load the Interface to R
  interface_path <- file.path(dir_path, "r_config_state.rda")

  # Check for r_config_state.rda
  if (!file.exists(interface_path)) {
    stop(
      "There is no file r_config_state.rda in the selected directory ",
      "The directory is: ", dir_path
    )
  }

  # Load interface
  name_interface <- load(interface_path)
  loaded_object <- get(x = name_interface)
  return(loaded_object)
}

#' Create config for R interfaces
#'
#' Function creates a config that can be saved to disk. It is used during loading
#' an object from disk in order to set the correct configuration.
#'
#' @param object Object of class `"TEClassifierRegular"`, `"TEClassifierProtoNet"`,
#' `"TEFeatureExtractor"`, `"TextEmbeddingModel"`, `"LargeDataSetForTextEmbeddings"`,
#' `"LargeDataSetForText"`, `"EmbeddedText"`.
#'
#' @return Returns a `list` that contains the class of the object, the public, and
#' private fields.
#'
#' @family Utils
#' @keywords internal
create_config_state <- function(object) {
  config <- object$get_all_fields()
  config["class"] <- class(object)[1L]

  # Remove embeddings to avoid duplicate data storage
  if (config["class"] == "EmbeddedText") {
    config$public$embeddings <- NA
  }

  return(config)
}
