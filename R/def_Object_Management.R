# ===================DataSets===================================================
#' @title List of all available types of data sets
#' @description Named `list` containing all available types of data sets as a string.
#' @family Parameter Dictionary
#' @export
DataSetsIndex <- list()

# ===================Tokenizer==================================================
#' @title List of all available Tokenizers
#' @description Named `list` containing all tokenizers as a string.
#' @family Parameter Dictionary
#' @export
TokenizerIndex <- list()

# ===================BaseModels=================================================
#' @title List of all available BaseModels
#' @description Named `list` containing all BaseModels as a string.
#' @family Parameter Dictionary
#' @export
BaseModelsIndex <- list()

# ===================Classifiers================================================

#' @title Class names of all classifier models based on text embeddings
#' @description `vector` containing all class names as a string.
#' @family Parameter Dictionary
#' @keywords internal
#' @noRd
TEClassifiers_class_names <- vector()

#' @title Get names of classifiers
#' @description Function returns the names of all classifiers which
#' are child classes of a specific super class.
#' @param super_class `string` Name of the super class the classifiers should
#' be a child of. To request the names of all classifiers set this argument to `NULL`.
#' @return Returns a vector containing the names of the classifiers.
#' @family Parameter Dictionary
#' @export
get_TEClassifiers_class_names <- function(super_class = NULL) {
  if (is.null(super_class)) {
    return(TEClassifiers_class_names)
  } else {
    class_names <- NULL
    for (class in TEClassifiers_class_names) {
      object <- create_object(class)
      if (inherits(object, super_class)) {
        class_names <- append(
          x = class_names,
          values = class
        )
      }
    }
    return(class_names)
  }
}

# ===================TextEmbedding===============================================
TextEmbeddingObjectsIndex <- list()


#' @title Names of all deprecated objects
#'
#' @description `vector` containing all class names as a string.
#'
#' @family Parameter Dictionary
#' @keywords internal
#' @noRd
DeprecatedObjects <- c("TEClassifierProtoNet", "TEClassifierRegular")


#' @title Get names of deprecated objects
#' @description Function returns the names of all objects that are deprecated.
#' @return Returns a `vector` containing the names.
#' @family Parameter Dictionary
#' @export
get_depr_obj_names <- function() {
  return(DeprecatedObjects)
}


#' @title Create object#'
#' @description  Support function for creating objects.
#' @param class `string` Name of the class to be created.
#' @return Returns an object of the requested class.
#' @family Utils Developers
#' @export
create_object <- function(class) {
  # Create list of all objects on user level

  base_model <- get_entry_from_BaseModelsIndex("class_name")
  names(base_model) <- get_entry_from_BaseModelsIndex("model_type")

  object_list <- c(
    DataSetsIndex,
    TokenizerIndex,
    base_model,
    TextEmbeddingObjectsIndex,
    TEClassifiers_class_names
  )

  if (class == "TEClassifierRegular") {
    return(suppressMessages(TEClassifierRegular$new()))
  } else if (class == "TEClassifierProtoNet") {
    return(suppressMessages(TEClassifierProtoNet$new()))
  } else if (class %in% object_list) {
    object <- eval(str2expression(class))
    return(object$new())
  } else if (class %in% tolower(names(object_list))) {
    index <- which(x = (class == tolower(names(object_list))))
    object <- eval(str2expression(object_list[[index]]))
    return(object$new())
  } else {
    stop("Object ", class, " is not implemented in this function.")
  }
}

# ===================BaseModels===============================================

#' @title Get entries from BaseModelsIndex
#' @description Returns the values for all BaseModels in the specific entry.
#' @param entry `string` Name of the entry for which the values should be returned.
#' @return Returns a `vector`.
#' @family Utils Developers
#' @keywords internal
#' @noRd
get_entry_from_BaseModelsIndex <- function(entry = "class_name") {
  results <- vector(length = length(BaseModelsIndex))
  for (i in seq_along(BaseModelsIndex)) {
    results[i] <- BaseModelsIndex[[i]][[entry]]
  }
  return(results)
}
