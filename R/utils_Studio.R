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

#' @title Generate sidebar
#' @description Function for generating a sidebar containing information on a model.
#'
#' @param model Model for which the sidebar should be generated.
#'
#' @return Returns a `shiny::tagList` containing the html elements for the user interface. The content of the list
#'   depends on the kind of model passed to this function.
#'
#' @family Utils Studio Developers
#' @keywords internal
#' @noRd
#'
generate_sidebar_information <- function(model) {
  ui <- shiny::tagList()

  if (inherits(model, "TextEmbeddingModel")) {
    # Prepare output
    if (is.null(model)) {
      model_label <- NULL
    } else {
      model_label <- model$get_model_info()$model_label
    }

    max_tokens <- (model$get_model_config()$max_length - model$get_model_config()$overlap) *
      model$get_model_config()$chunks + model$get_model_config()$max_length

    if (!is.null(model$get_model_config()$aggregation)) {
      aggegation <- shiny::tags$p("Hidden States Aggregation: ", model$get_model_config()$aggregation)
    } else {
      aggegation <- NULL
    }

    if (!is.null(model$get_model_config()$emb_pool_type)) {
      pool_type <- model$get_model_config()$emb_pool_type
      min_layer <- model$get_model_config()$emb_layer_min
      max_layer <- model$get_model_config()$emb_layer_max
    } else {
      pool_type <- NULL
      min_layer <- NULL
      max_layer <- NULL
    }

    if (methods::isClass(Class = "data.frame", where = model$BaseModel$get_sustainability_data())) {
      if (nrow(model$BaseModel$get_sustainability_data()) != 0) {
        kwh <- round(sum(model$BaseModel$get_sustainability_data()[, "sustainability_data.total_energy_kwh"]), 3)
      } else {
        kwh <- "not estimated"
      }
    } else {
      kwh <- "not estimated"
    }

    if (methods::isClass(Class = "data.frame", where = model$BaseModel$get_sustainability_data())) {
      if (nrow(model$BaseModel$get_sustainability_data()) != 0) {
        co2 <- round(sum(model$BaseModel$get_sustainability_data()[, "sustainability_data.co2eq_kg"]), 3)
      } else {
        co2 <- "not estimated"
      }
    } else {
      co2 <- "not estimated"
    }

    ui <- shiny::tagList(
      shiny::tags$p(shiny::tags$b("Model:")),
      shiny::tags$p(model_label),
      shiny::tags$hr(),
      shiny::tags$p("# Parameter: ", model$BaseModel$count_parameter()),
      shiny::tags$p("Method: ", model$BaseModel$get_model_type()),
      aggegation,
      shiny::tags$p("Max Tokens per Chunk: ", model$get_model_config()$max_length),
      shiny::tags$p("Max Chunks: ", model$get_model_config()$chunks),
      shiny::tags$p("Token Overlap: ", model$get_model_config()$overlap),
      shiny::tags$p("Max Tokens: ", max_tokens),
      shiny::tags$p("Pool Type: ", pool_type),
      shiny::tags$p("Embedding Layers - Min: ", min_layer),
      shiny::tags$p("Embedding Layers - Max: ", max_layer),
      shiny::tags$hr(),
      shiny::tags$p("Energy Consumption (kWh): ", kwh),
      shiny::tags$p("Carbon Footprint (CO2eq. kg): ", co2)
    )
  } else if (inherits(model, "BaseModelCore")) {
    # Prepare output
    if (is.null(model)) {
      model_label <- NULL
    } else {
      model_label <- model$get_model_info()$model_label
    }

    if (methods::isClass(Class = "data.frame", where = model$get_sustainability_data())) {
      if (nrow(model$get_sustainability_data()) != 0) {
        kwh <- round(sum(model$get_sustainability_data()[, "sustainability_data.total_energy_kwh"]), 3)
      } else {
        kwh <- "not estimated"
      }
    } else {
      kwh <- "not estimated"
    }

    if (methods::isClass(Class = "data.frame", where = model$get_sustainability_data())) {
      if (nrow(model$get_sustainability_data()) != 0) {
        co2 <- round(sum(model$get_sustainability_data()[, "sustainability_data.co2eq_kg"]), 3)
      } else {
        co2 <- "not estimated"
      }
    } else {
      co2 <- "not estimated"
    }

    ui <- shiny::tagList(
      shiny::tags$p(shiny::tags$b("Model:")),
      shiny::tags$p(model_label),
      shiny::tags$hr(),
      shiny::tags$p("# Parameter: ", model$count_parameter()),
      shiny::tags$p("Method: ", model$get_model_type()),
      shiny::tags$hr(),
      shiny::tags$p("Energy Consumption (kWh): ", kwh),
      shiny::tags$p("Carbon Footprint (CO2eq. kg): ", co2)
    )
  } else if (inherits(model, "ClassifiersBasedOnTextEmbeddings")) {
    if (is.null(model)) {
      model_label <- NULL
    } else {
      model_label <- model$get_model_info()$model_label
    }

    if (nrow(model$get_sustainability_data()) > 0) {
      kwh <- round(sum(model$get_sustainability_data()$sustainability_data.total_energy_kwh), 3)
      co2 <- round(sum(model$get_sustainability_data()$sustainability_data.co2eq_kg), 3)
    } else {
      kwh <- "not estimated"
      co2 <- "not estimated"
    }

    ui <- shiny::tagList(
      shiny::tags$p(shiny::tags$b("Model:")),
      shiny::tags$p(model_label),
      shiny::tags$hr(),
      shiny::tags$p("# Parameter: ", model$count_parameter()),
      shiny::tags$p("Synthetic Cases: ", model$last_training$config$use_sc),
      shiny::tags$p("Pseudo Labeling: ", model$last_training$config$use_pl),
      shiny::tags$hr(),
      shiny::tags$p("Energy Consumption (kWh): "),
      shiny::tags$p(kwh),
      shiny::tags$p("Carbon Footprint (CO2eq. kg): "),
      shiny::tags$p(co2)
    )
  } else if (inherits(model, "TEFeatureExtractor")) {
    if (!is.null(model)) {
      if (nrow(model$get_sustainability_data()) > 0) {
        kwh <- round(model$get_sustainability_data()$sustainability_data.total_energy_kwh, 3)
        co2 <- round(model$get_sustainability_data()$sustainability_data.co2eq_kg, 3)
      } else {
        kwh <- "not estimated"
        co2 <- "not estimated"
      }

      ui <- shiny::tagList(
        shiny::tags$p(shiny::tags$b("Model:")),
        shiny::tags$p(model$get_model_info()$model_label),
        shiny::tags$hr(),
        shiny::tags$p("# Parameter: ", model$count_parameter()),
        shiny::tags$hr(),
        shiny::tags$p("Target Features: ", model$model_config$features),
        shiny::tags$p("Method: ", model$model_config$method),
        shiny::tags$p("Noise Factor: ", model$model_config$noise_factor),
        shiny::tags$p("Optimizer: ", model$model_config$optimizer),
        shiny::tags$hr(),
        shiny::tags$p("Energy Consumption (kWh): "),
        shiny::tags$p(kwh),
        shiny::tags$p("Carbon Footprint (CO2eq. kg): "),
        shiny::tags$p(co2)
      )
    }
  }
  return(ui)
}


#' @title Generate model description
#' @description Function for generating the html elements describing a model.
#'
#' @param model Model for which the description should be generated.
#' @param eng `bool` If `TRUE` the generation assumes the description to be in English. If `FALSE` it assumes native
#'   language.
#'
#' @return Returns a `shiny::tagList` containing the html elements for the user interface. The content of the list
#'   depends on the kind of model passed to this function. If the `model` is `NULL` function returns `NULL`.
#'
#' @family Utils Studio Developers
#' @keywords internal
#' @noRd
#'
generate_model_description <- function(model, eng) {
  if (!is.null(model)) {
    if (eng == TRUE) {
      ui <- shiny::tagList(
        shiny::tags$h3("Abstract"),
        if (!is.null(model$get_model_description()$abstract_eng)) {
          shiny::tags$p(shiny::includeMarkdown(model$get_model_description()$abstract_eng))
        },
        shiny::tags$h3("Description"),
        if (!is.null(model$get_model_description()$eng)) {
          shiny::tags$p(shiny::includeMarkdown(model$get_model_description()$eng))
        }
      )
    } else {
      ui <- shiny::tagList(
        shiny::tags$h3("Abstract"),
        if (!is.null(model$get_model_description()$abstract_native)) {
          shiny::tags$p(shiny::includeMarkdown(model$get_model_description()$abstract_native))
        },
        shiny::tags$h3("Description"),
        if (!is.null(model$get_model_description()$native)) {
          shiny::tags$p(shiny::includeMarkdown(model$get_model_description()$native))
        }
      )
    }
    return(ui)
  } else {
    return(NULL)
  }
}

#' @title Generate description of bibliographic information
#' @description Function for generating the html elements reporting a model's bibliographic information.
#'
#' @param model Model for which the description should be generated.
#'
#' @return Returns a `shiny::tagList` containing the html elements for the user interface.
#'
#' @family Utils Studio Developers
#' @keywords internal
#' @noRd
#'
generate_model_bib_description <- function(model) {
  pub_info <- model$get_publication_info()
  ui <- shiny::tagList(
    if (!is.null(pub_info$developed_by$authors)) {
      shiny::tags$p("Developers: ", paste(
        format(
          x = pub_info$developed_by$authors,
          include = c("given", "family")
        ),
        collapse = ", "
      ))
    },
    if (!is.null(pub_info$developed_by$citation)) {
      shiny::tags$p("Citation: ", pub_info$developed_by$citation)
    },
    if (!is.null(pub_info$modifided_by$authors)) {
      shiny::tags$p("Modifiers: ", paste(
        format(
          x = pub_info$modifided_by$authors,
          include = c("given", "family")
        ),
        collapse = ", "
      ))
    },
    if (!is.null(pub_info$modifided_by$citation)) {
      shiny::tags$p("Citation: ", pub_info$modifided_by$citation)
    },
    if (!is.null(pub_info$modifided_by$citation)) {
      shiny::tags$p("Language: ", model$get_model_info()$model_language)
    },
  )

  return(ui)
}

#' @title Generate widgets for documenting the bibliographic information of a model
#' @description Function generates the input widgets for documenting the bibliographic information of  a model. This
#'   includes the names of the involved persons, mail e-mail addresses, urls, and citation.
#'
#' @param ns `function` for setting the namespace of the input elements. This should be `session$ns`.
#' @param model Model for which the description should be generated.
#' @param type `string` determining if the widgets should be generated for documenting the developers (`type =
#'   "developers"`) or the modifiers (`type = "modifiers"`).
#'
#' @return Returns a `shiny::tagList` containing the html elements for the user interface.
#'
#' @family Utils Studio Developers
#' @keywords internal
#' @noRd
#'
generate_doc_input_developers <- function(ns, model, type = "developers") {
  if (type == "developers") {
    pup_info_for <- "developed_by"
    pup_info_titles <- "Developers"
  } else if (type == "modifiers") {
    pup_info_for <- "modified_by"
    pup_info_titles <- "Modifiers"
  }

  widgets <- NULL
  for (j in 1:10) {
    pup_info <- model$get_publication_info()[[pup_info_for]]$authors
    widgets[[j]] <- list(
      shiny::fluidRow(
        shiny::column(
          width = 4,
          shiny::textInput(
            inputId = ns(paste0("doc_", pup_info_titles, "_fist_name_", j)),
            label = paste("Given Name", j),
            value = pup_info[[j]]$given,
            width = "100%"
          )
        ),
        shiny::column(
          width = 4,
          shiny::textInput(
            inputId = ns(paste0("doc_", pup_info_titles, "_last_name_", j)),
            label = paste("Family Name", j),
            value = pup_info[[j]]$family,
            width = "100%"
          )
        ),
        shiny::column(
          width = 4,
          shiny::textInput(
            inputId = ns(paste0("doc_", pup_info_titles, "_mail_", j)),
            label = paste("Mail", j),
            value = pup_info[[j]]$email,
            width = "100%"
          )
        )
      )
    )
  }

  ui <- shiny::tagList(
    shiny::tabPanel(
      title = pup_info_titles,
      shiny::textInput(
        inputId = ns(paste0("doc_", pup_info_for, "_citation")),
        label = "Citation",
        value = model$get_publication_info()[[pup_info_for]]$citation
      ),
      shiny::textInput(
        inputId = ns(paste0("doc_", pup_info_for, "_url")),
        label = "URL",
        value = model$get_publication_info()[[pup_info_for]]$url
      ),
      widgets,
      shiny::actionButton(
        inputId = ns(paste0("doc_", pup_info_for, "_save")),
        label = "Save",
        icon = shiny::icon("floppy-disk")
      )
    )
  )
  return(ui)
}

#' @title Generate widgets for documenting a model
#' @description Function generates the input widgets for documenting a model.
#'
#' @param ns `function` for setting the namespace of the input elements. This should be `session$ns`.
#' @param model Model for which the description should be generated.
#' @param language `string` determining if the documentation should be saved in English (`language = "eng"`) or in the
#'   model's native language (`language = "native"`).
#' @param type `string` determining if the input refers to the abstract (`type = "abstract"`) or the main documentation
#'   (`type = "documentation"`).
#'
#' @return Returns a `shiny::tagList` containing the html elements for the user interface.
#'
#' @family Utils Studio Developers
#' @keywords internal
#' @noRd
#'
generate_doc_input_text_editor <- function(ns, model, language = "eng", type = "abstract") {
  if (language == "eng") {
    if (type == "abstract") {
      documentation_keyword <- "keywords_eng"
      documention_part <- "abstract_eng"
      documentation_field <- "abstract_eng"
    } else {
      documention_part <- "description_eng"
      documentation_field <- "eng"
    }
  } else {
    if (type == "abstract") {
      documentation_keyword <- "keywords_native"
      documention_part <- "abstract_native"
      documentation_field <- "abstract_native"
    } else {
      documention_part <- "description_native"
      documentation_field <- "native"
    }
  }

  ui <- shiny::tagList(
    bslib::layout_column_wrap(
      bslib::card(
        bslib::card_header("Editor"),
        bslib::card_body(
          shiny::textAreaInput(
            inputId = ns(paste0("doc_editor_", documention_part)),
            label = "Editor",
            rows = 6,
            width = "100%",
            value = model$get_model_description()[[documentation_field]]
          ),
          if (type == "abstract") {
            shiny::textInput(
              inputId = ns(paste0("doc_editor_", documention_part, "_keywords")),
              value = model$get_model_description()[[documentation_keyword]],
              label = "Keywords",
              width = "100%"
            )
          },
          shiny::actionButton(
            inputId = ns(paste0("doc_editor_", documention_part, "_preview_button")),
            label = "Preview",
            icon = shiny::icon("eye")
          ),
          shiny::actionButton(
            inputId = ns(paste0("doc_editor_", documention_part, "_save_button")),
            label = "Save",
            icon = shiny::icon("floppy-disk")
          )
        )
      ),
      bslib::card(
        bslib::card_header("Preview"),
        bslib::card_body(
          shiny::uiOutput(outputId = ns(paste0("doc_editor_", documention_part, "_preview")))
        )
      )
    )
  )
  return(ui)
}

#' @title Load and check embeddings
#' @description Function for checking and loading text embeddings in AI for Education - Studio.
#'
#' @param dir_path `string` path to the directory containing the embeddings.
#'
#' @return If there are any errors an error modal is displayed by calling the function `display_errors`. If there are no
#'   errors the function returns embeddings as an object of class [LargeDataSetForTextEmbeddings] or [EmbeddedText]. In
#'   the case of erros the function returns `NULL`.
#'
#' @family Utils Studio Developers
#' @keywords internal
#' @noRd
#'
load_and_check_embeddings <- function(dir_path) {
  if (!is.null(dir_path)) {
    if (file.exists(dir_path) == TRUE) {
      display_processing(
        title = "Working. Please wait.",
        size = "l",
        easy_close = FALSE,
        message = ""
      )
      # Wait for modal
      Sys.sleep(1)
      embeddings <- load_from_disk(dir_path)
      if (
        inherits(embeddings, "EmbeddedText") ||
          inherits(embeddings, "LargeDataSetForTextEmbeddings")
      ) {
        shiny::removeModal()
        return(embeddings)
      } else {
        shiny::removeModal()
        display_errors(
          title = "Error",
          size = "l",
          easy_close = TRUE,
          error_messages = "The file contains data in an unsupported format.
              Text embeddings must be of class 'LargeDataSetForTextEmbeddings' or 'EmbeddedText'. Please
              check data. Data embeddings should always be created via data
              preparation of this user interfache or with the corresponding
              method of the TextEmbeddingModel."
        )
        rm(embeddings)
        gc()
        return(NULL)
      }
    } else {
      shiny::removeModal()
      display_errors(
        title = "Error",
        size = "l",
        easy_close = TRUE,
        error_messages = "The file does not exist on the path."
      )
      return(NULL)
    }
  } else {
    return(NULL)
  }
}

load_and_check_dataset_raw_texts <- function(dir_path) {
  if (!is.null(dir_path)) {
    if (file.exists(dir_path) == TRUE) {
      display_processing(
        title = "Working. Please wait.",
        size = "l",
        easy_close = FALSE,
        message = ""
      )
      # Wait for modal
      Sys.sleep(1)
      data_set_raw_text <- load_from_disk(dir_path)
      if (inherits(data_set_raw_text, "LargeDataSetForText")) {
        shiny::removeModal()
        return(data_set_raw_text)
      } else {
        shiny::removeModal()
        display_errors(
          title = "Error",
          size = "l",
          easy_close = TRUE,
          error_messages = "The file contains data in an unsupported format.
              A data set for raw texts must be of class 'LargeDataSetForText'. Please
              check data. A data set for raw texts should always be created via data
              preparation of this user interfache or with the corresponding
              method of the LargeDataSetForText."
        )
        rm(data_set_raw_text)
        gc()
        return(NULL)
      }
    } else {
      shiny::removeModal()
      display_errors(
        title = "Error",
        size = "l",
        easy_close = TRUE,
        error_messages = "The file does not exist on the path."
      )
      return(NULL)
    }
  } else {
    return(NULL)
  }
}

#' @title Load and check target data
#' @description Function for checking and loading target data in AI for Education - Studio.
#'
#' @param file_path `string` path to the file containing the target data.
#'
#' @return If there are any errors an error modal is displayed by calling the function `display_errors`. If there are no
#'   errors the function returns a `data.frame` containing the target data. In the case of erros the function returns
#'   `NULL`.
#'
#' @importFrom stringi stri_split_fixed
#' @importFrom stringi stri_trans_tolower
#' @importFrom utils read.csv
#'
#' @family Utils Studio Developers
#' @keywords internal
#' @noRd
load_and_check_target_data <- function(file_path) {
  if (!is.null(file_path)) {
    if (file.exists(file_path) == TRUE) {
      display_processing(
        title = "Working. Please wait.",
        size = "l",
        easy_close = FALSE,
        message = ""
      )

      # extension <- stringr::str_split_fixed(file_path, pattern = "\\.", n = Inf)
      # extension <- extension[1, ncol(extension)]
      # extension <- stringr::str_to_lower(extension)
      extension <- stringi::stri_split_fixed(file_path, pattern = ".")[[1]]
      extension <- stringi::stri_trans_tolower(extension[[length(extension)]])

      if (extension == "csv" || extension == "txt") {
        target_data <- try(
          as.data.frame(
            utils::read.csv(
              file = file_path,
              header = TRUE
            )
          ),
          silent = TRUE
        )
      } else if (extension == "xlsx") {
        target_data <- try(
          as.data.frame(
            readxl::read_xlsx(
              path = file_path,
              sheet = 1,
              col_names = TRUE
            )
          ),
          silent = TRUE
        )
      } else if (extension %in% c("rda", "rdata")) {
        object_name <- load(file = file_path)
        target_data <- get(x = object_name)
        target_data <- try(
          as.data.frame(target_data),
          silent = TRUE
        )
      } else {
        target_data <- NA
      }

      # Final Check
      if (is.character(target_data)) {
        shiny::removeModal()
        display_errors(
          title = "Error",
          size = "l",
          easy_close = TRUE,
          error_messages = "Data can not be loaded as data frame. Please check your data."
        )
        return(NULL)
      } else {
        if ("id" %in% colnames(target_data)) {
          rownames(target_data) <- target_data$id
          shiny::removeModal()
          return(target_data)
        } else {
          shiny::removeModal()
          display_errors(
            title = "Error",
            size = "l",
            easy_close = TRUE,
            error_messages = "Data does not contain a column named 'id'. This
                       column is necessary to match the text embeddings to their
                       corresponding targets. Please check your data."
          )
          return(NULL)
        }
      }
    } else {
      shiny::removeModal()
      display_errors(
        title = "Error",
        size = "l",
        easy_close = TRUE,
        error_messages = "The file does not exist on the path."
      )
      return(NULL)
    }
  } else {
    return(NULL)
  }
}

#' @title Check and ensure a valid empty argument
#' @description Function replaces empty input from an input widget into a valid empty argument for long running tasks.
#'   The valid empty argument is `NULL`.
#'
#' @param object Object to be transformed.
#'
#' @return Returns the object. Only in the case that the object is `NULL` or `object == ""` the function returns `NULL`
#'
#' @family Utils Studio Developers
#' @keywords internal
#' @noRd
#'
transform_input <- function(object) {
  res <- NULL
  if (!is.null(object) && object != "") res <- object
  return(res)
}

#' @title Checks for an empty input from an input widget
#' @description unction replaces checks for empty input produced by an input widget. These empty values are `NULL` and
#'   `""`.
#'
#' @param input Object to be transformed.
#'
#' @return Returns `TRUE` if input is `NULL` or `""`.
#'
#' @family Utils Studio Developers
#' @keywords internal
#' @noRd
#'
check_for_empty_input <- function(input) {
  return(is.null(input) || input == "")
}

#' @title Checks and transforms an numeric input
#' @description Function ensured that a numeric input is returned or an empty value (`NULL`). This function should only
#'   be applied if a numeric input is expected.
#'
#' @param input Object to be transformed.
#'
#' @return Returns the input as a numeric input or `NULL` if input is `NULL` or `""`.
#'
#' @family Utils Studio Developers
#' @keywords internal
#' @noRd
#'
check_numeric_input <- function(input) {
  res <- NULL
  if (!is.null(input) && input != "") res <- as.numeric(input)
  return(res)
}

#' @title Load target data for long running tasks
#' @description Function loads the target data for a long running task.
#'
#' @param file_path `string` Path to the file storing the target data.
#' @param selectet_column `string` Name of the column containing the target data.
#'
#' @details This function assumes that the target data is stored as a columns with the cases in the rows and the
#' categories in the columns. The ids of the cases must be stored in a column called "id".
#'
#' @return Returns a named factor containing the target data.
#' @importFrom utils read.csv
#' @family Utils Studio Developers
#' @export
long_load_target_data <- function(file_path, selectet_column) {
  extension <- stringi::stri_split_fixed(file_path, pattern = ".")[[1]]
  extension <- stringi::stri_trans_tolower(extension[[length(extension)]])

  if (extension == "csv" || extension == "txt") {
    target_data <- try(
      as.data.frame(
        utils::read.csv(
          file = file_path,
          header = TRUE
        )
      ),
      silent = TRUE
    )
  } else if (extension == "xlsx") {
    target_data <- try(
      as.data.frame(
        readxl::read_xlsx(
          path = file_path,
          sheet = 1,
          col_names = TRUE
        )
      ),
      silent = TRUE
    )
  } else if (extension %in% c("rda", "rdata")) {
    object_name <- load(file = file_path)
    target_data <- get(x = object_name)
    target_data <- try(
      as.data.frame(target_data),
      silent = TRUE
    )
  } else {
    stop("Could not load data.")
  }

  # Final Check
  if (is.character(target_data)) {
    stop("Data can not be loaded as data frame. Please check your data.")
  }
  if ("id" %in% colnames(target_data)) {
    rownames(target_data) <- target_data$id
  } else {
    stop("Data does not contain a column named 'id'. This
                       column is necessary to match the text embeddings to their
                       corresponding targets. Please check your data.")
  }

  target_factor <- as.factor(target_data[[selectet_column]])
  names(target_factor) <- target_data$id

  return(target_factor)
}

#' @title Generate description for text embeddings
#' @description Function generates a description for the underling [TextEmbeddingModel] of
#' give text embeddings.
#'
#' @param embeddings Object of class [LargeDataSetForTextEmbeddings] or [EmbeddedText].
#' @return Returns a `shiny::tagList` containing the html elements for the user interface.
#'
#' @family Utils Studio Developers
#' @keywords internal
#'
create_data_embeddings_description <- function(embeddings) {
  model_info <- embeddings$get_model_info()
  info_table <- matrix(
    nrow = 3,
    ncol = 4,
    data = ""
  )
  info_table[1, 1] <- "Model Method:"
  info_table[2, 1] <- "Pooling Type:"
  info_table[3, 1] <- "Model Language:"

  info_table[1, 2] <- model_info$model_method
  info_table[2, 2] <- model_info$param_emb_pool_type
  info_table[3, 2] <- model_info$model_language

  info_table[1, 3] <- "Tokens per Chunk:"
  info_table[2, 3] <- "Max Chunks:"
  info_table[3, 3] <- "Token Overlap:"

  info_table[1, 4] <- model_info$param_seq_length
  info_table[2, 4] <- model_info$param_chunks
  info_table[3, 4] <- model_info$param_overlap

  ui <- list(
    bslib::value_box(
      value = embeddings$n_rows(),
      title = "Number of Cases",
      showcase = shiny::icon("list")
    ),
    shiny::tags$h3("Model:", model_info$model_label),
    shiny::tags$p("Name:", model_info$model_name),
    shiny::tags$p("Created", model_info$model_date),
    shiny::renderTable(
      expr = info_table,
      colnames = FALSE
    )
  )
  return(ui)
}

create_data_raw_texts_description <- function(data_set_for_raw_texts) {
  ui <- bslib::value_box(
    value = data_set_for_raw_texts$n_rows(),
    title = "Number of Cases",
    showcase = shiny::icon("list")
  )
  return(ui)
}

create_data_base_model_description <- function(base_model) {
  ui <- bslib::value_box(
    value = base_model$get_model_type(),
    title = "Base Model Type",
    showcase = shiny::icon("brain")
  )
  return(ui)
}

#' @title Function for setting up AI for Education - Studio
#' @description This functions checks if all necessary R packages and python packages are available for using AI for
#'   Education - Studio. In the case python is not initialized it will first try to set a virtual environment `"aifeducation"`. If
#'   this does not exist it tries to use a 'conda' environment `"aifeducation"`. In
#'   the case python is already initialized it checks if the app can be run within the current environment.
#'
#' @param env_type `string` If set to `"venv"`  virtual environment is requested. If set to
#' `"conda"` a 'conda' environment is requested. If set to `"auto"` the function tries to
#' activate a virtual environment with the given name. If this environment does not exist
#' it tries to activate a conda environment with the given name. If this fails
#' the default virtual environment is used.
#'
#' @return Function does not return anything. It is used for preparing python and R
#' in order to run AI for Education - Studio.
#'
#' @family Utils Studio Developers
#' @importFrom utils packageVersion
#' @keywords internal
#' @noRd
#'
check_and_prepare_for_studio <- function(env_type = "auto") {
  message("Checking R Packages.")
  r_packages <- list(
    "ggplot2" = NULL,
    "rlang" = NULL,
    "shiny" = "1.9.0",
    "shinyFiles" = NULL,
    "shinyWidgets" = NULL,
    "shinycssloaders" = NULL,
    "sortable" = NULL,
    "bslib" = NULL,
    "future" = NULL,
    "promises" = NULL,
    "DT" = NULL,
    "readtext" = NULL,
    "readxl" = NULL
  )

  missing_r_packages <- NULL
  for (i in seq_along(r_packages)) {
    if (!requireNamespace(names(r_packages)[i], quietly = TRUE, )) {
      missing_r_packages <- append(
        x = missing_r_packages,
        values = names(r_packages)[i]
      )
    } else {
      if (!is.null(r_packages[[i]])) {
        if (!check_versions(
          a = as.character(utils::packageVersion(names(r_packages)[[i]])),
          operator = ">=",
          b = r_packages[[i]]
        )) {
          cat(paste(
            "version of", names(r_packages)[i], "is", utils::packageVersion(names(r_packages)[i]),
            "but must be at least", r_packages[[i]], "."
          ))
          missing_r_packages <- append(
            x = missing_r_packages,
            values = names(r_packages)[i]
          )
        }
      }
    }
  }

  if (length(missing_r_packages) > 0) {
    install_now <- utils::askYesNo(
      msg = paste(
        "The following R packages are missing or need a newer version for Aifeducation Studio.",
        "'", paste(missing_r_packages, collapse = ","), "'.",
        "Do you want to install them now?"
      ),
      default = TRUE,
      prompts = getOption("askYesNo", gettext(c("Yes", "No")))
    )
    if (install_now) {
      utils::install.packages(
        pkgs = missing_r_packages,
        dependencies = c("Depends", "Imports", "LinkingTo")
      )
    } else {
      stop("Some necessary R Packages are missing.")
    }
  }

  prepare_session(env_type = env_type, envname = "aifeducation", check_session = TRUE)

  message("Checking pytorch machine learning framework.")
  available_ml_frameworks <- NULL
  if (check_aif_py_modules(trace = FALSE)) {
    available_ml_frameworks <- append(available_ml_frameworks, values = "pytorch")
  }
  if (is.null(available_ml_frameworks)) {
    stop("No pytorch machine learning frameworks found.")
  }

  # Set Transformer Logger to Error
  set_transformers_logger(level = "ERROR")
  # Disable tqdm progressbar
  transformers$logging$disable_progress_bar()
  datasets$disable_progress_bars()
}

#' @title Generate widgets for licensing a model
#' @description Function generates the input widgets for licensing a model.
#'
#' @param ns `function` for setting the namespace of the input elements. This should be `session$ns`.
#' @param model Model for which the description should be generated.
#'
#' @return Returns a `shiny::tagList` containing the html elements for the user interface.
#'
#' @family Utils Studio Developers
#' @keywords internal
#' @noRd
#'
generate_doc_input_licensing_editor <- function(ns, model) {
  ui <- shiny::tagList(
    bslib::card(
      bslib::card_header("Editor"),
      bslib::card_body(
        shiny::textInput(
          inputId = ns(paste0("doc_editor_", "documentation_license")),
          label = "Model License",
          width = "100%",
          value = model$get_model_license()
        ),
        shiny::textInput(
          inputId = ns(paste0("doc_editor_", "software_license")),
          label = "Documentation License",
          width = "100%",
          value = model$get_documentation_license()
        ),
        shiny::actionButton(
          inputId = ns(paste0("doc_editor_", "licensing", "_save_button")),
          label = "Save",
          icon = shiny::icon("floppy-disk")
        )
      )
    )
  )
  return(ui)
}

#' @title Replace NULL with NA
#' @description Function replaces `NULL` with `NA`
#'
#' @return If value is `NULL` returns `NA`. In all other cases it returns value.
#'
#' @family Utils Studio Developers
#' @keywords internal
#' @noRd
#'
replace_null_with_na <- function(value) {
  if (is.null(value)) {
    return(NA)
  } else {
    return(value)
  }
}

#' @title Create widget card
#' @description This function creates a card which contains a widget for
#' all arguments of the method determined with `method`.
#'
#' @description In order to include
#' an argument within the card the argument must be specified in `get_param_dict`.
#' The entry `gui_box` must be set to a `string`. All argument with the same value
#' for `gui_box` are grouped within a box of the card. The value of `gui_box` also
#' functions as header of the box. To exclude an argumet set `gui_box=NULL` in
#' `get_param_dict`.
#'
#' @param id Id of the shiny session.
#' @param object_class `string` Class of the object.
#' @param method `string` Method for which the card should be created.
#' @param box_title `string` Title of the card containing the boxes.
#'
#' @return Returns a `bslib::card`.
#'
#' @family Utils Studio Developers
#' @keywords internal
#' @noRd
#'
create_widget_card <- function(id,
                               object_class,
                               method = "configure",
                               box_title) {
  # Create Object
  object <- create_object(object_class)

  # Get params of the corresponding method
  params <- rlang::fn_fmls_names(object[[method]])

  # Get param dict
  param_dict <- get_param_dict()[params]

  tmp_boxes <- list()
  for (param in params) {
    dict_entry <- param_dict[[param]]
    if (!is.null(dict_entry$gui_label)) {
      tmp_label <- dict_entry$gui_label
      # if (!is.null(dict_entry$values_desc)) {
      tmp_label_with_icon <- shiny::tags$p(
        bslib::popover(
          # trigger = shiny::icon("info-circle"),
          trigger = dict_entry$gui_label,
          shiny::includeMarkdown(
            get_parameter_documentation(
              param_name = param,
              param_dict = param_dict,
              inc_param_name = FALSE,
              as_list = FALSE
            )
          ),
          options = list(
            "trigger" = "hover" # ,
            # "delay"="{'show': 0, 'hide': 500}"
          )
        )
      )
      # } else {
      #  tmp_label_with_icon <- dict_entry$gui_label
      # }
    } else {
      tmp_label <- param
      tmp_label_with_icon <- param
    }
    if (!is.null(dict_entry$gui_box)) {
      if (dict_entry$type == "int") {
        if (dict_entry$min == -Inf) {
          tmp_min <- NA
        } else {
          tmp_min <- dict_entry$min
        }

        if (dict_entry$max == Inf) {
          tmp_max <- NA
        } else {
          tmp_max <- dict_entry$max
        }

        widget <- shiny::numericInput(
          inputId = shiny::NS(id, param),
          label = tmp_label_with_icon,
          value = dict_entry$default_value,
          min = tmp_min,
          max = tmp_max
        )
      } else if (dict_entry$type == "string") {
        if (param == "sustain_iso_code") {
          widget <- shiny::selectInput(
            inputId = shiny::NS(id, param),
            label = tmp_label_with_icon,
            choices = get_alpha_3_codes(),
            selected = dict_entry$default_value,
            multiple = FALSE
          )
        } else {
          if (!is.null(dict_entry$allowed_values)) {
            widget <- shiny::selectInput(
              inputId = shiny::NS(id, param),
              label = tmp_label_with_icon,
              choices = dict_entry$allowed_values,
              multiple = FALSE
            )
          } else {
            widget <- shiny::textInput(
              inputId = shiny::NS(id, param),
              label = tmp_label_with_icon
            )
          }
        }
      } else if (dict_entry$type == "bool") {
        widget <- shinyWidgets::materialSwitch(
          inputId = shiny::NS(id, param),
          label = tmp_label_with_icon,
          value = dict_entry$default_value
        )
      } else if (dict_entry$type == "double" ||
        dict_entry$type == "(double" ||
        dict_entry$type == "double)" ||
        dict_entry$type == "(double)") {
        if (dict_entry$min != -Inf && dict_entry$max != Inf) {
          if (!is.null(dict_entry$magnitude)) {
            widget <- shiny::selectInput(
              inputId = shiny::NS(id, param),
              label = tmp_label_with_icon,
              choices = get_magnitude_values(
                min = dict_entry$min,
                max = dict_entry$max,
                magnitude = dict_entry$magnitude,
                n_elements = 9
              )
            )
          } else {
            range <- dict_entry$max - dict_entry$min

            if (dict_entry$type == "(double" ||
              dict_entry$type == "(double)") {
              tmp_min <- dict_entry$min + range * 0.01
            } else {
              tmp_min <- dict_entry$min
            }

            if (dict_entry$type == "double)" ||
              dict_entry$type == "(double)") {
              tmp_max <- dict_entry$max - range * 0.01
            } else {
              tmp_max <- dict_entry$max
            }

            widget <- shiny::sliderInput(
              inputId = shiny::NS(id, param),
              label = tmp_label_with_icon,
              value = dict_entry$default_value,
              min = tmp_min,
              max = tmp_max,
            )
          }
        }
      }

      # Add widget to the correct box
      current_box <- tmp_boxes[[dict_entry$gui_box]]
      current_box[tmp_label] <- list(widget)
      tmp_boxes[dict_entry$gui_box] <- list(current_box)
    }
  }

  # Sort Boxes
  box_names <- names(tmp_boxes)
  if ("General Settings" %in% box_names) {
    reduced_names <- setdiff(x = box_names, "General Settings")
    ordered_names <- c(
      "General Settings",
      sort(reduced_names, na.last = TRUE)
    )
  } else {
    ordered_names <- sort(box_names, na.last = TRUE)
  }
  tmp_boxes <- tmp_boxes[ordered_names]

  # Sort Widgets
  for (i in seq_along(tmp_boxes)) {
    current_box <- tmp_boxes[[i]]
    tmp_names <- names(current_box)
    # Ensure that parameters starting with use are displayed first
    is_use_string <- stringi::stri_detect(str = tolower(tmp_names), regex = "^use([:alnum:]*)")
    if (max(is_use_string) >= 1) {
      use_string <- tmp_names[which(is_use_string)]
      reduced_names <- setdiff(x = tmp_names, y = use_string)
      ordered_names <- c(
        use_string,
        sort(reduced_names, na.last = TRUE)
      )
    } else {
      ordered_names <- sort(tmp_names, na.last = TRUE)
    }
    current_box <- current_box[ordered_names]
    tmp_boxes[i] <- list(current_box)
  }

  # Create boxes with widgets
  tmp_cards <- list()
  layer_dict <- get_layer_dict("all")
  layer_labels <- vector(length = length(layer_dict))
  names(layer_labels) <- names(layer_dict)
  for (layer in names(layer_labels)) {
    layer_labels[layer] <- layer_dict[[layer]]$title
  }

  for (i in seq_along(tmp_boxes)) {
    if (names(tmp_boxes)[i] %in% layer_labels) {
      idx_current_layer <- which(x = layer_labels == names(tmp_boxes)[i])
      current_layer_name <- names(layer_labels)[idx_current_layer]
      popover_text <- layer_dict[[current_layer_name]]$desc
      current_popover <- bslib::popover(
        trigger = shiny::icon("info-circle"),
        shiny::includeMarkdown(popover_text)
      )
    } else {
      current_popover <- ""
    }

    tmp_cards[length(tmp_cards) + 1] <- list(
      bslib::card(
        bslib::card_header(current_popover, names(tmp_boxes)[i]),
        bslib::card_body(
          # bslib::layout_column_wrap(
          tmp_boxes[[i]]
          # )
        )
      )
    )
  }


  # Create Main Card
  main_card <- bslib::card(
    bslib::card_header(box_title),
    bslib::card_body(
      do.call(
        what = bslib::layout_column_wrap,
        args = tmp_cards
      )
    )
  )
  return(main_card)
}

extract_args_from_input <- function(input, arg_names) {
  args_list <- list()
  for (param in arg_names) {
    args_list[param] <- list(input[[param]])
  }
  return(args_list)
}

#' @title Summarize arguments from shiny input
#' @description This function extracts the input relevant for a specific
#' method of a specific class from shiny input.
#'
#' @description In addition, it adds the path
#' to all objects which can not be exported to another R session. These object
#' must be loaded separately in the new session with the function `add_missing_args`.
#' The paths are intended to be used with `shiny::ExtendedTask`. The final preparation of the arguments
#' should be done with
#'
#' @description The function can also be used to override the default value of
#' a method or to add value for arguments which are not part of shiny input
#' (use parameter `override_args`).
#'
#' @param input Shiny input.
#' @param object_class `string` Class of the object.
#' @param method `string` Method of the class for which the arguments should
#' be extracted and prepared.
#' @param path_args `list` List containing the path to object that can not be exported
#' to another R session. These must be loaded in the session.
#' @param override_args `list` List containing all arguments that should be set manually.
#' The values override default values of the argument and values which are part of `input`.
#' @param meta_args `list` List containing information that are not relevant for the
#' arguments of the method but are necessary to set up the `shiny::ExtendedTask`
#' correctly.
#'
#' @note Please not that all list are named list of the format (argument_name=values).
#'
#' @return Returns a named `list` with the following entries:
#' * args: Named `list` of all arguments necessary for the method of the class.
#' * path_args: Named `list` of all paths for loading the objects missing in args.
#' * meta_args: Named `list` of all arguments that are not part of the arguments of
#'   the method but which are necessary to set up the `shiny::ExtendedTask` correctly.
#'
#' @family Utils Studio Developers
#' @export
summarize_args_for_long_task <- function(input,
                                         object_class,
                                         method = "configure",
                                         path_args = list(
                                           path_to_embeddings = NULL,
                                           path_to_textual_dataset = NULL,
                                           path_to_target_data = NULL,
                                           path_to_feature_extractor = NULL,
                                           destination_path = NULL,
                                           folder_name = NULL
                                         ),
                                         override_args = list(),
                                         meta_args = list(
                                           py_environment_type = get_py_env_type(),
                                           py_env_name = get_py_env_name(),
                                           target_data_column = input$data_target_column,
                                           object_class = input$classifier_type
                                         )) {
  # Create object in order to get relevant arguments
  object <- create_object(object_class)

  # Get dictionary of all parameters
  param_dict <- get_param_dict()

  # Create param_list
  param_list <- rlang::fn_fmls(object[[method]])

  # Get params of the method
  params <- names(param_list)

  for (param in params) {
    current_param <- param_dict[[param]]
    if (max(current_param$type %in% c("bool", "int", "double", "(double", "double)", "(double)", "string", "vector", "list")) &&
      !is.null(input[[param]])) {
      param_list[param] <- list(input[[param]])
    }
  }

  # Do type adjustments
  if ("lr_rate" %in% params) {
    param_list["lr_rate"] <- list(as.numeric(param_list[["lr_rate"]]))
  }
  if ("learning_rate" %in% params) {
    param_list["learning_rate"] <- list(as.numeric(param_list[["learning_rate"]]))
  }
  if ("lr_min" %in% params) {
    param_list["lr_min"] <- list(as.numeric(param_list[["lr_min"]]))
  }

  # Override params but only if the argument to override exists
  for (param in names(override_args)) {
    if (param %in% params) {
      param_list[param] <- list(override_args[[param]])
    }
  }

  # Add method to meta_args
  meta_args$method <- method

  # Add path arguments and further additional arguments
  return(list(
    args = param_list,
    path_args = path_args,
    meta_args = meta_args
  ))
}

#' @title Add missing arguments to a list of arguments
#' @description This function is designed for taking the output of
#' `summarize_args_for_long_task` as input. It adds the missing arguments.
#' In general these are arguments that rely on objects of class R6 which can not
#' be exported to a new R session.
#'
#' @param args Named `list` List for arguments for the method of a specific class.
#' @param path_args Named `list` List of paths where the objects are stored on disk.
#' @param meta_args Named `list` List containing arguments that are necessary in order to
#' add the missing objects correctly.
#'
#' @return Returns a named `list` of all arguments that a method of a specific class
#' requires.
#'
#' @family Utils Studio Developers
#' @export
#'
add_missing_args <- function(args, path_args, meta_args) {
  # Create a copy of all args
  complete_args <- args

  # Get dictionary of all parameters
  param_dict <- get_param_dict()

  for (param in names(args)) {
    current_param <- param_dict[[param]]
    if ("LargeDataSetForTextEmbeddings" %in% current_param$type) {
      if (!is.null(path_args$path_to_embeddings)) {
        complete_args[param] <- list(
          load_from_disk(path_args$path_to_embeddings)
        )
      }
    } else if ("TEFeatureExtractor" %in% current_param$type) {
      if (!is.null(path_args$path_to_feature_extractor)) {
        complete_args[param] <- list(
          load_from_disk(path_args$path_to_feature_extractor)
        )
      }
    } else if ("factor" %in% current_param$type && !is.null(path_args$path_to_target_data)) {
      complete_args[param] <- list(
        long_load_target_data(
          file_path = path_args$path_to_target_data,
          selectet_column = meta_args$target_data_column
        )
      )
    } else if (max(c("EmbeddedText", "LargeDataSetForText") %in% current_param$type)) {
      complete_args[param] <- list(
        load_from_disk(path_args$path_to_textual_dataset)
      )
    } else if (max(unlist(BaseModelsIndex) %in% current_param$type) && !is.null(path_args$path_to_base_model)) {
      complete_args[param] <- list(
        load_from_disk(path_args$path_to_base_model)
      )
    }
  }
  return(complete_args)
}
