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

#' @title Graphical user interface for documenting objects
#' @description Functions generates the page for documenting objects.
#'
#' @param id `string` determining the id for the namespace.
#' @return This function does nothing return. It is used to build a page for a shiny app.
#' @importFrom utils person
#' @family studio_gui_page_document
#' @keywords internal
#' @noRd
#'
DocumentPage_UI <- function(id, type = "TextEmbeddingModel") {
  bslib::page(
    bslib::page_sidebar(
      # Sidebar------------------------------------------------------------------
      sidebar = bslib::sidebar(
        position = "left",
        shiny::tags$h3("Control Panel"),
        shinyFiles::shinyDirButton(
          id = shiny::NS(id, "button_select_model"),
          label = "Choose a Model",
          title = "Please choose a folder",
          icon = shiny::icon("folder-open")
        ),
        shiny::uiOutput(outputId = shiny::NS(id, "sidebar_description"))
      ),
      # Main panel---------------------------------------------------------------
      bslib::navset_card_underline(
        bslib::nav_panel(
          title = "Developers",
          shiny::uiOutput(outputId = shiny::NS(id, "developers"))
        ),
        if (type == "TextEmbeddingModel") {
          bslib::nav_panel(
            title = "Modifiers",
            shiny::uiOutput(outputId = shiny::NS(id, "modifiers"))
          )
        },
        bslib::nav_panel(
          title = "Abstract English",
          shiny::uiOutput(outputId = shiny::NS(id, "abstract_eng"))
        ),
        bslib::nav_panel(
          title = "Description English",
          shiny::uiOutput(outputId = shiny::NS(id, "desc_eng"))
        ),
        bslib::nav_panel(
          title = "Abstract Native",
          shiny::uiOutput(outputId = shiny::NS(id, "abstract_native"))
        ),
        bslib::nav_panel(
          title = "Description Native",
          shiny::uiOutput(outputId = shiny::NS(id, "desc_native"))
        ),
        bslib::nav_panel(
          title = "Licensing",
          shiny::uiOutput(outputId = shiny::NS(id, "licensing"))
        )
      )
    )
  )
}


#' @title Server function for: graphical user interface for classifiers - create
#' @description Functions generates the functionality of a page on the server.
#'
#' @param id `string` determining the id for the namespace.
#' @param type `string` determing the type of documentation page. `"TextEmbeddigModel"` for objects of class
#'   [TextEmbeddingModel] and `"Classifiers"` for objects of class [TEClassifierRegular] and [TEClassifierProtoNet].
#' @param volumes `vector` containing a named vector of available volumes.
#' @return This function does nothing return. It is used to create the functionality of a page for a shiny app.
#'
#' @family studio_gui_page_document
#' @keywords internal
#' @noRd
#'
DocumentPage_Server <- function(id, volumes, type = "TextEmbeddingModel") {
  shiny::moduleServer(id, function(input, output, session) {
    # global variables-----------------------------------------------------------

    shinyFiles::shinyDirChoose(
      input = input,
      id = "button_select_model",
      roots = volumes,
      allowDirCreate = FALSE
    )

    # Load the model and check for errors during loading-------------------------
    model_path <- shiny::eventReactive(input$button_select_model, {
      model_path <- shinyFiles::parseDirPath(volumes, input$button_select_model)
      if (length(model_path) > 0L) {
        return(model_path)
      } else {
        return(NULL)
      }
    })

    model <- shiny::eventReactive(input$button_select_model, {
      # Get model path
      model_path <- shinyFiles::parseDirPath(volumes, input$button_select_model)
      if (length(model_path) > 0L) {
        display_processing(
          title = "Working. Please wait.",
          size = "l",
          easy_close = FALSE,
          message = ""
        )

        # Try to load the model
        model <- try(load_from_disk(model_path), silent = TRUE)

        if (!inherits(model, "try-error")) {
          if (type == "TextEmbeddingModel") {
            if (inherits(model, "TextEmbeddingModel")) {
              shiny::removeModal()
              return(model)
            } else if (inherits(model, "BaseModelCore")) {
              shiny::removeModal()
              return(model)
            } else {
              display_errors(
                title = "Error",
                size = "l",
                easy_close = TRUE,
                error_messages = "The file does not contain an object of class of TextEmbeddingModel or BaseModelCore."
              )
              return(NULL)
            }
          } else if (type == "Classifier") {
            if (
              inherits(model, "TEClassifierRegular") |
                inherits(model, "TEClassifierProtoNet")
            ) {
              shiny::removeModal()
              return(model)
            } else {
              display_errors(
                title = "Error",
                size = "l",
                easy_close = TRUE,
                error_messages = "The file does not contain an object of class
                                  TEClassifierRegular or TEClassifierProtoNet."
              )
              return(NULL)
            }
          } else if (type == "FeatureExtractors") {
            if (inherits(model, "TEFeatureExtractor")) {
              shiny::removeModal()
              return(model)
            } else {
              display_errors(
                title = "Error",
                size = "l",
                easy_close = TRUE,
                error_messages = "The file does not contain an object of class TEFeatureExtractor."
              )
              return(NULL)
            }
          }
        } else {
          display_errors(
            title = "Error",
            size = "l",
            easy_close = TRUE,
            error_messages = model
          )
          return(NULL)
        }
      }
    })

    # Render the information at the sidebar--------------------------------------
    output$sidebar_description <- shiny::renderUI({
      shiny::req(model())
      return(
        generate_sidebar_information(model())
      )
    })

    # Developed By Tab---------------------------------------------------------------
    output$developers <- shiny::renderUI({
      shiny::req(model())
      return(
        generate_doc_input_developers(ns = session$ns, model = model(), type = "developers")
      )
    })

    shiny::observeEvent(input$doc_developed_by_save, {
      tmp_model <- model()

      tmp_person_list <- NULL
      for (i in 1L:10L) {
        given <- input[[paste0("doc_", "Developers", "_fist_name_", i)]]
        family_name <- input[[paste0("doc_", "Developers", "_last_name_", i)]]
        mail <- input[[paste0("doc_", "Developers", "_mail_", i)]]
        if (!is.null(given) & !is.null(family_name)) {
          if (given != "" & family_name != "") {
            person_data <- person(given = given, family = family_name, email = mail)
            tmp_person_list <- append(
              x = tmp_person_list,
              values = person_data
            )
          }
        }
      }

      tmp_model$set_publication_info(
        type = "developer",
        authors = tmp_person_list,
        citation = input[[paste0("doc_", "developed_by", "_citation")]],
        url = input[[paste0("doc_", "developed_by", "_url")]]
      )
      r_interface_path <- file.path(model_path(), "r_config_state.rda")
      tmp_config <- create_config_state(tmp_model)
      save(tmp_config, file = r_interface_path)
      model <- shiny::reactive({
        tmp_model
      })
    })

    # Modified By Tab---------------------------------------------------------------
    if (type == "TextEmbeddingModel") {
      output$modifiers <- shiny::renderUI({
        shiny::req(model())
        return(
          generate_doc_input_developers(ns = session$ns, model = model(), type = "modifiers")
        )
      })

      shiny::observeEvent(input$doc_modified_by_save, {
        tmp_model <- model()

        tmp_person_list <- NULL
        for (i in 1L:10L) {
          given <- input[[paste0("doc_", "Modifiers", "_fist_name_", i)]]
          family_name <- input[[paste0("doc_", "Modifiers", "_last_name_", i)]]
          mail <- input[[paste0("doc_", "Modifiers", "_mail_", i)]]
          if (!is.null(given) & !is.null(family_name)) {
            if (given != "" & family_name != "") {
              person_data <- person(given = given, family = family_name, email = mail)
              tmp_person_list <- append(
                x = tmp_person_list,
                values = person_data
              )
            }
          }
        }

        tmp_model$set_publication_info(
          type = "modifier",
          authors = tmp_person_list,
          citation = input[[paste0("doc_", "modified_by", "_citation")]],
          url = input[[paste0("doc_", "modified_by", "_url")]]
        )
        r_interface_path <- file.path(model_path(), "r_config_state.rda")
        tmp_config <- create_config_state(tmp_model)
        save(tmp_config, file = r_interface_path)
        model <- shiny::reactive({
          tmp_model
        })
      })
    } else {
      output$modifiers <- NULL
    }

    # Abstract Eng---------------------------------------------------------------
    output$abstract_eng <- shiny::renderUI({
      shiny::req(model())
      return(
        generate_doc_input_text_editor(ns = session$ns, model = model(), language = "eng", type = "abstract")
      )
    })

    shiny::observeEvent(input$doc_editor_abstract_eng_save_button, {
      tmp_model <- model()
      tmp_model$set_model_description(
        abstract_eng = input$doc_editor_abstract_eng,
        keywords_eng = input$doc_editor_abstract_eng_keywords
      )
      r_interface_path <- file.path(model_path(), "r_config_state.rda")
      tmp_config <- create_config_state(tmp_model)
      save(tmp_config, file = r_interface_path)
      model <- shiny::reactive({
        tmp_model
      })
    })

    shiny::observeEvent(input$doc_editor_abstract_eng_preview_button, {
      output$doc_editor_abstract_eng_preview <- shiny::renderUI({
        return(shiny::includeMarkdown(input$doc_editor_abstract_eng))
      })
    })

    # Description Eng---------------------------------------------------------------
    output$desc_eng <- shiny::renderUI({
      shiny::req(model())
      return(
        generate_doc_input_text_editor(ns = session$ns, model = model(), language = "eng", type = "desc")
      )
    })

    shiny::observeEvent(input$doc_editor_description_eng_save_button, {
      tmp_model <- model()
      tmp_model$set_model_description(
        eng = input$doc_editor_description_eng
      )
      r_interface_path <- file.path(model_path(), "r_config_state.rda")
      tmp_config <- create_config_state(tmp_model)
      save(tmp_config, file = r_interface_path)
      model <- shiny::reactive({
        tmp_model
      })
    })

    shiny::observeEvent(input$doc_editor_description_eng_preview_button, {
      output$doc_editor_description_eng_preview <- shiny::renderUI({
        return(shiny::includeMarkdown(input$doc_editor_description_eng))
      })
    })

    # Abstract Native---------------------------------------------------------------
    output$abstract_native <- shiny::renderUI({
      shiny::req(model())
      return(
        generate_doc_input_text_editor(ns = session$ns, model = model(), language = "native", type = "abstract")
      )
    })

    shiny::observeEvent(input$doc_editor_abstract_native_save_button, {
      tmp_model <- model()
      tmp_model$set_model_description(
        abstract_native = input$doc_editor_abstract_native,
        keywords_native = input$doc_editor_abstract_native_keywords
      )
      r_interface_path <- file.path(model_path(), "r_config_state.rda")
      tmp_config <- create_config_state(tmp_model)
      save(tmp_config, file = r_interface_path)
      model <- shiny::reactive({
        tmp_model
      })
    })

    shiny::observeEvent(input$doc_editor_abstract_native_preview_button, {
      output$doc_editor_abstract_native_preview <- shiny::renderUI({
        return(shiny::includeMarkdown(input$doc_editor_abstract_native))
      })
    })

    # Description Native---------------------------------------------------------------
    output$desc_eng <- shiny::renderUI({
      shiny::req(model())
      return(
        generate_doc_input_text_editor(ns = session$ns, model = model(), language = "native", type = "desc")
      )
    })


    shiny::observeEvent(input$doc_editor_description_native_save_button, {
      tmp_model <- model()
      tmp_model$set_model_description(
        native = input$doc_editor_description_native
      )
      r_interface_path <- file.path(model_path(), "r_config_state.rda")
      tmp_config <- create_config_state(tmp_model)
      save(tmp_config, file = r_interface_path)
      model <- shiny::reactive({
        tmp_model
      })
    })

    shiny::observeEvent(input$doc_editor_description_native_preview_button, {
      output$doc_editor_description_native_preview <- shiny::renderUI({
        return(shiny::includeMarkdown(input$doc_editor_description_native))
      })
    })

    # Licensing-----------------------------------------------------------------
    output$licensing <- shiny::renderUI({
      shiny::req(model())
      return(
        generate_doc_input_licensing_editor(ns = session$ns, model = model())
      )
    })

    shiny::observeEvent(input$doc_editor_licensing_save_button, {
      tmp_model <- model()

      tmp_model$set_documentation_license(input$doc_editor_documentation_license)
      tmp_model$set_model_license(input$doc_editor_software_license)

      r_interface_path <- file.path(model_path(), "r_config_state.rda")
      tmp_config <- create_config_state(tmp_model)
      save(tmp_config, file = r_interface_path)
      model <- shiny::reactive({
        tmp_model
      })
    })

    shiny::observeEvent(input$doc_editor_description_native_preview_button, {
      output$doc_editor_description_native_preview <- shiny::renderUI({
        return(shiny::includeMarkdown(input$doc_editor_description_native))
      })
    })


    #--------------------------------------------------------------------------
  })
}
