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

#' @title Graphical user interface for transforming raw texts into numerical text embeddings.
#' @description Functions generates the tab within a page for generating text embeddings with an object of class
#'   [TextEmbeddingModel].
#'
#' @param id `string` determining the id for the namespace.
#' @return This function does nothing return. It is used to build a page for a shiny app.
#'
#' @family studio_gui_text_embedding_model_embed
#' @keywords internal
#' @noRd
#'
Embed_UI <- function(id) {
  bslib::page(
    bslib::card(
      bslib::card_header(
        "Raw Texts"
      ),
      bslib::card_body(
        shinyFiles::shinyDirButton(
          id = shiny::NS(id, "choose_file_raw_texts"),
          label = "Choose Folder",
          title = "Please choose a folder",
          icon = shiny::icon("folder"),
          multiple = FALSE
        ),
        shiny::textInput(
          inputId = shiny::NS(id, "file_path_raw_texts"),
          label = shiny::tags$p(shiny::icon("folder"), "Path to folder")
        ),
        shiny::numericInput(
          inputId = shiny::NS(id, "batch_size"),
          label = "Batch Size",
          min = 1,
          max = 512,
          value = 8
        ),
        shinyFiles::shinyDirButton(
          id = shiny::NS(id, "start_SaveModal"),
          label = "Save Embeddings",
          title = "Choose Destination",
          icon = shiny::icon("floppy-disk")
        )
      )
    )
  )
}


#' @title Server function for: graphical user interface for transforming raw texts into numerical text embeddings.
#' @description Functions generates the functionality of a page on the server.
#'
#' @param id `string` determining the id for the namespace.
#' @param model Model used for inference.
#' @param model_path `string` Path to the model.
#' @param log_dir `string` Path to the directory where the log files should be stored.
#' @param volumes `vector` containing a named vector of available volumes.
#' @return This function does nothing return. It is used to create the functionality of a page for a shiny app.
#'
#' @family studio_gui_text_embedding_model_embed
#' @keywords internal
#' @noRd
#'
Embed_Server <- function(id, model, model_path, log_dir, volumes) {
  shiny::moduleServer(id, function(input, output, session) {
    # global variables-----------------------------------------------------------
    log_path <- paste0(log_dir, "/aifeducation_state.log")

    # file system---------------------------------------------------------------
    shinyFiles::shinyDirChoose(
      input = input,
      id = "choose_file_raw_texts",
      roots = volumes,
      allowDirCreate = FALSE
    )

    shiny::observeEvent(input$choose_file_raw_texts, {
      tmp_file_path <- shinyFiles::parseDirPath(volumes, input$choose_file_raw_texts)
      if (length(tmp_file_path) > 0) {
        shiny::updateTextInput(
          inputId = "file_path_raw_texts",
          value = tmp_file_path
        )
      }
    })


    # Start screen for choosing the location for storing the data set-----------
    # Create Save Modal
    save_modal <- create_save_modal(
      id = id,
      ns = session$ns,
      title = "Choose Destination",
      easy_close = FALSE,
      size = "l"
    )

    # Implement file connection
    shinyFiles::shinyDirChoose(
      input = input,
      id = "start_SaveModal",
      roots = volumes,
      allowDirCreate = TRUE
    )

    # show save_modal
    shiny::observeEvent(input$start_SaveModal, {
      path <- shinyFiles::parseDirPath(volumes, input$start_SaveModal)
      if (!is.null(path) & !identical(path, character(0))) {
        if (path != "") {
          shiny::showModal(save_modal)
          shiny::updateTextInput(
            inputId = "save_modal_directory_path",
            value = path
          )
        }
      }
    })

    # Calculate Embedding------------------------------------------------------
    shiny::observeEvent(input$save_modal_button_continue, {
      # Remove Save Modal
      shiny::removeModal()

      # Check input
      error_list <- check_errors_text_embedding_model_embed(
        destination_path = input$save_modal_directory_path,
        folder_name = input$save_modal_folder_name,
        path_to_raw_texts = input$file_path_raw_texts,
        batch_size = input$batch_size
      )

      if (length(error_list) == 0) {
        start_and_monitor_long_task(
          id = id,
          ExtendedTask_type = "embed_raw_text",
          ExtendedTask_arguments = list(
            source_path = input$file_path_raw_texts,
            destination_path = input$save_modal_directory_path,
            destination_folder = input$save_modal_folder_name,
            log_path = log_path,
            batch_size = input$batch_size,
            model_path = model_path,
            log_write_interval = 2,
            py_environment_type = get_py_env_type(),
            py_env_name = get_py_env_name()
          ),
          log_path = log_path,
          pgr_use_middle = FALSE,
          pgr_use_bottom = FALSE,
          update_intervall = 4,
          success_type = "data_sets"
        )
      } else {
        display_errors(
          title = "Error",
          size = "l",
          easy_close = TRUE,
          error_messages = error_list
        )
      }
    })
    #--------------------------------------------------------------------------
  })
}
