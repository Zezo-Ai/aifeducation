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

#' @title Graphical user interface for feature extractors - create
#' @description Functions generates the page for a creating a new [TEFeatureExtractor].
#'
#' @param id `string` determining the id for the namespace.
#' @return This function does nothing return. It is used to build a page for a shiny app.
#'
#' @family studio_gui_page_feature_extractor_create
#' @keywords internal
#' @noRd
#'
FeatureExtractors_Create_UI <- function(id) {
  shiny::tagList(
    bslib::page_sidebar(
      # Sidebar------------------------------------------------------------------
      sidebar = bslib::sidebar(
        position = "left",
        shiny::tags$h3("Control Panel"),
        shiny::tags$hr(),
        shiny::textInput(
          inputId = shiny::NS(id, "label"),
          label = "Model Label",
          width = "100%"
        ),
        shinyFiles::shinyDirButton(
          id = shiny::NS(id, "start_SaveModal"),
          label = "Train Model",
          title = "Choose Destination",
          icon = shiny::icon("floppy-disk")
        )
      ),
      # Main Page---------------------------------------------------------------
      # Content depends in the TextEmbeddingModel of the embeddings
      # Embeddings
      # bslib::layout_column_wrap(
      bslib::card(
        bslib::card_header("Input Data"),
        bslib::card_body(
          shinyFiles::shinyDirButton(
            id = shiny::NS(id, "button_select_dataset_for_embeddings"),
            label = "Choose Embeddings",
            title = "Please choose a folder",
            icon = shiny::icon("folder-open")
          ),
          shiny::textInput(
            inputId = shiny::NS(id, "embeddings_dir"),
            label = shiny::tags$p(shiny::icon("folder"), "Path"),
            width = "100%"
          ),
          shiny::uiOutput(outputId = shiny::NS(id, "summary_data_embeddings"))
        )
      ),
      # Main config Cards
      shiny::uiOutput(outputId = shiny::NS(id, "model_configuration")),
      shiny::uiOutput(outputId = shiny::NS(id, "training_setup"))
      # )
    )
  )
}


#' @title Server function for: graphical user interface for feature extractors - create
#' @description Functions generates the functionality of a page on the server.
#'
#' @param id `string` determining the id for the namespace.
#' @param log_dir `string` Path to the directory where the log files should be stored.
#' @param volumes `vector` containing a named vector of available volumes.
#' @return This function does nothing return. It is used to create the functionality of a page for a shiny app.
#'
#' @family studio_gui_page_feature_extractor_create
#' @keywords internal
#' @noRd
#'
FeatureExtractor_Create_Server <- function(id, log_dir, volumes) {
  shiny::moduleServer(id, function(input, output, session) {
    # global variables-----------------------------------------------------------
    log_path <- paste0(log_dir, "/aifeducation_state.log")

    # File system management----------------------------------------------------
    # Embeddings
    shinyFiles::shinyDirChoose(
      input = input,
      id = "button_select_dataset_for_embeddings",
      roots = volumes,
      # session = session,
      allowDirCreate = FALSE
    )
    shiny::observeEvent(input$button_select_dataset_for_embeddings, {
      path <- shinyFiles::parseDirPath(volumes, input$button_select_dataset_for_embeddings)
      shiny::updateTextInput(
        inputId = "embeddings_dir",
        value = path
      )
    })

    path_to_embeddings <- shiny::eventReactive(input$embeddings_dir, {
      if (input$embeddings_dir != "") {
        return(input$embeddings_dir)
      } else {
        return(NULL)
      }
    })

    data_embeddings <- shiny::reactive({
      if (!is.null(path_to_embeddings())) {
        return(load_and_check_embeddings(path_to_embeddings()))
      } else {
        return(NULL)
      }
    })

    # Box for model configuration------------------------------------------------
    output$model_configuration <- shiny::renderUI({
      config_box <- create_widget_card(
        id = id,
        object_class = "TEFeatureExtractor",
        method = "configure",
        box_title = "Model Configuration"
      )
    })
    # Box for training set up---------------------------------------------------
    output$training_setup <- shiny::renderUI({
      config_box <- create_widget_card(
        id = id,
        object_class = "TEFeatureExtractor",
        method = "train",
        box_title = "Training SetUp"
      )
    })

    # Start screen for choosing the location for storing the data set-----------
    # Create Save Modal
    save_modal <- create_save_modal(
      id = id,
      # ns = session$ns,
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

    # Start training------------------------------------------------------------
    shiny::observeEvent(input$save_modal_button_continue, {
      # Remove Save Modal
      shiny::removeModal()

      # Check for errors
      errors <- check_errors_create_feature_extractor(
        destination_path = input$save_modal_directory_path,
        folder_name = input$save_modal_folder_name,
        path_to_embeddings = path_to_embeddings(),
        features = input$features,
        model_label = input$label
      )

      # If there are errors display them. If not start running task.
      if (!is.null(errors)) {
        display_errors(
          title = "Error",
          size = "l",
          easy_close = TRUE,
          error_messages = errors
        )
      } else {
        # Start task and monitor
        start_and_monitor_long_task(
          id = id,
          ExtendedTask_type = "feature_extractor",
          ExtendedTask_arguments = list(
            configure = summarize_args_for_long_task(
              input = input,
              object_class = "TEFeatureExtractor",
              method = "configure",
              path_args = list(
                path_to_embeddings = path_to_embeddings(),
                path_to_target_data = NULL,
                path_to_feature_extractor = NULL,
                destination_path = input$save_modal_directory_path,
                folder_name = input$save_modal_folder_name
              ),
              override_args = list(
                sustain_track = TRUE
              ),
              meta_args = list(
                py_environment_type = get_py_env_type(),
                py_env_name = get_py_env_name(),
                target_data_column = NULL,
                object_class = "TEFeatureExtractor"
              )
            ),
            train = summarize_args_for_long_task(
              input = input,
              object_class = "TEFeatureExtractor",
              method = "train",
              path_args = list(
                path_to_embeddings = path_to_embeddings(),
                path_to_target_data = NULL,
                path_to_feature_extractor = NULL,
                destination_path = input$save_modal_directory_path,
                folder_name = input$save_modal_folder_name
              ),
              override_args = list(
                sustain_track = TRUE,
                log_dir = log_dir,
                trace = FALSE,
                ml_trace = 0,
                n_cores = auto_n_cores()
              ),
              meta_args = list(
                py_environment_type = get_py_env_type(),
                py_env_name = get_py_env_name(),
                target_data_column = NULL,
                object_class = "TEFeatureExtractor"
              )
            )
          ),
          log_path = log_path,
          pgr_use_middle = TRUE,
          pgr_use_bottom = TRUE,
          pgr_use_graphic = TRUE,
          update_intervall = 4,
          success_type = "classifier"
        )
      }
    })

    # Display Data Summary------------------------------------------------------
    # Embeddings
    output$summary_data_embeddings <- shiny::renderUI({
      embeddings <- data_embeddings()
      # shiny::req(embeddings)
      if (!is.null(embeddings)) {
        ui <- create_data_embeddings_description(embeddings)

        return(ui)
      } else {
        return(NULL)
      }
    })

    # Error handling-----------------------------------------------------------


    #--------------------------------------------------------------------------
  })
}
