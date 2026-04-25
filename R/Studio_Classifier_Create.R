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

#' @title Graphical user interface for classifiers - create
#' @description Functions generates the page for a creating new classifiers.
#'
#' @param id `string` determining the id for the namespace.
#' @return This function does nothing return. It is used to build a page for a shiny app.
#'
#' @family studio_gui_page_classifier_create
#' @keywords internal
#' @noRd
#'
Classifiers_Create_UI <- function(id) {
  shiny::tagList(
    bslib::page_sidebar(
      # Sidebar------------------------------------------------------------------
      sidebar = bslib::sidebar(
        position = "left",
        shiny::tags$h3("Control Panel"),
        shiny::tags$hr(),
        shiny::selectInput(
          inputId = shiny::NS(id, "classifier_type"),
          choices = setdiff(
            x = get_TEClassifiers_class_names("ClassifiersBasedOnTextEmbeddings"),
            y = get_depr_obj_names()
          ),
          label = "Classifier Type"
        ),
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
        ),
        shiny::tags$hr(),
        shiny::actionButton(
          inputId = shiny::NS(id, "test_data_matching"),
          label = "Test Data Matching",
          icon = shiny::icon("circle-question")
        ),
        shiny::actionButton(
          inputId = shiny::NS(id, "test_featureextractor_matching"),
          label = "Test TEFeatureExtractor",
          icon = shiny::icon("circle-question")
        )
      ),
      # Main Page---------------------------------------------------------------
      # Content depends in the selected base model
      bslib::layout_column_wrap(
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
        bslib::card(
          bslib::card_header("Target Data"),
          bslib::card_body(
            shinyFiles::shinyFilesButton(
              id = shiny::NS(id, "button_select_target_data"),
              multiple = FALSE,
              label = "Choose Target Data",
              title = "Please choose a file",
              icon = shiny::icon("file")
            ),
            shiny::textInput(
              inputId = shiny::NS(id, "target_dir"),
              label = shiny::tags$p(shiny::icon("folder"), "Path"),
              width = "100%"
            ),
            bslib::layout_column_wrap(
              shiny::uiOutput(outputId = shiny::NS(id, "summary_data_targets")),
              shiny::uiOutput(outputId = shiny::NS(id, "output_target_levels"))
            )
          )
        )
      ),
      # FeatureExtractor
      bslib::card(
        bslib::card_header(
          "Feature Extractor"
        ),
        bslib::card_body(
          shinyFiles::shinyDirButton(
            id = shiny::NS(id, "button_select_feature_extractor"),
            label = "Choose TEFeatureExtractor",
            title = "Please choose a folder",
            icon = shiny::icon("folder-open")
          ),
          shiny::textInput(
            inputId = shiny::NS(id, "feature_extractor_dir"),
            label = shiny::tags$p(shiny::icon("folder"), "Path")
          )
        )
      ),
      # Main config Cards
      shinycssloaders::withSpinner(
        shiny::uiOutput(outputId = shiny::NS(id, "model_configuration"))
      ),
      shinycssloaders::withSpinner(
        shiny::uiOutput(outputId = shiny::NS(id, "training_setup"))
      )
    )
  )
}

#' @title Server function for: graphical user interface for classifiers - create
#' @description Functions generates the functionality of a page on the server.
#'
#' @param id `string` determining the id for the namespace.
#' @param log_dir `string` Path to the directory where the log files should be stored.
#' @param volumes `vector` containing a named vector of available volumes.
#' @return This function does nothing return. It is used to create the functionality of a page for a shiny app.
#'
#' @family studio_gui_page_classifier_create
#' @keywords internal
#' @noRd
#'
Classifiers_Create_Server <- function(id, log_dir, volumes) {
  shiny::moduleServer(id, function(input, output, session) {
    # global variables-----------------------------------------------------------
    ns <- session$ns
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

    # Target Data
    shinyFiles::shinyFileChoose(
      input = input,
      id = "button_select_target_data",
      roots = volumes,
      filetypes = c("csv", "rda", "rdata", "xlsx")
    )

    shiny::observeEvent(input$button_select_target_data,
      {
        tmp_file_path <- shinyFiles::parseFilePaths(volumes, input$button_select_target_data)
        if (nrow(tmp_file_path) > 0) {
          shiny::updateTextInput(
            inputId = "target_dir",
            value = tmp_file_path[[1, "datapath"]]
          )
        } else {
          shiny::updateTextInput(
            inputId = "target_dir",
            value = ""
          )
        }
      },
      ignoreNULL = FALSE
    )

    path_to_target_data <- shiny::eventReactive(input$target_dir, {
      if (input$target_dir != "") {
        return(input$target_dir)
      } else {
        return(NULL)
      }
    })

    data_targets <- shiny::reactive({
      if (!is.null(path_to_target_data())) {
        return(load_and_check_target_data(path_to_target_data()))
      } else {
        return(NULL)
      }
    })

    # Box for model configuration------------------------------------------------
    output$model_configuration <- shiny::renderUI({
      config_box <- create_widget_card(
        id = id,
        object_class = input$classifier_type,
        method = "configure",
        box_title = "Model Configuration"
      )
    })
    # Box for training set up---------------------------------------------------
    output$training_setup <- shiny::renderUI({
      config_box <- create_widget_card(
        id = id,
        object_class = input$classifier_type,
        method = "train",
        box_title = "Training SetUp"
      )
    })

    # FeatureExtractor
    shinyFiles::shinyDirChoose(
      input = input,
      id = "button_select_feature_extractor",
      roots = volumes,
      allowDirCreate = FALSE
    )
    shiny::observeEvent(input$button_select_feature_extractor, {
      path <- shinyFiles::parseDirPath(volumes, input$button_select_feature_extractor)
      shiny::updateTextInput(
        inputId = "feature_extractor_dir",
        value = path
      )
    })
    path_to_feature_extractor <- shiny::eventReactive(input$feature_extractor_dir, {
      if (input$feature_extractor_dir != "") {
        return(input$feature_extractor_dir)
      } else {
        return(NULL)
      }
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

      # Check vor valid arguments
      if (identical(as.double(input$loss_alpha), numeric(0))) {
        loss_alpha <- NULL
      } else {
        loss_alpha <- as.double(input$loss_alpha)
      }
      if (identical(as.double(input$loss_margin), numeric(0))) {
        loss_margin <- NULL
      } else {
        loss_margin <- as.double(input$loss_margin)
      }

      # Check for errors
      errors <- check_errors_create_classifier(
        classifier_type = input$classifier_type,
        destination_path = input$save_modal_directory_path,
        folder_name = input$save_modal_folder_name,
        path_to_embeddings = path_to_embeddings(),
        path_to_target_data = path_to_target_data(),
        path_to_feature_extractor = path_to_feature_extractor(),
        model_name = input$name,
        model_label = input$label,
        use_sc = input$use_sc,
        sc_min_k = input$sc_min_k,
        sc_max_k = input$sc_max_k,
        use_pl = input$use_pl,
        pl_min = input$pl_min,
        pl_max = input$pl_max,
        pl_anchor = input$pl_anchor
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
          ExtendedTask_type = "classifier",
          ExtendedTask_arguments = list(
            configure = summarize_args_for_long_task(
              input = input,
              object_class = input$classifier_type,
              method = "configure",
              path_args = list(
                path_to_embeddings = path_to_embeddings(),
                path_to_target_data = NULL,
                path_to_feature_extractor = path_to_feature_extractor(),
                destination_path = input$save_modal_directory_path,
                folder_name = input$save_modal_folder_name
              ),
              override_args = list(
                sustain_track = TRUE
              ),
              meta_args = list(
                py_environment_type = get_py_env_type(),
                py_env_name = get_py_env_name(),
                target_data_column = input$data_target_column,
                object_class = input$classifier_type
              )
            ),
            train = summarize_args_for_long_task(
              input = input,
              object_class = input$classifier_type,
              method = "train",
              path_args = list(
                path_to_embeddings = path_to_embeddings(),
                path_to_target_data = path_to_target_data(),
                path_to_feature_extractor = path_to_feature_extractor(),
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
                target_data_column = input$data_target_column,
                object_class = input$classifier_type
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

    # Target data
    output$summary_data_targets <- shiny::renderUI({
      target_data <- data_targets()
      # shiny::req(target_data)
      if (!is.null(target_data)) {
        column_names <- colnames(target_data)
        column_names <- setdiff(x = column_names, y = c("id", "text"))
        ui <- list(
          bslib::value_box(
            value = nrow(target_data),
            title = "Number of Cases",
            showcase = shiny::icon("list")
          ),
          shiny::selectInput(
            inputId = ns("data_target_column"),
            label = "Select a Column",
            choices = column_names
          ),
          shiny::tableOutput(outputId = ns("data_target_abs_freq"))
        )
      } else {
        return(NULL)
      }
    })

    output$data_target_abs_freq <- shiny::renderTable({
      # shiny::req(data_targets())
      relevant_data <- data_targets()
      relevant_data <- relevant_data[input$data_target_column]
      if (nrow(relevant_data) > 0) {
        return(table(relevant_data, useNA = "always"))
      } else {
        return(NULL)
      }
    })

    target_levels_unsorted <- shiny::reactive({
      if (!is.null(data_targets())) {
        relevant_data <- data_targets()
        relevant_data <- relevant_data[input$data_target_column]
        if (nrow(relevant_data) > 0) {
          target_levels <- names(table(relevant_data, useNA = "no"))
          return(target_levels)
        } else {
          return(NULL)
        }
      } else {
        return(NULL)
      }
    })

    output$output_target_levels <- shiny::renderUI({
      if (!is.null(target_levels_unsorted())) {
        return(
          sortable::rank_list(
            text = "Please select the order of categories/classes.",
            labels = target_levels_unsorted(),
            input_id = session$ns("target_levels"),
            class = c("default-sortable", "aifeducation-sortable")
          )
        )
      } else {
        return(NULL)
      }
    })

    # Test Data matching--------------------------------------------------------
    # Data Sets
    shiny::observeEvent(input$test_data_matching,
      {
        cond_1 <- (!is.null(data_embeddings()))
        cond_2 <- (!is.null(data_targets()))

        if (cond_1 & cond_2) {
          embeddings <- data_embeddings()
          targets <- data_targets()[input$target_data_column]
          ids <- embeddings$get_ids()
          matched_cases <- intersect(
            x = ids,
            y = rownames(targets)
          )
          n_matched_cases <- length(matched_cases)
          shinyWidgets::show_alert(
            title = "Matching Results",
            text = paste(
              n_matched_cases,
              "out of",
              embeddings$n_rows(),
              "could be matched"
            ),
            type = "info"
          )
        } else {
          display_errors(
            title = "Error",
            size = "l",
            easy_close = TRUE,
            error_messages = "Embeddings and target data must be selected before matching is possible."
          )
        }
      },
      ignoreInit = TRUE
    )


    # Error handling-----------------------------------------------------------


    #--------------------------------------------------------------------------
  })
}
