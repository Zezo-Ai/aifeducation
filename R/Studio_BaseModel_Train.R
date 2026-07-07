#' @title Graphical user interface for base models - train
#' @description Functions generates the page for using the BaseModels.
#'
#' @param id `string` determining the id for the namespace.
#' @return This function does nothing return. It is used to build a page for a shiny app.
#'
#' @family studio_gui_page_base_model_train
#' @keywords internal
#' @noRd
#'
BaseModel_Train_UI <- function(id) {
  ns <- shiny::NS(id)

  shiny::tagList(
    bslib::page_sidebar(
      # Sidebar------------------------------------------------------------------
      sidebar = bslib::sidebar(
        position = "left",
        shiny::tags$h3("Control Panel"),
        shiny::tags$hr(),
        shinyFiles::shinyDirButton(
          id = shiny::NS(id, "start_SaveModal"),
          label = "Start Training",
          title = "Choose Destination",
          icon = shiny::icon("paper-plane")
        )
      ),
      # Main Page---------------------------------------------------------------
      bslib::layout_column_wrap(
        # Base Model
        bslib::card(
          bslib::card_header("Base Model"),
          bslib::card_body(
            shinyFiles::shinyDirButton(
              id = shiny::NS(id, "button_select_base_model"),
              label = "Select a Base Model",
              title = "Please choose a folder",
              icon = shiny::icon("folder-open")
            ),
            shiny::textInput(
              inputId = shiny::NS(id, "base_model_dir"),
              label = shiny::tags$p(shiny::icon("folder"), "Path"),
              width = "100%"
            ),
            shiny::uiOutput(outputId = shiny::NS(id, "summary_base_model"))
          )
        ),
        # Raw Texts
        bslib::card(
          bslib::card_header("Input Data"),
          bslib::card_body(
            shinyFiles::shinyDirButton(
              id = shiny::NS(id, "button_select_dataset_for_raw_texts"),
              label = "Choose Collection of Raw Texts",
              title = "Please choose a folder",
              icon = shiny::icon("folder-open")
            ),
            shiny::textInput(
              inputId = shiny::NS(id, "raw_text_dir"),
              label = shiny::tags$p(shiny::icon("folder"), "Path"),
              width = "100%"
            ),
            shiny::uiOutput(outputId = shiny::NS(id, "summary_data_raw_texts"))
          )
        )
      ),
      #------------------------------------------------------------------------
      bslib::layout_column_wrap(
        # Tokenizer and Text Statistics calculation
        bslib::card(
          bslib::card_header("Tokenizer and Data Statistics"),
          bslib::card_body(
            shiny::actionButton(
              inputId = shiny::NS(id, "calc_tok_statistics"),
              label = "Calculate Data Statistics",
              icon = shiny::icon("paper-plane")
            ),
            shiny::uiOutput(outputId = shiny::NS(id, "values_toc_statistics"))
          )
        ),
        # Parameter and flop calculation
        bslib::card(
          bslib::card_header("Parameter and FLOPS"),
          bslib::card_body(
            shiny::actionButton(
              inputId = shiny::NS(id, "calc_param_flops_statistics"),
              label = "Calculate Flops Statistics",
              icon = shiny::icon("paper-plane")
            ),
            shiny::uiOutput(outputId = shiny::NS(id, "values_param_flops_statistics"))
          )
        )
      ),
      # Main config Cards
      shiny::uiOutput(outputId = shiny::NS(id, "base_model_train"))
    )
    # )
  )
}

#' @title Server function for: graphical user interface for base models - train
#' @description Functions generates the functionality of a page on the server.
#'
#' @param id `string` determining the id for the namespace.
#' @param log_dir `string` Path to the directory where the log files should be stored.
#' @param volumes `vector` containing a named vector of available volumes.
#' @param sustain_tracking `list` with the sustainability tracking parameters.
#' @return This function does nothing return. It is used to create the functionality of a page for a shiny app.
#'
#' @family studio_gui_page_base_model_train
#' @keywords internal
#' @noRd
#'
BaseModel_Train_Server <- function(id, log_dir, volumes) {
  shiny::moduleServer(id, function(input, output, session) {
    # global variables-----------------------------------------------------------
    log_path <- paste0(log_dir, "/aifeducation_state.log")

    # File system management----------------------------------------------------
    # Raw Texts
    shinyFiles::shinyDirChoose(
      input = input,
      id = "button_select_dataset_for_raw_texts",
      roots = volumes,
      # session = session,
      allowDirCreate = FALSE
    )
    shiny::observeEvent(input$button_select_dataset_for_raw_texts, {
      path <- shinyFiles::parseDirPath(volumes, input$button_select_dataset_for_raw_texts)
      shiny::updateTextInput(
        inputId = "raw_text_dir",
        value = path
      )
    })

    path_to_raw_texts <- shiny::eventReactive(input$raw_text_dir, {
      if (input$raw_text_dir != "") {
        return(input$raw_text_dir)
      } else {
        return(NULL)
      }
    })

    data_raw_texts <- shiny::reactive({
      if (!is.null(path_to_raw_texts())) {
        shinyWidgets::show_alert(
          title = "Loading",
          text = "Please wait",
          type = "info",
          closeOnClickOutside = FALSE,
          showCloseButton = FALSE,
          btn_labels = NA
        )
        shinyWidgets::closeSweetAlert()
        shinyWidgets::closeSweetAlert()
        return(load_and_check_dataset_raw_texts(path_to_raw_texts()))
      } else {
        return(NULL)
      }
    })

    # Base model
    shinyFiles::shinyDirChoose(
      input = input,
      id = "button_select_base_model",
      roots = volumes,
      # session = session,
      allowDirCreate = FALSE
    )
    shiny::observeEvent(input$button_select_base_model, {
      path <- shinyFiles::parseDirPath(volumes, input$button_select_base_model)
      shiny::updateTextInput(
        inputId = "base_model_dir",
        value = path
      )
    })

    path_to_base_model <- shiny::eventReactive(input$base_model_dir, {
      if (input$base_model_dir != "") {
        return(input$base_model_dir)
      } else {
        return(NULL)
      }
    })

    base_model <- shiny::reactive({
      if (!is.null(path_to_base_model())) {
        shinyWidgets::show_alert(
          title = "Loading",
          text = "Please wait",
          type = "info",
          closeOnClickOutside = FALSE,
          showCloseButton = FALSE,
          btn_labels = NA
        )
        model <- load_and_check_base_model(path_to_base_model())
        shinyWidgets::closeSweetAlert()
        shinyWidgets::closeSweetAlert()
        return(model)
      } else {
        return(NULL)
      }
    })

    # Detect type of model
    base_model_type <- shiny::reactive({
      model <- base_model()
      if (!is.null(model)) {
        type <- unlist(
          intersect(
            x = class(model),
            y = get_entry_from_BaseModelsIndex("class_name")
          )
        )
        return(type)
      } else {
        return(NULL)
      }
    })

    # Tokenizer statistics--------------------------------------------------------------------------
    tok_statistics <- shiny::eventReactive(input$calc_tok_statistics, {
      if (!is.null(base_model()) & !is.null(data_raw_texts())) {
        display_processing()

        tokenizer <- base_model()$Tokenizer

        shiny::removeModal()
        shiny::removeModal()
        return(tokenizer$calculate_statistics(
          text_dataset = data_raw_texts(),
          statistics_max_tokens_length = 512,
          step = "creation"
        ))
      } else {
        return(NULL)
      }
    })

    output$values_toc_statistics <- shiny::renderUI({
      table <- tok_statistics()
      if (!is.null(table)) {
        return(
          bslib::value_box(
            title = "Tokens per Word",
            value = table$mu_g,
            shiny::tags$p(
              "Total Words:", format(x = table$n_words, big.mark = ",")
            ),
            shiny::tags$p(
              "Total Tokens: ", format(x = table$n_tokens, big.mark = ",")
            )
          )
        )
      } else {
        return(NULL)
      }
    })

    # FLOPS and Parameter statistics--------------------------------------------------------------------------
    flop_statistics <- shiny::eventReactive(input$calc_param_flops_statistics, {
      if (!is.null(tok_statistics()) & !is.null(base_model())) {
        display_processing()

        flops_statistics <- base_model()$calc_flops_architecture_based(
          batch_size = input$batch_size,
          n_batches = ceiling(tok_statistics()$n_tokens / input$batch_size),
          n_epochs = input$n_epoch
        )

        shiny::removeModal()
        shiny::removeModal()
        return(flops_statistics)
      } else {
        return(NULL)
      }
    })

    output$values_param_flops_statistics <- shiny::renderUI({
      table <- flop_statistics()
      if (!is.null(table)) {
        return(
          bslib::value_box(
            title = "Parameter",
            value = table$n_parameter,
            shiny::tags$p("N Epochs:", input$n_epoch),
            shiny::tags$p("Total Flops (factor 2):", format(x = table$flops_bp_2, scientific = TRUE)),
            shiny::tags$p("Total Flops (factor 3):", format(x = table$flops_bp_3, scientific = TRUE))
          )
        )
      } else {
        return(NULL)
      }
    })

    # Card of Model Configuration------------------------------------------------
    output$base_model_train <- shiny::renderUI({
      if (!is.null(base_model_type())) {
        config_box <- create_widget_card(
          id = id,
          object_class = base_model_type(),
          method = "train",
          box_title = "Training Settings"
        )
        return(config_box)
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

    # Display Data Summary------------------------------------------------------
    output$summary_data_raw_texts <- shiny::renderUI({
      data_set_raw_texts <- data_raw_texts()
      # shiny::req(data_set_raw_texts)
      if (!is.null(data_set_raw_texts)) {
        ui <- create_data_raw_texts_description(data_set_raw_texts)
        return(ui)
      } else {
        return(NULL)
      }
    })

    output$summary_base_model <- shiny::renderUI({
      model <- base_model()
      # shiny::req(data_set_raw_texts)
      if (!is.null(model)) {
        ui <- create_data_base_model_description(model)
        return(ui)
      } else {
        return(NULL)
      }
    })

    # Start creation-------------------------------------------------------------
    shiny::observeEvent(input$save_modal_button_continue, {
      # Remove Save Modal
      shiny::removeModal()

      # Check for errors
      errors <- check_error_base_model_create_or_train(
        destination_path = input$save_modal_directory_path,
        folder_name = input$save_modal_folder_name,
        path_to_raw_texts = path_to_raw_texts()
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
        print(path_to_raw_texts())
        start_and_monitor_long_task(
          id = id,
          ExtendedTask_type = "train_transformer",
          ExtendedTask_arguments = list(
            train = summarize_args_for_long_task(
              input = input,
              object_class = base_model_type(),
              method = "train",
              path_args = list(
                path_to_embeddings = NULL,
                path_to_target_data = NULL,
                path_to_textual_dataset = path_to_raw_texts(),
                path_to_base_model = path_to_base_model(),
                path_to_feature_extractor = NULL,
                destination_path = input$save_modal_directory_path,
                folder_name = input$save_modal_folder_name
              ),
              override_args = list(
                output_dir = paste0(input$save_modal_directory_path, "/", input$save_modal_folder_name),
                model_dir_path = path_to_base_model(),
                sustain_track = TRUE,
                log_dir = log_dir,
                trace = FALSE,
                pytorch_safetensors = TRUE
              ),
              meta_args = list(
                py_environment_type = get_py_env_type(),
                py_env_name = get_py_env_name(),
                object_class = base_model_type()
              )
            )
          ),
          log_path = log_path,
          pgr_use_middle = TRUE,
          pgr_use_bottom = TRUE,
          pgr_use_graphic = TRUE,
          update_intervall = 2,
          success_type = "train_transformer"
        )
      }
    })
  })
}
