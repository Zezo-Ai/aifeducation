#' @title Graphical user interface for base models - create
#' @description Functions generates the page for using the BaseModels.
#'
#' @param id `string` determining the id for the namespace.
#' @return This function does nothing return. It is used to build a page for a shiny app.
#'
#' @family studio_gui_page_base_model_create
#' @keywords internal
#' @noRd
#'
BaseModel_Create_UI <- function(id) {
  ns <- shiny::NS(id)

  # Sidebar------------------------------------------------------------------
  shiny::tagList(
    bslib::page_sidebar(
      # Sidebar------------------------------------------------------------------
      sidebar = bslib::sidebar(
        position = "left",
        shiny::tags$h3("Control Panel"),
        shiny::tags$hr(),
        shiny::selectInput(
          inputId = shiny::NS(id, "base_model_type"),
          choices = get_entry_from_BaseModelsIndex("class_name"),
          label = "Base Model Type"
        ),
        shiny::selectInput(
          inputId = shiny::NS(id, "tokenizer_model_type"),
          choices = setdiff(x = unname(unlist(TokenizerIndex)), y = "HuggingFaceTokenizer"),
          label = "Tokenizer Model Type"
        ),
        shiny::tags$hr(),
        shinyFiles::shinyDirButton(
          id = shiny::NS(id, "start_SaveModal"),
          label = "Create Model",
          title = "Choose Destination",
          icon = shiny::icon("floppy-disk")
        )
      ),
      # Main Page---------------------------------------------------------------
      # bslib::layout_column_wrap(
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
          shinycssloaders::withSpinner(
            shiny::uiOutput(outputId = shiny::NS(id, "summary_data_raw_texts"))
          )
        )
      ),
      # Tokenizer
      bslib::card(
        bslib::card_header("Tokenizer"),
        bslib::card_body(
          bslib::layout_column_wrap(
            heights_equal = "row",
            shiny::uiOutput(outputId = shiny::NS(id, "base_model_create_tokenizer")),
            shiny::uiOutput(outputId = shiny::NS(id, "base_model_train_tokenizer")),
            bslib::card(
              bslib::card_header("Statistics"),
              bslib::card_body(
                shiny::actionButton(
                  inputId = shiny::NS(id, "calc_tok_statistics"),
                  label = "Calculate Statistics",
                  icon = shiny::icon("paper-plane")
                ),
                shiny::uiOutput(outputId = shiny::NS(id, "values_toc_statistics"))
              )
            )
          )
        )
      ),
      # Base Model
      shiny::uiOutput(outputId = shiny::NS(id, "base_model_create"))
    )
    # )
  )
}

#' @title Server function for: graphical user interface for base models - create.
#' @description Functions generates the functionality of a page on the server.
#'
#' @param id `string` determining the id for the namespace.
#' @param log_dir `string` Path to the directory where the log files should be stored.
#' @param volumes `vector` containing a named vector of available volumes.
#' @param sustain_tracking `list` with the sustainability tracking parameters.
#' @return This function does nothing return. It is used to create the functionality of a page for a shiny app.
#'
#' @family studio_gui_page_base_model_create
#' @keywords internal
#' @noRd
#'
BaseModel_Create_Server <- function(id, log_dir, volumes) {
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
        return(load_and_check_dataset_raw_texts(path_to_raw_texts()))
      } else {
        return(NULL)
      }
    })

    # Card for the tokenizer--------------------------------------------------
    output$base_model_create_tokenizer <- shiny::renderUI({
      config_box <- create_widget_card(
        id = id,
        object_class = input$tokenizer_model_type,
        method = "configure",
        box_title = "Configuration"
      )
    })

    output$base_model_train_tokenizer <- shiny::renderUI({
      config_box <- create_widget_card(
        id = id,
        object_class = input$tokenizer_model_type,
        method = "train",
        box_title = "Training Settings"
      )
    })

    # Card of Model Configuration------------------------------------------------
    output$base_model_create <- shiny::renderUI({
      config_box <- create_widget_card(
        id = id,
        object_class = input$base_model_type,
        method = "configure",
        box_title = "Base Model Configuration"
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

    # Tokenizer statistics--------------------------------------------------------------------------
    tok_statistics <- shiny::eventReactive(input$calc_tok_statistics, {
      if (!is.null(data_raw_texts())) {
        display_processing()

        tokenizer <- create_object(input$tokenizer_model_type)

        do.call(
          what = tokenizer$configure,
          args = extract_args_from_input(input = input, arg_names = rlang::fn_fmls_names(tokenizer$configure))
        )

        tokenizer$train(
          text_dataset = data_raw_texts(),
          statistics_max_tokens_length = 512,
          sustain_track = FALSE,
          sustain_iso_code = NULL,
          sustain_region = NULL,
          sustain_interval = 15,
          trace = FALSE
        )

        shiny::removeModal()
        shiny::removeModal()
        return(tokenizer$get_tokenizer_statistics())
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
        start_and_monitor_long_task(
          id = id,
          ExtendedTask_type = "create_transformer",
          ExtendedTask_arguments = list(
            # Tokenizer config
            tok_configure = summarize_args_for_long_task(
              input = input,
              object_class = input$tokenizer_model_type,
              method = "configure",
              path_args = list(
                path_to_embeddings = NULL,
                path_to_target_data = NULL,
                path_to_textual_dataset = path_to_raw_texts(),
                path_to_feature_extractor = NULL,
                destination_path = input$save_modal_directory_path,
                folder_name = input$save_modal_folder_name
              ),
              override_args = list(
                model_dir = paste0(input$save_modal_directory_path, "/", input$save_modal_folder_name),
                sustain_track = TRUE,
                log_dir = log_dir,
                trace = FALSE,
                pytorch_safetensors = TRUE
              ),
              meta_args = list(
                py_environment_type = get_py_env_type(),
                py_env_name = get_py_env_name(),
                object_class = input$tokenizer_model_type
              )
            ),
            # Tokenizer Train
            tok_train = summarize_args_for_long_task(
              input = input,
              object_class = input$tokenizer_model_type,
              method = "train",
              path_args = list(
                path_to_embeddings = NULL,
                path_to_target_data = NULL,
                path_to_textual_dataset = path_to_raw_texts(),
                path_to_feature_extractor = NULL,
                destination_path = input$save_modal_directory_path,
                folder_name = input$save_modal_folder_name
              ),
              override_args = list(
                model_dir = paste0(input$save_modal_directory_path, "/", input$save_modal_folder_name),
                sustain_track = TRUE,
                log_dir = log_dir,
                trace = FALSE,
                pytorch_safetensors = TRUE
              ),
              meta_args = list(
                py_environment_type = get_py_env_type(),
                py_env_name = get_py_env_name(),
                object_class = input$tokenizer_model_type
              )
            ),
            # Base Model config
            bm_configure = summarize_args_for_long_task(
              input = input,
              object_class = input$base_model_type,
              method = "configure",
              path_args = list(
                path_to_embeddings = NULL,
                path_to_target_data = NULL,
                path_to_textual_dataset = path_to_raw_texts(),
                path_to_feature_extractor = NULL,
                destination_path = input$save_modal_directory_path,
                folder_name = input$save_modal_folder_name
              ),
              override_args = list(
                model_dir = paste0(input$save_modal_directory_path, "/", input$save_modal_folder_name),
                sustain_track = TRUE,
                log_dir = log_dir,
                trace = FALSE,
                pytorch_safetensors = TRUE
              ),
              meta_args = list(
                py_environment_type = get_py_env_type(),
                py_env_name = get_py_env_name(),
                object_class = input$base_model_type
              )
            )
          ),
          log_path = log_path,
          pgr_use_middle = TRUE,
          pgr_use_bottom = FALSE,
          pgr_use_graphic = FALSE,
          update_intervall = 2,
          success_type = "classifier"
        )
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
  })
}
