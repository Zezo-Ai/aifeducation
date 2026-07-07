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

#' @title Graphical user interface for text embedding models - create
#' @description Functions generates the page for a creating a new [TextEmbeddingModel].
#'
#' @param id `string` determining the id for the namespace.
#' @return This function does nothing return. It is used to build a page for a shiny app.
#'
#' @family studio_gui_page_text_embedding_model_create
#' @keywords internal
#' @noRd
#'
TextEmbeddingModel_Create_UI <- function(id) {
  shiny::tagList(
    bslib::page_sidebar(
      # Sidebar------------------------------------------------------------------
      sidebar = bslib::sidebar(
        position = "left",
        shiny::tags$h3("Control Panel"),
        shinyFiles::shinyDirButton(
          id = shiny::NS(id, "button_select_base_model"),
          label = "Choose a Base Model",
          title = "Please choose a folder",
          icon = shiny::icon("folder-open")
        ),
        shiny::tags$hr(),
        shiny::textInput(
          inputId = shiny::NS(id, "lm_model_label"),
          label = "Label"
        ),
        shiny::textInput(
          inputId = shiny::NS(id, "lm_model_language"),
          label = "Language"
        ),
        shiny::tags$hr(),
        shinyFiles::shinyDirButton(
          id = shiny::NS(id, "start_SaveModal"),
          label = "Save Model",
          title = "Choose Destination",
          icon = shiny::icon("floppy-disk")
        )
      ),
      # Main Page---------------------------------------------------------------
      # Content depends in the selected base model
      bslib::layout_column_wrap(
        bslib::card(
          bslib::card_header(shiny::textOutput(outputId = shiny::NS(id, "path_to_base_model"))),
          bslib::card_body(
            shiny::uiOutput(outputId = shiny::NS(id, "lm_interface_setting"))
          )
        ),
        bslib::card(
          bslib::card_header("Embedding Proberties"),
          bslib::card_body(
            shiny::uiOutput(outputId = shiny::NS(id, "granularity_rate")),
            shiny::uiOutput(outputId = shiny::NS(id, "total_words")),
          )
        ),
        bslib::card(
          bslib::card_header("Chunk Calculation"),
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
            shiny::actionButton(
              inputId = shiny::NS(id, "btn_calc_quantile_chunks"),
              label = shiny::tags$p(shiny::icon("paper-plane"), "Calculate"),
              width = "100%"
            ),
            DT::DTOutput(outputId = shiny::NS(id, "quantile_chunks")),
          )
        )
      )
    )
  )
}

#' @title Server function for: graphical user interface for text embedding models - create
#' @description Functions generates the functionality of a page on the server.
#'
#' @param id `string` determining the id for the namespace.
#' @param log_dir `string` Path to the directory where the log files should be stored.
#' @param volumes `vector` containing a named vector of available volumes.
#' @return This function does nothing return. It is used to create the functionality of a page for a shiny app.
#'
#' @family studio_gui_page_text_embedding_model_create
#' @keywords internal
#' @noRd
#'
TextEmbeddingModel_Create_Server <- function(id, log_dir, volumes) {
  shiny::moduleServer(id, function(input, output, session) {
    # global variables-----------------------------------------------------------
    ns <- session$ns

    # File system management----------------------------------------------------
    shinyFiles::shinyDirChoose(
      input = input,
      id = "button_select_base_model",
      roots = volumes,
      # session = session,
      allowDirCreate = FALSE
    )
    path_to_base_model <- shiny::eventReactive(input$button_select_base_model, {
      path <- shinyFiles::parseDirPath(volumes, input$button_select_base_model)
      return(path)
    })

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

    # Start screen for choosing the location for storing the data set-----------
    # Create Save Modal
    save_modal <- create_save_modal(
      id = id,
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

    shinyFiles::shinyDirChoose(
      input = input,
      id = "button_model_destination_dir",
      roots = volumes,
      # session = session,
      allowDirCreate = TRUE
    )

    shiny::observeEvent(input$button_model_destination_dir, {
      shiny::updateTextInput(
        inputId = ns("destination_dir"),
        value = shinyFiles::parseDirPath(volumes, input$button_model_destination_dir)
      )
    })


    # Load Information on Base Model-----------------------------------------------------------
    # Load the model and extract information
    interface_architecture <- shiny::eventReactive(path_to_base_model(), {
      model_path <- path_to_base_model()

      if (file.exists(file.path(model_path, "r_config_state.rda"))) {
        model <- load_from_disk(model_path)

        model_architecture <- detect_base_model_type(model$get_model()$config)
        max_position_embeddings <- model$get_model_config()$max_position_embeddings
        max_layer <- model$get_n_layers()
        toc_granularity <- model$get_tokenizer_statistics()
      } else {
        path_bin <- file.path(model_path, "pytorch_model.bin")
        path_safetensor <- file.path(model_path, "model.safetensors")

        if (file.exists(path_safetensor)) {
          valid_path <- path_safetensor
        } else if (file.exists(path_bin)) {
          valid_path <- path_bin
        } else {
          valid_path <- NA
        }
        if (!is.na(valid_path)) {
          model <- transformers$AutoModel$from_pretrained(model_path)
          model_architecture <- detect_base_model_type(model)
          max_position_embeddings <- model$config$max_position_embeddings
          if (model_architecture == "funnel") {
            max_layer <- sum(model$config$block_repeats * model$config$block_sizes)
          } else {
            max_layer <- model$config$num_hidden_layers
          }
        } else {
          model_architecture <- NULL
          max_position_embeddings <- NULL
          max_layer <- NULL
          toc_granularity <- NULL
        }
      }
      return(list(model_architecture, max_position_embeddings, max_layer, toc_granularity))
    })

    # Create UI Main Page-----------------------------------------------------------------
    # Card Header
    output$path_to_base_model <- shiny::renderText({
      path_to_base_model()
    })

    # Card body
    output$lm_interface_setting <- shiny::renderUI({
      if (length(interface_architecture()[[2]]) > 0) {
        max_layer_transformer <- interface_architecture()[[3]]

        if (interface_architecture()[[1]] == "funnel") {
          pool_type_choices <- c("CLS")
        } else {
          pool_type_choices <- c("Average", "CLS")
        }

        ui <- shiny::tagList(
          shiny::selectInput(
            inputId = ns("lm_emb_pool_type"),
            label = paste("Pooling Type"),
            choices = pool_type_choices,
            multiple = FALSE
          ),
          shiny::sliderInput(
            inputId = ns("lm_chunks"),
            label = "N Chunks",
            value = 2,
            min = 2,
            max = 50,
            step = 1
          ),
          shiny::sliderInput(
            inputId = ns("lm_max_length"),
            label = paste("Maximal Sequence Length", "(Max:", interface_architecture()[2], ")"),
            value = interface_architecture()[[2]],
            min = 20,
            max = interface_architecture()[[2]],
            step = 1
          ),
          shiny::sliderInput(
            inputId = ns("lm_overlap"),
            label = paste("N Token Overlap", "(Max:", interface_architecture()[2], ")"),
            value = 0,
            min = 0,
            max = interface_architecture()[[2]],
            step = 1
          ),
          shiny::sliderInput(
            inputId = ns("lm_insert_mask_tokens"),
            label = "Insert Mask Tokens",
            value = 0.15,
            min = 0.0,
            max = 0.50,
            step = 0.01
          ),
          shiny::sliderInput(
            inputId = ns("lm_emb_layers"),
            label = "Layers for Embeddings",
            value = c(
              max(1, floor(0.5 * max_layer_transformer)),
              max(1, floor(2 / 3 * max_layer_transformer))
            ),
            min = 1,
            max = max_layer_transformer,
            step = 1
          )
        )
        return(ui)
      } else {
        return(NULL)
      }
    })

    # Embeddings Proberties
    output$granularity_rate <- shiny::renderUI({
      table <- interface_architecture()[[4]]
      if (!is.null_or_na(table)) {
        if (!is.na(table$mu_g)) {
          return(
            bslib::value_box(
              title = "Tokens per Word",
              value = table$mu_g
            )
          )
        } else {
          return(
            bslib::value_box(
              title = "Tokens per Word",
              value = "Not Available"
            )
          )
        }
      } else {
        return(NULL)
      }
    })
    output$total_words <- shiny::renderUI({
      table <- interface_architecture()[[4]]
      if (!is.null_or_na(table)) {
        if (!is.na(table$mu_g)) {
          return(
            bslib::value_box(
              title = "Estimated Average Total Number of Words",
              value = floor(((input$lm_max_length - input$lm_overlap) * (input$lm_chunks - 1) + input$lm_max_length) / table$mu_g)
            )
          )
        } else {
          return(
            bslib::value_box(
              title = "Estimated Average Total Number of Tokens",
              value = floor((input$lm_max_length - input$lm_overlap) * (input$lm_chunks - 1) + input$lm_max_length)
            )
          )
        }
      } else {
        return(NULL)
      }
    })

    # Chunk quantile
    quantile_table <- shiny::eventReactive(input$btn_calc_quantile_chunks, {
      model_path <- path_to_base_model()
      path_to_text <- path_to_raw_texts()


      if (!is.null(model_path) && dir.exists(path_to_text)) {
        display_processing()
        raw_texts <- try(load_from_disk(path_to_text), silent = TRUE)
        if (inherits(x = raw_texts, what = c("LargeDataSetForText", "EmbeddedText"))) {
          if (inherits(raw_texts, "EmbeddedText")) {
            raw_texts <- raw_texts$convert_to_LargeDataSetForTextEmbeddings()
          }

          if (file.exists(file.path(model_path, "r_config_state.rda"))) {
            base_model <- load_from_disk(model_path)
          } else {
            base_model <- create_object(model_architecture)
            base_model$create_from_hf(
              model_dir = model_path,
              tokenizer_dir = model_path
            )
          }

          quantile <- base_model$Tokenizer$calc_quantiles(
            text_dataset = raw_texts,
            batch_size = 32L,
            seq_len_tokens = input$lm_max_length,
            token_overlap = input$lm_overlap,
            trace = FALSE
          )

          quantile_table <- as.matrix(
            rbind(
              names(quantile),
              quantile
            )
          )
          quantile_table <- t(quantile_table)
          quantile_table <- unname(quantile_table)
          colnames(quantile_table) <- c("Quantile", "Chunks")

          shiny::removeModal()
          shiny::removeModal()
          return(quantile_table)
        } else {
          shiny::removeModal()
          shiny::removeModal()

          display_errors(
            error_messages = "Object at the given path is not of class
            LargeDataSetForText or EmbeddedText."
          )
        }
      } else {
        return(NULL)
      }
    })

    output$quantile_chunks <- DT::renderDT({
      quantile_table()
    })

    # Save the model to disk----------------------------------------------------
    shiny::observeEvent(input$save_modal_button_continue, {
      # Remove Save Modal
      shiny::removeModal()

      model_architecture <- interface_architecture()[[1]]

      # Check for errors
      errors <- check_errors_text_embedding_model_create(
        destination_path = input$save_modal_directory_path,
        folder_name = input$save_modal_folder_name,
        path_to_base_model = path_to_base_model(),
        interface_architecture = interface_architecture()
      )

      if (length(errors) == 0) {
        shinyWidgets::show_alert(
          title = "Working",
          text = "Please wait",
          type = "info"
        )

        model_path <- path_to_base_model()

        if (file.exists(file.path(model_path, "r_config_state.rda"))) {
          base_model <- load_from_disk(model_path)
        } else {
          base_model <- create_object(model_architecture)
          base_model$create_from_hf(
            model_dir = model_path,
            tokenizer_dir = model_path
          )
        }

        new_interface <- TextEmbeddingModel$new()
        new_interface$configure(
          model_name = NULL,
          model_label = input$lm_model_label,
          model_language = input$lm_model_language,
          max_length = input$lm_max_length,
          overlap = input$lm_overlap,
          chunks = input$lm_chunks,
          emb_layer_min = input$lm_emb_layers[1],
          emb_layer_max = input$lm_emb_layers[2],
          emb_pool_type = input$lm_emb_pool_type,
          emb_insert_mask_tokens = input$lm_insert_mask_tokens,
          base_model = base_model
        )

        save_to_disk(
          object = new_interface,
          dir_path = input$save_modal_directory_path,
          folder_name = input$save_modal_folder_name
        )
        rm(new_interface)
        gc()
        shinyWidgets::closeSweetAlert()
        shiny::removeModal()
      } else {
        display_errors(
          title = "Error",
          size = "l",
          easy_close = TRUE,
          error_messages = errors
        )
      }
    })

    # Error handling-----------------------------------------------------------
    shiny::observe({
      if (!identical(path_to_base_model(), character(0))) {
        if (
          is.null(interface_architecture()[[1]]) &
            is.null(interface_architecture()[[2]])
        ) {
          display_errors(
            title = "Error",
            size = "l",
            easy_close = TRUE,
            error_messages = "There is no model to load in the directory."
          )
        }
      }
    })

    #--------------------------------------------------------------------------
  })
}
