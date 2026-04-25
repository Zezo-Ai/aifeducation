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

#' @title Graphical user interface for displaying the training history of an object.
#' @description Functions generates the tab within a page for displaying the training history of an object.
#'
#' @param id `string` determining the id for the namespace.
#' @return This function does nothing return. It is used to build a page for a shiny app.
#'
#' @family studio_gui_training
#' @keywords internal
#' @noRd
#'
Training_UI <- function(id) {
  bslib::page(
    bslib::page_sidebar(
      sidebar = bslib::sidebar(
        position = "right",
        shiny::sliderInput(
          inputId = shiny::NS(id, "text_size"),
          label = "Text Size",
          min = 1,
          max = 20,
          step = 0.5,
          value = 12
        ),
        shiny::numericInput(
          inputId = shiny::NS(id, "y_min"),
          label = "Y Min",
          value = NA
        ),
        shiny::numericInput(
          inputId = shiny::NS(id, "y_max"),
          label = "Y Max",
          value = NA
        ),
        shiny::numericInput(
          inputId = shiny::NS(id, "x_min"),
          label = "X Min",
          value = NA
        ),
        shiny::numericInput(
          inputId = shiny::NS(id, "x_max"),
          label = "X Max",
          value = NA
        ),
        shinyWidgets::materialSwitch(
          inputId = shiny::NS(id, "ind_best_model"),
          label = "Indicate Best States",
          value = TRUE,
          status = "primary"
        ),
        shiny::uiOutput(
          outputId = shiny::NS(id, "classifier_specific")
        )
      ),
      shiny::plotOutput(
        outputId = shiny::NS(id, "training_plot")
      )
    )
  )
}

#' @title Server function for: graphical user interface for displaying the training history of an object.
#' @description Functions generates the functionality of a page on the server.
#'
#' @param id `string` determining the id for the namespace.
#' @param model Model used for inference.
#' @return This function does nothing return. It is used to create the functionality of a page for a shiny app.
#'
#' @importFrom rlang .data
#'
#' @family studio_gui_training
#' @keywords internal
#' @noRd
#'
Training_Server <- function(id, model) {
  shiny::moduleServer(id, function(input, output, session) {
    # global variables-----------------------------------------------------------
    ns <- session$ns

    # Control widgets for classifiers--------------------------------------------
    output$classifier_specific <- shiny::renderUI({
      if (inherits(model(), "ClassifiersBasedOnTextEmbeddings")) {
        ui <- shiny::tagList(
          shinyWidgets::materialSwitch(
            inputId = ns("ind_selected_model"),
            label = "Indicate Final States",
            value = FALSE,
            status = "primary"
          ),
          shinyWidgets::radioGroupButtons(
            inputId = ns("training_phase"),
            label = "Training Phase",
            choices = list(
              "Folds" = FALSE,
              "Final" = TRUE
            )
          ),
          shiny::selectInput(
            inputId = ns("measure"),
            label = "Measures",
            choices = list(
              "Loss" = "loss",
              "Average Iota" = "avg_iota",
              "Accuracy" = "accuracy",
              "Balanced Accuracy" = "balanced_accuracy"
            )
          ),
          shinyWidgets::materialSwitch(
            inputId = ns("training_min_max"),
            label = "Add Min/Max",
            value = FALSE,
            status = "primary"
          ),
          shiny::uiOutput(
            outputId = ns("widget_classifier_pl_step")
          )
        )
        return(ui)
      } else if (inherits(model(), "TEFeatureExtractor")) {
        ui <- shiny::tagList(
          shinyWidgets::radioGroupButtons(
            inputId = ns("measure"),
            label = "Measures",
            choices = list(
              "Loss" = "loss"
            )
          )
        )
      } else {
        return(NULL)
      }
    })


    output$widget_classifier_pl_step <- shiny::renderUI({
      if (any(get_TEClassifiers_class_names() %in% class(model))) {
        if (model()$last_training$config$use_pl == TRUE) {
          n_steps <- model()$last_training$config$pl_max_steps
          return(
            shinyWidgets::radioGroupButtons(
              inputId = ns("classifier_pl_step"),
              label = "Steps during Pseudo Labeling",
              choices = seq.int(from = 1, to = n_steps)
            )
          )
        } else {
          return(NULL)
        }
      } else {
        return(NULL)
      }
    })

    # plot------------------------------------------------
    output$training_plot <- shiny::renderPlot(
      {
        shiny::req(model)
        if (inherits(model(), "TextEmbeddingModel")) {
          plot <- model()$BaseModel$plot_training_history(
            y_min = input$y_min,
            y_max = input$y_max,
            x_min = input$x_min,
            x_max = input$x_max,
            ind_best_model = input$ind_best_model,
            text_size = input$text_size
          )
        } else if ((inherits(model(), "BaseModelCore"))) {
          plot <- model()$plot_training_history(
            y_min = input$y_min,
            y_max = input$y_max,
            x_min = input$x_min,
            x_max = input$x_max,
            ind_best_model = input$ind_best_model,
            text_size = input$text_size
          )
        } else if (inherits(model(), "ClassifiersBasedOnTextEmbeddings")) {
          # Necessary input
          shiny::req(input$measure)

          if (input$training_phase == "TRUE") {
            bool_training_phase <- TRUE
          } else {
            bool_training_phase <- FALSE
          }

          plot <- model()$plot_training_history(
            y_min = input$y_min,
            y_max = input$y_max,
            x_min = input$x_min,
            x_max = input$x_max,
            ind_best_model = input$ind_best_model,
            ind_selected_model = input$ind_selected_model,
            final_training = bool_training_phase,
            pl_step = input$classifier_pl_step,
            measure = input$measure,
            add_min_max = input$training_min_max,
            text_size = input$text_size
          )
        } else if (inherits(model(), "TEFeatureExtractor")) {
          plot <- model()$plot_training_history(
            y_min = input$y_min,
            y_max = input$y_max,
            x_min = input$x_min,
            x_max = input$x_max,
            ind_best_model = input$ind_best_model,
            text_size = input$text_size
          )
        }
        return(plot)
      },
      res = 72 * 2
    )
  })
}
