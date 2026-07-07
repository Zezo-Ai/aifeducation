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

#' @title Show error modal
#' @description Function for displaying a modal that reports errors to the user.
#'
#' @param title `string` Title of the modal.
#' @param size `string` Size of the modal. Possible are `"m"`, `"s"`, `"l"`, and `"xl"`.
#' @param easy_close `bool` If `TRUE`, the modal dialog can be dismissed by clicking outside the dialog box, or be
#'   pressing the Escape key. If `FALSE` the modal must be dismissed by clicking on a modalButton or from a call
#'   removeModal on the server.
#' @param error_messages `shiny::tagList` containing the html elements to display on the modal.
#'
#' @return Function does nothing return. It displays a modal.
#'
#' @family studio_modals
#' @keywords internal
#' @noRd
#'
display_errors <- function(
  title = "Error",
  size = "l",
  easy_close = TRUE,
  error_messages
) {
  error_modal <- shiny::modalDialog(
    title = title,
    size = size,
    easyClose = easy_close,
    footer = shiny::modalButton("Close"),
    shiny::tagList(error_messages)
  )
  shiny::showModal(error_modal)
}

#' @title Show processing modal
#' @description Function for displaying a modal that reports to the user that the app is currently working.
#'
#' @param title `string` Title of the modal.
#' @param size `string` Size of the modal. Possible are `"m"`, `"s"`, `"l"`, and `"xl"`.
#' @param easy_close `bool` If `TRUE`, the modal dialog can be dismissed by clicking outside the dialog box, or be
#'   pressing the Escape key. If `FALSE` the modal must be dismissed by clicking on a modalButton or from a call
#'   removeModal on the server.
#' @param message `shiny::tagList` containing the html elements to display on the modal.
#'
#' @return Function does nothing return. It displays a modal.
#'
#' @family studio_modals
#' @keywords internal
#' @noRd
#'
display_processing <- function(
  title = "Working. Please wait.",
  size = "s",
  easy_close = FALSE,
  message = ""
) {
  if (easy_close == FALSE) {
    footer <- NULL
  } else {
    footer <- shiny::modalButton("Close")
  }

  processing_modal <- shiny::modalDialog(
    title = title,
    size = size,
    easyClose = easy_close,
    footer = footer,
    shiny::tagList(message)
  )
  shiny::showModal(processing_modal)

  # Give system time to display the modal
  Sys.sleep(2)
}


display_processing_alert <- function(text = "Working") {
  shinyWidgets::show_alert(
    title = text,
    text = "Please wait",
    type = "info",
    closeOnClickOutside = FALSE,
    showCloseButton = FALSE,
    btn_labels = NA
  )
  # Give system time to display the modal
  Sys.sleep(2)
}
