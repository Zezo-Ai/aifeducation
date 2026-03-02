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

#' @title Base class for regular classifiers relying on [EmbeddedText] or [LargeDataSetForTextEmbeddings] as input
#' @description Abstract class for all regular classifiers that use numerical representations of texts instead of words.
#'
#' Objects of this class containing fields and methods used in several other classes in 'AI for Education'.
#'
#' This class is **not** designed for a direct application and should only be used by developers.
#'
#' @return A new object of this class.
#' @family R6 Classes for Developers
#' @export
TEClassifiersBasedOnRegular <- R6::R6Class(
  classname = "TEClassifiersBasedOnRegular",
  inherit = ClassifiersBasedOnTextEmbeddings,
  public = list(
    #---------------------------------------------------------------------------
    #' @description Method for training a neural net.
    #'
    #'   Training includes a routine for early stopping. In the case that loss<0.0001 and Accuracy=1.00 and Average
    #'   Iota=1.00 training stops. The history uses the values of the last trained epoch for the remaining epochs.
    #'
    #'   After training the model with the best values for Average Iota, Accuracy, and Loss on the validation data set
    #'   is used as the final model.
    #'
    #' @param data_embeddings `r get_param_doc_desc("data_embeddings")`
    #' @param data_targets `r get_param_doc_desc("data_targets")`.
    #' @param data_folds `r get_param_doc_desc("data_folds")`
    #' @param data_val_size `r get_param_doc_desc("data_val_size")`
    #' @param loss_cls_fct_name `r get_param_doc_desc("loss_cls_fct_name")`
    #' @param loss_balance_class_weights `r get_param_doc_desc("loss_balance_class_weights")`
    #' @param loss_balance_sequence_length `r get_param_doc_desc("loss_balance_sequence_length")`
    #' @param use_sc `r get_param_doc_desc("use_sc")`
    #' @param sc_method `r get_param_doc_desc("sc_method")`
    #' @param sc_min_k `r get_param_doc_desc("sc_min_k")`
    #' @param sc_max_k `r get_param_doc_desc("sc_max_k")`
    #' @param use_pl `r get_param_doc_desc("use_pl")`
    #' @param pl_max_steps `r get_param_doc_desc("pl_max_steps")`
    #' @param pl_anchor `r get_param_doc_desc("pl_anchor")`
    #' @param pl_max `r get_param_doc_desc("pl_max")`
    #' @param pl_min `r get_param_doc_desc("pl_min")`
    #' @param sustain_track `r get_param_doc_desc("sustain_track")`
    #' @param sustain_iso_code `r get_param_doc_desc("sustain_iso_code")`
    #' @param sustain_region `r get_param_doc_desc("sustain_region")`
    #' @param sustain_interval `r get_param_doc_desc("sustain_interval")`
    #' @param sustain_log_level `r get_param_doc_desc("sustain_log_level")`
    #' @param epochs `r get_param_doc_desc("epochs")`
    #' @param batch_size `r get_param_doc_desc("batch_size")`
    #' @param log_dir `r get_param_doc_desc("log_dir")`
    #' @param log_write_interval `r get_param_doc_desc("log_write_interval")`
    #' @param trace `r get_param_doc_desc("trace")`
    #' @param ml_trace `r get_param_doc_desc("ml_trace")`
    #' @param n_cores `r get_param_doc_desc("n_cores")`
    #' @param lr_rate `r get_param_doc_desc("lr_rate")`
    #' @param lr_min `r get_param_doc_desc("lr_min")`
    #' @param lr_scheduler `r get_param_doc_desc("lr_scheduler")`
    #' @param lr_warm_up_ratio `r get_param_doc_desc("lr_warm_up_ratio")`
    #' @param optimizer `r get_param_doc_desc("optimizer")`
    #' @return Function does not return a value. It changes the object into a trained classifier.
    #' @details
    #'
    #' * `sc_max_k`: All values from sc_min_k up to sc_max_k are successively used. If
    #' the number of sc_max_k is too high, the value is reduced to a number that allows the calculating of synthetic
    #' units.
    #' * `pl_anchor`: With the help of this value, the new cases are sorted. For
    #' this aim, the distance from the anchor is calculated and all cases are arranged into an ascending order.
    #'
    train = function(data_embeddings = NULL,
                     data_targets = NULL,
                     data_folds = 5L,
                     data_val_size = 0.25,
                     loss_balance_class_weights = TRUE,
                     loss_balance_sequence_length = TRUE,
                     loss_cls_fct_name = "FocalLoss",
                     use_sc = FALSE,
                     sc_method = "knnor",
                     sc_min_k = 1L,
                     sc_max_k = 10L,
                     use_pl = FALSE,
                     pl_max_steps = 3L,
                     pl_max = 1.00,
                     pl_anchor = 1.00,
                     pl_min = 0.00,
                     sustain_track = TRUE,
                     sustain_iso_code = NULL,
                     sustain_region = NULL,
                     sustain_interval = 15L,
                     sustain_log_level = "warning",
                     epochs = 40L,
                     batch_size = 32L,
                     trace = TRUE,
                     ml_trace = 1L,
                     log_dir = NULL,
                     log_write_interval = 10L,
                     n_cores = auto_n_cores(),
                     lr_rate = 1e-3,
                     lr_min=1e-4,
                     lr_warm_up_ratio = 0.02,
                     lr_scheduler="None",
                     optimizer = "AdamW") {
      private$do_training(args = get_called_args(n = 1L))
    }
  ),
  private = list()
)
