# This file is part of the R package "aifeducation".
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as published by
# the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>

#' @title Dictionary of layers
#' @description Function for receiving a `list` containing a description of all layers
#' user of the package can apply.
#' @param layer `string` Name of the layer that should be returned. If `layer="all"`
#' all layers are returned as a `list`.
#' @returns Returns a `list` with the following entries:
#'
#' * title: `string` Name of the layer.
#' * desc: `string`Description of the layer written in rmarkdown.
#' * img: `string` Name of the image used to illustrate the layer. File extension must be '.png'.
#' * references: `vector` of `strings`. Every entry contains the literature reference in rmarkdown if relevant.
#' * param_prefix: `string` Prefix used for all parameters to configure the specific layer.
#'
#' @family Utils Documentation
#' @keywords internal
get_layer_dict <- function(layer) {
  documentation <- list()

  documentation$tf_layers <- list(
    title = "Transformer Encoder Layers",
    desc = "The transformer encoder layers follow the structure of the encoder layers
    used in transformer models. A single layer is designed as described by Chollet, Kalinowski, and Allaire (2022, p. 373) with
    the exception that single components of the layers (such as the activation function,
    the kind of residual connection, the kind of normalization or the kind of attention) can be customized.",
    img = "layers_tf_encoder.png",
    references = c(
      "Chollet, F., Kalinowski, T. & Allaire, J. J. (2022). Deep learning with R (Second edition). Manning Publications Co. <https://learning.oreilly.com/library/view/-/9781633439849/?ar>",
      "Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In J. Burstein, C. Doran & T. Solorio (Hrsg.), Proceedings of the 2019 Conference of the North (S. 4171-4186). Association for Computational Linguistics. <https://doi.org/10.18653/v1/N19-1423>"
    ),
    param_prefix = "tf_"
  )

  documentation$feature_layers <- list(
    title = "Feature Layer",
    desc = "The feature layer is a dense layer that can be used to
    increase or decrease the number of features of the input data before passing the
    data into your model. The aim of this layer is to increase or reduce the complexity of the data for your model.
    The output size of this layer determines the number of features for all following layers. In the special case that
    the requested number of features equals the number of features of the text embeddings this layer
    is reduced to a dropout layer with masking capabilities.",
    img = "layers_features.png",
    references = NULL,
    param_prefix = "feat_"
  )

  documentation$dense_layers <- list(
    title = "Dense Layers",
    desc = "A fully connected layer. The layer is applied to every step of a sequence.",
    img = "layers_dense.png",
    references = NULL,
    param_prefix = "dense_"
  )

  documentation$n_gram_layers <- list(
    title = "Multiple N-Gram Layers",
    desc = "This type of layer focuses on sub-sequence and performs an 1d convolutional operation. On a word and token level
    these sub-sequences can be interpreted as n-grams (Jacovi, Shalom & Goldberg 2018). The convolution is done across all features.
    The number of filters equals the number of features of the input tensor. Thus, the shape of the tensor is retained (Pham, Kruszewski & Boleda 2016).
    \n The layer is able to consider multiple n-grams at the same time. In this case the convolution of the n-grams is done
    seprately and the resulting tensors are concatenated along the feature dimension. The number of filters for each n-gram is set to the next smallest natural number
    of num_features/num_n-grams. A residual is added to the first n-gram. Thus, the resulting tensor has the same shape as the input tensor.
    \n Sub-sequences that are masked in the input are
    also masked in the output.
    \n The output of this layer can be understand as the results of the n-gram filters. Stacking this layer
    allows the model to perform n-gram detection of n-grams (meta perspective).",
    img = "layers_ng_conv.png",
    references = c(
      "Jacovi, A., Shalom, O. S. & Goldberg, Y. (2018). Understanding Convolutional Neural Networks for Text Classification. https://doi.org/10.48550/arXiv.1809.08037",
      "Pham, N.-Q., Kruszewski, G. & Boleda, G. (2016). Convolutional Neural Network Language Models. In J. Su, K. Duh & X. Carreras (Hrsg.), Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (S. 1153-1162). Association for Computational Linguistics. https://doi.org/10.18653/v1/D16-1123"
    ),
    param_prefix = "ng_conv_"
  )

  documentation$rec_layers <- list(
    title = "Recurrent Layers",
    desc = "A regular recurrent layer either as Gated Recurrent Unit (GRU) or Long Short-Term Memory (LSTM) layer. Uses
    PyTorchs implementation.",
    img = NULL,
    references = NULL,
    param_prefix = "rec_"
  )

  documentation$cls_pooling_layer <- list(
    title = "Classifiction Pooling Layer",
    desc = "Layer transforms sequences into a lower dimensional space that can be passed to dense layers. It can
    perform two types of pooling. First, it extractes features across the time dimension selecting the maximal
    and/or minimal features. Second, it performs pooling over the remaining features selecting a specific number of
    the heighest and/or lowest features. Pooling over time is mandatory. Pooling over features is optional and can lead to a lose of information.
    \n In the case of selecting the minmal *and* maximal features at the same time the minmal
    features are concatenated to the tensor of the maximal features resulting in the shape $(Batch, Times, 2*Features)$ at the end of the first step.
    In the second step the
    number of requested features is halved. The first half is used for the maximal features and the second for the minimal
    features.",
    img = "layers_cls_pooling.png",
    references = NULL,
    param_prefix = "cls_pooling_"
  )

  documentation$merge_layer <- list(
    title = "Merge Layer",
    desc = "Layer for combining the output of different layers. All inputs must be sequential data of shape (Batch, Times, Features).
    First, pooling over time is applied extracting the minimal and/or maximal features.
    Second, the pooled tensors are combined by calculating their weighted sum. Different attention mechanism can be used
    to dynamically calculate the corresponding weights. This allows the model to decide which part of the data is most usefull.
    Finally, pooling over features can be applied extracting a specific number of maximal and/or minimal features.
    Pooling over times is mandatory. Pooling over features is optional and can lead to a lose of information.\n
    A normalization of all input at the begining of the layer is possible.",
    img = "layers_merge.png",
    references = NULL,
    param_prefix = "merge_"
  )

  if (layer != "all") {
    return(documentation[[layer]])
  } else {
    return(documentation)
  }
}

#' @title Dictionary of core models
#' @description Function for receiving a `list` containing a description of all core models
#' user of the package can apply.
#' @param model `string` Name of the model that should be returned. If `model="all"`
#' all models are returned as a `list`.
#' @returns Returns a `list` with the following entries:
#'
#' * title: `string` Name of the model.
#' * desc: `string`Description of the model written in rmarkdown.
#' * img: `string` Name of the image used to illustrate the model. File extension must be '.png'.
#'
#' @family Utils Documentation
#' @keywords internal
get_dict_core_models <- function(model) {
  dictionary <- NULL

  dictionary$sequential <- list(
    title = "Sequential Core Architecture",
    desc = "This model is based on a sequential architecture.
  The input is passed to a specific number of layers step by step.
  All layers are grouped by their kind into stacks.",
    img = "core_arch_sequential.png"
  )

  dictionary$parallel <- list(
    title = "Parallel Core Architecture",
    desc = "This model is based on a parallel architecture.
  An input is passed to different types of layers separately. At the end the outputs
  are combined to create the final output of the whole model.",
    img = "core_arch_parallel.png"
  )

  if (model == "all") {
    return(dictionary)
  } else {
    return(dictionary[[model]])
  }
}


#' @title Dictionary of classifier types
#' @description Function for receiving a `list` containing a description of all types of classifiers
#' user of the package can apply.
#' @param cls_type `string` Classification type
#' @returns Returns a `string` containing the description written in rmarkdown.
#'
#' @family Utils Documentation
#' @keywords internal
get_dict_cls_type <- function(cls_type) {
  if (cls_type == "prob") {
    desc <- "This is a probability classifier that predicts a probability distribution for
    different classes/categories. This is the standard case most common in literature."
  } else if (cls_type == "prototype") {
    desc <- "This object is a metric based classifer and represents in implementation of a prototypical network for
    few-shot learning as described by Snell,
   Swersky, and Zemel (2017). The network uses a multi way contrastive loss described by Zhang et al. (2019). The
   network learns to scale the metric as described by Oreshkin, Rodriguez, and Lacoste (2018)."
  }
  return(desc)
}

#' @title Dictionary of input types
#' @description Function for receiving a `list` containing a description of the input types necessary
#' for specific models.
#' @param input_type `string` Input type
#' @returns Returns a `string` containing the description of the required input written in rmarkdown.
#'
#' @family Utils Documentation
#' @keywords internal
get_dict_input_types <- function(input_type) {
  if (input_type == "text_embeddings") {
    desc <- "For the creation and training of a
   classifier an object of class [EmbeddedText] or [LargeDataSetForTextEmbeddings] on the one hand and a [factor] on
   the other hand are necessary.
   \n The object of class [EmbeddedText] or [LargeDataSetForTextEmbeddings]  contains the numerical text representations
   (text embeddings) of the raw texts generated by an object of class [TextEmbeddingModel]. For supporting large data
   sets it is recommended to use [LargeDataSetForTextEmbeddings] instead of [EmbeddedText].
   \n The `factor` contains the classes/categories for every text. Missing values (unlabeled cases) are supported and can
   be used for pseudo labeling.
   \n For predictions an object of class [EmbeddedText] or [LargeDataSetForTextEmbeddings] has to be used which was
   created with the same [TextEmbeddingModel] as for training."
  } else {
    stop("input type not supported-")
  }

  return(desc)
}


# ===============================================================================
#' @title Generate layer documentation
#' @description Function for generating the documentation of a specific layer.
#' @param param_name `string` Name of the parameter.
#' @param inc_param_name `bool` If `TRUE` the documentation includes the name of the parameter.
#' @param param_dict `list` storing the parameter description.
#' @param as_list `bool` If `TRUE` returns the element as part of a list.
#'
#' @returns Returns a `string` containing the description written in rmarkdown.
#' @family Utils Documentation
#' @export
get_parameter_documentation <- function(param_name, param_dict, as_list = TRUE, inc_param_name = TRUE) {
  # Add description
  if (as_list) {
    prefix <- "- *"
    suffix <- "*: "
    list_level <- "\t"
  } else {
    prefix <- ""
    suffix <- ": "
    list_level <- ""
  }

  if (inc_param_name) {
    param_desc <- paste0(prefix, param_name, suffix)
  } else {
    param_desc <- NULL
  }
  param_desc <- paste0(
    param_desc,
    param_dict[[param_name]]$desc, "\n"
  )

  if (!is.null(param_dict[[param_name]]$values_desc)) {
    param_desc <- paste0(
      param_desc,
      "Allowed values:\n\n"
    )
    for (j in seq_along(param_dict[[param_name]]$values_desc)) {
      param_desc <- paste0(
        param_desc,
        list_level, "- ",
        "`'", names(param_dict[[param_name]]$values_desc)[j], "'`", ": ",
        param_dict[[param_name]]$values_desc[[j]], "\n"
      )
    }
  }
  return(param_desc)
}

#' @title Generate layer documentation
#' @description Function for generating the documentation of a specific layer.
#' @param layer_name `string` Name of the layer.
#' @param title_format `string` Kind of format of the title.
#' @param subtitle_format `string` Kind of format for all sub-titles.
#' @param inc_img `bool` Include a visualization of the layer.
#' @param inc_params `bool` Include a description of every parameter of the layer.
#' @param inc_references `bool` Include a list of literature references for the layer.
#'
#' @returns Returns a `string` containing the description written in rmarkdown.
#' @family Utils Documentation
#' @export
get_layer_documentation <- function(layer_name, title_format = "bold", subtitle_format = "italic", inc_img = FALSE, inc_params = FALSE, inc_references = FALSE) {
  current_doc <- get_layer_dict(layer_name)
  param_dict <- get_param_dict()

  relevant_params_index <- stringi::stri_detect(
    str = names(param_dict),
    regex = paste0("^", current_doc$param_prefix)
  )
  relevant_params <- subset(x = names(param_dict), subset = relevant_params_index)

  # General description--------------------------------------------------------
  # Title
  if (title_format == "bold") {
    title_format_1 <- "**"
    title_format_2 <- title_format_1
  } else if (title_format == "header") {
    title_format_1 <- "## "
    title_format_2 <- ""
  }

  title_string <- paste0(title_format_1, current_doc$title, title_format_2, "\n\n")


  if (subtitle_format == "italic") {
    subtitle_format1 <- "*"
  } else if (subtitle_format == "bold") {
    subtitle_format1 <- "**"
  }
  subtitle_format2 <- subtitle_format1

  # Description general
  desc <- paste0(
    subtitle_format1, "Description", subtitle_format2, "\n\n",
    current_doc$desc, " ",
    "All parameters with the prefix *", current_doc$param_prefix, "* can be used to configure this layer.",
    "\n\n"
  )

  # Image of the layer
  if (inc_img) {
    img_block <- paste0(subtitle_format1, "Visualization", subtitle_format1, "\n\n")
    img_block <- paste0(
      img_block,
      "![Figure: ", current_doc$title, "](", current_doc$img, "){width='100%'}\n\n"
    )
  } else {
    img_block <- NULL
  }

  # Description of all parameters
  param_desc <- NULL
  # Parameter Documentation---------------------------------------------------
  if (inc_params) {
    param_desc <- paste0(subtitle_format1, "Parameters", subtitle_format1, "\n\n")
    for (i in seq_along(relevant_params)) {
      selected_param <- relevant_params[[i]]

      param_desc <- paste0(
        param_desc, "\n",
        get_parameter_documentation(param_name = selected_param, param_dict = param_dict)
      )
    }
  }


  # Gather documentation elements---------------------------------------------
  markdown_doc <- paste0(
    "\n",
    title_string,
    img_block,
    desc,
    param_desc
  )
  return(markdown_doc)
}


#' @title Generate documentation for core models
#' @description Function for generating the documentation of a specific core model.
#' @param name `string` Name of the core model.
#' @param title_format `string` Kind of format of the title.
#' @param inc_img `bool` Include a visualization of the layer.
#'
#' @returns Returns a `string` containing the description written in rmarkdown.
#' @family Utils Documentation
#' @export
get_desc_for_core_model_architecture <- function(name, title_format = "bold", inc_img = FALSE) {
  documentation <- get_dict_core_models(name)

  if (inc_img) {
    img_block <- "**Visualization**\n\n"
    img_block <- paste0(
      img_block,
      "![", documentation$title, "](", documentation$img, "){width='100%'}\n\n"
    )
  } else {
    img_block <- NULL
  }

  if (title_format == "bold") {
    title_format1 <- "**"
    title_format2 <- title_format1
  } else if (title_format == "header") {
    title_format1 <- "## "
    title_format2 <- ""
  }

  markdown_doc <- paste0(
    "\n",
    title_format1, documentation$title, title_format2, "\n\n",
    img_block, "\n\n",
    documentation$desc, "\n"
  )

  return(markdown_doc)
}

#' @title Generate documentation for a classifier class
#' @description Function for generating the documentation of a model.
#' @param model_name `string` Name of the model.
#' @param cls_type `string` Type of classification
#' @param core_type `string` Name of the core type.
#' @param input_type `bool` Name of the input type necessary for training and predicting.
#'
#' @returns Returns a `string` containing the description written in rmarkdown.
#' @note Function is designed to be used with roxygen2 in the regular documentation.
#' @family Utils Documentation
#' @export
build_documentation_for_model <- function(model_name, cls_type = NULL, core_type = NULL, input_type = "text_embeddings") {
  layer_dict <- get_layer_dict("all")
  prefixes <- NULL
  for (i in seq_along(layer_dict)) {
    prefixes <- append(x = prefixes, values = layer_dict[[i]]$param_prefix)
  }

  layer_included <- vector(length = length(prefixes))
  names(layer_included) <- names(layer_dict)

  model <- create_object(model_name)
  params <- rlang::fn_fmls_names(model$configure)

  for (i in seq_along(layer_included)) {
    check_inlucded <- stringi::stri_detect(str = params, regex = paste0("^", prefixes[i]))
    if (sum(check_inlucded) > 0L) {
      layer_included[i] <- TRUE
    } else {
      layer_included[i] <- FALSE
    }
  }

  model_documentation <- NULL

  # CLS Type
  if (!is.null(cls_type)) {
    model_documentation <- paste0(
      "**Classification Type**\n\n",
      get_dict_cls_type(cls_type)
    )
  }

  # Core Architecture
  if (!is.null(core_type)) {
    model_documentation <- paste0(
      model_documentation, "\n\n",
      get_desc_for_core_model_architecture(core_type)
    )
  }

  # Layer Description
  for (i in seq_along(layer_included)) {
    if (layer_included[i]) {
      model_documentation <- paste0(
        model_documentation, "\n",
        get_layer_documentation(names(layer_included)[i], subtitle_format = "italic")
      )
    }
  }

  # Input and Prediction
  desc_input_and_predict <- NULL
  if (!is.null(input_type)) {
    desc_input_and_predict <- paste0(
      "**Training and Prediction**\n\n",
      get_dict_input_types(input_type)
    )
  }
  model_documentation <- paste0(
    model_documentation, "\n",
    desc_input_and_predict
  )
  return(model_documentation)
}

#' @title Generate documentation of all layers for an vignette or article
#' @description Function for generating the whole documentation for an article
#' used on the package's home page.
#' @returns Returns a `string` containing the description written in rmarkdown.
#' @note Function is designed to be used with inline r code in rmarkdown vignettes/articles.
#' @family Utils Documentation
#' @export
build_layer_stack_documentation_for_vignette <- function() {
  # layer documentation
  layer_dict <- get_layer_dict(layer = "all")

  layer_doc <- "# Layers"
  for (layer in names(layer_dict)) {
    layer_doc <- paste0(
      layer_doc, "\n",
      get_layer_documentation(
        layer_name = layer,
        title_format = "header",
        subtitle_format = "bold",
        inc_img = TRUE,
        inc_params = TRUE,
        inc_references = TRUE
      )
    )
  }

  core_model_dict <- get_dict_core_models("all")
  core_models_doc <- "# Core Models"
  for (core_model in names(core_model_dict)) {
    core_models_doc <- paste0(
      core_models_doc, "\n",
      get_desc_for_core_model_architecture(core_model, title_format = "header", inc_img = TRUE)
    )
  }

  markdown_syntax <- paste0(
    layer_doc, "\n",
    core_models_doc
  )

  return(markdown_syntax)
}

#' @title Build a homepage for the package
#' @description Function build the homepage of the package. In order to use python
#' the build process is run in the current environment.
#' @param clear_docs `bool` If `TRUE` the docs folder will be completely cleared.
#' @return Function does nothing return. It builds the homepage for the package.
#' @importFrom stringi stri_replace_all
#' @family Parameter Dictionary
#' @noRd
#' @keywords internal
build_aife_site <- function(clear_docs = FALSE) {
  build_layer_stack_documentation_for_vignette()

  requireNamespace("pkgdown")
  pkgdown::clean_cache()
  if (clear_docs) {
    pkgdown::clean_site()
  }
  pkgdown::init_site()
  pkgdown::build_home()

  # build site for articles
  articles <- list.files(
    path = "vignettes",
    pattern = "*.Rmd",
    full.names = FALSE
  )
  articles <- stringi::stri_replace_all(str = articles, regex = ".Rmd", replacement = "")
  for (article in articles) {
    pkgdown::build_article(
      name = article,
      new_process = FALSE,
      quiet = FALSE
    )
  }

  pkgdown::build_news()
  pkgdown::build_reference()
  pkgdown::build_redirects()

  # devtools::load_all()
  pkgdown::preview_site()
}


#' @title Generate documentation of supported BaseModels for an vignette or article
#' @description Function for generating the whole documentation for an article
#' used on the package's home page.
#' @returns Returns a `string` containing the description written in rmarkdown.
#' @note Function is designed to be used with inline r code in rmarkdown vignettes/articles.
#' @importFrom stringi stri_replace_all
#' @family Utils Documentation
#' @export
build_layer_overview_base_models <- function() {
  requireNamespace("knitr")
  ov_clm_names <- c("class_name", "reference", "req_sentencepiece")
  final_clm_names <- c("BaseModel", "Reference", "Require Sentencepiece")
  overview_table <- matrix(
    nrow = length(BaseModelsIndex),
    ncol = length(ov_clm_names),
    dimnames = list(
      NULL,
      ov_clm_names
    )
  )
  for (i in seq_along(BaseModelsIndex)) {
    for (clm_name in ov_clm_names) {
      overview_table[i, clm_name] <- stringi::stri_replace_all(
        str = BaseModelsIndex[[i]][[clm_name]],
        replacement = "",
        regex = "\\n"
      )
    }
  }
  overview_table <- overview_table[order(overview_table[, "class_name"]), ]
  colnames(overview_table) <- final_clm_names
  return(
    knitr::kable(
      x = overview_table
    )
  )
}

# ==============================================================================

#' @keywords internal
get_description <- function(type) {
  if (type == "return_object") {
    return("Does return a new object of this class.")
  } else if (type == "return_nothing") {
    return("Does nothing return.")
  } else if (type == "return_save_on_disk") {
    return("Function does nothing return. It is used to save an object on disk.")
  } else if (type == "return_load_on_disk") {
    return("Function does nothin return. It loads an object from disk.")
  } else if (type == "save_dir") {
    return("Path to the directory where to save the object.")
  } else if (type == "load_dir") {
    return("Path where the object set is stored.")
  } else if (type == "model_dir") {
    return("Path where the model is stored.")
  } else {
    stop("Type ist not implemented.")
  }
}
