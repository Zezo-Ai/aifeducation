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

#' @title Recommended version of python packages
#' @description Returns the minimum and maximum versions of the core
#' python packages used in *aifeducation*. It is recommended to use packages of
#' these version. Packages of other versions can result in errors or unexpected results.
#' @param package_name `string` Name of the package the recommended version should be returned. If
#' set to `NULL` a table with all core packages and their supported version is returned.
#' @return Returns a `data.frame` with the packages in the columns and the minimum,
#' maximum, and recommended version in the rows. If a concrete name is passed returns a
#' string with leading '<='.
#'
#' @family Installation and Configuration
#'
#' @export
get_recommended_py_versions <- function(package_name = NULL) {
  check_class_and_type(object = package_name, object_name = "package_name", type_classes = "string", allow_NULL = TRUE)
  py_versions <- list(
    transformers = c("4.56.0", "5.8.1"),
    tokenizers = c("0.22.0", "0.22.2"),
    pandas = c("2.3.2", "3.0.2"),
    datasets = c("3.6.0", "4.8.4"),
    codecarbon = c("3.0.0", "3.2.2"),
    safetensors = c("0.6.2", "0.7.0"),
    torcheval = c("0.0.7", "0.0.7"),
    accelerate = c("1.10.1", "1.13.0"),
    sentencepiece = c("0.2.0", "0.2.1"),
    google.protobuf = c("7.34.0", "7.34.1"),
    calflops = c("0.3.2", "0.3.2")
  )

  if (is.null(package_name)) {
    data_matrix <- matrix(nrow = 3L, ncol = length(py_versions))
    for (i in seq_along(py_versions)) {
      data_matrix[1L, i] <- py_versions[[i]][1L]
      data_matrix[2L, i] <- py_versions[[i]][2L]
      if (length(py_versions[[i]]) == 2L) {
        data_matrix[3L, i] <- py_versions[[i]][2L]
      } else {
        data_matrix[3L, i] <- py_versions[[i]][3L]
      }
    }
    data_matrix <- as.data.frame(data_matrix)
    colnames(data_matrix) <- names(py_versions)
    rownames(data_matrix) <- c("min", "max", "recommended")
    return(data_matrix)
  } else {
    if (!(package_name %in% names(py_versions))) {
      stop(
        "Verson of the package ist not implemented. Supported packages are ",
        toString(names(py_versions))
      )
    }
    package_versions <- py_versions[[package_name]]
    if (length(package_versions) == 2L) {
      package_version <- package_versions[2L]
    } else {
      package_version <- package_versions[3L]
    }
    final_string <- paste0("<=", package_version)
    return(final_string)
  }
}

#' @title Install aifeducation on a machine
#' @description Function for installing 'aifeducation' on a machine.
#'
#' Using a virtual environment (`use_conda=FALSE`)
#' If 'python' is already installed the installed version is used. In the case that
#' the required version of 'python' is different from the existing version the new
#' version is installed. In all other cases python will be installed on the system.
#'
#' #' Using a conda environment (`use_conda=TRUE`)
#' If 'miniconda' is already existing on the machine no installation of 'miniconda'
#' is applied. In this case the system checks for update and updates 'miniconda' to
#' the newest version. If 'miniconda' is not found on the system it will be installed.
#'
#' @param install_aifeducation_studio `bool` If `TRUE` all necessary R packages are installed for using AI for Education
#'   Studio.
#' @param use_conda `bool` If `TRUE` installation installs 'miniconda' and uses 'conda' as package manager. If `FALSE`
#' installation installs python and uses virtual environments for package management.
#' @param cuda_version `string` determining the requested version of cuda.
#' @param python_version `string` Python version to use/install.
#' @return Function does nothing return. It installs python, optional R packages, and necessary 'python' packages on a
#'   machine.
#'
#' @note On MAC OS torch will be installed without support for cuda.
#'
#' @importFrom reticulate install_python
#' @importFrom reticulate install_miniconda
#' @importFrom utils install.packages
#' @importFrom methods is
#'
#' @family Installation and Configuration
#'
#' @export
install_aifeducation <- function(install_aifeducation_studio = TRUE,
                                 python_version = "3.12",
                                 cuda_version = "13.0",
                                 use_conda = FALSE) {
  if (install_aifeducation_studio) {
    install_aifeducation_studio()
  }

  if (!use_conda) {
    # install request version of python
    reticulate::install_python(
      version = python_version,
      force = FALSE
    )
  } else {
    miniconda <- try(reticulate::install_miniconda(
      update = TRUE,
      force = FALSE
    ), silent = TRUE)
    if (methods::is(object = miniconda, class2 = "try-error")) {
      message("Minicond is neither installed nor updated.")
    }
  }

  install_py_modules(
    envname = "aifeducation",
    remove_first = FALSE,
    python_version = python_version,
    use_conda = use_conda,
    pytorch_cuda_version = cuda_version
  )

  cat("\n======================================================")
  cat("\n Installation successful. Please restart R/R Studio.")
  cat("\n======================================================\n")
}

#' @title Updates an existing installation of 'aifeducation' on a machine
#' @description Function for updating 'aifeducation' on a machine.
#'
#' The function tries to find an existing environment on the machine, removes the
#' environment and installs the environment with the new python modules.
#'
#' In the case `env_type = "auto"` the function tries to update an existing virtual environment.
#' If no virtual environment exits it tries to update a conda environment.
#'
#' @param update_aifeducation_studio `bool` If `TRUE` all necessary R packages are installed for using AI for Education
#'   Studio.
#' @param envname `string` Name of the environment where the packages should be installed.
#' @param cuda_version `string` determining the requested version of cuda.
#' @param env_type `string` If set to `"venv"`  virtual environment is requested. If set to
#' `"conda"` a 'conda' environment is requested. If set to `"auto"` the function tries to
#' use a virtual environment with the given name. If this environment does not exist
#' it tries to activate a conda environment with the given name. If this fails
#' the default virtual environment is used.'
#' @return Function does nothing return. It installs python, optional R packages, and necessary 'python' packages on a
#'   machine.
#'
#' @note On MAC OS torch will be installed without support for cuda.
#'
#' @importFrom reticulate virtualenv_exists
#' @importFrom reticulate condaenv_exists
#'
#' @family Installation and Configuration
#' @export
update_aifeducation <- function(update_aifeducation_studio = TRUE,
                                env_type = "auto",
                                cuda_version = "13.0",
                                envname = "aifeducation") {
  # Search for environment
  if (env_type == "auto") {
    message("Try to use virtual environment '", envname, "'.")
    if (reticulate::virtualenv_exists("aifeducation")) {
      message("Use virtual environment'", envname, "'.")
      use_conda <- FALSE
    } else {
      message("There is no virtual environment '", envname, "'. Try to use a conda environment with the same name.")
      if (reticulate::condaenv_exists("aifeducation")) {
        message("USe conda environment'", envname, "'.")
        use_conda <- TRUE
      } else {
        message("The requestet environment does not exists. Neither as virtual environment nor as conda environment.")
        current_env <- get_current_venv()
        message("Use the standard virtual environment", current_env)
        use_conda <- FALSE
      }
    }
  } else if (env_type == "venv") {
    if (reticulate::virtualenv_exists("aifeducation")) {
      message("Use virtual environment'", envname, "'.")
      use_conda <- FALSE
    } else {
      stop("The requestet environment does not exists.")
    }
  } else if (env_type == "conda") {
    if (reticulate::condaenv_exists("aifeducation")) {
      message("Use conda environment'", envname, "'.")
      use_conda <- TRUE
    } else {
      stop("The requestet environment does not exists.")
    }
  }

  install_py_modules(
    envname = envname,
    remove_first = TRUE,
    use_conda = use_conda,
    pytorch_cuda_version = cuda_version
  )

  if (update_aifeducation_studio) {
    install_aifeducation_studio()
  }

  cat("\n======================================================\n")
  cat(" Update successful. Please restart R/R Studio.\n")
  cat("======================================================\n")
}

#' @title Install 'AI for Education - Studio' on a machine
#' @description Function installs/updates all relevant R packages necessary
#' to run the shiny app ''AI for Education - Studio'.
#' @return Function does nothing return. It installs/updates R packages.
#'
#' @family Installation and Configuration
#'
#' @export
install_aifeducation_studio <- function() {
  utils::install.packages(
    c(
      "ggplot2",
      "rlang",
      "shiny",
      "shinyFiles",
      "shinyWidgets",
      "shinycssloaders",
      "sortable",
      "bslib",
      "future",
      "promises",
      "DT",
      "readtext",
      "readxl"
    )
  )
}

#' @title Installing necessary python modules to an environment
#' @description Function for installing the necessary python modules.
#'
#' @param envname `string` Name of the environment where the packages should be installed.
#' @param transformer_version `string` determining the desired version of the python library 'transformers'.
#' @param tokenizers_version `string` determining the desired version of the python library 'tokenizers'.
#' @param pandas_version `string` determining the desired version of the python library 'pandas'.
#' @param datasets_version `string` determining the desired version of the python library 'datasets'.
#' @param codecarbon_version `string` determining the desired version of the python library 'codecarbon'.
#' @param safetensors_version `string` determining the desired version of the python library 'safetensors'.
#' @param torcheval_version `string` determining the desired version of the python library 'torcheval'.
#' @param accelerate_version `string` determining the desired version of the python library 'accelerate'.
#' @param calflops_version `string` determining the desired version of the python library 'calflops'.
#' @param pytorch_cuda_version `string` determining the desired version of 'cuda' for 'PyTorch'.
#' To install 'PyTorch' without cuda set to `NULL`.
#' @param python_version `string` Python version to use.
#' @param remove_first `bool` If `TRUE` removes the environment completely before recreating the environment and
#'   installing the packages. If `FALSE` the packages are installed in the existing environment without any prior
#'   changes.
#' @param use_conda `bool` If `TRUE` uses 'conda' for package management. If `FALSE` uses virtual environments for
#' package management.
#' @return Returns no values or objects. Function is used for installing the necessary python libraries in a conda
#'   environment.
#' @note Function tries to identify the type of operating system. In the case that
#' MAC OS is detected 'PyTorch' is installed without support for cuda.
#' @note Supported versions of the packages can be requested with `get_recommended_py_versions())`
#'
#' @importFrom reticulate conda_create
#' @importFrom reticulate conda_remove
#' @importFrom reticulate condaenv_exists
#' @importFrom reticulate py_install
#' @importFrom utils compareVersion
#' @family Installation and Configuration
#' @export
install_py_modules <- function(envname = "aifeducation",
                               transformer_version = get_recommended_py_versions("transformers"),
                               tokenizers_version = get_recommended_py_versions("tokenizers"),
                               pandas_version = get_recommended_py_versions("pandas"),
                               datasets_version = get_recommended_py_versions("datasets"),
                               codecarbon_version = get_recommended_py_versions("codecarbon"),
                               safetensors_version = get_recommended_py_versions("safetensors"),
                               torcheval_version = get_recommended_py_versions("torcheval"),
                               accelerate_version = get_recommended_py_versions("accelerate"),
                               calflops_version = get_recommended_py_versions("calflops"),
                               pytorch_cuda_version = "13.0",
                               python_version = "3.12",
                               remove_first = FALSE,
                               use_conda = FALSE) {
  # Define the relevant packages
  relevant_modules <- c(
    paste0("transformers", transformer_version),
    paste0("tokenizers", tokenizers_version),
    paste0("pandas", pandas_version),
    paste0("datasets", datasets_version),
    paste0("codecarbon", codecarbon_version),
    paste0("calflops", calflops_version)
  )
  relevant_modules_pt <- c(
    paste0("safetensors", safetensors_version),
    paste0("torcheval", torcheval_version),
    paste0("accelerate", accelerate_version)
  )

  if (detec_os() == "mac") {
    message("Operating System: ", "mac", ". Cuda is not requested.")
    pytorch_cuda_version <- NULL
  } else {
    message("Operating System: ", detec_os(), ". Cuda is requested.")
  }

  if (!is.null(pytorch_cuda_version)) {
    pip_cuda <- paste0(
      "--index-url https://download.pytorch.org/whl/cu",
      stringi::stri_replace(
        str = pytorch_cuda_version,
        regex = "[:punct:]",
        replacement = ""
      )
    )
  } else {
    pip_cuda <- NULL
  }

  if (!use_conda) {
    # Use virtualenv
    if (reticulate::virtualenv_exists(envname = envname)) {
      if (remove_first) {
        reticulate::virtualenv_remove(envname = envname, confirm = FALSE)
        Sys.sleep(5L)
        reticulate::virtualenv_create(
          envname = envname,
          python_version = python_version
        )
      }
    } else {
      reticulate::virtualenv_create(
        envname = envname,
        python_version = python_version
      )
    }

    # Install packages
    reticulate::virtualenv_install(
      envname = envname,
      packages = "torch",
      pip_options = pip_cuda
    )
    reticulate::virtualenv_install(
      envname = envname,
      packages = c(relevant_modules, relevant_modules_pt)
    )
  } else if (use_conda) {
    # use conda
    if (reticulate::condaenv_exists(envname = envname)) {
      if (remove_first) {
        reticulate::conda_remove(envname = envname)
        reticulate::conda_create(
          envname = envname,
          channel = "conda-forge",
          python_version = python_version
        )
      }
    } else {
      reticulate::conda_create(
        envname = envname,
        channel = "conda-forge",
        python_version = python_version
      )
    }

    # PyTorch Installation---------------------------------------------------
    reticulate::conda_install(
      packages = c(
        "pytorch",
        paste0("pytorch-cuda", "=", pytorch_cuda_version)
      ),
      envname = envname,
      channel = c("pytorch", "nvidia"),
      conda = "auto",
      pip = FALSE
    )

    # Necessary Packages----------------------------------------------------
    reticulate::conda_install(
      packages = c(relevant_modules, relevant_modules_pt),
      envname = envname,
      conda = "auto",
      pip = TRUE
    )
  }
}

#' @title Check if all necessary python modules are available
#' @description This function checks if all  python modules necessary for the package 'aifeducation' to work are
#'   available.
#'
#' @param trace `bool` `TRUE` if a list with all modules and their availability should be printed to the console.
#' @param for_sentencepiece `bool` if `TRUE` checks only for 'sentencepiece' and 'google.protobuf'.
#' @return The function prints a table with all relevant packages and shows which modules are available or unavailable.
#' @return If all relevant modules are available, the functions returns `TRUE`. In all other cases it returns `FALSE`
#' @family Installation and Configuration
#' @export
check_aif_py_modules <- function(trace = TRUE, for_sentencepiece = FALSE) {
  if (for_sentencepiece) {
    relevant_modules <- c(
      "sentencepiece",
      "google.protobuf"
    )
  } else {
    general_modules <- c(
      "os",
      "transformers",
      "tokenizers",
      "datasets",
      "codecarbon",
      "calflops"
    )
    pytorch_modules <- c(
      "torch",
      "torcheval",
      "safetensors",
      "accelerate",
      "pandas"
    )
    relevant_modules <- c(
      general_modules,
      pytorch_modules
    )
  }

  matrix_overview <- matrix(
    data = NA,
    nrow = length(relevant_modules),
    ncol = 2L
  )
  colnames(matrix_overview) <- c("module", "available")
  matrix_overview <- as.data.frame(matrix_overview)
  for (i in seq_along(relevant_modules)) {
    matrix_overview[i, 1L] <- relevant_modules[i]
    matrix_overview[i, 2L] <- reticulate::py_module_available(relevant_modules[i])
  }

  if (trace) {
    print(matrix_overview)
  }

  if (sum(matrix_overview[, 2L]) == length(relevant_modules)) {
    return(TRUE)
  } else {
    return(FALSE)
  }
}

#' @title Sets the level for logging information of the 'transformers' library
#' @description This function changes the level for logging information of the 'transformers' library. .
#'
#' @param level `string` Minimal level that should be printed to console.
#' Four levels are available: `"INFO"`, `"WARNING"`, `"ERROR"`, and `"DEBUG"`.
#' @return This function does not return anything. It is used for its side effects.
#' @family Installation and Configuration
#' @export
set_transformers_logger <- function(level = "ERROR") {
  valid_levels <- c("INFO", "WARNING", "ERROR", "DEBUG")
  if (level %in% valid_levels) {
    transformers$logging$set_verbosity(as.integer(transformers$utils$logging$log_levels[tolower(level)]))
  } else {
    stop("Valid levels are: ", toString(valid_levels))
  }
}

#' @title Sets the level for logging information of the 'codecarbon' library
#' @description This function changes the level for logging information of the 'codecarbon' library.
#'
#' @param level `string` Minimal level that should be printed to console. Four levels are available: `"INFO"`, `"WARNING"`,
#'   `"ERROR"`, `"CRITICAL"`, and `"DEBUG"`
#' @return This function does not return anything. It is used for its side effects.
#' @family Installation and Configuration
#' @export
set_codecarbon_logger <- function(level = "ERROR") {
  valid_levels <- c("INFO", "WARNING", "ERROR", "CRITICAL", "DEBUG")
  if (level %in% valid_levels) {
    codecarbon$core$config$logger$setLevel(level)
  } else {
    stop("Valid levels are ", toString(valid_levels))
  }
}

#' @title Function for setting up a python environment within R.
#' @description This functions checks for python and a specified environment. If the environment exists
#' it will be activated. If python is already initialized it uses the current environment.
#'
#' @param envname `string` envname name of the requested environment.
#' @param env_type `string` If set to `"venv"`  virtual environment is requested. If set to
#' `"conda"` a 'conda' environment is requested. If set to `"auto"` the function tries to
#' activate a virtual environment with the given name. If this environment does not exist
#' it tries to activate a conda environment with the given name. If this fails
#' the default virtual environment is used.
#' @param check_session `bool` If `TRUE` functions checks if all necessary python packages are
#' available. Set this argument to `FALSE` can speed up sessions' preparation.
#' Set this argument to `FALSE` only if you are certain that the requirements for the
#' package are satisfied.
#' @param set_logger_level `bool` If `TRUE` the logger level of all python packages is set to "Error" and
#' all prints to the console are disabled.
#'
#' @return Function does not return anything. It is used for preparing python and R.
#'
#' @family Installation and Configuration
#' @export
prepare_session <- function(env_type = "auto",
                            envname = "aifeducation",
                            check_session = TRUE,
                            set_logger_level = TRUE) {
  if (!reticulate::py_available(FALSE)) {
    message("Python is not initalized.")
    if (env_type == "auto") {
      message("Try to use virtual environment '", envname, "'.")
      if (reticulate::virtualenv_exists("aifeducation")) {
        message("Set virtual environment to '", envname, "'.")
        reticulate::use_virtualenv("aifeducation")
      } else {
        message("There is no virtual environment '", envname, "'. Try to use a conda environment with the same name.")
        if (reticulate::condaenv_exists("aifeducation")) {
          message("Set conda environment to '", envname, "'.")
          reticulate::use_condaenv("aifeducation")
        } else {
          message("The requestet environment does not exists. Neither as virtual environment nor as conda environment.")
          current_env <- get_current_venv()
          message("Set the standard virtual environment", current_env)
          reticulate::use_virtualenv(current_env)
        }
      }
    } else if (env_type == "venv") {
      if (reticulate::virtualenv_exists("aifeducation")) {
        message("Set virtual environment to '", envname, "'.")
        reticulate::use_virtualenv("aifeducation")
      } else {
        stop("The requestet environment does not exists.")
      }
    } else if (env_type == "conda") {
      if (reticulate::condaenv_exists("aifeducation")) {
        message("Set conda environment to '", envname, "'.")
        reticulate::use_virtualenv("aifeducation")
      } else {
        stop("The requestet environment does not exists.")
      }
    }

    message("Initializing python.")
    if (!reticulate::py_available(TRUE)) {
      stop("Python cannot be initalized. Please check your installation of python.")
    }
  } else {
    current_sessions <- reticulate::py_config()
    if (current_sessions$conda == "True") {
      current_conda <- get_current_conda_env()
      message(
        "Python is already initalized with the conda environment ",
        "'", current_conda, "'."
      )
    } else {
      current_venv <- get_current_venv()
      message(
        "Python is already initalized with the virtual environment ",
        "'", current_venv, "'."
      )
    }
    message("Try to use this environment.")
  }

  # Print information
  message("Detected OS: ", detec_os())
  if (check_session) {
    message("Checking python packages. This can take a moment.")
    if (check_aif_py_modules(trace = FALSE, for_sentencepiece = FALSE)) {
      message("All necessary python packages are available.")
    } else {
      stop("Not all required python packages are available. Call check_aif_py_modules for details.")
    }
    pkg_versions <- get_py_package_versions(sentencepiece = FALSE)
    print_strings <- pad_str(c(names(pkg_versions), "GPU Acceleration", "sentencepiece available"), width = NULL, pad = " ", end = " : ")
    message(
      paste0(
        print_strings,
        c(
          pkg_versions,
          torch$cuda$is_available(),
          check_aif_py_modules(trace = FALSE, for_sentencepiece = TRUE)
        ),
        collapse = "\n"
      )
    )
    if (check_versions(a = get_py_package_version("transformers"), operator = ">=", b = "5.0.0")) {
      message(
        "Version of python package 'transformers' is 5.0.0 or higher. ",
        "Some older models from hugging face may not work."
      )
    }
  }

  message("Load all python objects and functions.")
  load_all_py_scripts()
  # Set logger level of python packages
  if (set_logger_level) {
    message("Set logger level of python packages.")
    datasets$disable_progress_bars()
    transformers$logging$disable_progress_bar()
    set_transformers_logger("ERROR")
    set_codecarbon_logger("ERROR")
  }
  message("Location for Temporary Files:", create_and_get_tmp_dir())
}

#' @title Function for detecting the OS..
#' @description This functions tries to detect the operating system.
#'
#' @return Function returns a string with the name of the operation system.
#'
#' @family Installation and Configuration
#' @keywords internal
#' @noRd
detec_os <- function() {
  sys_name <- tolower(Sys.info()["sysname"])
  if (tolower(sys_name) == "windows") {
    return("windows")
  } else if (tolower(sys_name) == "unix" || tolower(sys_name) == "linux") {
    return("linux")
  } else if (tolower(sys_name) == "darwin") {
    return("mac")
  } else {
    return(sys_name)
  }
}

#' @title Install protocol buffer compiler
#' @description This function is a support function to install the protocol buffer
#' compiler which is necessary for using the python libraries 'protobuf' and
#' 'sentencepiece'. Details can be found here
#' <https://protobuf.dev/installation/>.
#' @description On Windows the function uses Winget requiring Windows 10 or higher.
#' On MacOS it uses Homebrew. On Linux it uses apt.
#' @return This function does not return anything. It installs
#' the protocol buffer compiler
#' @family Installation and Configuration
#' @export
install_protocol_buffer_compiler <- function() {
  os <- detec_os()
  if (os == "windows") {
    command <- "winget install protobuf"
  } else if (os == "linux") {
    command <- "apt install -y protobuf-compiler"
  } else if (os == "mac") {
    command <- "brew install protobuf"
  } else {
    stop("OS could not be detected.")
  }
  cat("Using the command'", command, "' to install.\n")
  results <- suppressWarnings(
    system(
      command = command,
      intern = TRUE,
      wait = TRUE,
      invisible = FALSE,
      show.output.on.console = FALSE
    )
  )
  cat(toString(results), "\n")
  version <- system(
    command = "protoc --version",
    intern = TRUE,
    wait = TRUE,
    invisible = FALSE,
    show.output.on.console = FALSE
  )
  return(toString(version))
}
