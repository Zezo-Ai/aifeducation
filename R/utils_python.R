#' @title Run python file
#' @description Used to run python files with `reticulate::py_run_file()` from folder `python`.
#'
#' @param py_file_name `string` Name of a python file to run. The file must be in the `python` folder of `aifeducation`
#'   package.
#' @return This function returns nothing.
#'
#' @importFrom reticulate py_run_file
#'
#' @family Utils Python Developers
#' @export
run_py_file <- function(py_file_name) {
  reticulate::py_run_file(system.file("python", py_file_name, package = "aifeducation"))
}

#' @title Load and re-load python scripts
#' @description Function loads or re-loads python scripts within the package 'aifeducation'.
#' @param files `vector` containing the file names of the scripts that should be loaded.
#' @return Function does nothing return. It loads the requested scripts.
#'
#' @family Utils Python Developers
#' @importFrom reticulate py_run_file
#' @export
load_py_scripts <- function(files) {
  for (file in files) {
    file_path <- system.file(
      paste0("python/", file),
      package = "aifeducation"
    )
    if (file_path != "") {
      reticulate::py_run_file(file_path)
    } else {
      stop(file, " does not exist")
    }
  }
}

#' @title Load and re-load all python scripts
#' @description Function loads or re-loads all python scripts within the package 'aifeducation'.
#' @return Function does nothing return. It loads the requested scripts.
#'
#' @family Utils Python Developers
#' @importFrom reticulate py_run_file
#' @export
load_all_py_scripts <- function() {
  python_dir <- system.file(
    "python",
    package = "aifeducation"
  )
  python_files <- list.files(
    path = python_dir,
    full.names = TRUE,
    pattern="\\.py"
  )

  for (file in python_files) {
    reticulate::py_run_file(file)
  }
}

#' @title Check numpy array to be writable
#' @description Function for checking if a numpy array is writable.
#'
#' @param np_array A numpy array.
#' @return Function returns `TRUE` if the numpy array is writable. It returns `FALSE`
#' if the array is not writable.
#'
#' @family Utils Python Developers
#' @keywords internal
#' @noRd
#'
numpy_writeable <- function(np_array) {
  if (!inherits(x = np_array, what = "numpy.ndarray")) {
    stop("Provided object is no numpy array")
  }
  return(reticulate::py_to_r(np_array$flags["WRITEABLE"]))
}

#' @title Check for a virtual python environment
#' @description This function checks if a virtual python environment is active.
#'
#' @return Returns `TRUE` if a virtual python environment is active.
#' In the case that python is not active the function raises an error.
#'
#' @family Utils Python Developers
#' @keywords internal
#' @noRd
#'
is_venv <- function() {
  if (reticulate::py_available()) {
    py_config <- reticulate::py_config()
    if (py_config$conda == "False") {
      return(TRUE)
    } else {
      return(FALSE)
    }
  } else {
    stop("Python session not active.")
  }
}

#' @title Get python environment type.
#' @description This function determines if a 'conda' or a virtual environment is active.
#'
#' @return Returns a `string` `"venv"` if a virtual environment is active and `"conda"` if a
#' 'conda' environment is active. In the case that python is not active the function raises an error.
#'
#' @family Utils Python Developers
#' @keywords internal
#' @noRd
#'
get_py_env_type <- function() {
  if (reticulate::py_available()) {
    py_config <- reticulate::py_config()
    if (py_config$conda == "False") {
      return("venv")
    } else {
      return("conda")
    }
  } else {
    stop("Python session not active.")
  }
}

#' @title Get name of the active python environment
#' @description Function for getting the name of active python environment.
#'
#' @return Returns the name of the active python environment. If the current
#' environment is a 'conda' environment the name of this environment is returned.
#' If the active environment is a virtual environment the name of this environment
#' is returned. In the case that python is not active the function raises an error.
#'
#' @family Utils Python Developers
#' @keywords internal
#' @noRd
#'
get_py_env_name <- function() {
  if (reticulate::py_available()) {
    if (is_venv()) {
      return(get_current_venv())
    } else {
      return(get_current_conda_env())
    }
  } else {
    stop("Python session not active.")
  }
}

#' @title Get current conda environment
#' @description Function for getting the active conda environment.
#'
#' @return Returns the name of the current conda environment as a `string`.
#' In the case that python is not active or conda is not used this function
#' raises an error.
#'
#' @family Utils Python Developers
#' @keywords internal
#' @noRd
#'
get_current_conda_env <- function() {
  if (reticulate::py_available()) {
    current_sessions <- reticulate::py_config()
    if (current_sessions$conda == "True") {
      current_conda_env <- base::strsplit(
        x = current_sessions$pythonhome,
        split = "/",
        fixed = TRUE
      )
      current_conda_env <- current_conda_env[[1L]][length(current_conda_env[[1L]])]
      return(current_conda_env)
    } else {
      stop("No conda environment active.")
    }
  } else {
    stop("Python session not active.")
  }
}

#' @title Get current virtual environment
#' @description Function for getting the active virtual environment.
#'
#' @return Returns the name of the current virtual environment as a `string`.
#' In the case that python is not active or a virtual environment is not used this function
#' raises an error.
#'
#' @family Utils Python Developers
#' @keywords internal
#' @noRd
#'
get_current_venv <- function() {
  if (reticulate::py_available()) {
    current_sessions <- reticulate::py_config()
    if (current_sessions$conda == "False") {
      current_venv <- base::strsplit(
        x = current_sessions$pythonhome,
        split = "/",
        fixed = TRUE
      )
      current_venv <- current_venv[[1L]][length(current_venv[[1L]])]
      return(current_venv)
    } else {
      stop("No virtual environment active.")
    }
  } else {
    stop("Python session not active.")
  }
}

#' @title Get versions of python components
#' @description Function for requesting a summary of the versions of all
#' critical python components.
#'
#' @return Returns a list that contains the version number of python and
#' the versions of critical python packages. If a package is not available
#' version is set to `NA`.
#' @param sentencepiece `bool` If `TRUE` the packages 'google.protobuf' and
#' 'sentencepiece' are checked.
#' @family Utils Python Developers
#' @importFrom reticulate py_module_available
#' @importFrom reticulate py_config
#' @importFrom reticulate import
#' @export
get_py_package_versions <- function(sentencepiece = FALSE) {
  if (sentencepiece) {
    list_of_packages <- c(
      "sentencepiece",
      "google.protobuf"
    )
  } else {
    list_of_packages <- c(
      "torch",
      "pyarrow",
      "transformers",
      "tokenizers",
      "pandas",
      "datasets",
      "calflops",
      "codecarbon",
      "safetensors",
      "torcheval",
      "accelerate",
      "numpy"
    )
  }

  versions <- vector(length = length(list_of_packages) + 1L)
  names(versions) <- c("python", list_of_packages)
  versions["python"] <- as.character(reticulate::py_config()$version)

  for (package in list_of_packages) {
    versions[package] <- get_py_package_version(package)
  }
  return(versions)
}

#' @title Get versions of a specific python package
#' @description Function for requesting the version of a specific python package.
#' @param package_name `string` Name of the package.
#' @return Returns the version as `string` or `NA` if the package does not exist
#' or no version is available.
#'
#' @family Utils Python Developers
#' @importFrom reticulate import
#' @export
get_py_package_version <- function(package_name) {
  if (reticulate::py_module_available(package_name)) {
    tmp_package <- reticulate::import(module = package_name, delay_load = FALSE)
    tmp_version <- try(as.character(tmp_package["__version__"]), silent = TRUE)

    if (is(tmp_version, "try-error")) {
      packages_installed <- try(reticulate::py_list_packages(), silent = TRUE)

      if (!inherits(packages_installed, "try-error") && is.data.frame(packages_installed)) {
        if (package_name %in% packages_installed$package) {
          tmp_version <- packages_installed$version[which(packages_installed$package == package_name)]
        } else {
          tmp_version <- NA
        }
      } else {
        tmp_version <- NA
      }
    }
  } else {
    tmp_version <- NA
  }
  return(tmp_version)
}
