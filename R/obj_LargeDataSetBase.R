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

#' @title Abstract base class for large data sets
#' @description This object contains public and private methods which may be useful for every large data sets. Objects
#'   of this class are not intended to be used directly.
#'
#' @return Returns a new object of this class.
#' @export
#' @family R6 Classes for Developers
LargeDataSetBase <- R6::R6Class(
  classname = "LargeDataSetBase",
  public = list(
    #--------------------------------------------------------------------------
    #' @description Number of columns in the data set.
    #' @return `int` describing the number of columns in the data set.
    n_cols = function() {
      return(private$data$num_columns)
    },


    #--------------------------------------------------------------------------
    #' @description Number of rows in the data set.
    #' @return `int` describing the number of rows in the data set.
    n_rows = function() {
      return(private$data$num_rows)
    },

    #--------------------------------------------------------------------------
    #' @description Get names of the columns in the data set.
    #' @return `vector` containing the names of the columns as `string`s.
    get_colnames = function() {
      return(private$data$column_names)
    },

    #--------------------------------------------------------------------------
    #' @description Extracts the data from a python data set.
    #' @param col_name `string` Name of the column.
    #' @param format `string` Format of the data.
    #' * `"R"` returns the data as a R object.
    #' * `"torch"` returns the data as PyTorch tensors.
    #' * `"numpy"` returns the data as numpy array.
    #' @return Returns a `vector`, `matrix` or `array` for `format="R"`. In
    #' all other cases the requestes format is returned..
    extract_column=function(col_name, format="R"){
      check_class_and_type(
        object = col_name,
        object_name = "col_name",
        type_classes = "string",
        allow_NULL = FALSE
        )
      if(col_name%in%self$get_colnames()){
        extraction=extract_column_from_py_dataset(
          py_dataset=private$data,
          column_name=col_name,
          format = format
        )
        return(extraction)
      } else {
        stop("Column is not part of the data set.")
      }
    },

    #--------------------------------------------------------------------------
    #' @description Get data set.
    #' @return Returns the data set of this object as an object of class `datasets.arrow_dataset.Dataset`.
    get_dataset = function() {
      return(private$data)
    },

    #--------------------------------------------------------------------------
    #' @description Reduces the data set to a data set containing only unique ids. In the case an id exists multiple
    #'  times in the data set the first case remains in the data set. The other cases are dropped.
    #'
    #'  **Attention** Calling this method will change the data set in place.
    #' @return Method does not return anything. It changes the data set of this object in place.
    reduce_to_unique_ids = function() {
      private$data <- reduce_to_unique(private$data, "id")
    },

    #--------------------------------------------------------------------------
    #' @description Returns a data set which contains only the cases belonging to the specific indices.
    #' @param  indicies `vector` of `int` for selecting rows in the data set. **Attention** The indices are zero-based.
    #' @return Returns a data set of class `datasets.arrow_dataset.Dataset` with the selected rows.
    select = function(indicies) {
      private$data$set_format("np")
      if (length(indicies) > 1L) {
        return(private$data$select(as.integer(indicies)))
      } else {
        return(private$data$select(list(as.integer(indicies))))
      }
    },

    #--------------------------------------------------------------------------
    #' @description Get ids
    #' @return Returns a `vector` containing the ids of every row as `string`s.
    get_ids = function() {
      return(extract_column_from_py_dataset(private$data,"id"))
    },

    #--------------------------------------------------------------------------
    #' @description Saves a data set to disk.
    #' @param dir_path Path where to store the data set.
    #' @param folder_name `string` Name of the folder for storing the data set.
    #' @param create_dir `bool` If `True` the directory will be created if it does not exist.
    #' @return Method does not return anything. It write the data set to disk.
    save = function(dir_path, folder_name, create_dir = TRUE) {
      # Create directory
      if (!dir.exists(dir_path)) {
        if (create_dir) {
          dir.create(dir_path)
        } else {
          stop("Directory does not exist.")
        }
      }

      # Create folder
      save_location <- file.path(dir_path, folder_name)
      create_dir(save_location, FALSE)

      # Save
      private$data$save_to_disk(dataset_path = save_location)
    },
    #--------------------------------------------------------------------------
    #' @description loads an object of class [LargeDataSetBase] from disk 'and updates the object to the current version
    #'   of the package.
    #' @param dir_path Path where the data set set is stored.
    #' @return Method does not return anything. It loads an object from disk.
    load_from_disk = function(dir_path) {
      self$load(dir_path)
    },
    #--------------------------------------------------------------------------
    #' @description Loads a data set from disk.
    #' @param dir_path Path where the data set is stored.
    #' @return Method does not return anything. It loads a data set from disk.
    load = function(dir_path) {
      private$data <- datasets$Dataset$load_from_disk(dataset_path = dir_path)
    },
    #---------------------------------------------------------------------------
    #' @description Method for setting the package version for 'aifeducation',
    #' 'reticulate', 'torch', and 'numpy' to the currently used versions.
    #' @return Method does not return anything. It is used to set the private
    #' fields fo package versions.
    set_package_versions = function() {
      private$r_package_versions$aifeducation <- packageVersion("aifeducation")
      private$r_package_versions$reticulate <- packageVersion("reticulate")
      if (!is.null_or_na(private$ml_framework)) {
        private$py_package_versions$torch <- torch["__version__"]
        private$py_package_versions$numpy <- np$version$short_version
      }
    },
    #---------------------------------------------------------------------------
    #' @description Method for requesting a summary of the R and python packages'
    #' versions used for creating the model.
    #' @return Returns a `list` containing the versions of the relevant
    #' R and python packages.
    get_package_versions = function() {
      return(
        list(
          r_package_versions = private$private$r_package_versions,
          py_package_versions = private$py_package_versions
        )
      )
    },
    #--------------------------------------------------------------------------
    #' @description Return all fields.
    #' @return Method returns a `list` containing all public and private fields of the object.
    get_all_fields = function() {
      public_list <- NULL
      private_list <- NULL

      for (entry in names(self)) {
        if (!is.function(self[[entry]]) & !is.environment(self[[entry]])) {
          public_list[entry] <- list(self[[entry]])
        }
      }

      for (entry in names(private)) {
        if (!is.function(private[[entry]]) & !is.environment(private[[entry]])) {
          private_list[entry] <- list(private[[entry]])
        }
      }

      return(
        list(
          public = public_list,
          private = private_list
        )
      )
    }
  ),
  private = list(
    data = NULL,
    r_package_versions = list(
      aifeducation = NA,
      reticulate = NA
    ),
    py_package_versions = list(
      tensorflow = NA,
      torch = NA,
      keras = NA,
      numpy = NA
    ),
    #--------------------------------------------------------------------------
    add = function(new_dataset) {
      # Check
      check_class(object = new_dataset, classes = "datasets.arrow_dataset.Dataset", allow_NULL = TRUE)

      if (is.null(private$data)) {
        private$data <- new_dataset
      } else {
        private$data <- datasets$concatenate_datasets(
          list(private$data, new_dataset)
        )
      }
    }
  )
)
