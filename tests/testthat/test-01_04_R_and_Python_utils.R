testthat::skip_on_cran()

testthat::skip_if_not(
  condition = check_aif_py_modules(trace = FALSE),
  message = "Necessary python modules not available"
)

load_all_py_scripts()

# Start time
test_time_start <- Sys.time()

test_that("class_vector_to_py_dataset", {
  vec <- c(0, 0, 1, 1, 2, 0)
  name_vector <- c("A", "B", "C", "d", "e", "f")
  names(vec) <- name_vector
  dataset <- class_vector_to_py_dataset(vec)

  expect_s3_class(object = dataset, class = "datasets.arrow_dataset.Dataset")
  expect_equal(extract_column_from_py_dataset(dataset,"id"), name_vector)
  expect_equal(extract_column_from_py_dataset(dataset,"labels"), unname(vec))
})

test_that("is_venv ", {
  if (Sys.getenv("CI") == "true") {
    expect_false(is_venv())
  } else {
    expect_true(is_venv())
  }
})

test_that("get_py_env_type ", {
  if (Sys.getenv("CI") == "true") {
    expect_equal(get_py_env_type(), "conda")
  } else {
    expect_equal(get_py_env_type(), "venv")
  }
})

test_that("get_py_env_type ", {
  if (Sys.getenv("CI") == "true") {
    expect_equal(get_py_env_name(), "r-reticulate")
  } else {
    expect_equal(get_py_env_name(), "aifeducation")
  }
})

test_that("get_current_conda_env and get_current_venv", {
  if (Sys.getenv("CI") == "true") {
    expect_equal(get_current_conda_env(), "r-reticulate")
    expect_error(get_current_venv())
  } else {
    expect_equal(get_current_venv(), "aifeducation")
    expect_error(get_current_conda_env())
  }
})

# Monitor test time
monitor_test_time_on_CI(
  start_time = test_time_start,
  test_name = "01_04_R_and_Python_utils"
)
