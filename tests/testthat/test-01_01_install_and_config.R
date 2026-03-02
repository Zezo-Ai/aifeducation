testthat::skip_on_cran()

testthat::skip_if_not(
  condition = check_aif_py_modules(trace = FALSE),
  message = "Necessary python modules not available"
)

# Start time
test_time_start <- Sys.time()

test_that("check_python_modules", {
  expect_type(
    check_aif_py_modules(trace = FALSE),
    "logical"
  )
})

test_that("set_transformers_logger", {
  for (state in c("ERROR", "WARNING", "INFO", "DEBUG")) {
    expect_no_error(set_transformers_logger(level = state))
  }
  set_transformers_logger("WARNING")
})

test_that("prepare_session", {
  expect_no_error(prepare_session(check_session=TRUE))
})

test_that("prepare_session", {
  expect_no_error(prepare_session(check_session=FALSE))
})

test_that("get_recommended_py_versions", {
  expect_s3_class(get_recommended_py_versions(), "data.frame")
})

monitor_test_time_on_CI(
  start_time = test_time_start,
  test_name = "01_01_install_and_config"
)
