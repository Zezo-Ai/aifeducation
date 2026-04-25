# This file does not contain any tests. It is used for creating FeatureExtractors
# that can be used for testing Classifiers
testthat::skip_on_cran()

testthat::skip_if_not(
  condition = check_aif_py_modules(trace = FALSE),
  message = "Necessary python modules not available"
)

# Start time
test_time_start <- Sys.time()

test_that("build_layer_stack_documentation_for_vignette",{
expect_no_error(
  build_layer_stack_documentation_for_vignette()
)
})

test_that("build_documentation_for_model",{
  expect_no_error(
    build_documentation_for_model(
      model_name = "TEClassifierSequentialPrototype",
      cls_type = "prototype",
      core_type = "sequential",
      input_type = "text_embeddings"
    )
  )
  expect_no_error(
    build_documentation_for_model(
      model_name = "TEClassifierSequential",
      cls_type = "prob",
      core_type = "sequential",
      input_type = "text_embeddings"
    )
  )
  expect_no_error(
    build_documentation_for_model(
      model_name = "TEClassifierParallelPrototype",
      cls_type = "prototype",
      core_type = "parallel",
      input_type = "text_embeddings"
    )
  )
  expect_no_error(
    build_documentation_for_model(
      model_name = "TEClassifierParallel",
      cls_type = "prob",
      core_type = "parallel",
      input_type = "text_embeddings"
    )
  )
})



# Monitor test time
monitor_test_time_on_CI(
  start_time = test_time_start,
  test_name = "00_02_setup_classifiers"
)
