testthat::skip_on_cran()
testthat::skip_if_not(
  condition = check_aif_py_modules(trace = FALSE),
  message = "Necessary python modules not available"
)
load_all_py_scripts()

# Start time
test_time_start <- Sys.time()

# Load python scripts
load_all_py_scripts()

test_that("Focal Loss", {
  # Test correct computation for gamma = 0
  output <- torch$randn(3L, 5L)
  targets_idx <- torch$empty(3L, dtype = torch$long)$random_(5L)
  targets <- torch$nn$functional$one_hot(targets_idx, num_classes = 5L)
  targets <- targets$to(dtype = output$dtype)
  class_weights <- torch$ones(5L)

  focal_loss_fct <- py$focal_loss(
    gamma = 0,
    class_weights = class_weights
  )
  ce_loss <- torch$nn$CrossEntropyLoss(
    reduction = "none",
    weight = class_weights
  )

  loss <- focal_loss_fct(output, targets)
  loss_ce <- ce_loss(output, targets)
  expect_equal(loss$numpy(), loss_ce$numpy())

  # Test for class weights
  class_weights <- torch$rand(5L)

  focal_loss_fct <- py$focal_loss(
    gamma = 0,
    class_weights = class_weights
  )
  ce_loss <- torch$nn$CrossEntropyLoss(
    reduction = "none",
    weight = class_weights
  )

  loss <- focal_loss_fct(output, targets)
  loss_ce <- ce_loss(output, targets)
  expect_equal(loss$numpy(), loss_ce$numpy())
})

# Multi-way contrastive loss----------------------------------------------------
test_that("Multi-way contrastive loss", {
  layer <- py$multi_way_contrastive_loss(alpha = 0.2, margin = 0.9)
  np_array <- reticulate::np_array(c(0, 0, 0, 1, 1, 1, 2, 2, 2))
  test_classes <- torch$from_numpy(np_array$copy())

  distance_matrix <- matrix(
    data = c(
      1.214546097,        1.049661321,    1.167033276,
      1.004301858,        1.192821119,    1.921787363,
      0.620796979,        0.745244181,    1.615827961,
      0.933124977,        1.017212968,    1.587566272,
      0.640668054,        0.530963485,    1.201041215,
      0.926942405,        0.782872205,    1.01721843,
      1.196350933,        0.95989004,     0.597355282,
      1.938080379,        1.678478147,    0.686294397,
      1.012614877,        0.863243239,    0.766789845
    ),
    nrow = 9,
    ncol = 3,
    byrow = TRUE
  )

  distance_matrix <- torch$from_numpy(reticulate::np_array(distance_matrix))

  loss <- layer(classes_q = test_classes, distance_matrix = distance_matrix, metric_scale_factor = 1L)
  expect_equal(object = as.vector(loss$numpy()), expected = 0.438978706, tolerance = 1e-4)
})

# Monitor test time
monitor_test_time_on_CI(
  start_time = test_time_start,
  test_name = "02_01_pytorch_loss_fcts"
)
