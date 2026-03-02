

# Start time
test_time_start <- Sys.time()

test_that("logging in R - state log", {
  dir <- testthat::test_path("test_artefacts")
  create_dir(dir, FALSE)

  log_file <- file.path(dir, "state.log")
  reset_log(log_file)
  log <- read_log(log_file)
  reset_log <- rbind(
    c(0L, 1L, NA),
    c(0L, 1L, NA),
    c(0L, 1L, NA)
  )
  colnames(reset_log) <- c("value", "total", "message")

  expect_equal(as.matrix(log), reset_log)

  write_log(
    log_file = log_file,
    value_top = 0L, total_top = 1L, message_top = 2,
    value_middle = 3L, total_middle = 4L, message_middle = 5,
    value_bottom = 6L, total_bottom = 7L, message_bottom = 8,
    last_log = NULL, write_interval = 0L
  )
  write_log_file <- rbind(
    c(0L, 1L, 2),
    c(3L, 4L, 5),
    c(6L, 7L, 8)
  )
  colnames(write_log_file) <- c("value", "total", "message")
  log <- read_log(log_file)
  expect_equal(as.matrix(log), write_log_file)
})

test_that("logging in R - loss log", {
  dir <- testthat::test_path("test_artefacts")
  create_dir(dir, FALSE)

  log_file <- file.path(dir, "loss.log")

  expect_no_error(reset_loss_log(log_file))

  res <- read_loss_log(log_file)
  expect_equal(colnames(res), c("train", "validation", "test", "epoch"))
  expect_equal(unname(as.matrix(res)[1, 1:3]), c(-100, -100, -100))
})

# Monitor test time
monitor_test_time_on_CI(
  start_time = test_time_start,
  test_name = "01_06_logging"
)
