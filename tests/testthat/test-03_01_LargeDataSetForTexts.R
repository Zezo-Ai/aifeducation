testthat::skip_on_cran()
testthat::skip_if_not(
  condition = check_aif_py_modules(trace = FALSE),
  message = "Necessary python modules not available"
)
load_all_py_scripts()

# Start time
test_time_start <- Sys.time()

# SetUp Test---------------------------------------------------------------------
root_path_data <- testthat::test_path("test_data/LargeDataSetForTexts")
create_dir(testthat::test_path("test_artefacts"), FALSE)

root_path_results <- testthat::test_path("test_artefacts/LargeDataSetForTexts")
create_dir(root_path_results, FALSE)

# SetUp datasets
# Disable tqdm progressbar
transformers$logging$disable_progress_bar()
datasets$disable_progress_bars()

# Start test---------------------------------------------------------------------

test_that("LargeDataSetForTexts - Create", {
  expect_no_error(LargeDataSetForText$new())
})

# Test read of multiple text files-----------------------------------------------
root_path_data_multiple_texts <- paste0(root_path_data, "/texts")

test_that("LargeDataSetForTexts - Add txt", {
  new_dataset <- LargeDataSetForText$new()
  expect_no_error(
    new_dataset$add_from_files_txt(
      dir_path = root_path_data_multiple_texts,
      trace = FALSE
    )
  )
  expect_equal(new_dataset$n_rows(), 2)
  expect_equal(new_dataset$n_cols(), 7)
  expect_equal(
    new_dataset$get_colnames(),
    c(
      "id",
      "text",
      "bib_entry",
      "license",
      "url_license",
      "text_license",
      "url_source"
    )
  )
  expect_equal(
    new_dataset$extract_column("id"),
    c("text_a", "text_b")
  )
  expect_equal(
    new_dataset$extract_column("text"),
    c("This is the first text", "This is the 2. text")
  )
  expect_equal(
    new_dataset$extract_column("bib_entry"),
    c("BIBSENTRY txt A", "BIBSENTRY txt B")
  )
  expect_equal(
    new_dataset$extract_column("license"),
    c("CC BY", "NA")
  )
  expect_equal(
    new_dataset$extract_column("url_license"),
    c("NA", "https://url.to.license.txt_b.html")
  )
  expect_equal(
    new_dataset$extract_column("text_license"),
    c("NA", "License text for txt b")
  )
  expect_equal(
    new_dataset$extract_column("url_source"),
    c("NA", "https://url.to.source.txt_b.txt")
  )

  id <- new_dataset$get_ids()
  true_ids <- c("text_a", "text_b")
  ids_complete <- sum(true_ids %in% id)
  expect_equal(ids_complete, length(true_ids))
})

test_that("LargeDataSetForTexts - Add txt - with log", {
  log_dir <- root_path_results
  create_dir(log_dir, FALSE)
  log_file <- paste0(log_dir, "/aifeducation_state.log")

  new_dataset <- LargeDataSetForText$new()
  expect_no_error(
    new_dataset$add_from_files_txt(
      dir_path = root_path_data_multiple_texts,
      trace = FALSE,
      log_file = log_file,
      log_write_interval = 2,
      log_top_value = 0,
      log_top_total = 1,
      log_top_message = NA,
      clean_text = TRUE
    )
  )

  state_log_exists <- file.exists(log_file)
  expect_true(state_log_exists)
  if (state_log_exists) {
    log_state <- read.csv(log_file)
    expect_equal(nrow(log_state), 3)
    expect_equal(ncol(log_state), 3)
    expect_equal(colnames(log_state), c("value", "total", "message"))
  }

  expect_equal(new_dataset$n_rows(), 2)
  expect_equal(new_dataset$n_cols(), 7)
  expect_equal(
    new_dataset$get_colnames(),
    c(
      "id",
      "text",
      "bib_entry",
      "license",
      "url_license",
      "text_license",
      "url_source"
    )
  )

  expect_equal(
    new_dataset$extract_column("id"),
    c("text_a", "text_b")
  )
  expect_equal(
    new_dataset$extract_column("text"),
    c("This is the first text", "This is the 2. text")
  )
  expect_equal(
    new_dataset$extract_column("bib_entry"),
    c("BIBSENTRY txt A", "BIBSENTRY txt B")
  )
  expect_equal(
    new_dataset$extract_column("license"),
    c("CC BY", "NA")
  )
  expect_equal(
    new_dataset$extract_column("url_license"),
    c("NA", "https://url.to.license.txt_b.html")
  )
  expect_equal(
    new_dataset$extract_column("text_license"),
    c("NA", "License text for txt b")
  )
  expect_equal(
    new_dataset$extract_column("url_source"),
    c("NA", "https://url.to.source.txt_b.txt")
  )

  id <- new_dataset$get_ids()
  true_ids <- c("text_a", "text_b")
  ids_complete <- sum(true_ids %in% id)
  expect_equal(ids_complete, length(true_ids))
})

test_that("LargeDataSetForTexts - Add pdf", {
  new_dataset <- LargeDataSetForText$new()
  expect_no_error(
    new_dataset$add_from_files_pdf(
      dir_path = root_path_data_multiple_texts,
      trace = FALSE
    )
  )
  expect_equal(new_dataset$n_rows(), 2)
  expect_equal(new_dataset$n_cols(), 7)
  expect_equal(
    new_dataset$get_colnames(),
    c(
      "id",
      "text",
      "bib_entry",
      "license",
      "url_license",
      "text_license",
      "url_source"
    )
  )

  expect_equal(
    new_dataset$extract_column("id"),
    c("pdf_a", "pdf_b")
  )
  expect_equal(
    new_dataset$extract_column("text"),
    c("This is PDF A ", "This is PDF B ")
  )
  expect_equal(
    new_dataset$extract_column("bib_entry"),
    c("BIBSENTRY PDF A", "BIBSENTRY PDF B")
  )
  expect_equal(
    new_dataset$extract_column("license"),
    c("NA", "CC BY SA")
  )
  expect_equal(
    new_dataset$extract_column("url_license"),
    c("NA", "https://url.to.license.pdf_b.html")
  )
  expect_equal(
    new_dataset$extract_column("text_license"),
    c("NA", "License text for pdf b")
  )
  expect_equal(
    new_dataset$extract_column("url_source"),
    c("NA", "https://url.to.source.pdf_b.pdf")
  )

  id <- new_dataset$get_ids()
  true_ids <- c("pdf_a", "pdf_b")
  ids_complete <- sum(true_ids %in% id)
  expect_equal(ids_complete, length(true_ids))
})

test_that("LargeDataSetForTexts - Add pdf with log", {
  log_dir <- root_path_results
  create_dir(log_dir, FALSE)
  log_file <- paste0(log_dir, "/aifeducation_state.log")

  new_dataset <- LargeDataSetForText$new()
  expect_no_error(
    new_dataset$add_from_files_pdf(
      dir_path = root_path_data_multiple_texts,
      clean_text = TRUE,
      trace = FALSE
    )
  )

  state_log_exists <- file.exists(log_file)
  expect_true(state_log_exists)
  if (state_log_exists) {
    log_state <- read.csv(log_file)
    expect_equal(nrow(log_state), 3)
    expect_equal(ncol(log_state), 3)
    expect_equal(colnames(log_state), c("value", "total", "message"))
  }

  expect_equal(new_dataset$n_rows(), 2)
  expect_equal(new_dataset$n_cols(), 7)
  expect_equal(
    new_dataset$get_colnames(),
    c(
      "id",
      "text",
      "bib_entry",
      "license",
      "url_license",
      "text_license",
      "url_source"
    )
  )

  expect_equal(
    new_dataset$extract_column("id"),
    c("pdf_a", "pdf_b")
  )
  expect_equal(
    new_dataset$extract_column("text"),
    c("This is PDF A ", "This is PDF B ")
  )
  expect_equal(
    new_dataset$extract_column("bib_entry"),
    c("BIBSENTRY PDF A", "BIBSENTRY PDF B")
  )
  expect_equal(
    new_dataset$extract_column("license"),
    c("NA", "CC BY SA")
  )
  expect_equal(
    new_dataset$extract_column("url_license"),
    c("NA", "https://url.to.license.pdf_b.html")
  )
  expect_equal(
    new_dataset$extract_column("text_license"),
    c("NA", "License text for pdf b")
  )
  expect_equal(
    new_dataset$extract_column("url_source"),
    c("NA", "https://url.to.source.pdf_b.pdf")
  )

  id <- new_dataset$get_ids()
  true_ids <- c("pdf_a", "pdf_b")
  ids_complete <- sum(true_ids %in% id)
  expect_equal(ids_complete, length(true_ids))
})

test_that("LargeDataSetForTexts - Add excel", {
  new_dataset <- LargeDataSetForText$new()
  expect_no_error(
    new_dataset$add_from_files_xlsx(
      dir_path = root_path_data_multiple_texts,
      trace = FALSE
    )
  )
  expect_equal(new_dataset$n_rows(), 3)
  expect_equal(new_dataset$n_cols(), 7)
  expect_equal(
    new_dataset$get_colnames(),
    c(
      "id",
      "text",
      "bib_entry",
      "license",
      "url_license",
      "text_license",
      "url_source"
    )
  )

  expect_equal(
    new_dataset$extract_column("id"),
    c("excel_a", "excel_b", "excel_c")
  )
  expect_equal(
    new_dataset$extract_column("text"),
    c("Excel Text A", "Excel Text B", "Excel Text C")
  )
  expect_equal(
    new_dataset$extract_column("bib_entry"),
    c("NA", "test excel", "NA")
  )
  expect_equal(
    new_dataset$extract_column("license"),
    c("NA", "CC BY", "CC BY NC")
  )
  expect_equal(
    new_dataset$extract_column("url_license"),
    c("https://test.a.html", "https://test.b.html", "NA")
  )
  expect_equal(
    new_dataset$extract_column("text_license"),
    c(
      "This is a license text for entry a",
      "This is a license text for entry b",
      "This is a license text for entry c"
    )
  )
  expect_equal(
    new_dataset$extract_column("url_source"),
    c(
      "NA",
      "www.source.source.excel_b.xlsx",
      "NA"
    )
  )

  id <- new_dataset$get_ids()
  true_ids <- c("excel_a", "excel_b", "excel_c")
  ids_complete <- sum(true_ids %in% id)
  expect_equal(ids_complete, length(true_ids))
})

test_that("LargeDataSetForTexts - Add excel with log", {
  log_dir <- root_path_results
  create_dir(log_dir, FALSE)
  log_file <- paste0(log_dir, "/aifeducation_state.log")

  new_dataset <- LargeDataSetForText$new()
  expect_no_error(
    new_dataset$add_from_files_xlsx(
      dir_path = root_path_data_multiple_texts,
      trace = FALSE
    )
  )

  state_log_exists <- file.exists(log_file)
  expect_true(state_log_exists)
  if (state_log_exists) {
    log_state <- read.csv(log_file)
    expect_equal(nrow(log_state), 3)
    expect_equal(ncol(log_state), 3)
    expect_equal(colnames(log_state), c("value", "total", "message"))
  }

  expect_equal(new_dataset$n_rows(), 3)
  expect_equal(new_dataset$n_cols(), 7)
  expect_equal(
    new_dataset$get_colnames(),
    c(
      "id",
      "text",
      "bib_entry",
      "license",
      "url_license",
      "text_license",
      "url_source"
    )
  )

  expect_equal(
    new_dataset$extract_column("id"),
    c("excel_a", "excel_b", "excel_c")
  )
  expect_equal(
    new_dataset$extract_column("text"),
    c("Excel Text A", "Excel Text B", "Excel Text C")
  )
  expect_equal(
    new_dataset$extract_column("bib_entry"),
    c("NA", "test excel", "NA")
  )
  expect_equal(
    new_dataset$extract_column("license"),
    c("NA", "CC BY", "CC BY NC")
  )
  expect_equal(
    new_dataset$extract_column("url_license"),
    c("https://test.a.html", "https://test.b.html", "NA")
  )
  expect_equal(
    new_dataset$extract_column("text_license"),
    c(
      "This is a license text for entry a",
      "This is a license text for entry b",
      "This is a license text for entry c"
    )
  )
  expect_equal(
    new_dataset$extract_column("url_source"),
    c(
      "NA",
      "www.source.source.excel_b.xlsx",
      "NA"
    )
  )

  id <- new_dataset$get_ids()
  true_ids <- c("excel_a", "excel_b", "excel_c")
  ids_complete <- sum(true_ids %in% id)
  expect_equal(ids_complete, length(true_ids))
})

test_that("LargeDataSetForTexts - Add data.frame", {
  text_data_frame <- rbind(
    c("df_A", "data.frame text A", "bib_entry Text A", NA, NA, NA, NA),
    c(
      "df_B", "data.frame text B", "bib_entry Text B",
      "CC BY", "www.url_to_license", "This is license text.", "www.to.source"
    )
  )
  colnames(text_data_frame) <- c(
    "id",
    "text",
    "bib_entry",
    "license",
    "url_license",
    "text_license",
    "url_source"
  )
  text_data_frame <- as.data.frame(text_data_frame)

  new_dataset <- LargeDataSetForText$new()
  expect_no_error(
    new_dataset$add_from_data.frame(text_data_frame)
  )
  expect_equal(new_dataset$n_rows(), 2)
  expect_equal(new_dataset$n_cols(), 7)
  expect_equal(
    new_dataset$get_colnames(),
    c(
      "id",
      "text",
      "bib_entry",
      "license",
      "url_license",
      "text_license",
      "url_source"
    )
  )

  expect_equal(
    new_dataset$extract_column("id"),
    c("df_A", "df_B")
  )
  expect_equal(
    new_dataset$extract_column("text"),
    c("data.frame text A", "data.frame text B")
  )
  expect_equal(
    new_dataset$extract_column("bib_entry"),
    c("bib_entry Text A", "bib_entry Text B")
  )
  expect_equal(
    new_dataset$extract_column("license"),
    c("NA", "CC BY")
  )
  expect_equal(
    new_dataset$extract_column("url_license"),
    c(
      "NA",
      "www.url_to_license"
    )
  )
  expect_equal(
    new_dataset$extract_column("text_license"),
    c(
      "NA",
      "This is license text."
    )
  )
  expect_equal(
    new_dataset$extract_column("url_source"),
    c(
      "NA",
      "www.to.source"
    )
  )

  id <- new_dataset$get_ids()
  true_ids <- c("df_A", "df_B")
  ids_complete <- sum(true_ids %in% id)
  expect_equal(ids_complete, length(true_ids))
})

# Read of single text files-----------------------------------------------------
root_path_data_single_texts <- paste0(root_path_data, "/single_text")
test_that("LargeDataSetForTexts - Add single txt", {
  new_dataset <- LargeDataSetForText$new()
  expect_no_error(
    new_dataset$add_from_files_txt(
      dir_path = root_path_data_single_texts,
      trace = FALSE
    )
  )
  expect_equal(new_dataset$n_rows(), 1)
  expect_equal(new_dataset$n_cols(), 7)
  expect_equal(
    new_dataset$get_colnames(),
    c(
      "id",
      "text",
      "bib_entry",
      "license",
      "url_license",
      "text_license",
      "url_source"
    )
  )

  id <- new_dataset$get_ids()
  true_ids <- "text_a"
  ids_complete <- sum(true_ids %in% id)
  expect_equal(ids_complete, length(true_ids))
})

test_that("LargeDataSetForTexts - Add single pdf", {
  new_dataset <- LargeDataSetForText$new()
  expect_no_error(
    new_dataset$add_from_files_pdf(
      dir_path = root_path_data_single_texts,
      trace = FALSE
    )
  )
  expect_equal(new_dataset$n_rows(), 1)
  expect_equal(new_dataset$n_cols(), 7)
  expect_equal(
    new_dataset$get_colnames(),
    c(
      "id",
      "text",
      "bib_entry",
      "license",
      "url_license",
      "text_license",
      "url_source"
    )
  )

  id <- new_dataset$get_ids()
  true_ids <- "pdf_a"
  ids_complete <- sum(true_ids %in% id)
  expect_equal(ids_complete, length(true_ids))
})

test_that("LargeDataSetForTexts - Add single excel", {
  new_dataset <- LargeDataSetForText$new()
  expect_no_error(
    new_dataset$add_from_files_xlsx(
      dir_path = root_path_data_single_texts,
      trace = FALSE
    )
  )
  expect_equal(new_dataset$n_rows(), 1)
  expect_equal(new_dataset$n_cols(), 7)
  expect_equal(
    new_dataset$get_colnames(),
    c(
      "id",
      "text",
      "bib_entry",
      "license",
      "url_license",
      "text_license",
      "url_source"
    )
  )

  id <- new_dataset$get_ids()
  true_ids <- "excel_a"
  ids_complete <- sum(true_ids %in% id)
  expect_equal(ids_complete, length(true_ids))
})

# Test for the reduction to unique ids------------------------------------------
test_that("LargeDataSetForTexts -Unique IDS", {
  new_dataset <- LargeDataSetForText$new()
  expect_no_error(
    new_dataset$add_from_files_txt(
      dir_path = root_path_data_multiple_texts,
      trace = FALSE
    )
  )
  expect_equal(new_dataset$n_rows(), 2)
  expect_no_error(
    new_dataset$add_from_files_txt(
      dir_path = root_path_data_multiple_texts,
      trace = FALSE
    )
  )
  expect_equal(new_dataset$n_rows(), 4)
  new_dataset$reduce_to_unique_ids()
  expect_equal(new_dataset$n_rows(), 2)

  id <- new_dataset$get_ids()
  true_ids <- c("text_a", "text_b")
  ids_complete <- sum(true_ids %in% id)
  expect_equal(ids_complete, length(true_ids))

  expect_equal(new_dataset$n_cols(), 7)
  expect_equal(
    new_dataset$get_colnames(),
    c(
      "id",
      "text",
      "bib_entry",
      "license",
      "url_license",
      "text_license",
      "url_source"
    )
  )
})

#-----------------------------------------------------------------------------
test_that("LargeDataSetForTexts - Method Save and Load", {
  save_path <- paste0(root_path_results, "/dataset")
  folder_name <- generate_id()
  load_path <- paste0(save_path, "/", folder_name)

  new_dataset <- LargeDataSetForText$new()
  new_dataset$add_from_files_txt(
    dir_path = root_path_data_multiple_texts,
    trace = FALSE
  )
  expect_no_error(new_dataset$save(save_path,
    folder_name = folder_name
  ))

  new_dataset <- NULL
  new_dataset <- LargeDataSetForText$new()
  new_dataset$load(load_path)

  expect_equal(new_dataset$n_rows(), 2)

  id <- new_dataset$get_ids()
  true_ids <- c("text_a", "text_b")
  ids_complete <- sum(true_ids %in% id)
  expect_equal(ids_complete, length(true_ids))

  expect_equal(
    new_dataset$get_colnames(),
    c(
      "id",
      "text",
      "bib_entry",
      "license",
      "url_license",
      "text_license",
      "url_source"
    )
  )

  expect_equal(
    new_dataset$extract_column("id"),
    c("text_a", "text_b")
  )
  expect_equal(
    new_dataset$extract_column("text"),
    c("This is the first text", "This is the 2. text")
  )
  expect_equal(
    new_dataset$extract_column("bib_entry"),
    c("BIBSENTRY txt A", "BIBSENTRY txt B")
  )
  expect_equal(
    new_dataset$extract_column("license"),
    c("CC BY", "NA")
  )
  expect_equal(
    new_dataset$extract_column("url_license"),
    c("NA", "https://url.to.license.txt_b.html")
  )
  expect_equal(
    new_dataset$extract_column("text_license"),
    c("NA", "License text for txt b")
  )
  expect_equal(
    new_dataset$extract_column("url_source"),
    c("NA", "https://url.to.source.txt_b.txt")
  )

  # Clean Directory
  unlink(
    x = load_path,
    recursive = TRUE
  )
})

#-----------------------------------------------------------------------------
test_that("LargeDataSetForTexts - Function Save and Load", {
  save_path <- paste0(root_path_results, "/dataset")
  folder_name <- generate_id()
  load_path <- paste0(save_path, "/", folder_name)

  new_dataset <- LargeDataSetForText$new()
  new_dataset$add_from_files_txt(
    dir_path = root_path_data_multiple_texts,
    trace = FALSE
  )
  # Save
  expect_no_error(save_to_disk(
    object = new_dataset,
    dir_path = save_path,
    folder_name = folder_name
  ))
  # Load
  new_dataset <- NULL
  new_dataset <- load_from_disk(dir_path = load_path)

  expect_equal(new_dataset$n_rows(), 2)

  id <- new_dataset$get_ids()
  true_ids <- c("text_a", "text_b")
  ids_complete <- sum(true_ids %in% id)
  expect_equal(ids_complete, length(true_ids))

  expect_equal(
    new_dataset$get_colnames(),
    c(
      "id",
      "text",
      "bib_entry",
      "license",
      "url_license",
      "text_license",
      "url_source"
    )
  )

  expect_equal(
    new_dataset$extract_column("id"),
    c("text_a", "text_b")
  )
  expect_equal(
    new_dataset$extract_column("text"),
    c("This is the first text", "This is the 2. text")
  )
  expect_equal(
    new_dataset$extract_column("bib_entry"),
    c("BIBSENTRY txt A", "BIBSENTRY txt B")
  )
  expect_equal(
    new_dataset$extract_column("license"),
    c("CC BY", "NA")
  )
  expect_equal(
    new_dataset$extract_column("url_license"),
    c("NA", "https://url.to.license.txt_b.html")
  )
  expect_equal(
    new_dataset$extract_column("text_license"),
    c("NA", "License text for txt b")
  )
  expect_equal(
    new_dataset$extract_column("url_source"),
    c("NA", "https://url.to.source.txt_b.txt")
  )

  # Clean Directory
  unlink(
    x = load_path,
    recursive = TRUE
  )
})

# Clean Directory
if (dir.exists(root_path_results)) {
  unlink(
    x = root_path_results,
    recursive = TRUE
  )
}

# Monitor test time
monitor_test_time_on_CI(
  start_time = test_time_start,
  test_name = "03_01_LargeDataSetForTexts"
)
