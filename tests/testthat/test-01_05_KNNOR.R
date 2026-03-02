testthat::skip_on_cran()

testthat::skip_if_not(
  condition = check_aif_py_modules(trace = FALSE),
  message = "Necessary python modules not available"
)

load_all_py_scripts()

# Start time
test_time_start <- Sys.time()

# SetUp-------------------------------------------------------------------------
root_path_general_data <- testthat::test_path("test_data/Embeddings")

# Load test data
imdb_embeddings <- load_from_disk(paste0(root_path_general_data, "/imdb_embeddings"))
imdb_embeddings <- imdb_embeddings$convert_to_LargeDataSetForTextEmbeddings()
imdb_embeddings <- imdb_embeddings$convert_to_EmbeddedText()

text_embeddings <- imdb_embeddings$embeddings

dims <- dim(text_embeddings)
batches <- dims[1] # cases
times <- dims[2] # chunks
features <- dims[3] # features in each chunk

example_data <- imdb_movie_reviews

k <- 5

mat <- tensor_to_matrix_c(text_embeddings, times, features)
t_features <- times * features

labels <- as.factor(as.numeric(example_data$label))

dataset <- list(
  embeddings = mat,
  labels = labels
)

aug_nums <- c(10, 100)
cycle_number_limit <- 80

for (aug_num in aug_nums) {
  test_that(paste0("knnor: aug_num = ", aug_num), {
    aug_emb <- knnor(dataset, k, aug_num, cycle_number_limit)
    # print(paste0("Number of generated cases is ", dim(aug_emb)[1], " out of ", aug_num))
    expect_equal(dim(aug_emb), c(aug_num, t_features))

    # check if new points are valid
    for (i in seq_len(dim(aug_emb)[1])) {
      ok <- knnor_is_same_class(
        new_point = aug_emb[i, ],
        dataset = mat,
        labels = labels,
        k = k
      )

      expect_equal(ok, TRUE)
    }
  })
}

# Monitor test time
monitor_test_time_on_CI(
  start_time = test_time_start,
  test_name = "01_05_KNNOR"
)
