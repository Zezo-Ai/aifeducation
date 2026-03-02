testthat::skip_on_cran()
testthat::skip_if_not(
  condition = check_aif_py_modules(trace = FALSE),
  message = "Necessary python modules not available"
)

# Start time
test_time_start <- Sys.time()

# Load python scripts
load_all_py_scripts()

# Prototyp Metric---------------------------------------------------------------
test_that("Prototype Metric", {
  device <- ifelse(torch$cuda$is_available(), "cuda", "cpu")
  layer <- py$layer_protonet_metric()$to(device)

  samples <- matrix(
    data = c(
      0.59, 0.71, 0.51,
      0.45, -1, -0.77,
      0.27, 0.11, -0.99,
      -0.15, -0.61, -0.89,
      0.39, 0.57, -0.31,
      0.57, 0.21, 0.46,
      -0.59, 0.44, -0.06,
      -0.85, 0.45, 0.94,
      0.1, -0.36, 0.49
    ),
    nrow = 9,
    ncol = 3,
    byrow = TRUE
  )
  samples <- reticulate::np_array(samples)
  samples <- torch$from_numpy(
    samples
  )

  prototypes <- matrix(
    data = c(
      0.436667,   -0.060000,      -0.416667,
      0.270000,   0.056667,       -0.246667,
      -0.446667,  0.176667,       0.456667
    ),
    nrow = 3,
    ncol = 3,
    byrow = TRUE
  )
  prototypes <- reticulate::np_array(prototypes)
  prototypes <- torch$from_numpy(
    prototypes
  )
  result_matrix <- matrix(
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

  scaling_factor <- layer$get_scaling_factor()

  distances <- layer(x = samples$to(device), prototypes = prototypes$to(device))
  expect_equal(
    object = tensor_to_numpy(distances),
    expected = result_matrix,
    tolerance = 1e-6
  )
})

# Masking Layer-----------------------------------------------------------------
test_that("Masking Layer", {
  device <- ifelse(torch$cuda$is_available(), "cuda", "cpu")
  pad_value <- sample(x = seq(from = -200, to = -10, by = 10), size = 1)
  times <- sample(x = seq(from = 3, to = 10, by = 1), size = 1)
  features <- sample(x = seq(from = 3, to = 1024, by = 1), size = 1)
  sequence_length <- sample(x = seq(from = 1, to = times, by = 1), size = 30, replace = TRUE)
  example_tensor <- generate_tensors(
    times = times,
    features = features,
    seq_len = sequence_length,
    pad_value = pad_value
  )

  layer <- py$masking_layer(pad_value)$to(device)
  y <- layer(example_tensor)

  # Check if input is the same as the output
  expect_equal(tensor_to_numpy(y[[1]]), tensor_to_numpy(example_tensor))

  # Check sequence length
  expect_equal(as.numeric(tensor_to_numpy(py$get_SeqLen_from_mask(y[[2]]))), sequence_length)

  # Check Masking times
  expect_equal(rowSums(tensor_to_numpy(y[[2]])), times - sequence_length)

  #Check Masking Features
  expect_equal(rowSums(tensor_to_numpy(py$get_FeatureMask_from_mask(y[[2]],as.integer(features)))), (times - sequence_length)*features)
})

# Identity Layer----------------------------------------------------------------
test_that("identity layer", {
  device <- ifelse(torch$cuda$is_available(), "cuda", "cpu")
  pad_value <- sample(x = seq(from = -200, to = -10, by = 10), size = 1)
  times <- sample(x = seq(from = 3, to = 10, by = 1), size = 1)
  features <- sample(x = seq(from = 3, to = 1024, by = 1), size = 1)
  sequence_length <- sample(x = seq(from = 1, to = times, by = 1), size = 30, replace = TRUE)
  example_tensor <- generate_tensors(
    times = times,
    features = features,
    seq_len = sequence_length,
    pad_value = pad_value
  )$to(device)
  masking_layer <- py$masking_layer(pad_value)$to(device)
  values <- masking_layer(example_tensor)

  layer <- py$identity_layer(apply_masking = FALSE)$to(device)
  y <- layer(
    x = values[[1]],
    mask_times = values[[2]],
  )

  # Test that  values are the same
  expect_equal(tensor_to_numpy(y[[1]]), tensor_to_numpy(values[[1]]))
  expect_equal(tensor_to_numpy(y[[2]]), tensor_to_numpy(values[[2]]))

  # Test that padding is not destroyed
  y_2 <- masking_layer(y[[1]])
  expect_equal(tensor_to_numpy(y[[2]]), tensor_to_numpy(y_2[[2]]))
})

# Residual connection with Mask-----------------------------------------------------------
test_that("residual connection with Mask", {
  device <- ifelse(torch$cuda$is_available(), "cuda", "cpu")
  pad_value <- sample(x = seq(from = -200, to = -10, by = 10), size = 1)
  times <- sample(x = seq(from = 3, to = 10, by = 1), size = 1)
  features <- sample(x = seq(from = 3, to = 1024, by = 1), size = 1)
  sequence_length <- sample(x = seq(from = 1, to = times, by = 1), size = 30, replace = TRUE)
  example_tensor <- generate_tensors(
    times = times,
    features = features,
    seq_len = sequence_length,
    pad_value = pad_value
  )$to(device)
  types <- c("None", "Addition", "ResidualGate")
  masking_layer <- py$masking_layer(pad_value)$to(device)
  values <- masking_layer(example_tensor)

  for (type in types) {
    layer <- py$layer_residual_connection(
      type = type,
      pad_value = as.integer(pad_value)
    )$to(device)
    y <- layer(
      x = values[[1]],
      y = values[[1]],
      mask_times = values[[2]]
    )
    layer$eval()

    # Test that masking values are the same
    expect_equal(tensor_to_numpy(y[[2]]), tensor_to_numpy(values[[2]]))

    # Test that padding is not destroyed
    y_2 <- masking_layer(y[[1]])
    expect_equal(tensor_to_numpy(y[[2]]), tensor_to_numpy(y_2[[2]]))

    # Test that values do not change at random for same input
    y_1 <- layer(
      x = values[[1]],
      y = values[[1]],
      mask_times = values[[2]]
    )
    y_2 <- layer(
      x = values[[1]],
      y = values[[1]],
      mask_times = values[[2]]
    )
    expect_equal(tensor_to_numpy(y_1[[1]]), tensor_to_numpy(y_2[[1]]))
  }
})

# LayerNorm with Mask-----------------------------------------------------------
test_that("LayerNorm with Mask", {
  device <- ifelse(torch$cuda$is_available(), "cuda", "cpu")
  pad_value <- sample(x = seq(from = -200, to = -10, by = 10), size = 1)
  times <- sample(x = seq(from = 3, to = 10, by = 1), size = 1)
  features <- sample(x = seq(from = 3, to = 1024, by = 1), size = 1)
  sequence_length <- sample(x = seq(from = 1, to = times, by = 1), size = 30, replace = TRUE)
  example_tensor <- generate_tensors(
    times = times,
    features = features,
    seq_len = sequence_length,
    pad_value = pad_value
  )$to(device)
  masking_layer <- py$masking_layer(pad_value)$to(device)
  values <- masking_layer(example_tensor)

  layer <- py$LayerNorm_with_Mask(
    times = as.integer(times),
    features = as.integer(features),
    pad_value = as.integer(pad_value)
  )$to(device)
  y <- layer(
    x = values[[1]],
    mask_times = values[[2]]
  )

  # Test that masking values are the same
  expect_equal(tensor_to_numpy(y[[2]]), tensor_to_numpy(values[[2]]))

  # Test that padding is not destroyed
  y_2 <- masking_layer(y[[1]])
  expect_equal(tensor_to_numpy(y[[2]]), tensor_to_numpy(y_2[[2]]))

  # Test that the computations are correct for sequences with full length
  example_tensor <- generate_tensors(
    times = times,
    features = features,
    seq_len = rep.int(times, times = 10),
    pad_value = pad_value
  )$to(device)
  comparison_layer <- torch$nn$LayerNorm(
    normalized_shape = example_tensor$size(2L),
    eps = 1e-05,
    elementwise_affine = TRUE,
    bias = TRUE,
    device = NULL,
    dtype = example_tensor$dtype
  )$to(device)
  res_expected <- tensor_to_numpy(comparison_layer(example_tensor))
  values <- masking_layer(example_tensor)

  results <- tensor_to_numpy(layer(
    x = values[[1]],
    mask_times = values[[2]],
  )[[1]])
  expect_equal(results, res_expected, tolerance = 1e-5)
})

# BatchNorm with Mask-----------------------------------------------------------
test_that("BatchNorm with Mask", {
  device <- ifelse(torch$cuda$is_available(), "cuda", "cpu")
  pad_value <- sample(x = seq(from = -200, to = -10, by = 10), size = 1)
  times <- sample(x = seq(from = 3, to = 10, by = 1), size = 1)
  features <- sample(x = seq(from = 3, to = 1024, by = 1), size = 1)
  sequence_length <- sample(x = seq(from = 1, to = times, by = 1), size = 30, replace = TRUE)
  example_tensor <- generate_tensors(
    times = times,
    features = features,
    seq_len = sequence_length,
    pad_value = pad_value
  )$to(device)
  masking_layer <- py$masking_layer(pad_value)$to(device)
  values <- masking_layer(example_tensor)

  layer <- py$BatchNorm_with_Mask(
    features = as.integer(features),
    pad_value = as.integer(pad_value),
    alpha=0.1,
    eps = 1e-05
  )$to(device)
  layer$eval()
  y <- layer(
    x = values[[1]],
    mask_times = values[[2]]
  )

  # Test that masking values are the same
  expect_equal(tensor_to_numpy(y[[2]]), tensor_to_numpy(values[[2]]))

  # Test that padding is not destroyed
  y_2 <- masking_layer(y[[1]])
  expect_equal(tensor_to_numpy(y[[2]]), tensor_to_numpy(y_2[[2]]))

  # Test that the computations are correct for sequences with full length
  example_tensor <- generate_tensors(
    times = times,
    features = features,
    seq_len = rep.int(times, times = 10),
    pad_value = pad_value
  )$to(device)
  comparison_layer <- torch$nn$BatchNorm1d(
    num_features = example_tensor$size(2L),
    eps = 1e-05,
    affine = TRUE,
    momentum=0.1,
    track_running_stats=TRUE,
    device = NULL,
    dtype = example_tensor$dtype
  )$to(device)
  comparison_layer$eval()
  res_expected <- tensor_to_numpy(
    torch$permute(
      comparison_layer(
        torch$permute(
          example_tensor,reticulate::tuple(list(0L,2L,1L))
        )
      ),
      reticulate::tuple(list(0L,2L,1L))
    )
  )
  values <- masking_layer(example_tensor)

  results <- tensor_to_numpy(layer(
    x = values[[1]],
    mask_times = values[[2]]
  )[[1]])
  expect_equal(results, res_expected, tolerance = 1e-4)
})

# RMSNorm with Mask-----------------------------------------------------------
test_that("RMSNorm with Mask", {
  device <- ifelse(torch$cuda$is_available(), "cuda", "cpu")
  pad_value <- sample(x = seq(from = -200, to = -10, by = 10), size = 1)
  times <- sample(x = seq(from = 3, to = 10, by = 1), size = 1)
  features <- sample(x = seq(from = 3, to = 1024, by = 1), size = 1)
  sequence_length <- sample(x = seq(from = 1, to = times, by = 1), size = 30, replace = TRUE)
  example_tensor <- generate_tensors(
    times = times,
    features = features,
    seq_len = sequence_length,
    pad_value = pad_value
  )$to(device)
  masking_layer <- py$masking_layer(pad_value)$to(device)
  values <- masking_layer(example_tensor)

  layer <- py$RMSNorm_with_Mask(
    features = as.integer(features),
    pad_value = as.integer(pad_value)
  )$to(device)
  y <- layer(
    x = values[[1]],
    mask_times = values[[2]],
  )

  # Test that masking values are the same
  expect_equal(tensor_to_numpy(y[[2]]), tensor_to_numpy(values[[2]]))

  # Test that padding is not destroyed
  y_2 <- masking_layer(y[[1]])
  expect_equal(tensor_to_numpy(y[[2]]), tensor_to_numpy(y_2[[2]]))


  # Test that the computations are correct for sequences with full length
  #example_tensor <- generate_tensors(
  #  times = times,
  #  features = features,
  #  seq_len = rep.int(times, times = 10),
  #  pad_value = pad_value
  #)$to(device)
  #comparison_layer <- torch.nn.RMSNorm(
  #  normalized_shape = example_tensor$size(2L),
  #  eps = 1e-05,
  #  elementwise_affine = TRUE,
  #  device = NULL,
  #  dtype = example_tensor$dtype
  #)$to(device)
  #res_expected <- tensor_to_numpy(comparison_layer(example_tensor))
  #values <- masking_layer(example_tensor)

  #results <- tensor_to_numpy(layer(
  #  x = values[[1]],
  #  seq_len = values[[2]],
  #  mask_times = values[[3]],
  #  mask_features = values[[4]]
  #)[[1]])
  #expect_equal(results, res_expected, tolerance = 1e-5)
})

#PowerNorm with mask-----------------------------------------------------------
test_that("PowerNorm with Mask", {
  device <- ifelse(torch$cuda$is_available(), "cuda", "cpu")
  pad_value <- sample(x = seq(from = -200, to = -10, by = 10), size = 1)
  times <- sample(x = seq(from = 3, to = 10, by = 1), size = 1)
  features <- sample(x = seq(from = 3, to = 1024, by = 1), size = 1)
  sequence_length <- sample(x = seq(from = 1, to = times, by = 1), size = 30, replace = TRUE)
  example_tensor <- generate_tensors(
    times = times,
    features = features,
    seq_len = sequence_length,
    pad_value = pad_value
  )$to(device)
  masking_layer <- py$masking_layer(pad_value)$to(device)
  values <- masking_layer(example_tensor)

  layer <- py$PowerNorm_with_Mask(
    features = as.integer(features),
    pad_value = as.integer(pad_value),
    alpha=0.9,
    eps = 1e-05
  )$to(device)
  layer$eval()
  y <- layer(
    x = values[[1]],
    mask_times = values[[2]]
  )

  # Test that masking values are the same
  expect_equal(tensor_to_numpy(y[[2]]), tensor_to_numpy(values[[2]]))

  # Test that padding is not destroyed
  y_2 <- masking_layer(y[[1]])
  expect_equal(tensor_to_numpy(y[[2]]), tensor_to_numpy(y_2[[2]]))
})

# Dense Layer with Mask-----------------------------------------------------------
test_that("DenseLayer with Mask", {
  device <- ifelse(torch$cuda$is_available(), "cuda", "cpu")
  normalization_types <- c("None", "LayerNorm")
  residual_types <- c("None", "Addition", "ResidualGate")
  connection_types <- c("Regular", "PairwiseOrthogonal")
  pad_value <- sample(x = seq(from = -200, to = -10, by = 10), size = 1)
  times <- sample(x = seq(from = 3, to = 10, by = 1), size = 1)
  features <- sample(x = seq(from = 3, to = 1024, by = 1), size = 1)
  sequence_length <- sample(x = seq(from = 1, to = times, by = 1), size = 30, replace = TRUE)
  example_tensor <- generate_tensors(
    times = times,
    features = features,
    seq_len = sequence_length,
    pad_value = pad_value
  )$to(device)
  masking_layer <- py$masking_layer(pad_value)$to(device)
  values <- masking_layer(example_tensor)

  # Test for equal, more, and fewer features as input size
  features_output <- c(
    features,
    sample(x = seq(from = 1, to = (features - 1)), size = 1),
    sample(x = seq(from = (features + 1), to = 2 * features), size = 1)
  )
  for (connection_type in connection_types) {
    for (norm_types in normalization_types) {
      for (res_types in residual_types) {
        for (target_features in features_output) {
          # Create layer
          layer <- py$dense_layer_with_mask(
            input_size = as.integer(features),
            output_size = as.integer(target_features),
            times = as.integer(times),
            pad_value = as.integer(pad_value),
            connection_type = connection_type,
            act_fct = "ELU",
            dropout = 0.3,
            bias = TRUE,
            parametrizations = "None",
            dtype = values[[1]]$dtype,
            residual_type = "None",
            normalization_type = norm_types
          )$to(device)
          layer$eval()

          y <- layer(
            x = values[[1]],
            mask_times = values[[2]]
          )

          # Test that masking values are the same
          expect_equal(tensor_to_numpy(y[[2]]), tensor_to_numpy(values[[2]]))

          # Test that padding is not destroyed
          y_2 <- masking_layer(y[[1]])
          expect_equal(tensor_to_numpy(y[[2]]), tensor_to_numpy(y_2[[2]]))

          # Test that values do not change at random for same input
          y_1 <- layer(
            x = values[[1]],
            mask_times = values[[2]]
          )
          y_2 <- layer(
            x = values[[1]],
            mask_times = values[[2]]
          )
          expect_equal(tensor_to_numpy(y_1[[1]]), tensor_to_numpy(y_2[[1]]))
        }
      }
    }
  }
})

# layer_tf_encoder-------------------------------------------------------
test_that("layer_tf_encoder", {
  device <- ifelse(torch$cuda$is_available(), "cuda", "cpu")
  attention_types <- c("MultiHead", "Fourier")
  normalization_types <- c("None", "LayerNorm")
  normalization_positions <- c("Post", "Pre")

  pad_value <- sample(x = seq(from = -200, to = -10, by = 10), size = 1)
  times <- sample(x = seq(from = 3, to = 10, by = 1), size = 1)
  features <- sample(x = seq(from = 4, to = 1024, by = 2), size = 1)
  sequence_length <- sample(x = seq(from = 1, to = times, by = 1), size = 30, replace = TRUE)
  example_tensor <- generate_tensors(
    times = times,
    features = features,
    seq_len = sequence_length,
    pad_value = pad_value
  )$to(device)
  masking_layer <- py$masking_layer(pad_value)$to(device)
  values <- masking_layer(example_tensor)

  # Test for equal, more, and fewer features as input size
  features_output <- c(
    features,
    sample(x = seq(from = 1, to = (features - 1)), size = 1),
    sample(x = seq(from = (features + 1), to = 2 * features), size = 1)
  )

  for (attention_type in attention_types) {
    for (normalization_type in normalization_types) {
      for (normalization_position in normalization_positions) {
        # Create layer
        layer <- py$layer_tf_encoder(
          dense_dim = 38L,
          times = as.integer(times),
          pad_value = as.integer(pad_value),
          attention_type = attention_type,
          features = as.integer(features),
          normalization_type = normalization_type,
          normalization_position = normalization_position,
          residual_type = "ResidualGate",
          num_heads = as.integer(2),
          act_fct = "ELU",
          dropout_rate_1 = 0.3,
          dropout_rate_2 = 0.3,
          bias = TRUE,
          parametrizations = "None",
          dtype = values[[1]]$dtype,
        )$to(device)
        layer$eval()

        y <- layer(
          x = values[[1]],
          mask_times = values[[2]]
        )

        # Test that masking values are the same
        expect_equal(tensor_to_numpy(y[[2]]), tensor_to_numpy(values[[2]]))

        # Test that padding is not destroyed
        y_2 <- masking_layer(y[[1]])
        expect_equal(tensor_to_numpy(y[[2]]), tensor_to_numpy(y_2[[2]]))

        # Test that values do not change at random for same input
        y_1 <- layer(
          x = values[[1]],
          mask_times = values[[2]]
        )
        y_2 <- layer(
          x = values[[1]],
          mask_times = values[[2]]
        )
        expect_equal(tensor_to_numpy(y_1[[1]]), tensor_to_numpy(y_2[[1]]))
      }
    }
  }
})

# exreme_pooling_over_time------------------------------------------------------------
test_that("exreme_pooling_over_time", {
  device <- ifelse(torch$cuda$is_available(), "cuda", "cpu")
  pad_value <- sample(x = seq(from = -200, to = -10, by = 10), size = 1)
  times <- sample(x = seq(from = 3, to = 10, by = 1), size = 1)
  features <- sample(x = seq(from = 3, to = 1024, by = 1), size = 1)
  sequence_length <- sample(x = seq(from = 1, to = times, by = 1), size = 30, replace = TRUE)
  example_tensor <- generate_tensors(
    times = times,
    features = features,
    seq_len = sequence_length,
    pad_value = pad_value
  )$to(device)
  masking_layer <- py$masking_layer(pad_value)$to(device)
  values <- masking_layer(example_tensor)

  for (pooling_type in c("Max", "Min", "MinMax")) {
    layer <- py$exreme_pooling_over_time(
      times = as.integer(times),
      features = as.integer(features),
      pad_value = as.integer(pad_value),
      pooling_type = pooling_type
    )$to(device)
    y_1 <- layer(values[[1]], py$get_FeatureMask_from_mask(values[[2]],as.integer(features)))
    if (pooling_type != "MinMax") {
      expect_equal(dim(tensor_to_numpy(y_1)), c(length(sequence_length), features))
    } else {
      expect_equal(dim(tensor_to_numpy(y_1)), c(length(sequence_length), 2 * features))
    }
  }
})

# layer_adaptive_extreme_pooling_1d---------------------------------------------
test_that("layer_adaptive_extreme_pooling_1d", {
  device <- ifelse(torch$cuda$is_available(), "cuda", "cpu")
  tensor <- torch$rand(30L, 768L)$to(device)
  output_size <- 10

  for (pooling_type in c("Max", "Min", "MinMax")) {
    layer <- py$layer_adaptive_extreme_pooling_1d(
      output_size = as.integer(output_size),
      pooling_type = pooling_type
    )$to(device)
    result <- layer(tensor)

    # Check the corres output shape
    expect_equal(ncol(tensor_to_numpy(result)), output_size)


    # Check if the correct values are selected
    result_matrix <- tensor_to_numpy(result)
    tensor_matrix <- tensor_to_numpy(tensor)
    for (i in seq_len(nrow(result_matrix))) {
      if (pooling_type == "Max") {
        ordered_values <- tensor_matrix[i, order(tensor_matrix[i, ], decreasing = TRUE)]
        relevant_values <- ordered_values[seq(from = 1, to = output_size)]
        expect_equal(sum(round(result_matrix[i, ], digits = 5) %in% round(relevant_values, digits = 5)), output_size)
      } else if (pooling_type == "Min") {
        ordered_values <- tensor_matrix[i, order(tensor_matrix[i, ], decreasing = FALSE)]
        relevant_values <- ordered_values[seq(from = 1, to = output_size)]
        expect_equal(sum(round(result_matrix[i, ], digits = 5) %in% round(relevant_values, digits = 5)), output_size)
      } else {
        n_max <- ceiling(output_size / 2)
        n_min <- output_size - n_max

        # Max
        ordered_values_max <- tensor_matrix[i, order(tensor_matrix[i, ], decreasing = TRUE)]
        relevant_values_max <- ordered_values_max[seq(from = 1, to = n_max)]
        expect_equal(sum(round(result_matrix[i, ], digits = 5) %in% round(relevant_values_max, digits = 5)), n_max)

        # Min
        ordered_values_min <- tensor_matrix[i, order(tensor_matrix[i, ], decreasing = FALSE)]
        relevant_values_min <- ordered_values_min[seq(from = 1, to = n_min)]
        expect_equal(sum(round(result_matrix[i, ], digits = 5) %in% round(relevant_values_min, digits = 5)), n_min)
      }
    }
  }
})

# Layer layer_n_gram_convolution--------------------------------------------------
test_that("layer_n_gram_convolution", {
  device <- ifelse(torch$cuda$is_available(), "cuda", "cpu")
  pad_value <- sample(x = seq(from = -200, to = -10, by = 10), size = 1)
  times <- sample(x = seq(from = 3, to = 10, by = 1), size = 1)
  features <- sample(x = seq(from = 3, to = 1024, by = 1), size = 1)
  sequence_length <- sample(x = seq(from = 1, to = times, by = 1), size = 30, replace = TRUE)
  example_tensor <- generate_tensors(
    times = times,
    features = features,
    seq_len = sequence_length,
    pad_value = pad_value
  )$to(device)
  masking_layer <- py$masking_layer(pad_value)$to(device)
  values <- masking_layer(example_tensor)

  n_filter <- sample(x = seq(from = 2, to = features, by = 1), size = 1)
  layer <- py$layer_n_gram_convolution(
    kernel_size_times = as.integer(2),
    times = as.integer(times),
    features = as.integer(features),
    pad_value = as.integer(pad_value),
    n_filter = as.integer(n_filter),
    bias = TRUE,
    parametrizations = "None",
    dtype = values[[1]]$dtype,
    device = device
  )$to(device)
  layer$eval()

  y <- layer(
    x = values[[1]],
    mask_times = values[[2]]
  )

  # Test that masking values are the same
  expect_equal(tensor_to_numpy(y[[2]]), tensor_to_numpy(values[[2]]))

  # Test that padding is not destroyed
  y_2 <- masking_layer(y[[1]])
  expect_equal(tensor_to_numpy(y[[2]]), tensor_to_numpy(y_2[[2]]))

  # Test that values do not change at random for same input
  y_1 <- layer(
    x = values[[1]],
    mask_times = values[[2]]
  )
  y_2 <- layer(
    x = values[[1]],
    mask_times = values[[2]]
  )
  expect_equal(tensor_to_numpy(y_1[[1]]), tensor_to_numpy(y_2[[1]]))

  # Test method return shape
  expect_equal(dim(tensor_to_numpy(y_1[[1]])), c(length(sequence_length), times, n_filter))
})

# Layer layer_mutiple_n_gram_convolution--------------------------------------------------
test_that("layer_mutiple_n_gram_convolution", {
  device <- ifelse(torch$cuda$is_available(), "cuda", "cpu")
  pad_value <- sample(x = seq(from = -200, to = -10, by = 10), size = 1)
  times <- sample(x = seq(from = 3, to = 10, by = 1), size = 1)
  features <- sample(x = seq(from = 3, to = 1024, by = 1), size = 1)
  sequence_length <- sample(x = seq(from = 1, to = times, by = 1), size = 30, replace = TRUE)
  example_tensor <- generate_tensors(
    times = times,
    features = features,
    seq_len = sequence_length,
    pad_value = pad_value
  )$to(device)
  masking_layer <- py$masking_layer(pad_value)$to(device)
  values <- masking_layer(example_tensor)

  for (max_n_gram in 3:times) {
    layer <- py$layer_mutiple_n_gram_convolution(
      ks_min = 2L,
      ks_max = as.integer(max_n_gram),
      times = as.integer(times),
      features = as.integer(features),
      pad_value = as.integer(pad_value),
      bias = TRUE,
      parametrizations = "None",
      dtype = values[[1]]$dtype,
      device = device,
      act_fct_name = "ELU"
    )$to(device)
    layer$eval()

    y <- layer(
      x = values[[1]],
      mask_times = values[[2]]
    )

    # Test that masking values are the same
    expect_equal(tensor_to_numpy(y[[2]]), tensor_to_numpy(values[[2]]))

    # Test that padding is not destroyed
    y_2 <- masking_layer(y[[1]])
    expect_equal(tensor_to_numpy(y[[2]]), tensor_to_numpy(y_2[[2]]))

    # Test that values do not change at random for same input
    y_1 <- layer(
      x = values[[1]],
      mask_times = values[[2]]
    )
    y_2 <- layer(
      x = values[[1]],
      mask_times = values[[2]]
    )
    expect_equal(tensor_to_numpy(y_1[[1]]), tensor_to_numpy(y_2[[1]]))

    # Test method return shape
    expect_equal(dim(tensor_to_numpy(y_1[[1]])), c(length(sequence_length), times, features))
  }
})

# Layer merge leyer---------------------------------------
test_that("merge_layer", {
  device <- ifelse(torch$cuda$is_available(), "cuda", "cpu")
  pad_value <- sample(x = seq(from = -200, to = -10, by = 10), size = 1)
  times <- sample(x = seq(from = 3, to = 10, by = 1), size = 1)
  features <- sample(x = seq(from = 52, to = 1024, by = 1), size = 1)
  sequence_length <- sample(x = seq(from = 1, to = times, by = 1), size = 30, replace = TRUE)
  example_tensor <- generate_tensors(
    times = times,
    features = features,
    seq_len = sequence_length,
    pad_value = pad_value
  )$to(device)
  n_input_streams <- sample(seq(from = 2, to = 10, by = 1), size = 1)
  n_extracted_features <- sample(seq(from = 2, to = features, by = 1), size = 1)
  masking_layer <- py$masking_layer(pad_value)$to(device)
  values <- masking_layer(example_tensor)

  for (attention_type in c("MultiHead", "Fourier")) {
    for (pooling_type in c("Max", "Min", "MinMax")) {
      layer <- py$merge_layer(
        times = as.integer(times),
        features = as.integer(features),
        n_extracted_features = as.integer(n_extracted_features),
        n_input_streams = as.integer(n_input_streams),
        pad_value = as.integer(pad_value),
        pooling_type = pooling_type,
        attention_type = attention_type,
        num_heads = 1L,
        dtype = values[[1]]$dtype,
        device = device
      )$to(device)
      layer$eval()

      y <- layer(
        tensor_list = rep(values[1], times = n_input_streams),
        mask_times = values[[2]]
      )

      # Test the correct shape
      expect_equal(dim(tensor_to_numpy(y)), c(length(sequence_length), n_extracted_features))
    }
  }
})

# Layer rnn preparation Layer---------------------------------------
test_that("rnn_preparation", {
  device <- ifelse(torch$cuda$is_available(), "cuda", "cpu")
  pad_value <- sample(x = seq(from = -200, to = -10, by = 10), size = 1)
  times <- sample(x = seq(from = 3, to = 10, by = 1), size = 1)
  features <- sample(x = seq(from = 3, to = 1024, by = 1), size = 1)
  sequence_length <- sample(x = seq(from = 1, to = times, by = 1), size = 30, replace = TRUE)
  example_tensor <- generate_tensors(
    times = times,
    features = features,
    seq_len = sequence_length,
    pad_value = pad_value
  )$to(device)
  masking_layer <- py$masking_layer(pad_value)$to(device)
  values <- masking_layer(example_tensor)


  layer_do <- py$layer_pack_and_masking()$to(device)
  layer_do$eval()

  layer_undo <- py$layer_unpack_and_masking(
    sequence_length = as.integer(times),
    pad_value = pad_value
  )$to(device)
  layer_undo$eval()

  y <- layer_do(
    x = values[[1]],
    mask_times = values[[2]]
  )
  y <- layer_undo(
    x = y[[1]],
    mask_times = y[[2]]
  )

  # Test that masking values are the same
  expect_equal(tensor_to_numpy(y[[2]]), tensor_to_numpy(values[[2]]))

  # Test that padding is not destroyed
  y_2 <- masking_layer(y[[1]])
  expect_equal(tensor_to_numpy(y[[2]]), tensor_to_numpy(y_2[[2]]))
})

test_that("layer_class_mean", {
  device <- ifelse(torch$cuda$is_available(), "cuda", "cpu")
  layer <- py$layer_class_mean()$to(device)


  test_tensor <- matrix(
    data = c(
      0.59, 0.71, 0.51,
      0.45, -1, -0.77,
      0.27, 0.11, -0.99,
      -0.15, -0.61, -0.89,
      0.39, 0.57, -0.31,
      0.57, 0.21, 0.46,
      -0.59, 0.44, -0.06,
      -0.85, 0.45, 0.94,
      0.1, -0.36, 0.49
    ), nrow = 9, ncol = 3, byrow = TRUE
  )
  test_tensor <- reticulate::np_array(test_tensor)
  test_tensor <- torch$from_numpy(
    test_tensor
  )
  test_classes <- torch$from_numpy(reticulate::np_array(c(0, 0, 0, 1, 1, 1, 2, 2, 2))$copy())
  num_classes <- 3

  result_matrix <- matrix(
    data = c(
      0.436667,   -0.060000,      -0.416667,
      0.270000,   0.056667,       -0.246667,
      -0.446667,  0.176667,       0.456667
    ),
    nrow = 3, ncol = 3, byrow = TRUE
  )

  cls_means <- layer(
    x = test_tensor$to(device),
    classes = test_classes$to(device),
    total_classes = 3L
  )
  expect_equal(
    object = tensor_to_numpy(cls_means),
    expected = result_matrix,
    tolerance = 1e-5
  )
})

# layer_global_average_pooling_1d------------------------------------------------
test_that("layer_global_average_pooling_1d", {
  device <- ifelse(torch$cuda$is_available(), "cuda", "cpu")
  pad_value <- sample(x = seq(from = -200, to = -10, by = 10), size = 1)
  times <- sample(x = seq(from = 3, to = 10, by = 1), size = 1)
  features <- sample(x = seq(from = 3, to = 1024, by = 1), size = 1)
  sequence_length <- sample(x = seq(from = 1, to = times, by = 1), size = 30, replace = TRUE)
  example_tensor <- generate_tensors(
    times = times,
    features = features,
    seq_len = sequence_length,
    pad_value = pad_value
  )$to(device)
  masking_layer <- py$masking_layer(pad_value)$to(device)
  values <- masking_layer(example_tensor)

  layer <- py$layer_global_average_pooling_1d(mask_type = "mask")$to(device)

  results <- layer(
    x = values[[1]],
    mask =values[[2]]
  )

  true_mean_source <- tensor_to_numpy(example_tensor)
  true_mean_source <- replace(x = true_mean_source, true_mean_source == pad_value, values = 0)
  true_mean <- matrix(data = 0, nrow = 30, ncol = features)
  for (b in seq(30)) {
    for (t in 1:times) {
      for (f in 1:features) {
        true_mean[b, f] <- true_mean_source[b, t, f] + true_mean[b, f]
      }
    }
  }
  true_mean <- true_mean / sequence_length

  expect_equal(
    object = tensor_to_numpy(results),
    expected = true_mean,
    tolerance = 1e-7
  )
})

# Monitor test time
monitor_test_time_on_CI(
  start_time = test_time_start,
  test_name = "02_02_pytorch_layers"
)
