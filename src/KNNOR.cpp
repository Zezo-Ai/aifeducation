// This file is part of the R package "aifeducation".
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 3 as published by
// the Free Software Foundation.
//
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>

// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"
#include <cmath>

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// ========================================== Function declarations ====================================================

// public (exported) function ------------------------------------------------------------------------------------------
// KNNOR
arma::mat knnor(const Rcpp::List &, size_t, size_t, size_t);

// KNNOR Validation
bool knnor_is_same_class(const arma::rowvec &, const arma::mat &, const arma::uvec &, size_t);

// private functions ---------------------------------------------------------------------------------------------------
// KNN
Rcpp::List knn(const arma::mat &, size_t);

// KNNOR Filtering
Rcpp::List knnor_filter(const Rcpp::List &, size_t);

double calc_threshold_distance(const arma::vec &);
arma::vec savgol_filter(const arma::vec &, size_t, size_t);
arma::vec savgol_coeffs(size_t, size_t);
arma::vec first_derivative(const arma::vec &);

// KNNOR Augmentation
arma::mat knnor_augmentation(const arma::mat &, const arma::mat &, const arma::mat &, size_t, size_t,
                             const arma::mat &, const arma::uvec &, size_t);

double alpha_coefficient(const arma::mat &, const arma::mat &);
arma::rowvec interpolate_point(const arma::rowvec &, const arma::rowvec &, double);
// ---------------------------------------------------------------------------------------------------------------------

// ========================================== Function definitions =====================================================

// KNNOR ===============================================================================================================


//' K-Nearest Neighbor OveRsampling approach (KNNOR)
//'
//' @param dataset `list` containing the following fields:
//'   - `embeddings`: an 2-D array (matrix) with size `batch` x `times*features`
//'   - `labels`: an 1-D array (vector) of integers with `batch` elements
//' @param k `unsigned integer` number of nearest neighbors
//' @param aug_num `unsigned integer` number of datapoints to be augmented
//' @param cycles_number_limit `unsigned integer` number of maximum try cycles
//'
//' @return Returns artificial points (`2-D array (matrix) with size `aug_num` x `times*features`)
//'
//' @references  Islam, A., Belhaouari, S. B., Rehman, A. U. & Bensmail, H. (2022).
//' KNNOR: An oversampling technique for imbalanced datasets.
//' Applied Soft Computing, 115, 108288. https://doi.org/10.1016/j.asoc.2021.108288
//'
//' @family oversampling_approaches
//'
//' @export
// [[Rcpp::export]]
arma::mat knnor(const Rcpp::List &dataset, size_t k, size_t aug_num, size_t cycles_number_limit = 100)
{
  Rcpp::List res = knnor_filter(dataset, k);

  arma::mat embeddings_augmented = knnor_augmentation(
      res["embeddings_min"],
      res["embeddings_maj"],
      res["embeddings_min_sorted"],
      k, aug_num,
      dataset["embeddings"],
      dataset["labels"],
      cycles_number_limit);

  return embeddings_augmented;
}

// KNN =================================================================================================================

// K-Nearest Neighbors
//
// Function written in C++ for calculating `k`-nearest neighbors and distances
//
// @param embeddings `2-D array (matrix)` with size `batch` x `times*features`
// @param k `unsigned integer` number of nearest neighbors
//
// @return Returns a list with the following fields:
//   - `neighbors`: an 2-D array (matrix) with size `batch` x `k`. Each row contains indices of neighbor cases
//   - `distances`: an 2-D array (matrix) with size `batch` x `k`. Each row contains distance values between two cases
//
Rcpp::List knn(const arma::mat &embeddings, size_t k)
{
  size_t n = embeddings.n_rows;

  if (n == 0) {
    Rcpp::stop("knn(): embeddings has zero rows");
  }

  if (k == 0) {
    Rcpp::stop("knn(): k must be >= 1");
  }

  if (k > n - 1) {
    Rcpp::stop("knn(): k cannot exceed n - 1");
  }

  // Creating result matrices
  arma::mat neighbors(n, k);
  arma::mat distances(n, k);

  for (size_t i = 0; i < n; ++i)
  {
    std::vector<std::pair<size_t, double>> dist_pairs;
    dist_pairs.reserve(n - 1);

    for (size_t j = 0; j < n; ++j)
    {
      if (i == j) continue;

      double dist = arma::norm(embeddings.row(i) - embeddings.row(j), 2);
      dist_pairs.push_back({j, dist});
    }

    // Sort by distance (ascending)
    std::sort(
      dist_pairs.begin(),
      dist_pairs.end(),
      [](const std::pair<size_t, double> &a, const std::pair<size_t, double> &b) {
        return a.second < b.second;
      }
    );

    // Store k nearest neighbors
    for (size_t t = 0; t < k; ++t)
    {
      neighbors(i, t) = dist_pairs[t].first;
      distances(i, t) = dist_pairs[t].second;
    }
  }

  return Rcpp::List::create(
      Rcpp::_["neighbors"] = neighbors,
      Rcpp::_["distances"] = distances);
}

// Algorithm 1 KNNOR Filtering =========================================================================================

// KNNOR-Filtering
//
// Function written in C++ for KNNOR-filtering.
// Split `dataset` into two embedding matrices for minority `embeddings_min` and majority `embeddings_maj` classes.
// Sort `embeddings_min` by distance in ascending order and take batches to be used for augmentation in
// `embeddings_min_sorted`
//
// @param dataset `list` containing the following fields:
//   - `embeddings`: an 2-D array (matrix) with size `batch` x `times*features`
//   - `labels`: an 1-D array (vector) of integers with `batch` elements
// @param k `unsigned integer` number of nearest neighbors
//
// @return Returns a list with the fields `embeddings_min`, `embeddings_maj` and `embeddings_min_sorted`
//
Rcpp::List knnor_filter(const Rcpp::List &dataset, size_t k)
{
  arma::mat embeddings = dataset["embeddings"];
  arma::vec labels = dataset["labels"];

  arma::vec unique_classes = arma::unique(labels);

  if (unique_classes.n_elem != 2) {
    Rcpp::stop("knnor_filter() supports exactly 2 classes.");
  }

  // List of two matrices
  std::vector<arma::mat> splitted_embeddings(2);
  for (size_t i = 0; i < labels.n_elem; ++i) {
    if (labels[i] == unique_classes[0]) {
      splitted_embeddings[0] = arma::join_vert(splitted_embeddings[0], embeddings.row(i));
    } else {
      splitted_embeddings[1] = arma::join_vert(splitted_embeddings[1], embeddings.row(i));
    }
  }

  size_t min_index = splitted_embeddings[0].n_rows < splitted_embeddings[1].n_rows ? 0 : 1;
  size_t maj_index = (min_index == 0) ? 1 : 0;

  size_t minority_n = splitted_embeddings[min_index].n_rows;
  if (minority_n <= k) {
    Rcpp::stop("Minority class has too few samples for requested k.");
  }

  Rcpp::List knn_res = knn(splitted_embeddings[min_index], k);
  arma::mat distances = knn_res["distances"];

  size_t k_th = k - 1;
  arma::vec distances2kth = distances.col(k_th);

  arma::vec distances_sorted = arma::sort(distances2kth, "ascend");
  double d = calc_threshold_distance(distances_sorted);

  size_t n_data_points = static_cast<size_t>(std::ceil(minority_n * d));

  // clamp to valid range
  n_data_points = std::min(n_data_points, minority_n);
  n_data_points = std::max(n_data_points, k + 1);

  if (n_data_points == k + 1) {
    Rcpp::warning("n_data_points was clamped to k + 1\n");
  }

  arma::uvec distances_sorted_inds = arma::sort_index(distances2kth, "ascend");

  arma::mat embeddings_min_sorted(n_data_points, splitted_embeddings[min_index].n_cols);
  for (size_t i = 0; i < n_data_points; ++i)
  {
    size_t j = distances_sorted_inds(i);
    embeddings_min_sorted.row(i) = splitted_embeddings[min_index].row(j);
  }

  return Rcpp::List::create(
      Rcpp::_["embeddings_min"] = splitted_embeddings[min_index],
      Rcpp::_["embeddings_maj"] = splitted_embeddings[maj_index],
      Rcpp::_["embeddings_min_sorted"] = embeddings_min_sorted);
}

// Threshold distance
//
// Function written in C++ for calculating the threshold distance (KNNOR-Filtering)
//
// @param y `array` sorted distances
//
// @return Returns the threshold distance
//
double calc_threshold_distance(const arma::vec &y)
{
  /*
   * Formula:
   * d = max_pos / length(dy),
   * where
   * dy = y_smoothed'
   * max_pos = max( sin(tan^(-1)dy) )
   */
  arma::vec y_smoothed = savgol_filter(y, 3, 2);
  arma::vec dy = first_derivative(y_smoothed);

  arma::vec sin_values = sin(atan(dy));
  arma::uword max_pos = sin_values.index_max() + 1;

  return double(max_pos) / double(dy.n_elem);
}

// 1-D Savitzky-Golay filter
//
// Function written in C++ for applying a 1-D Savitzky-Golay filter to an array (KNNOR-Filtering)
//
// @param y `array` the data to be filtered
// @param window_length `unsigned integer` the length of the filter window (i.e., the number of coefficients)
//   `window_length` must be odd
// @param poly_order `unsigned integer` the order of the polynomial used to fit the samples.
//   `poly_order` must be less than `window_length` parameter
//
// @return Returns an array with the filtered data
//
arma::vec savgol_filter(const arma::vec &y, size_t window_length, size_t poly_order)
{
  /*
   * Formula:
   * Y_{j} = sum{i=(1-m)/2, ..., (m-1)/2}( C_{i} * y_{j+i} ),
   * where (m + 1)/2 <= j <= n - (m - 1)/2
   */
  int n = y.size();

  arma::vec Y(n);
  arma::vec C = savgol_coeffs(window_length, poly_order);

  int half_win = window_length / 2;

  for (int j = 0; j < n; ++j)
  {
    double smoothed_value = 0.0;

    for (int i = -half_win; i <= half_win; ++i)
    {
      int index = j + i;
      if (index >= 0 && index < n)
        smoothed_value += C[i + half_win] * y[index];
    }

    Y[j] = smoothed_value;
  }

  return Y;
}

// Compute the coefficients for a 1-D Savitzky-Golay filter
//
// Function written in C++ for calculating the coefficients for Savitzky-Golay filter (KNNOR-Filtering)
//
// @param window_length `unsigned integer` the length of the filter window (i.e., the number of coefficients).
//   `window_length` must be odd
// @param poly_order `unsigned integer` the order of the polynomial used to fit the samples.
//   `poly_order` must be less than `window_length` parameter
//
// @return Returns an array with the filter coefficients
//
arma::vec savgol_coeffs(size_t window_length, size_t poly_order)
{
  if (window_length % 2 == 0)
    Rcpp::stop("`window_length` must be odd");

  // Center the window around zero
  int half_win = window_length / 2;

  /*
   * z = (1 - m)/2, ... , 0, ... , (m - 1)/2 =>
   * z = -half_win, ... , 0, ... , half_win
   */
  arma::vec z = arma::linspace(-half_win, half_win, window_length);

  // Vandermonde matrix
  arma::mat A(window_length, poly_order + 1);
  for (size_t j = 0; j <= poly_order; ++j)
    A.col(j) = arma::pow(z, j);

  // Pseudo-inverse (generalised inverse) of matrix A
  arma::mat pinvA = arma::pinv(A);

  // First row of the pseudo-inverse matrix A is the required coefficients
  arma::vec C = pinvA.row(0).t();

  return C;
}

// Calculate the first derivative
//
// Function written in C++ for calculating the first derivative, used while calculating threshold distance
//   (KNNOR-Filtering).
//
// @param y `array` containing the data points
//
// @return Returns an array, the first derivative
//
arma::vec first_derivative(const arma::vec &y)
{
  /*
   * Formula:
   * f'(x) = (y_{i+1} − y_{i−1}) / (x_{i+1} − x_{i−1}) =>
   * f'(x) = (y_{i+1} − y_{i−1}) / 2.0
   */

  size_t n = y.size();
  arma::vec derivative(n);

  for (size_t i = 1; i < n - 1; ++i)
  {
    derivative[i] = (y[i + 1] - y[i - 1]) / 2.0;
  }

  derivative[0] = y[1] - y[0];
  derivative[n - 1] = y[n - 1] - y[n - 2];

  return derivative;
}

// Algorithm 2 KNNOR Augmentation ======================================================================================

// KNNOR-Augmentation
//
// Function written in C++ for KNNOR-Augmentation
//
// @param embeddings_min `2-D array (matrix)` minority embeddings (with size `batch` x `times*features`)
// @param embeddings_maj `2-D array (matrix)` majority embeddings (with size `batch` x `times*features`)
// @param embeddings_min_sorted `2-D array (matrix)` sorted minprity embeddings to be used for augmentation
//   (with size `batch` x `times*features`)
// @param k `unsigned integer` number of nearest neighbors
// @param aug_num `unsigned integer` number of datapoints to be augmented
// @param embeddings `2-D array (matrix)` embeddings (with size `batch` x `times*features`)
// @param labels `1-D array (vector)` of integers with `batch` elements
// @param cycles_number_limit `unsigned integer` number of maximum try cycles
//
// @return Returns artificial points (`2-D array (matrix) with size `aug_num` x `times*features`)
//
arma::mat knnor_augmentation(const arma::mat &embeddings_min,
                             const arma::mat &embeddings_maj,
                             const arma::mat &embeddings_min_sorted,
                             size_t k, size_t aug_num,
                             const arma::mat &embeddings,
                             const arma::uvec &labels,
                             size_t cycles_number_limit)
{
  arma::mat embeddings_augmented;

  double upper_alpha = alpha_coefficient(embeddings_min, embeddings_maj);

  Rcpp::List res = knn(embeddings_min_sorted, k);
  arma::mat neighbors = res["neighbors"];
  arma::mat distances = res["distances"];

  size_t cycle_number = 0;
  while (aug_num > 0)
  {
    for (size_t i = 0; i < embeddings_min_sorted.n_rows; ++i)
    {
      arma::rowvec source = embeddings_min_sorted.row(i);

      arma::rowvec source_neighbors = neighbors.row(i);

      arma::rowvec new_embedding_point;
      for (size_t j = 0; j < k; ++j)
      {
        double raw_neighbor = source_neighbors(j);
        size_t neighbor_index = static_cast<size_t>(raw_neighbor);

        if (neighbor_index >= embeddings_min_sorted.n_rows) {
          Rcpp::stop("neighbor_index out of bounds");
        }

        arma::rowvec source_neighbor = embeddings_min_sorted.row(neighbor_index);

        double random_alpha = arma::randu() * (upper_alpha - 1e-6) + 1e-6; // interval (0; upper_alpha]
        new_embedding_point = interpolate_point(source, source_neighbor, random_alpha);

        source = new_embedding_point;
      }

      // Validation of new_embedding_point
      bool is_valid = knnor_is_same_class(new_embedding_point, embeddings, labels, k);
      if (is_valid)
      {
        embeddings_augmented = arma::join_vert(embeddings_augmented, new_embedding_point);
        aug_num--;
        if (aug_num == 0)
          break;
      }
    }

    cycle_number++;

    if (cycle_number == cycles_number_limit)
      break;
  }
  return embeddings_augmented;
}

// Calculate alpha coefficient
//
// Function written in C++ for calculating the minimum distance between all minority and majority points
// (KNNOR-Augmentation)
//
// @param embeddings_min `2-D array (matrix)` minority embeddings (with size `batch` x `times*features`)
// @param embeddings_maj `2-D array (matrix)` majority embeddings (with size `batch` x `times*features`)
//
// @return Returns alpha coefficient
//
double alpha_coefficient(const arma::mat &embeddings_min,
                         const arma::mat &embeddings_maj)
{
  arma::vec all_dist(embeddings_maj.n_rows * embeddings_min.n_rows);

  size_t k = 0;
  for (size_t i = 0; i < embeddings_maj.n_rows; ++i)
  {
    for (size_t j = 0; j < embeddings_min.n_rows; ++j)
    {
      double dist = arma::norm(embeddings_maj.row(i) - embeddings_min.row(j), 2);
      all_dist(k) = dist;
      k++;
    }
  }

  double min_dist = all_dist.min();

  return min_dist;
}

// Interpolate point
//
// Function written in C++ for point interpolation (KNNOR-Augmentation)
//
// @param Xi_new `1-D array (vector)` i-th origin point (with `times*features` elements)
// @param p `1-D array (vector)` a neighbor with `Xi_new` (with `times*features` elements)
// @param alpha `double` distance for a new point between `Xi_new` and `p`
//
// @return Returns an 1-D array (vector), interpolated point between two points on distance `alpha`
//   (with `times*features` elements)
//
arma::rowvec interpolate_point(const arma::rowvec &Xi_new, const arma::rowvec &p, double alpha)
{
  return Xi_new + (p - Xi_new) * alpha;
}

// Algorithm 3 KNNOR Validation ========================================================================================

//' Validate a new point
//'
//' Function written in C++ for validating a new point (KNNOR-Validation)
//'
//' @param new_point `1-D array (vector)` new data point to be validated before adding (with `times*features` elements)
//' @param dataset `2-D array (matrix)` current embeddings (with size `batch` x `times*features`)
//' @param labels `1-D array (vector)` of integers with `batch` elements
//' @param k `unsigned integer` number of nearest neighbors
//'
//' @return Returns `TRUE` if a new point can be added, otherwise - `FALSE`
//' @family oversampling_approaches Developers
//'
//' @export
// [[Rcpp::export]]
bool knnor_is_same_class(const arma::rowvec &new_point, const arma::mat &dataset, const arma::uvec &labels, size_t k)
{
  std::vector<std::pair<double, int>> distances;

  for (size_t i = 0; i < dataset.n_rows; ++i)
  {
    double dist = arma::norm(new_point - dataset.row(i), 2);
    distances.push_back({dist, labels(i)});
  }

  std::sort(distances.begin(), distances.end());

  std::vector<int> nearest_classes;

  for (size_t i = 0; i < k; ++i)
    nearest_classes.push_back(distances[i].second);

  return std::all_of(nearest_classes.begin(), nearest_classes.end(),
                     [&](int c)
                     { return c == nearest_classes[0]; });
}
