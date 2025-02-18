% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{matrix_to_array_c}
\alias{matrix_to_array_c}
\title{Reshape matrix to array}
\usage{
matrix_to_array_c(matrix, times, features)
}
\arguments{
\item{matrix}{\code{matrix} containing the sequential data.}

\item{times}{\code{uword} Number of sequences.}

\item{features}{\code{uword} Number of features within each sequence.}
}
\value{
Returns an array. The first dimension corresponds to the cases,
the second to the times, and the third to the features.
}
\description{
Function written in C++ for reshaping a matrix containing sequential data into
an array for use with keras.
}
\seealso{
Other Auxiliary Functions: 
\code{\link{get_alpha_3_codes}()},
\code{\link{summarize_tracked_sustainability}()},
\code{\link{to_categorical_c}()}
}
\concept{Auxiliary Functions}
