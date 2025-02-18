% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/install_and_config.R
\name{install_py_modules}
\alias{install_py_modules}
\title{Installing necessary python modules to an environment}
\usage{
install_py_modules(
  envname = "aifeducation",
  install = "pytorch",
  transformer_version = "<=4.46",
  tokenizers_version = "<=0.20.4",
  pandas_version = "<=2.2.3",
  datasets_version = "<=3.1.0",
  codecarbon_version = "<=2.8.2",
  safetensors_version = "<=0.4.5",
  torcheval_version = "<=0.0.7",
  accelerate_version = "<=1.1.1",
  pytorch_cuda_version = "12.1",
  python_version = "3.9",
  remove_first = FALSE
)
}
\arguments{
\item{envname}{\code{string} Name of the environment where the packages should be installed.}

\item{install}{\code{character} determining which machine learning frameworks should be installed.
\itemize{
\item \code{install = "all"}: for 'pytorch' and 'tensorflow'.
\item \code{install = "pytorch"}: for 'pytorch'.
\item \code{install = "tensorflow"}: for 'tensorflow'.
}}

\item{transformer_version}{\code{string} determining the desired version of the python library 'transformers'.}

\item{tokenizers_version}{\code{string} determining the desired version of the python library 'tokenizers'.}

\item{pandas_version}{\code{string} determining the desired version of the python library 'pandas'.}

\item{datasets_version}{\code{string} determining the desired version of the python library 'datasets'.}

\item{codecarbon_version}{\code{string} determining the desired version of the python library 'codecarbon'.}

\item{safetensors_version}{\code{string} determining the desired version of the python library 'safetensors'.}

\item{torcheval_version}{\code{string} determining the desired version of the python library 'torcheval'.}

\item{accelerate_version}{\code{string} determining the desired version of the python library 'accelerate'.}

\item{pytorch_cuda_version}{\code{string} determining the desired version of 'cuda' for ' PyTorch'.}

\item{python_version}{\code{string} Python version to use.}

\item{remove_first}{\code{bool} If \code{TRUE} removes the environment completely before recreating the environment and
installing the packages. If \code{FALSE} the packages are installed in the existing environment without any prior
changes.}
}
\value{
Returns no values or objects. Function is used for installing the necessary python libraries in a conda
environment.
}
\description{
Function for installing the necessary python modules.
}
\seealso{
Other Installation and Configuration: 
\code{\link{check_aif_py_modules}()},
\code{\link{install_aifeducation}()},
\code{\link{set_transformers_logger}()}
}
\concept{Installation and Configuration}
