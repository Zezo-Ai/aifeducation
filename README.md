
<!-- README.md is generated from README.Rmd. Please edit that file -->

# aifeducation

<!-- badges: start -->

**GitHub**
[![R-CMD-check](https://github.com/FBerding/aifeducation/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/FBerding/aifeducation/actions/workflows/R-CMD-check.yaml)
**CRAN** [![CRAN
status](https://www.r-pkg.org/badges/version/aifeducation)](https://CRAN.R-project.org/package=aifeducation)
[![Downloads](http://cranlogs.r-pkg.org/badges/grand-total/aifeducation)](https://cran.r-project.org/package=aifeducation)
<!-- badges: end -->

The R package *Artificial Intelligence for Education (aifeducation)* is
designed for the special requirements of educators, educational
researchers, and social researchers. The target audience of this package
are educators and researchers who would like to develop their own models
as well as people who would like to use models created by other
researchers/educators. The package supports the application of
Artificial Intelligence (AI) for Natural Language Processing tasks such
as text embedding and classification under the special conditions of the
educational and social sciences.

## Features Overview

- Simple usage of artificial intelligence by providing routines for the
  most important tasks for educators and researchers from the social and
  educational sciences.
- Supporting both ‘PyTorch’ and ‘Tensorflow’ as machine learning
  frameworks.
- Supporting the usage of trained models on both frameworks providing a
  high level of flexibility
- Supporting pre-trained language models from Hugging Face.
- Supporting BERT, RoBERTa, DeBERTa, Longformer, and Funnel Transformer
  for creating context sensitive text embedding.
- Making sharing pre-trained models very easy.
- Integrating sustainability tracking.
- Integrating special statistical techniques for dealing with data
  structures common in the social and educational sciences.
- Supporting the classification of long textual documents.

Currently, the package focuses on classification tasks which can either
be used to diagnose characteristics of learners from written material or
to estimate the properties of learning and teaching material. In the
future, more tasks will be implemented.

## Installation

You can install the package from CRAN with:

``` r
install.packages("aifeducation")
```

You can install the development version of aifeducation from
[GitHub](https://github.com/) with:

``` r
install.packages("devtools")
devtools::install_github(repo="FBerding/aifeducation",
                         ref="master",
                         dependencies = TRUE)
```

Further instructions for installation can be found in vignette [01 get
started](https://fberding.github.io/aifeducation/articles/aifeducation.html).

> **Note:** There is currently a bug in reticulate 1.32.0 which does not
> allow to load python packages such as tensorflow, keras or torch. The
> error seems to occur only when working with RStudio and Windows
> machines. Using R console directly seems not to evoke the error. The
> bug has been fixed in the developer version of reticulate. You can
> install this version with `install.packages("pak")` and
> `pak::pak("rstudio/reticulate")`. With the release of reticulate
> 1.33.0 you can ignore this note.

## Sustainability

Training AI models consumes time and energy. To help researchers
estimate the ecological impact of their work, a sustainability tracker
is implemented. It is based on the python library codecarbon by Courty
et al. (2023). This tracker allows to estimate the energy consumption
for CPUs, GPUs and RAM during training and derives a value for the CO2
emission. This value is based on the energy mix in the country where the
computer is located.

## PyTorch and Tensorflow Compatiblity

This package allows all supported models based either on ‘PyTorch’ or
‘tensorflow’, providing a high level of flexibility. Even pre-trained
models can be used with both frameworks. In general, users only have to
decide which framework they would like to use when starting a new
session.

At the moment, using ‘PyTorch’ for classifier objects requires
‘keras-core’ or ‘keras3’. If your system does not support these python
packages you can still use classifiers with ‘tensorflow’.

## Classification Tasks

### Transforming Texts into Numbers

Classification tasks require the transformation of raw texts into a
representation with numbers. For this step, *aifeducation* supports both
newer approaches such as BERT (Devlin et al. 2019), RoBERTa (Liu et al.
2019), DeBERTa version 2 (He et al. 2020), Funnel-Transformer (Dai et
al. 2020), and Longformer (Beltagy, Peters & Cohan 2020) and older
approaches such as GlobalVectors (Pennington, Socher & Manning 2014) or
Latent Dirichlet Allocation/Topic Modeling in classification tasks.

*aifeducation* supports the use of pre-trained transformer models
provided by [Hugging Face](https://huggingface.co/) and the creation of
new transformers, allowing educators and researchers to develop
specialized and domain-specific models.

The package supports the analysis of long texts. Depending on the
method, long texts are transformed into vectors at once or, if too long,
are split into several chunks which result in a sequence of vectors.

### Training AI under Challenging Conditions

For the second step within a classification task, *aifeducation*
integrates some important statistical and mathematical methods for
dealing with the main challenges in educational and social sciences for
applying AI. These are:

- **digital data availability:** In the educational and social sciences,
  data is often only available in handwritten form. For example, in
  schools or universities, students often solve tasks by creating
  handwritten documents. Thus, educators and researchers first have to
  transform analogue data into a digital form, involving human action.
  This makes data generation financially expensive and time-consuming,
  leading to *small data sets*.
- **high privacy policy standards:** Furthermore, in the educational and
  social sciences, data often refers to humans and/or their actions.
  These kinds of data are protected by privacy policies in many
  countries, limiting access to and the usage of data, which also
  results in *small data sets*.
- **long research tradition:** Educational and social sciences have a
  long research tradition in generating insights into social phenomena
  as well as learning and teaching. These insights have to be
  incorporated into applications of AI (e.g., Luan et al. 2020; Wong et
  al. 2019). This makes supervised machine learning a very important
  technology since it provides a link between educational and social
  theories or models on the one hand and machine learning on the other
  hand (Berding et al. 2022). However, this kind of machine learning
  requires humans to generate a valid data set for the training process,
  leading to *small data sets*.
- **complex constructs:** Compared to classification tasks where, for
  instance, AI has to differentiate between a ‘good’ or ‘bad’ movie
  review, constructs in the educational and social sciences are more
  complex. For example, some research instruments in motivational
  psychology require to infer personal motifs from written essays (e.g.,
  Gruber & Kreuzpointner 2013). A reliable and valid interpretation of
  this kind of information requires well qualified human raters, making
  data generation expensive. This also *limits the size of a data set*.
- **imbalanced data:** Finally, data in the educational and social
  sciences often occurs in an imbalanced pattern as several empirical
  studies show (Bloemen 2011; Stütz et al. 2022). Imbalanced means that
  some categories or characteristics of a data set have very high
  absolute frequencies compared to other categories and characteristics.
  Imbalance during AI training guides algorithms to focus and prioritize
  the categories and characteristics with high absolute frequencies,
  increasing the risk to miss categories/characteristics with low
  frequencies (Haixiang et al. 2017). This can lead AI to prefer special
  groups of people/material, imply false recommendations and
  conclusions, or to miss rare categories or characteristics.

In order to deal with the problem of imbalanced data sets, the package
integrates the *Synthetic Minority Oversampling Technique* into the
learning process. Currently, the *Basic Synthetic Minority Oversampling
Technique* (Chawla et al. 2002), *Density-Bases Synthetic Minority
Oversampling Technique* (Bunkhumpornpat, Sinapiromsaran & Lursinsap
2012), and *Adaptive Synthetic Sampling Approach for Imbalanced
Learning* (Hem Garcia & Li 2008) are implemented via the *R* package
*smotefamiliy*.

In order to address the problem of small data sets, training loops of AI
integrate *pseudo-labeling* (e.g., Lee 2013). Pseudo-labeling is a
technique which can be used for supervised learning. More specifically,
educators and researchers rate a part of a data set and train AI with
this very part. The remainder of the data is not processed by humans.
Instead, AI uses this part of data to learn on its own. Thus, educators
and researchers only have to provide additional data for the AI’s
learning process without coding it themselves. This offers the
possibility to add more data to the training process and to reduce labor
costs.

### Evaluating Performance

Classification tasks in machine learning are comparable to the empirical
method of *content analysis* from the social sciences. This method looks
back on a long research tradition and an ongoing discussion on how to
evaluate the reliability and validity of generated data. In order to
provide a link to this research tradition and to provide educators as
well as educational and social researchers with performance measures
they are more familiar with, every AI trained with this package is
evaluated with the following measures and concepts:

- Iota Concept of the Second Generation (Berding & Pargmann 2022)
- Krippendorff’s Alpha (Krippendorff 2019)
- Percentage Agreement
- Gwet’s AC1/AC2 (Gwet 2014)
- Kendall’s coefficient of concordance W
- Cohen’s Kappa with equal weights
- Fleiss’ Kappa for multiple raters with exact estimation
- Light’s Kappa for multiple raters

## Sharing Trained AI

Since the package is based on keras, tensorflow, and the transformer
libraries, every trained AI can be shared with other educators and
researchers. The package supports an easy use of pre-trained AI within
*R*, but also provides the possibility to export trained AI to other
environments.

Using a pre-trained AI for classification only requires the classifier
and the corresponding text embedding model. Just load both to *R* and
start predictions. Vignette [02 classification
tasks](https://fberding.github.io/aifeducation/articles/classification_tasks.html)
describes how to save and load the objects. In vignette [03 Sharing and
Using Trained
AI/Models](https://fberding.github.io/aifeducation/articles/sharing_and_publishing.html)
you can find a detailed guide on how to document and share your models.

## Tutorial and Guides

- Installation and configuration of the package: [01 get
  started](https://fberding.github.io/aifeducation/articles/aifeducation.html).
- A short introduction into the package with examples for classification
  tasks:[02 classification
  tasks](https://fberding.github.io/aifeducation/articles/classification_tasks.html).
- A description for sharing models: [03 sharing and using trained
  AI/models](https://fberding.github.io/aifeducation/articles/sharing_and_publishing.html)

## References

Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The
Long-Document Transformer. <https://doi.org/10.48550/arXiv.2004.05150>

Berding, F., & Pargmann, J. (2022). Iota Reliability Concept of the
Second Generation. Berlin: Logos. <https://doi.org/10.30819/5581>

Berding, F., Riebenbauer, E., Stütz, S., Jahncke, H., Slopinski, A., &
Rebmann, K. (2022). Performance and Configuration of Artificial
Intelligence in Educational Settings.: Introducing a New Reliability
Concept Based on Content Analysis. Frontiers in Education, 1-21.
<https://doi.org/10.3389/feduc.2022.818365>

Bloemen, A. (2011). Lernaufgaben in Schulbüchern der Wirtschaftslehre:
Analyse, Konstruktion und Evaluation von Lernaufgaben für die Lernfelder
industrieller Geschäftsprozesse. Hampp.

Bunkhumpornpat, C., Sinapiromsaran, K., & Lursinsap, C. (2012). DBSMOTE:
Density-Based Synthetic Minority Over-sampling Technique. Applied
Intelligence, 36(3), 664–684.
<https://doi.org/10.1007/s10489-011-0287-y>

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).
SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial
Intelligence Research, 16, 321–357. <https://doi.org/10.1613/jair.953>

Courty, B., Schmidt, V., Goyal-Kamal, Coutarel, M., Feld, B., Lecourt,
J., & … (2023). mlco2/codecarbon: v2.2.7.
<https://doi.org/10.5281/zenodo.8181237>

Dai, Z., Lai, G., Yang, Y. & Le, Q. V. (2020). Funnel-Transformer:
Filtering out Sequential Redundancy for Efficient Language Processing.
<https://doi.org/10.48550/arXiv.2006.03236>

Devlin, J., Chang, M.‑W., Lee, K., & Toutanova, K. (2019). BERT:
Pre-training of Deep Bidirectional Transformers for Language
Understanding. In J. Burstein, C. Doran, & T. Solorio (Eds.),
Proceedings of the 2019 Conference of the North (pp. 4171–4186).
Association for Computational Linguistics.
<https://doi.org/10.18653/v1/N19-1423>

Gruber, N., & Kreuzpointner, L. (2013). Measuring the reliability of
picture story exercises like the TAT. PloS One, 8(11), e79450.
<https://doi.org/10.1371/journal.pone.0079450>

Gwet, K. L. (2014). Handbook of inter-rater reliability: The definitive
guide to measuring the extent of agreement among raters (Fourth
edition). STATAXIS.

Haixiang, G., Yijing, L., Shang, J., Mingyun, G., Yuanyue, H., & Bing,
G. (2017). Learning from class-imbalanced data: Review of methods and
applications. Expert Systems with Applications, 73, 220–239.
<https://doi.org/10.1016/j.eswa.2016.12.035>

He, H., Bai, Y., Garcia, E. A., & Li, S. (2008). ADASYN: Adaptive
synthetic sampling approach for imbalanced learning. In 2008 IEEE
International Joint Conference on Neural Networks (IEEE World Congress
on Computational Intelligence) (pp. 1322–1328). IEEE.
<https://doi.org/10.1109/IJCNN.2008.4633969>

He, P., Liu, X., Gao, J. & Chen, W. (2020). DeBERTa: Decoding-enhanced
BERT with Disentangled Attention.
<https://doi.org/10.48550/arXiv.2006.03654>

Krippendorff, K. (2019). Content Analysis: An Introduction to Its
Methodology (4th Ed.). SAGE.

Lee, D.‑H. (2013). Pseudo-Label: The Simple and Efficient
Semi-Supervised Learning Method for Deep Neural Networks. CML 2013
Workshop: Challenges in Representation Learning.

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O.,
Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A Robustly
Optimized BERT Pretraining Approach.
<https://doi.org/10.48550/arXiv.1907.11692>

Luan, H., Geczy, P., Lai, H., Gobert, J., Yang, S. J. H., Ogata, H.,
Baltes, J., Guerra, R., Li, P., & Tsai, C.‑C. (2020). Challenges and
Future Directions of Big Data and Artificial Intelligence in Education.
Frontiers in Psychology, 11, 1–11.
<https://doi.org/10.3389/fpsyg.2020.580820>

Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global
Vectors for Word Representation. Proceedings of the 2014 Conference on
Empirical Methods in Natural Language Processing.
<https://aclanthology.org/D14-1162.pdf>

Stütz, S., Berding, F., Reincke, S., & Scheper, L. (2022).
Characteristics of learning tasks in accounting textbooks: an AI
assisted analysis. Empirical Research in Vocational Education and
Training, 14(1). <https://doi.org/10.1186/s40461-022-00138-2>

Wong, J., Baars, M., Koning, B. B. de, van der Zee, T., Davis, D.,
Khalil, M., Houben, G.‑J., & Paas, F. (2019). Educational Theories and
Learning Analytics: From Data to Knowledge. In D. Ifenthaler, D.-K. Mah,
& J. Y.-K. Yau (Eds.), Utilizing Learning Analytics to Support Study
Success (pp. 3–25). Springer.
<https://doi.org/10.1007/978-3-319-64792-0_1>
