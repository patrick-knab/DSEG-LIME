# DSEG-LIME Integration Configuration Documentation

Welcome to the DSEG-LIME Integration Configuration Guide. This documentation is designed to help users configure and utilize the DSEG-LIME integration for Explainable AI (XAI) analysis effectively. Below, you'll find detailed information on each configuration option available, allowing for a customized setup tailored to your specific requirements.

## Table of Contents

- [XAI Algorithm Selection](#xai-algorithm-selection)
- [Computation Configuration](#computation-configuration)
- [Model to Explain](#model-to-explain)
- [LIME Segmentation Parameters](#lime-segmentation-parameters)
- [Evaluation](#evaluation)

## XAI Algorithm Selection

This section allows users to enable or disable specific XAI algorithms for their analysis.

```yaml
XAI_algorithm:
  DSEG: True
  LIME: True
  SLIME: True
  BayesLime: True
  GLIME: True
```
- DSEG, LIME, SLIME, BayesLime, GLIME: Enable (True) or disable (False) specific XAI algorithms for analysis.

## Computation Configuration

Settings related to computational resources, such as the number of workers and GPU usage.

```yaml
computation:
  num_workers: 3
  gpu_device: False
  gpu_num: "4"
```
- **num_workers**: Number of worker threads for parallel processing.
- **gpu_device**: Enable (True) or disable (False) the use of GPU for computations.
- **gpu_num**: Identifier for the GPU device to be used, specified as a string.

## Model to Explain

-Define which models are to be analyzed and explained by the selected XAI algorithms.

```yaml
model_to_explain:
  EfficientNet: True
  ResNet: False
  VisionTransformer: False
```
- Indicates the model architectures to be subjected to explanation. Enable (True) or disable (False) as needed.

## LIME Segmentation Parameters

This section details the configuration for the LIME segmentation process, including algorithm parameters and segmentation techniques.

```yaml
lime_segmentation:
  num_samples: 256
  num_features: 1000
  min_weight: 0.01
  top_labels: 1
  hide_color: None
  batch_size: 10
  verbose: True

  slic: True
  quickshift: False
  felzenszwalb: False
  watershed: False

  all_dseg: False
  DETR: False
  SAM: True
  points_per_side: 32
  min_size: 512

  fit_segmentation: True
  slic_compactness: 16
  num_segments: 20
  markers: 16
  kernel_size: 6
  max_dist: 32

  iterations: 1
  shuffle: False
  max_segments: 8
  min_segments: 1
  auto_segment: False

  num_features_explanation: 2
  adaptive_num_features: False
  adaptive_fraction: True

  hide_rest: True
  positive_only: True

```

- **num_samples**: Specifies the total number of perturbations to generate for LIME analysis.
- **num_features**: Determines the number of features to be considered by LIME for constructing explanations.
- **min_weight**: Sets the minimum threshold for weights associated with each feature in the image explanation.
- **top_labels**: Defines the maximum number of top-ranking classes to include in the explanation outcome.
- **hide_color**: Specifies the color used to hide superpixels during LIME analysis, impacting the perturbation process.
- **batch_size**: Controls the number of samples processed in parallel during the LIME explanation generation.
- **verbose**: Enables detailed progress output, useful for monitoring the explanation generation process.
- **slic**: Enables the use of the SLIC segmentation technique when set to `True`.
- **quickshift**: Activates Quickshift as the segmentation method when set to `True`.
- **felzenszwalb**: Activates Felzenszwalb as the segmentation method when set to `True`.
- **watershed**: Applies the Watershed segmentation technique when this boolean is set to `True`.
- **all_dseg**: Ignores previous segmentation settings and applies DSEG universally when `True`.
- **DETR**: Employs DETR for segmentation, suitable only for iteration = 1. However, vanilla DSEG works with SAM!
- **SAM**: Selects SAM (Sequential Area Maximum) for the segmentation process.
- **points_per_side**: Configures the number of points per side used in SAM segmentation.
- **min_size**: Specifies the minimum size for segments generated during segmentation.
- **fit_segmentation**: Determines whether segmentations should be adjusted for DSEG during quantitative evaluations.
- **slic_compactness**: Sets the compactness parameter for SLIC, affecting the segmentation's spatial proximity.
- **num_segments**: Controls the total number of segments to produce during segmentation.
- **markers**: Defines the number of markers used in certain segmentation techniques.
- **kernel_size**: Adjusts the kernel size for algorithms requiring a convolution operation.
- **max_dist**: Sets the maximum distance parameter for segmentation algorithms that utilize spatial distance.
- **iterations**: Indicates the depth of hierarchy considered in DSEG.
- **shuffle**: Enables shuffling of model predictions during quantitative evaluations to test explanation robustness.
- **max_segments**: Sets an upper limit on the number of segments that can be generated for an image.
- **min_segments**: Establishes a lower boundary for the minimum number of segments to produce.
- **auto_segment**: Previously used to automatically decide between using minimum or maximum segment count based on testing results, now deprecated.
- **num_features_explanation**: Defines the depth of the hierarchical explanation in DSEG.
- **adaptive_num_features**: Was intended to incrementally add features to change the class of the explanation, no longer in use.
- **adaptive_fraction**: Adjusts the number of explanation coefficients adaptively, based on the interquartile range (IQR) as discussed in the paper.
- **hide_rest**: Determines whether to obscure the remainder of the image, focusing the explanation on key features only.
- **positive_only**: Restricts the explanation to positively impacting features, omitting those with a negative influence.

## Evaluation

Parameters under this section control aspects such as the number of samples for LIME, segmentation techniques to be applied, and parameters specific to each segmentation method.

```yaml
evaluation:
  noisy_background: True

  model_randomization: True
  explanation_randomization: True

  single_deletion: True
  fraction: 0.1
  fraction_std: 0.05

  incremental_deletion: True
  incremental_deletion_fraction: 0.15

  stability: True
  repetitions: 8

  preservation_check: True
  deletion_check: True

  variation_stability: True

  target_discrimination: True

  size: True
```

- **noisy_background**: Enables the addition of a noisy background to assess the robustness of the explanation against visual noise.
- **model_randomization**: Activates the randomization of model weights to evaluate the explanation's dependency on the specific model parameters.
- **explanation_randomization**: Tests the stability of the explanation by randomizing the explanation features and observing the impact on the explanation's coherence.
- **single_deletion**: Conducts a single deletion test, removing one feature at a time to measure the impact on the model's output, thereby assessing feature importance.
- **fraction**: Sets the fraction of features to be deleted in the single deletion test, providing a quantitative measure of feature significance.
- **fraction_std**: Specifies the standard deviation for the fraction of features to be deleted, introducing variability in the single deletion test for a more robust assessment.
- **incremental_deletion**: Enables the incremental removal of features based on their importance, allowing for the evaluation of explanation quality as more information is omitted.
- **incremental_deletion_fraction**: Defines the fraction for features to be incrementally deleted, offering a granular view of feature contribution to the model prediction.
- **stability**: Assesses the explanation's stability by generating multiple explanations under slight variations and measuring the consistency of feature importance rankings.
- **repetitions**: Determines the number of repetitions for generating explanations in stability and other tests, ensuring a statistically significant evaluation.
- **preservation_check**: Verifies the preservation of the original model's output when important features (as identified by the explanation) are retained, testing the explanation's fidelity.
- **deletion_check**: Examines the impact on the model's output when important features are deleted, further assessing the accuracy of the explanation.
- **variation_stability**: Evaluates the stability of explanations across different model variations, testing the explanation's robustness to model changes.
- **target_discrimination**: Measures the explanation's ability to discriminate between model to be explained and another model.
- **size**: Assesses the compactness of the explanation by evaluating the number of features included, with a focus on maintaining explanation simplicity while preserving informativeness.

