# Text→Image→Text→Image Semantic Drift Analysis

This repository contains the implementation and results of the project:
**“Text to Picture and Back Again: Quantifying Semantic Drift in Generative Pipelines.”**

## Overview

This project investigates semantic preservation in a multimodal **Text→Image→Text→Image (T→I→T→I)** pipeline. Given structured and vague prompts across categories (place vs. non-place), the system:

1. Generates images with **Stable Diffusion 2.1**,
2. Converts images back to text with **BLIP-2**, and
3. Computes semantic similarity across the pipeline using **SBERT**, **BERTScore**, **Word overlap**, and **CLIP image similarity**.

We evaluate:

* Semantic drift across the T→I→T→I cycle,
* Differences across content types and prompt complexity,
* Correlations between visual and textual metrics.

## Experimental Design

* **Prompts Matrix:** 120 prompts varying along:
  * Place vs. non-place
  * Structured vs. vague
* **Models:** Stable Diffusion 2.1 (image), BLIP-2 (captioning), SentenceBERT & BERTScore (text), CLIP (image).
* **Metrics:** SBERT similarity, BERTScore F1, word overlap, CLIP image similarity.
* **Statistical tests:** Mann-Whitney U tests, Spearman correlations, bootstrap confidence intervals.

## Repository Structure

```
├── data_loader.py           # Load and structure prompts
├── pipeline.py              # T→I→T→I pipeline implementation
├── similarity_metrics.py    # Text and image similarity computations
├── analysis.py              # Statistical analyses & visualizations
├── main.py                  # Main execution script
├── prompt_matrix.csv        # Input prompt data
├── results-figures          # figures and statistics
├── README.md                # Project overview
```

## Results & Figures

See the relevant directory for:

* Metric comparisons
* Prompt-type and content-type analysis
* Multimodal similarity scatterplots

## Key Findings

* Semantic preservation is modest (\~SBERT 0.23 mean).
* Structured prompts outperform vague prompts slightly.
* Place prompts show minimal advantage over non-place prompts.
* CLIP image similarity (\~0.67 mean) indicates that coarse visual features are mostly preserved.
* Moderate text–image correlation (\~0.349).

## Future Work

Future work may include:

* Applying newer image-text models like DALL·E 3.
* Exploring more robust similarity metrics.
* Investigating more diverse prompt types.

## Authors

- Harun Eroglu
- Sebastiano Franchini

For questions, please contact one of the contributors.

## License

MIT License
