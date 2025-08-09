# A Generalist Deep Learning Framework for Multimodal Attribution: Joint Effects of Text, Image and Advertising on Customer Conversion

This study examines the drivers of online conversion by modeling how multimodal marketing inputs, advertising spending, media channel mix, website text, and image content, jointly shape consumer behavior. We propose a generalist AI framework built on a large language model that scalably integrates structured and unstructured inputs in a unified pipeline. This design enables efficient and interpretable multimodal attribution across diverse marketing elements. To quantify each modality’s contribution, we introduce the Relative Impact Score (RIS), a novel metric that extends Integrated Gradients to capture modality-level attribution in online conversion.
Our findings show that advertising accounts for over half of the total impact on conversion, though its effects are mixed, ranging from positive to antagonistic depending on execution. In contrast, website content, particularly text, plays a more consistent and constructive role, contributing to about 60% of the positive lift in conversion. Digital and social media channels emerge as the most effective, outperforming traditional media in driving lower-funnel outcomes.
This work makes both substantive and methodological contributions by offering a scalable generalist AI approach to multimodal attribution, and by generating actionable insights for firms aiming to optimize content and media strategy in data-rich digital environments.

## Evaluation Platform
This Evaluation Platform is a full end-to-end machine learning experimentation framework for multi-modal sequence classification and generation tasks, supporting models like BERT, GPT-2, and T5.
It integrates data preprocessing, model training, evaluation, attribution analysis, and visualization — all within a configurable Experiment class. The code is designed to run multiple experimental combinations of modalities, models, and seeds, then consolidate the results.

#### 1. Data Handling & Preprocessing
The script defines a custom Dataset class to wrap textual and optional multi-modal data into tokenized sequences, producing inputs and masks tailored to the chosen model type (t5, bert, gpt2). The Experiment class can load CSV datasets containing marketing spending data, funnel metrics, text, and image captions. It computes differences (e.g., text changes) using ROUGE-L, converts continuous variables into categorical scales, and creates sliding window sequences over multiple weeks.
Data is split into train/validation/test sets, converted into PyTorch DataLoaders, and tokenized with Hugging Face tokenizers.
The framework supports different modality combinations — e.g., spending+text, text+image — and can output either classification labels ("one-hot") or generated text sequences.

#### 2. Model Loading, Training, and Testing
Depending on the model type and output type, the script loads the corresponding Hugging Face model (T5ForConditionalGeneration, GPT2LMHeadModel, BertForSequenceClassification, etc.) with proper configuration. Training uses AdamW with a linear warmup schedule, supports fine-tuning or head-only tuning, and logs metrics for each epoch.
The train method tracks loss, runs validation after each epoch, and saves the best model based on validation accuracy. The test method evaluates the model on test (and optionally train/validation) data, computes accuracy, writes predictions to CSV, and logs results into Excel for later aggregation.

#### 3. Integrated Gradients Attribution & Visualization
A key component is the T5IntegratedGradients class, a subclass of Captum’s LayerIntegratedGradients, extended with custom baseline generation methods (pad, zero, gaussblur) and a more controlled gradient computation process.
The attribute method in Experiment runs feature attribution on T5 models, identifying important tokens or subwords for predictions. Attributions are normalized, categorized into scales, and saved in HTML, CSV, and Excel formats.
Phrase-level attribution is computed for semantic groups like “Spending”, “Image”, “Text”.
The plot_attribution_stats method then generates histograms, KDE plots, cumulative distributions, and violin/boxen plots to analyze attribution score distributions by channel or token type.

#### 4. Experiment Orchestration & Result Consolidation
The script’s main function defines multiple experiment configurations by varying model types, output types, modality combinations, and random seeds. Each experiment runs train + test. After all experiments, the consolidate function aggregates accuracy metrics across seeds into a summary Excel file and generates boxen plots comparing performance across modalities and models.
This allows systematic benchmarking over multiple runs and clear visual comparison of results.

Overall, the code is a comprehensive experimental pipeline for multi-modal NLP tasks, with strong support for interpretability (Captum IG), performance tracking, and visualization. It is designed for repeatable, large-scale experiments where multiple configurations and seeds are tested, and interpretability analyses are performed to understand model decision-making.

#### Command for Evaluation

Execute `python3 run.py` to assess the generalist framework for multimodal attribution.

#### Experimental Results

The repository contains three folders: log/5_epochs, log/15_epochs, and log/30_epochs. These represent result files curated from models trained for 5, 15, and 30 epochs, respectively. Within each folder, you’ll find the test result files and figures of seven different combinations of modalities.

