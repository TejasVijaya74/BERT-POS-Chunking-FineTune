# Fine-Tuning DistilBERT for Token Classification

This project demonstrates a complete NLP pipeline for fine-tuning a **DistilBERT** model on token classification tasks, specifically addressing **POS Tagging** and **Phrase Chunking** concepts.

##  Project Overview
The objective is to leverage transformer architectures to perform sequence labeling. While originally designed for Named Entity Recognition (NER), the **wikiann (English)** dataset is used here as a high-performance proxy to illustrate token-level predictions. 

##  Tech Stack
* **Language:** Python 
* **Libraries:** Hugging Face (Transformers, Datasets, Evaluate), PyTorch, Seqeval 
* **Model:** `distilbert-base-uncased`

##  Key Features
- **Label Alignment:** Custom logic using `word_ids()` to handle subword tokenization by masking non-initial subwords with `-100`.
- **Efficient Training:** Reduced dataset footprint for faster iteration in resource-constrained environments.
- **Robust Evaluation:** Performance tracking via Precision, Recall, F1-Score, and Accuracy using the `seqeval` framework. 

##  Workflow
1. **Preprocessing:** Tokenization and subword label alignment. 
2. **Setup:** Configuring `AutoModelForTokenClassification` with dynamic label mapping. 
3. **Training:** Fine-tuning with a $2 \times 10^{-5}$ learning rate over 3 epochs. 
4. **Inference:** Predicting tags on custom sentences (e.g., *"james work at Google in Pune"*). 

##  Observations
- **POS vs. Chunking:** The project highlights the difference between word-level grammatical tagging (Easy) and phrase-level grouping (Medium).
- **Insights:** Transformers significantly outperform traditional methods by capturing intricate contextual relationships within sentences. 
