🧬 **BioBERT-Based Genomic Cancer Classifier Web App**

This project showcases an advanced, end-to-end Machine Learning pipeline that uses Transfer Learning to classify cancer types from short DNA sequence mutations. We then deploy this model as a real-time web service using Flask.

The goal was to build a fully functional system that takes raw DNA sequence input and predicts one of three major synthetic cancer types: Breast Cancer (BRCA), Lung Cancer (LUNG), and Colon Adenocarcinoma (COAD).


✨ **Key Features & Technologies**

**Transfer Learning with BioBERT:** We fine-tuned the dmis-lab/biobert-base-cased-v1.1 model, which is pre-trained on massive biomedical text data, for our specific genomic multi-class classification task.

**Genomic Sequence Classification:** This project treats DNA sequences as "text" tokens, validating the use of powerful NLP models (like BERT) for bioinformatics and genomic data analysis.

**End-to-End MLOps Pipeline:** The architecture is modular, clearly separating the model training logic (train.py) from the model serving logic (app.py).

**Web Deployment:** We use the lightweight Flask web framework to provide a responsive API and User Interface (UI) for real-time diagnostic predictions.

**Deep Learning Stack:** The project relies on PyTorch and the HuggingFace Transformers library for efficient training, handling, and loading of the large BioBERT model.


🚀 **Getting Started**

Follow these steps to get a working copy of the project running on your local machine.PrerequisitesYou'll need Python 3.8+ and pip installed.Bash# Optional: Create a new virtual environment to keep dependencies isolated
**python -m venv venv
source venv/bin/activate**

**Installation**
**Clone the repository**:

**git clone https://github.com/Roohi2204/BioBERT-Cancer-Classifier-Flask.git**

**cd BioBERT-Cancer-Classifier-Flask**

**Install the dependencies**:

**pip install -r requirements.txt**

**Step 1:** Train and Save the ModelRun the training script first. This process downloads the pre-trained BioBERT model and fine-tunes it on the synthetic data, saving the results locally.Generates 1,000 synthetic DNA sequence samples.Fine-tunes the BioBERT model for 3 epochs.Saves the final model weights, tokenizer, and label map to a new ./model/ directory.

**python train.py**

**Step 2:** Run the Web ApplicationOnce the model/ directory is created, you can start the web service.

**python app.py**

The application will launch at **http://127.0.0.1:5000/**. 
Open this URL in your browser to access the prediction interface.


**How to Test (Example Patterns)**

You can test the classifier using sequences that contain the synthetic markers the model was trained on:

Cancer Type         -         Synthetic Marker Sequence


BRCA                -           ACGT

LUNG                -         GATTACA

COAD                -         TTAG

