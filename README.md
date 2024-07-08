# Voice Gender Recognition
## About Dataset

### Voice Gender
**Gender Recognition by Voice and Speech Analysis**

This database was created to identify a voice as male or female, based upon acoustic properties of the voice and speech. The dataset consists of **3,168** recorded voice samples, collected from male and female speakers. The voice samples are pre-processed by acoustic analysis in R using the seewave and tuneR packages, with an analyzed frequency range of **0Hz-280Hz** (human vocal range).

## Dataset Details

The CSV file (`voice.csv`) includes the following acoustic properties for each voice sample:

- `meanfreq`: Mean frequency (in kHz)
- `sd`: Standard deviation of frequency
- `median`: Median frequency (in kHz)
- `Q25`: First quantile (in kHz)
- `Q75`: Third quantile (in kHz)
- `IQR`: Interquartile range (in kHz)
- `skew`: Skewness
- `kurt`: Kurtosis
- `sp.ent`: Spectral entropy
- `sfm`: Spectral flatness
- `mode`: Mode frequency
- `centroid`: Frequency centroid
- `peakf`: Peak frequency (frequency with highest energy)
- `meanfun`: Average fundamental frequency measured across acoustic signal
- `minfun`: Minimum fundamental frequency measured across acoustic signal
- `maxfun`: Maximum fundamental frequency measured across acoustic signal
- `meandom`: Average dominant frequency measured across acoustic signal
- `mindom`: Minimum dominant frequency measured across acoustic signal
- `maxdom`: Maximum dominant frequency measured across acoustic signal
- `dfrange`: Range of dominant frequency measured across acoustic signal
- `modindx`: Modulation index
- `label`: Gender label (`male` or `female`)


## Setup and Installation

### Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

### Install Dependencies:
Ensure you have Python and pip installed. Use pip to install the required packages:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost
```
### Download the Dataset:
You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/primaryobjects/voicegender/data).

Place the `voice.csv` file in the appropriate directory (`voicegender/voice.csv`).

### Run the Code:
Open the Jupyter notebook or Python script that contains the code to analyze the dataset and train the classifiers.

# Analysis

- **Accuracy**: Random Forest Classifier achieved the highest test accuracy of **98.58%**, closely followed by Support Vector Classifier (**98.26%**) and XGB Classifier (**98.11%**). Decision Tree Classifier, despite perfect training accuracy (100%), has a lower test accuracy (**96.69%**), indicating potential overfitting.

- **F1 Score**: Random Forest Classifier also led in F1 scores on the test set with **98.65%**, followed by Support Vector Classifier (**98.36%**) and XGB Classifier (**98.20%**). These metrics indicate good balance between precision and recall.

- **ROC AUC Score**: Random Forest Classifier and Support Vector Classifier both achieved high ROC AUC scores on the test set (**98.62%** and **98.29%** respectively), indicating excellent discrimination ability.

# Conclusion and Recommendations

Based on the performance metrics:

1. **Random Forest Classifier** emerges as the top performer across all evaluated metrics (accuracy, F1 score, ROC AUC score), indicating robust performance and good generalization ability. It consistently achieved high scores on both training and test sets.

2. **Support Vector Classifier** and **XGB Classifier** also demonstrated strong performance, especially in terms of accuracy and ROC AUC score. They are suitable alternatives if computational efficiency or interpretability is a concern compared to Random Forest.

3. **Decision Tree Classifier**, while achieving perfect training accuracy, shows signs of overfitting as indicated by its lower test accuracy and F1 score. Further regularization or ensemble methods could potentially improve its performance.

4. **Ridge Classifier** and **KNN Classifier** performed well but slightly below the top performers in terms of accuracy and F1 score metrics.

## License
[MIT License](LICENSE)
