# Rock vs Mine Prediction System

A Machine Learning project that classifies underwater objects as **Rock** or **Mine** using sonar signal measurements. The project uses a Support Vector Machine (SVM) with an RBF kernel and is deployed as an interactive web application using Streamlit.

## Live Demo
[Open Rock vs Mine Prediction System](https://rock-vs-mine-prediction-3qvjdigueodipfyanfzhhj.streamlit.app/)

## Project Overview
The system analyzes 60 numerical features extracted from sonar signals to predict whether an underwater object is a rock or a metal mine.

The model is trained using the UCI Sonar dataset, which contains 208 samples.

## Dataset
- **Source:** [UCI Machine Learning Repository - Sonar Dataset](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks)
- Total samples: 208
- Features: 60 numerical sonar measurements
- Classes:
  - `R`: Rock
  - `M`: Mine

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Machine Learning Workflow
1. Dataset loading and exploration
2. Exploratory Data Analysis (EDA)
3. Data preprocessing
4. Label encoding
5. Train-test splitting
6. Feature standardization using StandardScaler
7. Model training and evaluation
8. Algorithm comparison
9. Model serialization using Joblib
10. Web application development and deployment

## Models Evaluated
| Algorithm | Test Accuracy |
|---|---:|
| Logistic Regression | 83.33% |
| Linear SVM | 76.19% |
| RBF SVM | **85.71%** |

The RBF SVM achieved the highest test accuracy among the three evaluated models.

## Project Structure
```text
rock-vs-mine-prediction/
│
├── app.py
├── requirements.txt
├── rbf_svm_model.pkl
├── scaler.pkl
└── README.md
```

## Run Locally
Clone the repository:

```bash
git clone https://github.com/pallasahitya/rock-vs-mine-prediction.git
```

Navigate to the project directory:

```bash
cd rock-vs-mine-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Application Features
- Interactive input for 60 sonar measurements
- Feature scaling using the saved StandardScaler
- Prediction using the trained RBF SVM model
- Rock or Mine classification results

## Model Performance
The RBF SVM achieved **85.71% accuracy** on the held-out test set (42 samples).

This is an educational machine learning project. Predictions should not be used for real-world mine detection or safety-critical decisions.

## Author

**Sivani Sahitya Palla**

GitHub: [@pallasahitya](https://github.com/pallasahitya)

---

If you found this project interesting, feel free to explore the repository!
