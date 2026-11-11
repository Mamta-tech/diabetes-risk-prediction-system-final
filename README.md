# Diabetes Risk Prediction System

A BCA 5th Semester group project that demonstrates a machine-learning approach to estimating diabetes risk from selected health measurements.

## Group Members
- **Mamta** — Roll No. **24011041049**
- **Kirti** — Roll No. **24011041041**

**Class:** BCA 5th Semester  
**College:** Hindu College of Engineering, Sonipat  
**GitHub owner (temporary):** [harshbalan](https://github.com/harshbalan)

## Technology Used
Python, Pandas, Scikit-learn, Random Forest Classifier, Joblib and Gradio.

## Features
- Simple web interface for entering eight health measurements.
- Random Forest classification model.
- Displays a predicted risk category and model-estimated probability.
- Missing measurement handling through median imputation.

## Run Locally
1. Install Python 3.10 or later.
2. Open a terminal in this project folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the application:
   ```bash
   python app.py
   ```
5. Run `python train_model.py` once to download the public dataset, train the model and save `diabetes_model.pkl`.
6. Run `python app.py` and open the local Gradio link shown in the terminal.

## Deploy on Render
- Create a **Web Service** from this GitHub repository.
- Runtime: Python.
- Build command: `pip install -r requirements.txt`
- Start command: `python app.py`
- The build command installs dependencies and trains the model before the web service starts. The app then uses Render’s assigned `PORT`.

## Dataset and Model
The training script uses the Pima Indians Diabetes dataset (768 records, eight input features and one target). Dataset mirror: [CSV source](https://github.com/mikeizbicki/datasets/blob/master/csv/uci/pima-indians-diabetes.csv). The original dataset is historically associated with the UCI Machine Learning Repository and the National Institute of Diabetes and Digestive and Kidney Diseases. The dataset is limited to women aged 21 or older of Pima heritage; model performance and generalizability are limited accordingly.

## Important Disclaimer
This is an educational student project only. It is not a medical device and must not be used to diagnose diabetes, assess an individual's actual medical risk, or guide treatment. A healthcare professional and appropriate clinical tests are required for medical decisions.

## Repository
Temporary GitHub owner: https://github.com/harshbalan  
Suggested repository name: `diabetes-risk-prediction-system`

## Group Project Credits
Developed by **Mamta** and **Kirti**, BCA 5th Semester, Hindu College of Engineering, Sonipat.
