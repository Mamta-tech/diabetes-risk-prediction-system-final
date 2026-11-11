from pathlib import Path
import os
import joblib
import gradio as gr
import pandas as pd

BASE = Path(__file__).resolve().parent
MODEL_FILE = BASE / "diabetes_model.pkl"


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Trained model is missing. Render must run train_model.py during the build step.")
    return joblib.load(MODEL_FILE)

bundle = load_model()
model = bundle["model"]
FEATURES = bundle["features"]


def predict(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age):
    values = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]
    if any(v is None for v in values):
        return "Please enter all required values.", ""
    row = pd.DataFrame([values], columns=FEATURES)
    probability = float(model.predict_proba(row)[0][1])
    prediction = int(probability >= 0.5)
    if prediction:
        headline = "Higher predicted risk — please consult a qualified healthcare professional."
    else:
        headline = "Lower predicted risk in this model — this does not rule out diabetes."
    detail = f"Model-estimated probability: {probability * 100:.1f}%"
    return headline, detail

css = """
.gradio-container {max-width: 1050px !important; margin: auto !important;}
#hero {padding: 26px; border-radius: 18px; background: linear-gradient(120deg,#10243a,#1d5362); color: white;}
#hero h1 {font-size: 32px; margin-bottom: 8px;}
#result {border-radius: 14px;}
"""
with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="blue"), css=css, title="Diabetes Risk Prediction System") as demo:
    gr.HTML("""
    <div id='hero'>
      <h1>Diabetes Risk Prediction System</h1>
      <p>Machine-learning based educational risk estimation</p>
      <p><b>Group Project</b> · Mamta (24011041049) &nbsp; | &nbsp; Kirti (24011041041)<br>BCA 5th Semester · Hindu College of Engineering, Sonipat</p>
    </div>
    """)
    gr.Markdown("### Enter the health measurements\nFill in the fields below, then select **Predict Risk**.")
    with gr.Row():
        with gr.Column():
            pregnancies = gr.Number(label="Pregnancies", value=1, precision=0)
            glucose = gr.Number(label="Glucose level (mg/dL)", value=120)
            blood_pressure = gr.Number(label="Diastolic blood pressure (mm Hg)", value=70)
            skin_thickness = gr.Number(label="Skin thickness (mm)", value=20)
        with gr.Column():
            insulin = gr.Number(label="Insulin (mu U/mL)", value=80)
            bmi = gr.Number(label="BMI", value=25.0)
            pedigree = gr.Number(label="Diabetes pedigree function", value=0.5)
            age = gr.Number(label="Age (years)", value=30, precision=0)
    run = gr.Button("Predict Risk", variant="primary")
    result = gr.Textbox(label="Prediction", interactive=False, elem_id="result")
    probability = gr.Textbox(label="Model output", interactive=False)
    run.click(predict, inputs=[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age], outputs=[result, probability])
    gr.Markdown("""
    ---
    **Important medical notice:** This student project is for educational demonstration only. It is not a medical device, diagnosis, screening service, or substitute for professional medical advice. The model is trained on the Pima Indians dataset, which represents a limited population (women aged 21 or older of Pima heritage); its predictions may not generalize to other populations. Do not use its output to make healthcare decisions.

    **Developed by:** Mamta and Kirti · BCA 5th Semester · Hindu College of Engineering, Sonipat.
    """)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)
