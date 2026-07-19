import numpy as np
import onnxruntime as ort

def run_inference(heart_rate: float, model_path: str = "models/risk_model.onnx"):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    input_data = np.array([[heart_rate]], dtype=np.float32)
    result = session.run(None, {input_name: input_data})

    probs = result[0]
    predicted_class = int(np.argmax(probs, axis=1)[0])
    print(f"Heart Rate: {heart_rate} -> Predicted class idx: {predicted_class}, Probabilities: {probs}")
    return predicted_class, probs

if __name__ == "__main__":
    run_inference(95.0)