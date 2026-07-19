import qai_hub as hub

# Upload ONNX model
model = hub.upload_model("risk_model/models/risk_model.onnx")

print("Model uploaded successfully!")
print("Model ID:", model.model_id)