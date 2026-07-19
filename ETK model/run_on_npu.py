import os
import numpy as np
import onnxruntime_qnn as q
import onnxruntime as ort

# QNN provider register karo — ye line pehle missing thi
os.add_dll_directory(os.path.dirname(q.__file__))
ort.register_execution_provider_library("QNNExecutionProvider", q.get_library_path())

print("Available providers after registration:", ort.get_available_providers())

sess = ort.InferenceSession(
    "risk_model/models/model.onnx",
    providers=["QNNExecutionProvider", "CPUExecutionProvider"],
    provider_options=[{"backend_path": "QnnHtp.dll"}, {}]
)

heart_rate = 95.0
input_data = np.array([[heart_rate]], dtype=np.float32)
result = sess.run(None, {"float_input": input_data})

print("Heart Rate:", heart_rate)
print("Model Output (probabilities):", result)

npu_devices = [
    d for d in ort.get_ep_devices()
    if d.ep_name == "QNNExecutionProvider" and str(d.device.type).endswith("NPU")
]
print("NPU device found:", bool(npu_devices))

active_providers = sess.get_providers()
print("Session actually using:", active_providers)