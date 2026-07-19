import joblib
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

model = joblib.load("models/risk_model.pkl")

initial_type = [("float_input", FloatTensorType([1, 1]))]

options = {id(model): {"zipmap": False}}

onnx_model = convert_sklearn(
    model,
    initial_types=initial_type,
    target_opset=17,
    options=options
)

with open("models/risk_model_temp.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

outputs = [o.name for o in onnx_model.graph.output]
prob_output = next((n for n in outputs if "prob" in n.lower()), outputs[-1])

onnx.utils.extract_model(
    "models/risk_model_temp.onnx",
    "models/risk_model.onnx",
    input_names=["float_input"],
    output_names=[prob_output]
)

# ---- CRITICAL FIX: value_info se duplicate input/output names hatao ----
final_model = onnx.load("models/risk_model.onnx")

io_names = {t.name for t in final_model.graph.input} | {t.name for t in final_model.graph.output}

# value_info me se wo entries hatao jo already input/output me hain
keep = [vi for vi in final_model.graph.value_info if vi.name not in io_names]
del final_model.graph.value_info[:]
final_model.graph.value_info.extend(keep)

onnx.checker.check_model(final_model)  # ab ye pass hona chahiye

onnx.save(final_model, "models/risk_model.onnx")

print(f"✅ Qualcomm-compatible ONNX generated! Final output: {prob_output}")