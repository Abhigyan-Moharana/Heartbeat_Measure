import onnx

model = onnx.load("models/risk_model.onnx")

print("==== INPUTS ====")
for i in model.graph.input:
    print(i.name)

print("\n==== OUTPUTS ====")
for o in model.graph.output:
    print(o.name)

print("\n==== NODES ====")
bad_ops_found = False
for node in model.graph.node:
    domain = node.domain if node.domain else "ai.onnx (core)"
    print(node.op_type, "|", domain)
    if node.domain == "ai.onnx.ml":
        bad_ops_found = True

print("\n==== VERDICT ====")
if bad_ops_found:
    print("❌ ai.onnx.ml domain ops mile — Qualcomm AI Hub reject karega")
else:
    print("✅ Sirf core ONNX ops hain — Qualcomm compatible")