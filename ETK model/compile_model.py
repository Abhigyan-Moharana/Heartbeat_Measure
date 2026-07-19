import qai_hub as hub

model = hub.get_model("mq9gy0wln")  # apna latest uploaded model ID daalo

compile_job = hub.submit_compile_job(
    model=model,
    device=hub.Device("Snapdragon X Elite CRD"),   # <-- Step 1 se mila exact string yaha daalo
    input_specs={"float_input": (1, 1)},
)

print("Compile Job ID:", compile_job.job_id)
print("Status:", compile_job.wait().state)