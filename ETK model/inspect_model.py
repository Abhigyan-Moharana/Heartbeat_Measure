import qai_hub as hub

model = hub.get_model("mq313086n")

print(model)
print(type(model))

# jitni information mile sab print karo
print(dir(model))