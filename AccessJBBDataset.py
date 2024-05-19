import jailbreakbench as jbb
import json

dataset = jbb.read_dataset()

df = dataset.as_dataframe()

print(df.head())
print(len(df))

prompts = []
for index,row in df.iterrows():
    prompts.append({"query": row["Goal"], "reference_responses": row["Target"],"category": row["Category"], "behavior":row["Behavior"]})

with open("/data/lihongliang/LLMSafety/datasets/JailbreakBench/en_data.json", "w") as file:
    data = json.dump(prompts, file, ensure_ascii=False,indent=4)

# df.to_csv("/data/lihongliang/LLMSafety/datasets/JailbreakBench/jbb.csv", index=False)