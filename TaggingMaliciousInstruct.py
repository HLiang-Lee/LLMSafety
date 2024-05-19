import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import fire
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
from llama_recipes.inference.prompt_format_utils import build_default_prompt, create_conversation, LlamaGuardVersion
from typing import List, Tuple
from enum import Enum

from tqdm import tqdm

class AgentType(Enum):
    AGENT = "Agent"
    USER = "User"

def guard_evalaute(
    prompts: List[Tuple[List[str], AgentType]],
    model_id: str = "meta-llama/LlamaGuard-7b",
    llama_guard_version: LlamaGuardVersion = LlamaGuardVersion.LLAMA_GUARD_1,
    
):
    try:
        llama_guard_version = LlamaGuardVersion[llama_guard_version]
    except KeyError as e:
        raise ValueError(f"Invalid Llama Guard version '{llama_guard_version}'. Valid values are: {', '.join([lgv.name for lgv in LlamaGuardVersion])}") from e

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
    
    safety_res = []

    for prompt in tqdm(prompts):
        formatted_prompt = build_default_prompt(
                prompt[1], 
                create_conversation(prompt[0]),
                llama_guard_version)


        input = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
        prompt_len = input["input_ids"].shape[-1]
        output = model.generate(**input, max_new_tokens=100, pad_token_id=0)
        results = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
       
        safety_res.append(results)
        # print(prompt[0])
        # print(f"> {results}")
        # print("\n==================================\n")
    return safety_res


def read_MaliciousInstruct(path):
    with open(os.path.join(path, "data.json"), "r") as file:
        data = json.load(file)
    queries = [data["query"][str(i)] for i in range(0,100)]
    return queries

def read_AdvBench(path):
    with open(os.path.join(path, "data.json"), "r") as file:
        data = json.load(file)
    queries = [e["query"] for e in data]
    return queries

def read_JailbreakBench(path):
    with open(os.path.join(path, "en_data.json"), "r") as file:
        data = json.load(file)
    queries = [e["query"] for e in data]
    return queries


def tag_data_by_Llama_Guard(data_path, guard_path, out_path):
    # queries = read_MaliciousInstruct(data_path)
    # queries = read_AdvBench(data_path)
    queries = read_JailbreakBench(data_path)

    
    responses_ = [([s], AgentType.AGENT) for s in queries]
    safe_res = guard_evalaute(responses_, model_id=guard_path, llama_guard_version='LLAMA_GUARD_2')
    out_ = []
    for i in range(len(queries)):
        unsafe_categories = ""
        is_safe = ""
        if "\n" in safe_res[i]:
            res = safe_res[i].split("\n")
            is_safe = res[0]
            unsafe_categories = res[1]
        else:
            is_safe = safe_res[i]
        out_.append({"query": queries[i], "safe or unsafe": is_safe, "unsafe category": unsafe_categories})
    with open(out_path, "w", encoding="utf-8") as f_out:
        json.dump(out_, f_out, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # data_path = "/data/lihongliang/LLMSafety/datasets/EasyJailbreak_Datasets/MaliciousInstruct"
    # data_path = "/data/lihongliang/LLMSafety/datasets/EasyJailbreak_Datasets/AdvBench"
    data_path = "/data/lihongliang/LLMSafety/datasets/EasyJailbreak_Datasets/JailbreakBench"
    guard_path = "/data/lihongliang/LLMSafety/models/Meta-Llama-Guard-2-8B"
    out_path = f"{data_path}/guardtagged_data.json"
    tag_data_by_Llama_Guard(data_path, guard_path,out_path)
    
