"""
All_in_one_datasets format:
{
    "datasets_1":{
        "S1":[...],
        "S2":[...],
        "S3":[...],
        ...,
        "S11":[...]
    },
    /... ...
    "datasets_n":{
        "S1":[...],
        "S2":[...],
        "S3":[...],
        ...,
        "S11":[...]
    }
}
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import json
import fire
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
from llama_recipes.inference.prompt_format_utils import build_default_prompt, create_conversation, LlamaGuardVersion
from typing import List, Tuple
from enum import Enum
from TaggingMaliciousInstruct import guard_evalaute
from tqdm import tqdm

class AgentType(Enum):
    AGENT = "Agent"
    USER = "User"
base_url = "/data/lihongliang/LLMSafety/datasets/EasyJailbreak_Datasets"



def all_in_one():
    
    # f_all = open(os.path.join(base_url, "all_data.json"))
    data_adv = json.load(open(os.path.join(base_url,"AdvBench/data.json")))
    queries_adv = [k["query"] for k in data_adv]
    data_forb = json.load(open(os.path.join(base_url,"ForbiddenQuestion/data.json")))
    queries_forb = [data_forb["query"][str(k)] for k in range(0, len(data_forb["query"]))]
    data_jailbreak = json.load(open(os.path.join(base_url,"JailbreakBench/data.json")))
    queries_jailbreak = [k["query"] for k in data_jailbreak]
    data_malicious = json.load(open(os.path.join(base_url,"MaliciousInstruct/data.json")))
    queries_malicious = [data_malicious["query"][str(k)] for k in range(0, len(data_malicious["query"]))]
    data_mjp = json.load(open(os.path.join(base_url,"MJP/data.json")))
    queries_mjp = [data_mjp["query"][str(k)] for k in range(0, len(data_mjp["query"]))]
    data_qlist = json.load(open(os.path.join(base_url,"QuestionList/data.json")))
    queries_qlist = [data_mjp["query"][str(k)] for k in range(0, len(data_qlist["query"]))]

    Data_Collections = {
        "AdvBench": queries_adv,
        "ForbiddenQuestion": queries_forb,
        "JailbreakBench": queries_jailbreak,
        "MaliciousInstruct": queries_malicious,
        "MJP": queries_mjp,
        "QuestionList": queries_qlist
    }

    guard_path = "/data/lihongliang/LLMSafety/models/Meta-Llama-Guard-2-8B"
    new_dicts = {}
    for dsets_k in Data_Collections.keys():
        queries = Data_Collections[dsets_k]
        ## Tagging data into multiclass
        responses_ = [([s], AgentType.AGENT) for s in queries]
        safe_res = guard_evalaute(responses_, model_id=guard_path, llama_guard_version='LLAMA_GUARD_2')
        
        keys = ["S" + str(i) for i in range(12)]  # 生成"S0"到"S11"的键  
        dsets_k_clusters = {k: [] for k in keys}
        print(f"Begin to deal with datasets: {dsets_k}")
        for i in tqdm(range(len(queries))):
            # unsafe_categories = ""
            
            if "\n" in safe_res[i]:
                res = safe_res[i].split("\n")
                unsafe_categories = res[1]
                if "," in unsafe_categories:
                    categories_ = unsafe_categories.split(",")
                    for c in categories_:
                        c = c.strip()
                        dsets_k_clusters[c].append(queries[i])
                else:
                    c = unsafe_categories
                    dsets_k_clusters[c].append(queries[i])
            else:
                dsets_k_clusters["S0"].append(queries[i])
        new_dicts[dsets_k] = dsets_k_clusters
    
    out_path = os.path.join(base_url, "all_data.json")
    with open(out_path, "w", encoding="utf-8") as f_out:
        json.dump(new_dicts, f_out, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    all_in_one()
