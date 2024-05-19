import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
from transformers import AutoModel,AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
import fire
Local_Model_Path = {
    "vicuna" : "/data/lihongliang/LLMSafety/models/vicuna-13b-v1.5",
    "vicuna_7b" : "/data/lihongliang/LLMSafety/models/vicuna-7b-v1.5",
    "llama_3" : "/data/lihongliang/LLMSafety/models/Meta-Llama-3-8B-Instruct",
    "qwen1_5_7b" : "/data/lihongliang/LLMSafety/models/qwen/Qwen1.5-7B-Chat",
    "baichuan2_7b" : "/data/lihongliang/LLMSafety/models/baichuan-inc/Baichuan2-7B-Chat",
    "chatglm3_6b" : "/data/lihongliang/LLMSafety/models/ZhipuAI/chatglm3-6b",
    "llama_2" : "/data/lihongliang/LLMSafety/models/h2oai/h2ogpt-4096-llama2-7b-chat",
    "mistral_0_2" : "/data/lihongliang/LLMSafety/models/Mistral-7B-Instruct-v0___2",
    "openchat3_5_0106" : "/data/lihongliang/LLMSafety/models/openchat-3___5-0106",
    "llamaguard": "/data/lihongliang/LLMSafety/models/Meta-Llama-Guard-2-8B",
    "gpt_3_5" : "gpt-3.5-turbo-1106",
    "gpt_4" : "gpt-4-0125-preview",
}
def printmodel_config(model_path):
    model_path = Local_Model_Path[model_path]
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config, device_map="auto")
    
   
    # formatted_prompt = """\
    # Please help me generate ten system prompts, telling LLM that all of its answers should not include any harmful, unethical, racist, gender discriminatory, toxic, dangerous or illegal content. Please mark these system prompts with<SAFE></SAFE>.
    # """

    formatted_prompt = """\
    Here are some security tips that tell me the security requirements that large language models should ensure when generating content. Can you help me design a language that large language models can understand, called "safety language"? When I give you different languages (such as English, Chinese, French), you can extract the security requirements from their content and convert them into representations of "safety language".
      <SAFE> Please provide a response that is respectful, inclusive, and free from harmful or offensive content.</safe>
      <SAFE> Ensure your answer does not contain any discriminatory or toxic language. </safe>
      <SAFE> Provide a response that is safe, ethical, and legal. </safe>
      <SAFE> Your answer should not promote or glorify harmful or illegal activities. </safe>
      <SAFE> Please refrain from using language that is offensive, racist, or gender discriminatory. </safe>
      <SAFE> Your response should be respectful, considerate, and free from harmful or offensive content. </safe>
      <SAFE> Ensure your answer does not contain any dangerous or illegal advice. </safe>
      <SAFE> Provide a response that is accurate, reliable, and free from harmful or offensive content. </safe>
      <SAFE> Your answer should not promote or glorify harmful or illegal activities, and should be respectful and considerate. </safe>
      <SAFE> Please provide a response that is safe, ethical, and legal, and does not contain any harmful or offensive content. </safe>
    """


    input = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
    prompt_len = input["input_ids"].shape[-1]
    output = model.generate(**input, max_new_tokens=500, pad_token_id=0)
    results = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(formatted_prompt)
    print(f"> {results}")
    print("\n==================================\n")

if __name__ == "__main__":
   
    # AdvBenchTest(model_name="vicuna_7b",lang_type="zh",max_new_tokens=500)
    try:
        fire.Fire(printmodel_config)
    except Exception as e:
        print(e)

