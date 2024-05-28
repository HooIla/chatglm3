from vllm import LLM, SamplingParams

prompts = [
    "8只兔子一共几条腿",
    "简单介绍一下你自己"
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model = "/home/yuyongjian/ChatGLM3/models/chatglm3-6b")

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"prompt: {prompt!r}, Generated text: {generated_text!r}")


'''
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
--model /home/yuyongjian/ChatGLM3/models/chatglm3-6b \
--served-model-name glm3 \
--trust-remote-code \
--max-model-len 128 \
--port 7866 \
-q awq 


curl http://180.184.180.203:7866/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "smaug",
"prompt": "8只兔子一共几条腿",
"max_tokens": 1024,
"temperature": 0
}'

'''