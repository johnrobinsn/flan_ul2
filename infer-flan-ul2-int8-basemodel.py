from transformers import T5ForConditionalGeneration, AutoTokenizer
import subprocess


#model = "google/flan-ul2"
model_name = "google/flan-t5-xxl"


def model_fn():
    #load model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_name,
                                                       load_in_8bit=True, device_map="auto", cache_dir="/tmp/model_cache/")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # print 'nvidia-smi' output to see how much VRAM is being used by the model
    sp = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    out_list = out_str[0].decode("utf-8")
    print(out_list)
    
    return model, tokenizer


def predict_fn(data, model_and_tokenizer):
    # load model and tokenizer and retrieve prompt
    model, tokenizer = model_and_tokenizer
    text = data.pop("inputs", data)

    # tokenize prompt and use it (together with other generation parameters) to create the model response
    inputs = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(inputs, **data)
    
    # return model output and skip special tokens (such as "<s>")
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



# prompt = """Answer the following question by reasoning step by step.
# The cafeteria had 23 apples.  If they used 20 for lunch, and bought 6 more, how many apples do they have now?"""


model,tokenizer = model_fn()

data = {
    "inputs": '', 
    "min_length": 20, 
    "max_length": 250, 
    "do_sample": True,
    "temperature": 0.8,
    }

input_str = ''
print('Type quit or exit to exit this loop.')
while input_str != 'quit' and input_str != 'exit':
    input_str = input('Enter a prompt: ')
    data['inputs'] = input_str
    res = predict_fn(data,(model,tokenizer))
    print('Output: ', res)