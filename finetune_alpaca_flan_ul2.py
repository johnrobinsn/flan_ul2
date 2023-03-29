
import torch
from datasets import load_dataset
import peft
import peft.tuners.lora
import transformers
from transformers import T5ForConditionalGeneration, AutoTokenizer,  GenerationConfig
#import accelerate
import subprocess

from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    PeftModel,
)

GRADIENT_CHECKPOINTING = False
GRADIENT_CHECKPOINTING_RATIO = 1
warmup_steps = 50
save_steps = 50
save_total_limit = 3
logging_steps = 10


MICRO_BATCH_SIZE = 1
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
# EPOCHS = 3
LEARNING_RATE = 2e-4

# hyperparams
CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 2000
TARGET_MODULES = [
    "q",
    "v",
]

if LORA_DROPOUT > 0 and GRADIENT_CHECKPOINTING:
    LORA_DROPOUT = 0
    print('Disable Dropout.')

# config = LoraConfig(
#     r=LORA_R,
#     lora_alpha=LORA_ALPHA,
#     target_modules=TARGET_MODULES,
#     lora_dropout=LORA_DROPOUT,
#     bias="none",
#     task_type="CAUSAL_LM",
# )        

# config taken from https://github.com/huggingface/peft/issues/217
# need to look at PromptTuningConfig
# https://github.com/huggingface/peft/blob/main/examples/causal_language_modeling/peft_prompt_tuning_clm.ipynb
config = LoraConfig(
    bias="none",
    #"enable_lora": null,
    fan_in_fan_out=False,
    inference_mode=False,
    lora_alpha=32,
    lora_dropout=0.1,
    merge_weights=False,
    #"modules_to_save": null,
    peft_type="LORA",
    r=8,
    target_modules=[
    "q",
    "v"
    ],
    task_type="SEQ_2_SEQ_LM"
    #task_type="CAUSAL_LM"
)

def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


def tokenize(prompt, tokenizer):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }
    



def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# lora_config = LoraConfig(
#     r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
# )


# model = get_peft_model(model, lora_config)
# print_trainable_parameters(model)

def model_fn(training=False):
    #load model and tokenizer
    #model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained("google/flan-ul2",
                                                       load_in_8bit=True, device_map="auto", cache_dir="/tmp/model_cache/")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")
    

    if training:
        model = prepare_model_for_int8_training(model)

    model = get_peft_model(model, config)

    print("************** trainable parameters **************")
    print_trainable_parameters(model)

    # print 'nvidia-smi' output to see how much VRAM is being used by the model
    sp = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    out_list = out_str[0].decode("utf-8")
    print(out_list)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token


    return model, tokenizer


def predict_fn(data, model_and_tokenizer):
    # load model and tokenizer and retrieve prompt
    model, tokenizer = model_and_tokenizer
    text = data.pop("inputs", data)

    # tokenize prompt and use it (together with other generation parameters) to create the model response
    inputs = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    
    # works with non-peft
    #outputs = model.generate(inputs, **data)

    # works with peft

    temperature = 0.1
    top_p = 0.75
    top_k = 40
    num_beams = 4
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        #**kwargs,
    )

    #input_ids = inputs["input_ids"].to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs, #input_ids,
            #generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=2048,
            early_stopping=True,
        )
    
    # return model output and skip special tokens (such as "<s>")
    
    #no peft
    #return tokenizer.decode(outputs[0], skip_special_tokens=True)

    #peft
    s = outputs.sequences[0]
    return tokenizer.decode(s, skip_special_tokens=True)


# prompt = """Answer the following question by reasoning step by step.
# The cafeteria had 23 apples.  If they used 20 for lunch, and bought 6 more, how many apples do they have now?"""

def inference():
    model,tokenizer = model_fn()

    data = {
        "inputs": '', 
        "min_length": 20, 
        "max_length": 50, 
        "do_sample": True,
        "temperature": 0.6,
        }

    input_str = ''
    print('Type quit or exit to exit this loop.')
    while input_str != 'quit' and input_str != 'exit':
        input_str = input('Enter a prompt: ')
        data['inputs'] = input_str
        res = predict_fn(data,(model,tokenizer))
        print(res)

def train():

    data_path = "./alpaca_data_cleaned.json"
    batch_size = BATCH_SIZE
    micro_batch_size = MICRO_BATCH_SIZE
    epochs = 3
    lr = LEARNING_RATE
    eval_steps = 200
    save_steps = 200
    output_dir = "./lora-ul2"
    report_to = "wandb"

    model,tokenizer = model_fn(True)

    data = load_dataset("json", data_files=data_path)

    train_val = data["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"]
    val_data = train_val["test"]
    
    train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))
    val_data = val_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))

    gradient_accumulation_steps = batch_size // micro_batch_size

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=epochs,
            learning_rate=lr,
            #fp16=True,
            logging_steps=logging_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_steps,
            save_steps=save_steps,
            output_dir=output_dir,
            report_to=report_to if report_to else "none",
            save_total_limit=3,
            load_best_model_at_end=True,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    trainer.train()
    
    model.save_pretrained(output_dir)    

train()