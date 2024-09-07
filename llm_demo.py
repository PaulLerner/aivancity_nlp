from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def pipeline(text, model, tokenizer):
    messages = [
        {"role": "user", "content": text}
    ]
    inputs = tokenizer.apply_chat_template(messages, return_dict=True, tokenize=True, return_tensors="pt", add_generation_prompt=True)
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    input_length = inputs["input_ids"].shape[1]
    output = model.generate(**inputs, max_new_tokens=128)#, pad_token_id=tokenizer.eos_token_id)#, max_new_tokens=32
    output = tokenizer.batch_decode(output[:, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output


def user_loop(model, tokenizer):
    while True:
        answer = input(f">>> ").strip()#"what is aivancity?" #
        output = pipeline(answer,model, tokenizer)[0]
        print(f"{output}\n")


def main():
    model_name = "google/gemma-2-2b-it"#/home/paul/code/neot/data/models/Meta-Llama-3.1-8B-Instruct"
    model  = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        token=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True)
            #load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    )#low_cpu_mem_usage=True
    #print(f"{model.config.max_length=}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.pad_token_id = tokenizer.eos_token_id
    user_loop(model, tokenizer)


if __name__ == "__main__":
    main()