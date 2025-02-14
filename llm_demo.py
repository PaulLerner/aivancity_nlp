from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

def pipeline(text, model, tokenizer, chat=True, top_p=0.9, max_new_tokens=128, do_sample=True):
    if chat:
        messages = [
            {"role": "user", "content": text}
        ]
        inputs = tokenizer.apply_chat_template(messages, return_dict=True, tokenize=True, return_tensors="pt", add_generation_prompt=True)
    else:
        inputs = tokenizer([text], return_tensors="pt")

    for k, v in inputs.items():
        inputs[k] = v.cuda()
    input_length = inputs["input_ids"].shape[1]
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, top_p=top_p, do_sample=do_sample)
    output = tokenizer.batch_decode(output[:, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output[0]


def user_loop(*args, **kwargs):
    while True:
        answer = input(f">>> ").strip()
        output = pipeline(answer, *args, **kwargs)
        print(f"{output}\n")


def main():
    # TODO customize arguments
    warnings.filterwarnings("ignore")
    model_name = "google/gemma-2-2b-it"
    model  = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        token=True,
        torch_dtype="float16",
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    user_loop(model, tokenizer)


if __name__ == "__main__":
    # TODO add jsonargparse
    main()
