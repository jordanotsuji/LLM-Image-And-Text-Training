import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "HuggingFaceM4/idefics-9b-instruct"
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
processor = AutoProcessor.from_pretrained(checkpoint)

# load jmhessel/newyorker_caption_contest dataset from huggingface
# https://huggingface.co/datasets/jmhessel/newyorker_caption_contest
from datasets import load_dataset

dataset = load_dataset("jmhessel/newyorker_caption_contest", "explanation")


# prompts that give the model a random image and its caption from the dataset and ask it to explain the joke
prompts = []
for i in range(5):
    inst = dataset["validation"][i]
    prompts.append(
        [
            # give the model an image and its caption from the dataset and ask it to explain the joke
            "User: Explain the joke in this image and caption",
            inst["image"],
            inst["caption_choices"],
            "<end_of_utterance>",
            "\nAssistant:",
        ]
    )
    # print the prompt that was just added
    print(prompts[-1])


# --batched mode
inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)
# --single sample mode
# inputs = processor(prompts[0], return_tensors="pt").to(device)

# Generation args
exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=100)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
for i, t in enumerate(generated_text):
    print(f"{i}:\n{t}\n")
