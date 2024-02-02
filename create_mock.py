from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("google/t5-efficient-tiny")

model = T5ForConditionalGeneration.from_pretrained(
    "google/t5-efficient-tiny",
    d_ff=1,
    d_kv=1,
    d_model=1,
    num_decoder_layers=1,
    num_layers=1,
    ignore_mismatched_sizes=True,
)
# checking it works
input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")
model(input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids)

model.save_pretrained("mocked-t5", from_pt=True)
tokenizer.save_pretrained("mocked-t5")
