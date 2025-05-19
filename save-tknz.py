from transformers import AutoTokenizer

model_name = "minpeter/Llama-3.2-1B-AlternateTokenizer-chatml"
tokenizer = AutoTokenizer.from_pretrained(model_name)

save_path = "./custom_tokenizer"
tokenizer.save_pretrained(save_path)

print(f"토크나이저가 '{save_path}' 경로에 저장되었습니다.")
