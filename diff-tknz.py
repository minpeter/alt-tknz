import json
import os
from transformers import AutoTokenizer

# 5개의 모델 리스트
model_names = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-72B",
    # "Qwen/Qwen2.5-Coder-0.5B",
    # "Qwen/Qwen2.5-Coder-1.5B",
    # "Qwen/Qwen2.5-Coder-3B",
    # "Qwen/Qwen2.5-Coder-7B",
    # "Qwen/Qwen2.5-Coder-14B",
    # "Qwen/Qwen2.5-Coder-32B",
]

# 토크나이저 저장 경로
save_paths = ["./tokenizer_" + name.replace("/", "_") for name in model_names]

# 토크나이저 로드 및 저장
for model_name, save_path in zip(model_names, save_paths):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_path)

# JSON 파일 비교 함수
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def compare_files(file_name):
    base_file = load_json(os.path.join(save_paths[0], file_name))
    return all(load_json(os.path.join(path, file_name)) == base_file for path in save_paths[1:])

# 모든 파일 비교
is_identical = (
    compare_files("special_tokens_map.json") and
    compare_files("tokenizer_config.json") and
    compare_files("tokenizer.json")
)

# 결과 출력
print("전부동일" if is_identical else "불일치")
