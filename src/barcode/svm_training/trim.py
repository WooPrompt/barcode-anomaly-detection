import os

target_file = 'src/barcode/svm_preprocessing/data_manager.py'

with open(target_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 실제 이스케이프된 줄바꿈(\n), 탭(\t), 따옴표(\") 복원
fixed = content.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')

with open(target_file, 'w', encoding='utf-8') as f:
    f.write(fixed)

print("✅ 줄바꿈 복원 완료")
