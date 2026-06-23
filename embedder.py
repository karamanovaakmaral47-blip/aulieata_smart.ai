from sentence_transformers import SentenceTransformer
import json
import os

# Проверяем dataset
DATASET_FILE = "dataset/dataset.json"

if not os.path.exists(DATASET_FILE):
    print(f"Файл не найден: {DATASET_FILE}")
    exit()

# Создаем папку embeddings
os.makedirs("embeddings", exist_ok=True)

print("Загружаю модель...")

try:
    # Более стабильная модель
    model = SentenceTransformer("all-MiniLM-L6-v2")

except Exception as e:
    print("Ошибка загрузки модели:")
    print(e)
    exit()

print("Модель загружена")

# Читаем dataset
try:
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

except Exception as e:
    print("Ошибка чтения dataset:")
    print(e)
    exit()

texts = []

for item in dataset:

    if isinstance(item, dict):

        text = item.get("content", "")

        if text:
            texts.append(text)

print(f"Найдено записей: {len(texts)}")

if len(texts) == 0:
    print("В dataset нет текста")
    exit()

print("Создаю embeddings...")

try:
    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

except Exception as e:
    print("Ошибка создания embeddings:")
    print(e)
    exit()

result = []

for text, embedding in zip(texts, embeddings):

    result.append({
        "content": text,
        "embedding": embedding.tolist()
    })

OUTPUT_FILE = "embeddings/knowledge_base.json"

with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
) as f:

    json.dump(
        result,
        f,
        ensure_ascii=False
    )

print()
print("Готово")
print(f"Сохранено: {OUTPUT_FILE}")
print(f"Всего embeddings: {len(result)}")