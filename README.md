# Text Detoxification

Многоязычная модель детоксификации текста на основе Qwen3 для переписывания токсичных текстов в вежливые и уважительные формулировки.

## Описание проекта

Проект реализует систему детоксификации текста, которая переписывает токсичные или оскорбительные тексты, делая их вежливыми, уважительными и подходящими, сохраняя при этом исходный смысл и намерение. Модель основана на архитектуре Qwen3 и использует технику LoRA для эффективного дообучения.

### Основные возможности

- Поддержка множества языков (английский, русский, украинский, немецкий, испанский, амхарский, китайский, арабский, хинди)
- Эффективное обучение с использованием 4-bit quantization и LoRA
- Интеграция с MLflow для логирования экспериментов
- Конвертация модели в ONNX и TensorRT для продакшена
- Полный пайплайн от загрузки данных до инференса

## Технические детали

### Структура проекта

```
text-detoxification/
├── text_detoxification/    # Основной пакет
│   ├── __init__.py
│   ├── cli.py               # Точка входа CLI
│   ├── data_loader.py      # Загрузка данных с DVC
│   ├── preprocess.py        # Препроцессинг данных
│   ├── model.py             # Определение модели
│   ├── trainer.py           # Обучение с MLflow
│   ├── inference.py         # Инференс
│   ├── convert.py           # Конвертация в ONNX/TensorRT
│   └── triton_server.py     # Triton Inference Server
├── configs/                 # Hydra конфигурации
├── .env                     # Переменные окружения
├── pyproject.toml           # Зависимости (uv)
├── .pre-commit-config.yaml  # Pre-commit хуки
└── README.md
```

## Setup

### Требования

- Python >= 3.10
- CUDA-capable GPU (рекомендуется)
- Git

### Установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd text-detoxification
```

2. Установите `uv` (если еще не установлен):
```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Создайте виртуальное окружение и установите зависимости:
```bash
uv venv
source .venv/bin/activate  # Linux/Mac
# или
.venv\Scripts\activate  # Windows

uv pip install -e ".[dev]"
```

4. Установите pre-commit хуки:
```bash
pre-commit install
```

5. Настройте DVC (если используете удаленное хранилище):
```bash
# Для локального хранилища (по умолчанию)
dvc remote add -d local ./data

# Для S3 (пример)
# dvc remote add -d s3 s3://your-bucket/path

# Для Google Drive (пример)
# dvc remote add -d gdrive gdrive://your-folder-id
```

6. Загрузите данные через DVC:
```bash
dvc pull
```

7. Запустите MLflow сервер (в отдельном терминале):
```bash
mlflow ui --host 127.0.0.1 --port 8080
```

## Train

Для обучения модели используйте команду:

```bash
text-detox train
```

Команда автоматически:
1. Загрузит данные из HuggingFace и DVC (если настроено)
2. Применит препроцессинг
3. Обучит модель с логированием в MLflow
4. Сохранит лучшую модель в `outputs/final_model/`

### Настройка параметров обучения

Параметры обучения настраиваются через Hydra конфиги в папке `configs/`. Основные файлы:

- `configs/model.yaml` - параметры модели (LoRA, quantization)
- `configs/data.yaml` - настройки данных (языки, пути)
- `configs/training.yaml` - гиперпараметры обучения
- `configs/mlflow.yaml` - настройки MLflow

Пример изменения конфига:
```bash
text-detox train training.learning_rate=1e-4 training.num_train_epochs=3
```

### Логирование

Все метрики и параметры логируются в MLflow:
- Loss на обучении и валидации
- Гиперпараметры модели и обучения
- Git commit ID
- Версия кода

Графики сохраняются в папку `plots/` (если генерируются дополнительно).

## Production Preparation

### Конвертация в ONNX

Для конвертации обученной модели в ONNX:

```bash
text-detox convert \
    --model_path=outputs/final_model \
    --output_path=outputs/model.onnx \
    --format=onnx \
    --onnx_opset_version=14
```

### Конвертация в TensorRT

Для конвертации в TensorRT (требуется установленный TensorRT):

```bash
text-detox convert \
    --model_path=outputs/final_model \
    --output_path=outputs/model.trt \
    --format=tensorrt \
    --tensorrt_precision=fp16
```

### Комплектация поставки

Для продакшена необходимы следующие артефакты:

1. **Модель**: 
   - `outputs/final_model/` (PyTorch) или
   - `outputs/model.onnx` (ONNX) или
   - `outputs/model.trt` (TensorRT)

2. **Токенизатор**: 
   - `outputs/final_model/tokenizer_config.json`
   - `outputs/final_model/tokenizer.json` (если используется)

3. **Конфигурация**: 
   - Файлы из `configs/` (особенно `model.yaml`)

4. **Код инференса**: 
   - `text_detoxification/inference.py`

Минимальные зависимости для инференса:
- `torch` или `onnxruntime` (для ONNX) или `tensorrt` (для TensorRT)
- `transformers`
- `unsloth` (для PyTorch модели)

## Infer

Для запуска инференса на новых данных:

```bash
text-detox infer \
    --model_path=outputs/final_model \
    --input_path=data/test.tsv \
    --output_path=results/detoxified.tsv \
    --text_column=tat_toxic \
    --id_column=id \
    --max_new_tokens=128 \
    --temperature=0.3 \
    --top_p=0.9
```

### Формат входных данных

Входной файл должен быть в формате TSV с колонками:
- `id` (опционально) - идентификатор примера
- `tat_toxic` (или другое имя, указанное в `--text_column`) - токсичный текст

Пример файла `data/test.tsv`:
```
id	tat_toxic
1	This is a toxic comment
2	Another offensive text
```

### Формат выходных данных

Выходной файл будет содержать:
- `id` - идентификатор примера
- `tat_toxic` - исходный токсичный текст
- `tat_detox1` - детоксифицированный текст

## Dependencies

Проект использует `uv` для управления зависимостями. Все зависимости указаны в `pyproject.toml`.

Основные зависимости:
- `torch` - PyTorch
- `transformers` - HuggingFace Transformers
- `unsloth` - Unsloth для эффективного обучения LLM
- `pytorch-lightning` - PyTorch Lightning
- `hydra-core` - Hydra для конфигураций
- `mlflow` - MLflow для логирования
- `dvc` - DVC для управления данными
- `fire` - Fire для CLI

Для установки зависимостей:
```bash
uv pip install -e .
```

Для разработки (с инструментами качества кода):
```bash
uv pip install -e ".[dev]"
```

## Code Quality Tools

Проект использует pre-commit с следующими хуками:

- **pre-commit-hooks** - базовые проверки (trailing whitespace, end-of-file, etc.)
- **black** - форматирование Python кода
- **isort** - сортировка импортов
- **flake8** - линтинг Python кода
- **prettier** - форматирование YAML, JSON, Markdown

Проверка кода перед коммитом:
```bash
pre-commit run -a
```

Все хуки должны проходить успешно перед коммитом.

## Training Framework

Проект использует комбинацию:
- **PyTorch Lightning** - для структурирования кода обучения
- **Transformers/Unsloth** - для работы с языковыми моделями
- **TRL SFTTrainer** - для supervised fine-tuning

Модель обернута в `DetoxificationModel` (LightningModule) для интеграции с инфраструктурой логирования и управления обучением.

## Data Management

Проект использует **DVC** для управления данными:

1. Данные хранятся в DVC (локально, S3, или Google Drive)
2. Загрузка данных интегрирована в команды `train` и `infer`
3. Используется Python API DVC для автоматической загрузки

Настройка DVC:
```bash
# Инициализация (если еще не сделано)
dvc init

# Добавление данных
dvc add data/custom_dataset.tsv
git add data/custom_dataset.tsv.dvc .gitignore
git commit -m "Add dataset"

# Настройка удаленного хранилища
dvc remote add -d local ./data  # локальное
# или
dvc remote add -d s3 s3://bucket/path  # S3
```

Для локального хранилища реализована функция `download_data()` в `data_loader.py`, которая может скачивать данные из открытых источников.

## Hydra

Все гиперпараметры вынесены в YAML конфиги в папке `configs/`:

- **Иерархические конфиги**: `train.yaml` включает `model.yaml`, `data.yaml`, `training.yaml`, `mlflow.yaml`
- **Группировка**: отдельные файлы для модели, данных, обучения, логирования
- **Переопределение**: можно переопределять параметры через CLI

Примеры:
```bash
# Изменить learning rate
text-detox train training.learning_rate=1e-4

# Изменить список языков
text-detox train data.languages=[en,ru]

# Изменить MLflow URI
text-detox train mlflow.uri=http://localhost:5000
```

## Logging

Логирование реализовано через **MLflow**:

- **Метрики**: train_loss, eval_loss, и другие метрики из trainer
- **Параметры**: все гиперпараметры модели и обучения
- **Артефакты**: сохранение модели
- **Git commit ID**: автоматическое логирование версии кода

MLflow сервер должен быть запущен на `127.0.0.1:8080` (по умолчанию).

Графики (если генерируются) сохраняются в `plots/`.

## Inference Server

### MLflow Serving

Для использования MLflow Serving:

1. Зарегистрируйте модель в MLflow:
```bash
mlflow models serve -m runs:/<run-id>/model --host 127.0.0.1 --port 5000
```

2. Используйте API:
```bash
curl -X POST http://127.0.0.1:5000/invocations \
  -H 'Content-Type: application/json' \
  -d '{"inputs": ["toxic text here"]}'
```

### Triton Inference Server

Для Triton Inference Server используется PyTriton с TensorRT моделью:

1. **Конвертируйте модель в TensorRT** (если еще не сделано):
```bash
text-detox convert \
    --model_path=outputs/final_model \
    --output_path=outputs/model.trt \
    --format=tensorrt \
    --tensorrt_precision=fp16
```

2. **Запустите Triton Inference Server**:
```bash
text-detox serve \
    --tensorrt_model_path=outputs/model.trt \
    --tokenizer_path=outputs/final_model \
    --model_name=text_detoxification \
    --host=0.0.0.0 \
    --port=8000
```

3. **Используйте API для инференса**:
```bash
# Пример запроса через curl
curl -X POST http://localhost:8000/v2/models/text_detoxification/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {
        "name": "text",
        "shape": [1],
        "datatype": "BYTES",
        "data": ["This is a toxic comment"]
      }
    ]
  }'
```

Или используйте Python клиент:
```python
import requests
import json

url = "http://localhost:8000/v2/models/text_detoxification/infer"
payload = {
    "inputs": [
        {
            "name": "text",
            "shape": [1],
            "datatype": "BYTES",
            "data": ["Your toxic text here"]
        }
    ]
}

response = requests.post(url, json=payload)
result = response.json()
print(result["outputs"][0]["data"])
```

**Параметры сервера:**
- `--tensorrt_model_path`: Путь к TensorRT engine файлу (.trt)
- `--tokenizer_path`: Путь к директории с токенизатором
- `--model_name`: Имя модели в Triton (по умолчанию: text_detoxification)
- `--max_seq_length`: Максимальная длина последовательности (по умолчанию: 2048)
- `--max_new_tokens`: Максимальное количество токенов для генерации (по умолчанию: 128)
- `--temperature`: Температура сэмплирования (по умолчанию: 0.3)
- `--top_p`: Top-p параметр сэмплирования (по умолчанию: 0.9)
- `--host`: Хост для биндинга сервера (по умолчанию: 0.0.0.0)
- `--port`: Порт для биндинга сервера (по умолчанию: 8000)

**Требования:**
- Установленный TensorRT
- Установленный PyTriton
- CUDA-capable GPU
- PyCUDA для работы с CUDA

Подробная документация по PyTriton доступна в официальной документации.

## Лицензия

[Указать лицензию]

## Авторы

[Указать авторов]

