# Инструкция по настройке проекта

## Быстрый старт

1. **Установите uv** (если еще не установлен):
   ```bash
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # Linux/Mac
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Клонируйте репозиторий и перейдите в директорию**:
   ```bash
   git clone <repository-url>
   cd text-detoxification
   ```

3. **Создайте виртуальное окружение и установите зависимости**:
   ```bash
   uv venv
   source .venv/bin/activate  # Linux/Mac
   # или
   .venv\Scripts\activate  # Windows
   
   uv pip install -e ".[dev]"
   ```

4. **Установите pre-commit хуки**:
   ```bash
   pre-commit install
   ```

5. **Проверьте установку**:
   ```bash
   pre-commit run -a
   ```

6. **Настройте DVC** (опционально, если используете удаленное хранилище):
   ```bash
   dvc init
   dvc remote add -d local ./data  # для локального хранилища
   ```

7. **Запустите MLflow сервер** (в отдельном терминале):
   ```bash
   mlflow ui --host 127.0.0.1 --port 8080
   ```

## Проверка работоспособности

После настройки вы должны быть able to:

- ✅ Запустить `pre-commit run -a` без ошибок
- ✅ Импортировать пакет: `python -c "import text_detoxification"`
- ✅ Запустить команды: `text-detox train --help`

## Устранение проблем

### Ошибки при установке зависимостей

Если возникают проблемы с установкой `unsloth` или других зависимостей:

1. Убедитесь, что у вас установлен CUDA (для GPU)
2. Попробуйте установить зависимости по отдельности
3. Проверьте версию Python (должна быть >= 3.10)

### Ошибки pre-commit

Если pre-commit не проходит:

1. Запустите форматирование вручную:
   ```bash
   black .
   isort .
   ```

2. Исправьте ошибки flake8 вручную

### Проблемы с DVC

Если данные не загружаются:

1. Проверьте настройки DVC: `dvc remote list`
2. Убедитесь, что данные добавлены в DVC: `dvc list .`
3. Для локального хранилища убедитесь, что путь существует

