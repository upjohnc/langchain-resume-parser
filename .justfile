default:
    just --list

poetry-install:
    poetry install --no-root --with dev

pre-commit:
    pre-commit install

ollama-start:
    ollama serve

llama3:
    ollama pull llama3

run *args:
    PYTHONPATH=. poetry run python src/code.py "{{ args }}"

run-data-engr:
    PYTHONPATH=. poetry run python src/code.py -t "data engineer" -s "snowflake, python"

run-software-guy:
    PYTHONPATH=. poetry run python src/code.py -t "software guy" -s "node, restful, java"

