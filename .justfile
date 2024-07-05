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

# test return Chad Resume
run-data-engr:
    PYTHONPATH=. poetry run python src/code.py -t "data engineer" -s "snowflake, python"

# test to return Steve Resume
run-software-guy:
    PYTHONPATH=. poetry run python src/code.py -t "software guy" -s "node, restful, java"

# false positive that should return "not Chad" resume
run-false-positive:
    PYTHONPATH=. poetry run python src/code.py -t "frontend engineer" -s "node, java"

# should return Dave resume
run-frontend-arndt:
    PYTHONPATH=. poetry run python src/code.py -t "frontend engineer" -s "node, restful, java"
