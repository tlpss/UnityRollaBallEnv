pre-commit:
    autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place --ignore-init-module-imports .
    isort --profile black --line-length 119 .
    black --line-length 119 .
    mypy --ignore-missing-imports .

inter-clean:
    autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place .
    isort --profile black --line-length 119 .
