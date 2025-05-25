format:
	black .

lint:
	flake8 .
	pylint data.py model.py train.py predict.py

security:
	bandit -r .

test:
	pytest

all: format lint security test