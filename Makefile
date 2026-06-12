.PHONY: setup test lint data retrieve-eval train-deberta serve ui

setup:
	python3 -m pip install -r requirements.txt

test:
	python3 -m pytest

lint:
	python3 -m compileall data retrieval ranking models training rag agent evaluation serving ui tests

data:
	@echo "Phase 1 will add the data pipeline."

retrieve-eval:
	@echo "Phase 2 will add retrieval evaluation."

train-deberta:
	@echo "Phase 4 will add DeBERTa training."

serve:
	python3 -m uvicorn serving.api:app --reload

ui:
	python3 -m ui.app
