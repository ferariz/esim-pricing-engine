.PHONY: data install clean

install:
	pip install -r requirements.txt

data:
	python src/data_generation.py

clean:
	rm -f data/raw/transactions.parquet

notebook:
	jupyter lab notebooks/

lint:
	python -m py_compile src/data_generation.py && echo "OK"
