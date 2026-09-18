# Makefile del proyecto P01 - Simulador de finanzas conductuales
PY ?= python

.PHONY: all test run notebook figures clean help

help:
	@echo "make test      -> corre la bateria de pruebas (pytest)"
	@echo "make run       -> corre la simulacion completa y genera outputs/"
	@echo "make figures   -> regenera solo las figuras a partir de outputs/"
	@echo "make notebook  -> ejecuta el notebook y lo guarda con sus outputs"
	@echo "make all       -> test + run + notebook"

test:
	$(PY) -m pytest tests/ -v

run:
	$(PY) run_simulation.py --agents 1000 --days 504 --assets 60 --bootstrap 1000 --seed 42 --all

figures:
	$(PY) run_simulation.py --figures-only

notebook:
	$(PY) -m nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=3600 notebooks/P01_Analisis.ipynb

all: test run notebook

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
