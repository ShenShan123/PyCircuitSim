# pycmg-wrapper build and test automation

.PHONY: build test test-fast test-slow test-dc clean install

build:
	mkdir -p build && cd build && cmake .. && cmake --build . --target osdi

test:
	python -m pytest tests/ -v

test-fast:
	python -m pytest tests/test_api.py tests/test_nfin_scaling.py -v

test-slow:
	python -m pytest tests/ -v -m slow

test-dc:
	python -m pytest tests/test_dc_regions.py tests/test_dc_jacobian.py -v

clean:
	rm -rf build/osdi/*.osdi build/ngspice_eval/

install:
	pip install -e .
