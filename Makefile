
bin/python:
	python3 -m venv .
	bin/pip install -r requirements.txt

.PHONY: create
create: bin/python
	rm -rf mocked-t5
	bin/python create_mock.py
	bin/optimum-cli export onnx --model mocked-t5 mocked-t5/onnx --task text2text-generation
	find mocked-t5/onnx -type f ! -name '*.onnx' -delete
	bin/python quantize.py mocked-t5/onnx
