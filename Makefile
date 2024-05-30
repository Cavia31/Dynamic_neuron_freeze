
run-gpu0:
	python main.py --gpu cuda:0 config-neq1.toml
	python main.py --gpu cuda:0 config-neq4.toml

run-gpu1:
	python main.py --gpu cuda:1 config-neq2.toml
	python main.py --gpu cuda:1 config-neq5.toml

run-gpu2:
	python main.py --gpu cuda:2 config-neq3.toml
	python main.py --gpu cuda:2 config-neq6.toml