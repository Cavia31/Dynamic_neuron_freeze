
run-gpu0:
	python main.py --gpu cuda:0 config-neq-budget1.toml
	python main.py --gpu cuda:0 config-neq-budget2.toml

run-gpu1:
	python main.py --gpu cuda:1 config-neq-budget3.toml
	python main.py --gpu cuda:1 config-neq-budget4.toml

run-gpu2:
	python main.py --gpu cuda:2 config-neq-budget3.toml
	python main.py --gpu cuda:2 config-neq-budget4.toml