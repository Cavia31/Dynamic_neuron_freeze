
run-gpu0:
	python main.py --gpu cpu config-cpu-speed1.toml
	python main.py --gpu cpu config-cpu-speed2.toml
	python main.py --gpu cpu config-cpu-speed3.toml
	python main.py --gpu cpu config-cpu-speed4.toml
	python main.py --gpu cpu config-cpu-speed5.toml

run-gpu1:
	python main.py --gpu cuda:1 config-gvel4.toml
	python main.py --gpu cuda:1 config-gvel3.toml
	python main.py --gpu cuda:1 config-neq-budget3.toml

run-gpu2:
	python main.py --gpu cuda:2 config-neq-budget1.toml
	python main.py --gpu cuda:2 config-neq-budget2.toml
