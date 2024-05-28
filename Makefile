run:
	python main.py config2.toml
	python main.py config3.toml
	python main.py config4.toml
	python main.py config5.toml
	python main.py config6.toml
	python main.py config7.toml
	python main.py config8.toml
	python main.py config9.toml
	python main.py config10.toml
	python main.py config12.toml
	python main.py config13.toml
	python main.py config14.toml

run-gpu0:
	python main.py --gpu cuda:0 config19.toml
	python main.py --gpu cuda:0 config23.toml
	python main.py --gpu cuda:2 config26.toml

run-gpu1:
	python main.py --gpu cuda:0 config22.toml
	python main.py --gpu cuda:0 config24.toml
	python main.py --gpu cuda:0 config25.toml

run-gpu2:
	python main.py --gpu cuda:0 config21.toml
	python main.py --gpu cuda:0 config20.toml