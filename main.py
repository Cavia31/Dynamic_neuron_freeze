from tomlkit.toml_file import TOMLFile
from csv import DictWriter
from builder import Builder
from trainer import Trainer
import argparse

parser = argparse.ArgumentParser(
    prog="Main",
    description="Run a training following the specified config file")

parser.add_argument('configfile')
parser.add_argument('--gpu',
                    type=str,
                    default='cuda:0',
                    help="the cuda identifier of the gpu to run the config on")

try:
    args = parser.parse_args()

    filepath = args.configfile
    device = args.gpu
except:
    filepath = 'config2.toml'

try:
    d = TOMLFile("config/" + filepath).read()
except:
    print("config file not found, exitting...")
    exit()

print(device)
d['model']['device'] = device

b = Builder(d["model"], d["dataset"])

with open("runs/results_" + filepath.split('.')[0] + ".csv", "w", newline='') as csv:
    dwriter = DictWriter(csv, fieldnames=['epoch','train_loss','test_loss','acc1','acc5','unfrozen_neurons','total_params','unfrozen_params','epoch_time'])
    dwriter.writeheader()
    
    if "valid" in b.dataloader:
        t = Trainer(d["train"], b.model, dwriter, b.dataloader["train"], b.dataloader["test"], valid_set=b.dataloader["valid"], device=device)
    else:
        t = Trainer(d["train"], b.model, dwriter, b.dataloader["train"], b.dataloader["test"], device=device)
    t.train()