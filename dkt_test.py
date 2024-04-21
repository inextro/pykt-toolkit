import json
import torch
import torch.nn as nn

from pykt.models.dkt import DKT
from pykt.models.init_model import load_model


# load config
CONFIG_PATH = 'C:/Users/inno/Documents/jungmin/pykt-toolkit/examples/saved_model/ml-1m_dkt_qid_saved_model_42_0_0.2_200_0.001_1_1_18f8eeab-2a71-4c46-b809-a907801ff7c8/config.json'

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

train_config = config['train_config']
model_config = config['model_config']
data_config = config['data_config']
params = config['params']

# initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load checkpoint
CKPT_PATH = 'C:/Users/inno/Documents/jungmin/pykt-toolkit/examples/saved_model/ml-1m_dkt_qid_saved_model_42_0_0.2_200_0.001_1_1_18f8eeab-2a71-4c46-b809-a907801ff7c8'

model = load_model(model_name='dkt', model_config=model_config, data_config=data_config, emb_type=params['emb_type'], ckpt_path=CKPT_PATH).to(device)
