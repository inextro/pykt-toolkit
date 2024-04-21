import json
import torch
import torch.nn as nn

from pykt.models.dkvmn import DKVMN
from pykt.models.init_model import load_model


# load config
CONFIG_PATH = 'C:/Users/inno/Documents/jungmin/pykt-toolkit/examples/saved_model/ml-1m_dkvmn_qid_saved_model_42_0_0.2_200_0.001_50_1_1_39bff130-2696-44f8-954b-df500a5458b8/config.json'

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

train_config = config['train_config']
model_config = config['model_config']
data_config = config['data_config']
params = config['params']

# initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load checkpoint
CKPT_PATH = 'C:/Users/inno/Documents/jungmin/pykt-toolkit/examples/saved_model/ml-1m_dkvmn_qid_saved_model_42_0_0.2_200_0.001_50_1_1_39bff130-2696-44f8-954b-df500a5458b8'

model = load_model(model_name='dkvmn', model_config=model_config, data_config=data_config, emb_type=params['emb_type'], ckpt_path=CKPT_PATH).to(device)

print('no error')