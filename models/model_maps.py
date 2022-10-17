from models.model import *

supported_models = [
    'SentenceTransformerModel',
    'Word2VecModel',
    'Doc2VecModel'
]

def get_model(cl_name, m_name, device):
    if cl_name not in supported_models:
        raise Exception("Model not supported")
    else:
        mod = eval(cl_name)(m_name, device)
    return mod
