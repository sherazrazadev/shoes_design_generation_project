import torch
from einops import rearrange
from transformers import T5Tokenizer, T5EncoderModel

MAX_LENGTH = 256

DEFAULT_T5_NAME = 't5_small'

T5_VERSIONS = {
    't5_small': {'tokenizer': None, 'model': None, 'handle': 't5-small', 'dim': 512, 'size': .24},
    't5_base': {'tokenizer': None, 'model': None, 'handle': 't5-base', 'dim': 768, 'size': .890},
    't5_large': {'tokenizer': None, 'model': None, 'handle': 't5-large', 'dim': 1024, 'size': 2.75},
}

def _check_downloads(name):
    if T5_VERSIONS[name]['tokenizer'] is None:
        T5_VERSIONS[name]['tokenizer'] = T5Tokenizer.from_pretrained(T5_VERSIONS[name]['handle'])
    if T5_VERSIONS[name]['model'] is None:
        T5_VERSIONS[name]['model'] = T5EncoderModel.from_pretrained(T5_VERSIONS[name]['handle'])

def t5_encode_text(text, name: str = 't5_base', max_length=MAX_LENGTH):
    _check_downloads(name)
    tokenizer = T5_VERSIONS[name]['tokenizer']
    model = T5_VERSIONS[name]['model']

    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
    else:
        device = torch.device('cpu')

    tokenized = tokenizer.batch_encode_plus(
        text,
        padding='longest',
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)

    model.eval()

    with torch.no_grad():
        t5_output = model(input_ids=input_ids, attention_mask=attention_mask)
        final_encoding = t5_output.last_hidden_state.detach()

    final_encoding = final_encoding.masked_fill(~rearrange(attention_mask, '... -> ... 1').bool(), 0.)

    return final_encoding, attention_mask.bool()

def get_encoded_dim(name: str) -> int:
    return T5_VERSIONS[name]['dim']
