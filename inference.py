import tiktoken
import os
import torch
from torch.nn import functional as F

from model import GPTConfig, GPT

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

out_dir = '.'

# STOP
max_length = 500

enc = tiktoken.get_encoding('gpt2') 

# CHANGES IN CURRENT CODE
ckpt_path = os.path.join(out_dir, 'ckptlast.pt')
print(ckpt_path)
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.to(device)
model = torch.compile(model)

start_text = "JULIET: \n"
start_tokens = enc.encode(start_text)
# print(start_tokens, len(start_tokens))
start_tokens = torch.tensor(start_tokens)
x = start_tokens.view(1, len(start_tokens))
# print(x, x.shape)
x = x.to(device)

while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x)[0] # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)
        # print(x.size(1))

# print the generated text
tokens = x[0, :max_length].tolist()
decoded = enc.decode(tokens)
print(">", decoded)
