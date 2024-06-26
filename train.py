import math           
import time
import os
import tiktoken
import torch
from torch.nn import functional as F
from model import GPTConfig, GPT

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# SEED
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

out_dir = "."

# STOP
num_return_sequences = 5
max_length = 30

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2') 
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B*T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

block_size: int = 1024 # max sequence length
vocab_size: int = 50304 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
n_layer: int = 12 # number of layers
n_head: int = 12 # number of heads
n_embd: int = 768 # embedding dimension

# model init
model_args = dict(block_size=block_size, vocab_size=vocab_size, n_layer=n_layer, n_head=n_head,
                  n_embd=n_embd) # start with model_args from command line

torch.set_float32_matmul_precision('high')
gptconfig = GPTConfig(**model_args)
model = GPT(gptconfig)
model.to(device)
model = torch.compile(model)

# CODE UPDATE HERE
max_lr = 6e-4 
min_lr = max_lr * 0.1 # As per Chinchila
decay_lr = min_lr * 0.1 # Not used as per Chinchila
warmup_steps = 60
max_steps = 5000
decay_steps = max_steps # As per Chinchila

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > decay_steps:
        return min_lr
        # decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        # assert 0 <= decay_ratio <=1
        # coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        # return decay_lr + coeff * (min_lr - decay_lr)
    decay_ratio = (it - warmup_steps) / (decay_steps - warmup_steps)
    assert 0 <= decay_ratio <=1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# for step in range(max_steps):
#     lr = get_lr(step)
#     if ((step % 20 == 0) or (step == (max_steps-1) )):
#         print(f'step{step} | lr: {lr:.8f}')

# import sys; sys.exit(0)

train_loader = DataLoaderLite(B = 16, T = 1024)

# NEW CODE
import time
# optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)
for step in range(max_steps):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    # NEW CODE ADDED HERE
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y) 
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
    # NEW CODE
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    optimizer.step()
    torch.cuda.synchronize() 
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f'step{step} | loss: {loss.item()} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec: .2f} | norm: {norm:.2f} | lr: {lr:.8f}')

    if (step % 500 == 0):
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': step,
            'best_val_loss': loss,
        }
        ckptname = 'ckpt' + str(step) + '.pt'
        print(f"saving checkpoint {ckptname} to {out_dir}")
        torch.save(checkpoint, os.path.join(out_dir, ckptname))


print(loss)
# import sys; sys.exit(0)
# torch.save(model.state_dict(), 'GPT2_5000Iters.pt')

checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_args,
    'iter_num': max_steps,
    'best_val_loss': loss,
}
print(f"saving checkpoint to {out_dir}")
torch.save(checkpoint, os.path.join(out_dir, 'ckptlast.pt'))

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

# print the generated text
tokens = x[0, :max_length].tolist()
enc = tiktoken.get_encoding('gpt2') 
decoded = enc.decode(tokens)
print(">", decoded)
