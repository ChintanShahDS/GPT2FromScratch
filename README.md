# GPT2FromScratch
Created a GPT2 from Scratch using Shakespeare data - But can be used with any other text data

## Achievements
- Able to reduce the loss by half to 5.6 within 60 steps
- Able to reduce the loss to the target of 0.099 within 1900 steps
- Outputs can be seen at [https://huggingface.co/spaces/Chintan-Shah/HindiTokenizerFromScratch](https://huggingface.co/spaces/Chintan-Shah/GPT2FromScratch)
- Implemented most of the GPT2 and partly GPT3 related transformer code and optimizations: List below
  - Tokenizer: gpt2 tiktoken tokenizer
  - LR Scheduler: Chinchila OneCycleLR method
  - Autocast: torch.autocast(device_type=device, dtype=torch.bfloat16)
  - Normalization: torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
  - Flash Attention: F.scaled_dot_product_attention(q, k, v, is_causal = True)
  - Weight sharing for embedding to token and vice versa: self.transformer.wte.weight = self.lm_head.weight
  - Weight initialization: With mean 0.0 and std 0.2
  - Vocab size change: To power of 2 for better speed up
  - Optimized backward pass: Only focus on matmuls and embeddings leaving biases and layernorms not to be focussed on by Optimizer reducing those parameter overheads
  - Other optimizations
    - torch.set_float32_matmul_precision('high')
    - torch.compile to help get some optimization
    
## Files
- model.py
  - File having the model related classes
- train.py
  - Training related code including checkpointing of the model and 1 inference at the end to check the outcomes
- inference.py
  - File used to inference from the trained model
  - Used as base for huggingface app as well

## Hyperparameters used
- block_size: int = 1024 # max sequence length
- vocab_size: int = 50304 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
- n_layer: int = 12 # number of layers
- n_head: int = 12 # number of heads
- n_embd: int = 768 # embedding dimension
- max_lr = 6e-4 
- min_lr = 6e-5
- warmup steps = 60
- max steps = 5000
- batch size = 16
- token length = 1024
- AdamW optimizer
  - betas=(0.9, 0.95)
  - eps=1e-8
  - fused

## Hardware used
- Google Colab T4
- Some experimentation on CPU as well

## Detailed Training log
- Full log file kept at logGPT2_5000Its.txt
- Important Excerpts from the log
"""
step0 | loss: 10.947460174560547 | dt: 17781.67ms | tok/sec:  921.40 | norm: 28.57 | lr: 0.00001000
step59 | loss: 5.556936264038086 | dt: 2003.82ms | tok/sec:  8176.37 | norm: 1.13 | lr: 0.00060000
step999 | loss: 2.0623185634613037 | dt: 2236.46ms | tok/sec:  7325.86 | norm: 1.77 | lr: 0.00055327
step1899 | loss: 0.0848759114742279 | dt: 2206.88ms | tok/sec:  7424.07 | norm: 0.67 | lr: 0.00043546
step1999 | loss: 0.06174922734498978 | dt: 2227.44ms | tok/sec:  7355.54 | norm: 0.71 | lr: 0.00041945
step2999 | loss: 0.005564077757298946 | dt: 2226.34ms | tok/sec:  7359.16 | norm: 0.28 | lr: 0.00025066
step3999 | loss: 0.0014109887415543199 | dt: 2231.74ms | tok/sec:  7341.36 | norm: 0.02 | lr: 0.00011288
step4999 | loss: 0.001211557537317276 | dt: 2222.14ms | tok/sec:  7373.08 | norm: 0.01 | lr: 0.00006000
"""

## Observations
- lr scheduler can help faster convergence
- torch.compile works on specific devices only
- Even after compile the timings taken by the training per step was not consistent
- Loss reduction is not consistent and does not follow a pattern (Need to find out if lr has something to do with it)
- Outputs are not as great as expected thought the model has overfitted but the params are only 124M (While the magic starts after 7B)
