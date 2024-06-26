# GPT2FromScratch
Created a GPT2 from Scratch using Shakespeare data - But can be used with any other text data

## Achievements
- Able to reduce the loss by half to 5.6 within 60 steps
- Able to reduce the loss to the target of 0.099 within 1900 steps
- Outputs can be seen at [https://huggingface.co/spaces/Chintan-Shah/HindiTokenizerFromScratch](https://huggingface.co/spaces/Chintan-Shah/GPT2FromScratch)
- Implemented most of the GPT2 and partly GPT3 related transformer code and optimizations: List below
  - 

## Files
- model.py
  - File having the model related classes
- train.py
  - Training related code including checkpointing of the model and 1 inference at the end to check the outcomes
- inference.py
  - File used to inference from the trained model
  - Used as base for huggingface app as well

## Hyperparameters used
- 

## Detailed Training log
- Full log file kept at logGPT2_5000Its.txt
- Excerpts from the log
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
- TBD
