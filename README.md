# Flan-* training and inference using int8

Please see the [train-peft-flan-ul2-int8-alpaca.ipynb](./train-peft-flan-ul2-int8-alpaca.ipynb) notebook.


# Flan-UL2 inference using int8

Will run on a 24G GPU.

Will print out the memory usage of the model via nividia-smi.

The specific versions in the requirements file are supposedly due to a regression that caused an increase in memory usage.  I haven't verified all this myself, so need to see if I can remove these.

```bash
pip install -r requirements.txt

# limit to 1 GPU
export CUDA_VISIBLE_DEVICES=0
python inference_flan_ul2.py
```



Trying to finetune with 24G GPU.

without peft 20,469 MiB vs 20,521MiB with... ?

todo
try gradient checkpointing
try multiprocessing on 2 24G GPUs
