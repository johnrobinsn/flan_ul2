# Flan-UL2 inference using int8

Will run on a 24G GPU.

Will print out the memory usage of the model via nividia-smi.

```bash
pip install -r requirements.txt

# limit to 1 GPU
export CUDA_VISIBLE_DEVICES=0
python inference_flan_ul2.py
```
