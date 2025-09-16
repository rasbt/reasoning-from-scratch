# README



A collection of standalone example scripts for testing purposes.


&nbsp;
## 01_quick-example.py

Script runs the model on a simple example prompt.


```bash
➜ reasoning-from-scratch git:(main) uv run tests/examples/01_quick-example.py --help  

usage: 01_quick-example.py [-h] [--device DEVICE] [--cache] [--compile]

               [--reasoning] [--optimized]



Run Qwen3 text generation



options:

 -h, --help    show this help message and exit

 --device DEVICE Device to run on (e.g. 'cpu', 'cuda', 'mps'). If not

           provided, will auto-detect with get_device().

 --cache     Use KV cache during generation (default: False).

 --compile    Compile PyTorch model (default: False).

 --reasoning   Use reasoning model variant.

 --optimized   Use reasoning model variant.
```

&nbsp;
### Example:

```
➜  uv run tests/examples/01_quick-example.py --cache --device cpu


✓ qwen3/qwen3-0.6B-base.pth already up-to-date
✓ qwen3/tokenizer-base.json already up-to-date
============================================================
Iteration : 1
optimized : False
torch     : 2.7.1
device    : cpu
cache     : True
compile   : False
reasoning : False
============================================================
Output length: 41
Time: 1.44 sec
28 tokens/sec

 Large language models are artificial intelligence systems that can understand, generate, and process human language, enabling them to perform a wide range of tasks, from answering questions to writing articles, and even creating creative content.
============================================================
Iteration : 2
optimized : False
torch     : 2.7.1
device    : cpu
cache     : True
compile   : False
reasoning : False
============================================================
Output length: 41
Time: 1.46 sec
28 tokens/sec

 Large language models are artificial intelligence systems that can understand, generate, and process human language, enabling them to perform a wide range of tasks, from answering questions to writing articles, and even creating creative content.
============================================================
Iteration : 3
optimized : False
torch     : 2.7.1
device    : cpu
cache     : True
compile   : False
reasoning : False
============================================================
Output length: 41
Time: 1.43 sec
28 tokens/sec

 Large language models are artificial intelligence systems that can understand, generate, and process human language, enabling them to perform a wide range of tasks, from answering questions to writing articles, and even creating creative content.
```