# Troubleshooting Guide

This is a placeholder for providing help and tips on common issues encountered in the book. Right now, this page is intentionally left blank as no known issues have been reported.

&nbsp;
## File Download Issues

Please use [this discussion page](https://github.com/rasbt/reasoning-from-scratch/discussions/145) if you have any issues with file downloads.

&nbsp;
## Chapter 6

&nbsp;
### Corrupted Checkpoints

In `train_rlvr_grpo` (Chapter 6), a `Ctrl+C` triggers the `KeyboardInterrupt` handler to save a `-interrupt` checkpoint. If you press `Ctrl+C` a second time before the save completes, it can interrupt `torch.save` mid-write and leave a truncated `.pth` file. Wait for the `-interrupt` checkpoint message before exiting.

Corrupted model checkpoints usually raise load errors or fail during evaluation; another telltale sign is that they are much smaller than the expected ~1.5 GB.

&nbsp;
## Other Issues

For other issues, please feel free to open a new GitHub [Issue](https://github.com/rasbt/reasoning-from-scratch/issues).
