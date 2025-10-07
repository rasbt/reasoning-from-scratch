# Using `torch.compile()` on Windows

`torch.compile()` relies on *TorchInductor*, which JIT-compiles kernels and requires a working C/C++ compiler toolchain. 

So, on Windows, the setup required to make `torch.compile` work can be a bit more involved than on Linux or macOS, which usually don't require any extra steps besides installing PyTorch. 

If you are a Windows user and using `torch.compile` sounds too tricky or complicated, don't worry, all code examples in this repository will work fine without compilation.

Below are some tips that I compiled based on recommendations by [Daniel Kleine](https://github.com/d-kleine) and the following [PyTorch guide](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html).

&nbsp;
## 1 Basic Setup (CPU or CUDA)

&nbsp;
### 1.1 Install Visual Studio 2022

- Select the **“Desktop development with C++”** workload.
- Make sure to include the **English language pack**  (without it, you may run into UTF-8 encoding errors.)

&nbsp;
### 1.2 Open the correct command prompt


Launch Python from the 

**"x64 Native Tools Command Prompt for VS 2022"**

or from the

**"Visual Studio 2022 Developer Command Prompt"**.

Alternatively, you can initialize the environment manually by running:

```bash
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

&nbsp;
### 1.3 Verify that the compiler works

Run

   ```bash
   cl.exe
   ```

If you see version information printed, the compiler is ready.

&nbsp;
## 2 Troubleshooting Common Errors

&nbsp;
### 2.1 Error: `cl not found`

Install **Visual Studio Build Tools** with the "C++ build tools" workload and run Python from a developer command prompt. (See this Microsoft [guide](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170) for details)

&nbsp;
### 2.2 Error: `triton not found` (when using CUDA)

Install the Windows build of Triton manually:

```bash
pip install "triton-windows<3.4"
```

or, if you are using `uv`:

```bash
uv pip install "triton-windows<3.4"
```

(As mentioned earlier, triton is required by TorchInductor for CUDA kernel compilation.)



&nbsp;
## 3 Additional Notes

On Windows, the `cl.exe` compiler is only accessible from within the Visual Studio Developer environment. This means that using `torch.compile()` in notebooks such as Jupyter may not work unless the notebook was launched from a Developer Command Prompt.

As mentioned at the beginning of this article, there is also a [PyTorch guide](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html) that some users found helpful when getting `torch.compile()` running on Windows CPU builds. However, note that it refers to PyTorch's unstable branch, so use it as a reference only.

**If compilation continues to cause issues, please feel free to skip it. It's a nice bonus, but it's not important to follow the book.**

