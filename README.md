# Diffusion Exploration
Exploring diffusion based models using a framework built on top of pytorch. The framework is called miniAI and is based off Jeremy Howard's class ["From Deep Learning Foundations to Stable Diffusion
"](https://course.fast.ai/Lessons/part2.html). Lesson 9 to 18 covers the basics of building a flexible ML framework using pytorch. The following steps are needed for installation:

1. Install CUDA 11.7. Link is [here](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local).

2.  Initialize new conda environment. Python version is 3.11.3

3. Install pytorch 2.0 cuda 11.7. Command is as of below.
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

4. Install datasets, fastcore, fastprogressbar, torcheval.

NOTE: multiprocessing for DataLoader will break on Windows 10. Can only use single core for applying transform when using Windows 10.

OBJECTIVE: I hope to implement the following papers using this framework.

1. [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
2. [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
3. [Elucidating the Design Space of Diffusion-Based Generative Models
](https://arxiv.org/abs/2206.00364) 
