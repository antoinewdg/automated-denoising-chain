# Automated denoising chain

Implementation of state of the art noise detection and denoising algorithm,
with GPU acceleration of the bottlenecks using Numba CUDA (enables writing CUDA 
kernels in Python)

* [Analysis and Extension of the Ponomarenko et al. Method, Estimating a Noise Curve from a Single Image](http://www.ipol.im/pub/art/2013/45/)
* [Non-Local Means Denoising](http://www.ipol.im/pub/art/2011/bcm_nlm/)