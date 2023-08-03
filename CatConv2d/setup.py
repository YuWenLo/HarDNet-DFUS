from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='catconv2d_cuda',
    ext_modules=[
        CUDAExtension('catconv2d_cuda', [
            'catconv2d_cuda.cpp',
            'catconv2d_cuda_kernel.cu',
            ],extra_compile_args={'cxx': [],
                                  'nvcc': ['-Xptxas=-O3,-v'
                                      #, '-maxrregcount=127'
                                      ]}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
