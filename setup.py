from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import subprocess
import os
import sys
import pybind11

class CustomExtension(Extension):
    def __init__(self, *args, **kwargs):
        self.cuda_sources = kwargs.pop('cuda_sources', [])
        super().__init__(*args, **kwargs)

class CUDA_build_ext(build_ext):
    def build_extensions(self):
        cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
        nvcc = os.path.join(cuda_home, 'bin', 'nvcc')

        for ext in self.extensions:
            if not isinstance(ext, CustomExtension):
                super().build_extension(ext)
                continue

            ext_path = self.get_ext_fullpath(ext.name)
            build_temp = self.build_temp
            os.makedirs(build_temp, exist_ok=True)

            print(f"Build temp directory: {build_temp}")
            print(f"CUDA Home: {cuda_home}")
            print(f"NVCC path: {nvcc}")

            # Compile CUDA sources
            cuda_objects = []
            for cuda_source in ext.cuda_sources:
                try:
                    obj = os.path.join(build_temp, os.path.splitext(os.path.basename(cuda_source))[0] + '.o')
                    command = [
                        nvcc,
                        '-c', cuda_source,
                        '-o', obj,
                        '-Xcompiler', '-fPIC',
                        '-I', os.path.join(cuda_home, 'include'),
                        '-O3'
                    ]
                    print(f"Running command: {' '.join(command)}")
                    subprocess.check_call(command)
                    cuda_objects.append(obj)
                except subprocess.CalledProcessError as e:
                    print(f"Error output: {e.output}")
                    raise Exception(f"Compilation of {cuda_source} failed: {e}")

            # Add compiled CUDA objects to the sources
            ext.extra_objects.extend(cuda_objects)

            # Modify the extension to include CUDA options
            ext.include_dirs.append(os.path.join(cuda_home, 'include'))
            ext.library_dirs.append(os.path.join(cuda_home, 'lib64'))
            ext.libraries.append('cudart')
            ext.extra_compile_args = ['-std=c++11', '-O3']  # Only for C++ compiler
            ext.extra_link_args.append(f'-L{os.path.join(cuda_home, "lib64")}')

            # Set runtime library path
            if sys.platform == 'linux':
                ext.runtime_library_dirs = [os.path.join(cuda_home, 'lib64')]

        # Build updated extensions
        try:
            super().build_extensions()
        except Exception as e:
            print(f"Error during build_extensions: {e}")
            print(f"Extension attributes:")
            for ext in self.extensions:
                print(f"  Name: {ext.name}")
                print(f"  Sources: {ext.sources}")
                print(f"  Include dirs: {ext.include_dirs}")
                print(f"  Library dirs: {ext.library_dirs}")
                print(f"  Libraries: {ext.libraries}")
                print(f"  Extra objects: {ext.extra_objects}")
                print(f"  Extra compile args: {ext.extra_compile_args}")
                print(f"  Extra link args: {ext.extra_link_args}")
            raise

cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')

ext_modules = [
    CustomExtension(
        'cudafinance.cuda_module',
        sources=['src/cudafinance/cuda_bindings.cpp'],
        cuda_sources=['src/cudafinance/cuda_kernels.cu'],
        include_dirs=[os.path.join(cuda_home, 'include'), pybind11.get_include()],
        library_dirs=[os.path.join(cuda_home, 'lib64')],
        libraries=['cudart'],
        extra_link_args=['-lcudart'],
    )
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cudafinance",
    version="0.1.0",
    description="CUDA-accelerated financial indicators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Beniamino Viganò",
    author_email="beniamino.vigano@protonmail.com",
    url="https://github.com/benvigano/cudafinance",
    keywords='Indicators, Finance, CUDA, GPU-accelerated',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
    cmdclass={'build_ext': CUDA_build_ext},
    zip_safe=False,
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "pybind11",
    ],
    setup_requires=["pybind11>=2.5.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    package_data={
        "": ["README.md", "LICENSE"],
    },
)