from setuptools import setup, Extension
import sysconfig

module = Extension(
    "hilbert_native",
    sources=[
        "hilbert_pybind.c",
        "hilbert_native.c",
        "graph_ops.c",
        "spectral_ops.c",
        "prime_helix.c",
        "prime_store.c",
    ],
    include_dirs=["."],
    extra_compile_args=["/O2"],
)

setup(
    name="hilbert_native",
    version="1.0",
    ext_modules=[module],
)
