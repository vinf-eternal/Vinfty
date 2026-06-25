from setuptools import setup, find_packages

setup(
    name="vinfty",
    version="0.2.0",
    description="AI consistency auxiliary auditor — ont_self, C_ij coupling, HMM drift, barrier economics",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="V∞ Project",
    author_email="vinfty@twincosmos.dev",
    url="https://github.com/twincosmos/vinfty",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="cognitive-architecture observability llm-orchestration v-infty",
)
