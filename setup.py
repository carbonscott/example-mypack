import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mypack",
    version="24.04.22",
    author="Your Name",
    author_email="...",
    description="...",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="xxx",
    keywords = ['MyPack'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
)
