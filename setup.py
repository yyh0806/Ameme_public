import setuptools

#############################################
# File Name: setup.py
# Author: yangyuhui
# Mail:
# Created Time:  2020/6/11
#############################################
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Ameme",
    version="0.0.6",
    author="yangyuhui",
    author_email="yangyuhui0806@gmail.com",
    description="my tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yyh0806/Ameme",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)