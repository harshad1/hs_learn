import setuptools

with open("README.md") as f:
    long_description = f.read()

setuptools.setup(
    name="hs_learn_01_harshad",
    version="0.0.1",
    author="Harshad Srinivasan",
    author_email="harshad1@zoho.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="A project for learning ML and python",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)
