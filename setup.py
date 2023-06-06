from setuptools import setup, find_packages


version = {}
with open("tint/__version__.py") as fp:
    exec(fp.read(), version)


with open("README.md") as f:
    readme = f.read()


setup(
    name="time_interpret",
    version=version["__version__"],
    description="Time interpret package.",
    long_description=readme,
    author="Joseph Enguehard",
    author_email="joseph@skippr.com",
    url="https://github.com/josephenguehard/time_interpret",
    python_requires=">=3.7",
    install_requires=[
        "captum",
        "numpy",
        "pandas",
        "pytorch-lightning",
        "scikit-learn",
        "scipy",
        "torch",
    ],
    dependency_links=[],
    include_package_data=True,
    packages=find_packages(exclude=("tests", "docs")),
)
