import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name="jensen-shannon-centroid",
    version="1.0",
    author="Dustin Wright",
    author_email="dw@di.ku.dk",
    description="This package is used to calculate the Jensen-Shannon centroid of a set of categorical distributions.",
    python_requires=">=3.6",
    include_package_data=True,
    packages=['jensen_shannon_centroid'],
    install_requires=[
        'numpy'
    ],
    long_description=long_description,
    long_description_content_type="text/markdown"
    #package_data = {'':['data']}
)