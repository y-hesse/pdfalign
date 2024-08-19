from setuptools import setup, find_packages

setup(
    name='pdfalign',
    version='0.0.3',
    author='yhesse',
    description='pdfalign is a very simple tool to grid align extracted pdf text. This is useful for invoice table extraction or further processing with llms / rag systems',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/y-hesse/pdfalign',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'numpy',
        'pandas',
        'pymupdf',
        'pillow',
    ],
)

