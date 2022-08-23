import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='genepro',
    version='0.0.9',
    author='Marco Virgolin',
    author_email='marco.virgolin@cwi.nl',
    url='https://github.com/marcovirgolin/genepro',
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    setup_requires=['wheel'],
    install_requires=[
    	'numpy>=1.22.2',
        'scikit-learn>=1.0.2',
        'joblib>=1.1.0',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
