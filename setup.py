from setuptools import setup, find_packages

setup(
    name='alphazeropp',
    version='1.0.0',
    description='AlphaZero++ - A general development project',
    author='',
    author_email='',
    url='',
    package_dir={"": "src"}, 
    packages=find_packages(where="src"),
    python_requires='>=3.8',
    install_requires=[],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)
