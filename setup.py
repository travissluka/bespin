import setuptools  # type: ignore

setuptools.setup(
    name="bespin",
    version="0.0.1",
    author="JCSDA",
    description="Binned Error Statistics Package for INtegrated diagnostics",
    url="https://github.com/jcsda-internal/bespin",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        'Natural Language :: English',
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'click',
        'netcdf4',
        'numpy>=1.19',
        'scipy',
        'xarray',
    ],
    package_dir={"": "src"},
    packages = setuptools.find_packages(where='src'),
    package_data={"bespin": ["py.typed"]},
    python_requires=">=3.7",
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'bespin=bespin.bin.bespin:cli',
        ]
    }
)
