from setuptools import setup, find_packages

with open("README-pypi.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pycorerelator",
    version="0.0.2",
    author="Larry Syu-Heng Lai",
    author_email="larrysyuhenglai@gmail.com",
    description="A package for quantitative stratigraphic correlation analysis across geological core and physical log data", 
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GeoLarryLai/pyCoreRelator",
    packages=find_packages(exclude=['example_data', 'example_data.*', 'outputs', 'outputs.*', 'old codes', 'old codes.*', 'dtw_graphics', 'dtw_graphics.*', 'mapinfo', 'mapinfo.*', 'pickeddepth', 'pickeddepth.*', 'pickeddepth_ages', 'pickeddepth_ages.*', 'SegmentPair_DTW_frames', 'SegmentPair_DTW_frames.*', 'TestCores', 'TestCores.*', '__pycache__', '__pycache__.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9,<3.14",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "Pillow>=8.3.0",
        "imageio>=2.9.0",
        "librosa>=0.9.0",
        "tqdm>=4.60.0",
        "joblib>=1.1.0",
        "IPython>=7.25.0",
        "psutil>=5.8.0",
        "pydicom>=2.3.0",
        "opencv-python>=4.5.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.6.0",
        "lightgbm>=3.3.0"
    ],
    extras_require={
        "interactive": ["ipympl>=0.9.0"],
        "advanced": ["scikit-image>=0.18.0"]
    },
)
