from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='facemat',
    version='1.0.0',
    author='Hyebin Cho, Jaehyup Lee',
    author_email='hyebin.cho@kaist.ac.kr',
    description='Uncertainty-Guided Face Matting for Occlusion-Aware Face Transformation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hyebin-c/FaceMat',
    packages=find_packages(include=['FaceMat', 'FaceMat.*']),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.21.0',
        'opencv-python>=4.5.0',
        'Pillow>=9.0.0',
        'scikit-image>=0.19.0',
        'tqdm>=4.60.0',
        'pyyaml>=5.4.0',
        'matplotlib>=3.4.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.0.0',
            'flake8>=3.9.0',
            'mypy>=0.900',
            'black>=21.0',
            'isort>=5.0.0',
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=0.5.0',
            'sphinxcontrib-napoleon>=0.7',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Graphics',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'facemat=FaceMat.main:main',
        ],
    },
    include_package_data=True,
    license='MIT',
    keywords=['face matting', 'occlusion handling', 'computer vision', 'deep learning'],
)