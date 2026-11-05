"""Setup script for var_offside package."""

from setuptools import setup, find_packages

setup(
    name="var_offside",
    version="1.0.0",
    description="VAR Offside Detection System for Football",
    author="VAR Offside Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "ultralytics>=8.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "Pillow>=9.0.0",
        "PyQt5>=5.15.0",
        "fastapi>=0.115.0",
        "uvicorn>=0.30.0",
        "websockets>=12.0",  # live 3D view in the web app
        "python-multipart>=0.0.9",
        # 3D VAR replay rendering
        "trimesh>=4.0.0",
        "pyrender>=0.1.45",
        "pyglet<2.0",  # pyrender needs the 1.x OpenGL API
        "scipy>=1.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
        "umap": [
            "umap-learn>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "var-offside=var_offside.demo:main",
            "var-offside-app=var_offside.app.__main__:main",
            "var-offside-api=var_offside.api.__main__:main",
        ],
    },
)
