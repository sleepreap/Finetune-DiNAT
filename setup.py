from setuptools import setup, find_namespace_packages

setup(name='DiNAT',
      python_requires=">=3.8",
      install_requires=[
          "transformers",
          "datasets",
          "evaluate",
          "lightning",
          "tqdm",
          "matplotlib",
          "opencv-python",
          "scipy"
      ],
      )
