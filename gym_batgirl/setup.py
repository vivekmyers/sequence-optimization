from setuptools import setup

setup(name='gym_batgirl',
      version='0.0.1',
      include_package_data=True,
      install_requires=['gym', 'numpy', 'matplotlib', 'pandas', 'seaborn',
            'tqdm', 'torch', 'biopython', 'sklearn']
)

