from setuptools import setup

setup(name='fluorescence_polarity',
      version='0.1',
      description='Methods for quantifying polarity of fluorescence markers',
      url='https://github.com/bgraziano/fluorescence_polarity',
      author='Brian Graziano',
      author_email='brgrazian@gmail.com',
      license='MIT',
      packages=['fluorescence_polarity'],
      install_requires=['numpy', 'mpu', 'scikit-image', 'pandas', 'scipy'],
      include_package_data=True,
      zip_safe=False)