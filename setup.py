from setuptools import setup

files = ['fluorescence_polarity/*']

setup_parameters = dict(
      name='fluorescence_polarity',
      version='0.1',
      description='Methods for quantifying spatial polarization of fluorescence markers',
      url='https://github.com/bgraziano/fluorescence_polarity',
      author='Brian Graziano',
      author_email='brgrazian@gmail.com',
      license='MIT',
      packages=['fluorescence_polarity'],
      install_requires=['numpy>=1.16.4', 'mpu>=0.20.0', 'scikit-image>=0.15.0', 'pandas>=0.25.0', 'scipy>=1.3.0'],
      include_package_data=True,
      zip_safe=False,
      python_requires='>=3.7',
      )
      
setup(**setup_parameters)