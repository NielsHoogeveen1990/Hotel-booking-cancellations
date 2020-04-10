from setuptools import setup, find_packages
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='hotelbooking',
    keywords='',
    version='0.1',
    author='Niels Hoogeveen',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    entry_points={'console_scripts': ['hotelbooking = hotelbooking.cli:main']},
    description='ML model to predict amount of special requests per booking.',
    long_description=read('README.md'),
    long_description_content_type='text/markdown'
)



