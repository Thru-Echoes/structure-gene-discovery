from setuptools import setup

setup(name = "structure-gene-discovery",
        version = '0.0.1',
        packages = ['structure-gene-discovery'],
        entry_points = {
            'console_scripts': [
                'structure-gene-discovery = structure-gene-discovery.__main__:main'
            ]
        },
    )
    
