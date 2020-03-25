from setuptools import setup, find_packages

required_packages = ['numpy', 'torch>1.3', 'easydict', 'SimpleITK]


setup(name='medseg3d',
    version='1.0',
    description='3D Medical Image Segmentation Toolkit.',
    packages=find_packages(),
    url='https://github.com/qinliuliuqin/Medical-Segmentation3d-Toolkit',
    author='IDEA Lab, the University of North Carolina at Chapel Hill.',
    author_email='qinliu19@email.unc.edu',
    license='GNU GENERAL PUBLIC LICENSE V3.0',
    install_requires=required_packages,
    entry_points={
        'console_scripts':
            ['seg_train=segmentation3d.seg_train:main',
             'seg_infer=segmentation3d.seg_infer:main',
             'seg_eval=segmentation3d.seg_eval:main']
    }
)
