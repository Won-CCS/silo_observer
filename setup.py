from setuptools import find_packages, setup

package_name = 'silo_observer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='crs3',
    maintainer_email='stew00010011@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'silo_observer_node = silo_observer.silo_observer_node:main',
            'image_window_operator_node = silo_observer.image_window_operator_node:main',
            'static_base_link_broadcaster = silo_observer.static_base_link_broadcaster:main',
            'silo_coordinates_broadcaster = silo_observer.silo_coordinates_broadcaster:main'
        ],
    },
)
