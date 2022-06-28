from setuptools import find_packages, setup

packages = find_packages()
package_data = {package: ["py.typed"] for package in packages}

setup(
    name="mrl",
    packages=packages,
    version="0.0.1",
    package_data=package_data,
)
