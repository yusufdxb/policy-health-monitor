from setuptools import find_packages, setup

package_name = "phm_core"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Yusuf Guenena",
    maintainer_email="yusuf.a.guenena@gmail.com",
    description=(
        "Pure-Python detector logic for the Policy Health Monitor: Detector ABC, "
        "hysteresis, calibration, severity. No ROS dependency."
    ),
    license="MIT",
)
