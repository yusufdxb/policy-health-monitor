from setuptools import setup

package_name = "phm_recovery"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", ["config/recovery.yaml"]),
        (f"share/{package_name}/launch", ["launch/recovery.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Yusuf Guenena",
    maintainer_email="youssefguenena@gmail.com",
    description="PHM Recovery Node: /phm/health -> safe cmd_vel hold + rewind hook.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "recovery_node = phm_recovery.recovery_node:main",
        ],
    },
)
