from setuptools import find_packages, setup

setup(
    name="phm_arbiter",
    version="0.1.0",
    packages=find_packages(exclude=["tests*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/phm_arbiter"]),
        ("share/phm_arbiter", ["package.xml"]),
        ("share/phm_arbiter/launch", ["launch/arbiter.launch.py"]),
        ("share/phm_arbiter/config", ["config/phm_arbiter.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Yusuf Guenena",
    maintainer_email="yusuf.a.guenena@gmail.com",
    description="PHM arbiter: fuses DetectorVerdict -> /phm/health (PolicyHealthStatus).",
    license="MIT",
    entry_points={
        "console_scripts": [
            "phm_arbiter = phm_arbiter.arbiter_node:main",
        ],
    },
)
