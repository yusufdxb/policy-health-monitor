from setuptools import find_packages, setup

package_name = "phm_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", ["config/phm_sim.yaml"]),
        (f"share/{package_name}/launch", ["launch/sim.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Yusuf Guenena",
    maintainer_email="youssefguenena@gmail.com",
    description=(
        "Replay/perturbation publisher feeding PolicyEmbedding so the OOD "
        "pipeline runs without a real policy."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "embedding_publisher = phm_sim.embedding_publisher_node:main",
        ],
    },
)
