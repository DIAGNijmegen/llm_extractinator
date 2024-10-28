from setuptools import setup, find_packages


def read_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()


setup(
    name="llm_data_extractor",
    version="0.1.0",
    packages=find_packages(),  # Automatically find all packages in the project
    install_requires=read_requirements(),  # Define dependencies
    python_requires=">=3.6",  # Specify the Python version requirement
    description="A brief description of your package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Luc Builtjes",
    author_email="Luc.Builtjes@radboudumc.nl",
    url="https://github.com/yourusername/your-repo",  # GitHub repo or homepage URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,  # Include files from MANIFEST.in
    entry_points={
        "console_scripts": [
            "extract_data=data_extractor.main:extract_data",
            "evaluate=data_extractor.main:evaluate",
            "post_process=data_extractor.post_processing.DRAGON_post_processing:main",
        ],
    },
)
