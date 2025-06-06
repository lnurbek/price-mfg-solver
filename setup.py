from setuptools import setup, find_packages

def read_requirements(relax_versions=True):
    with open('requirements.txt') as f:
        lines = f.readlines()
    reqs = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue  # skip empty lines and comments
        if relax_versions:
            # Replace == with >= for flexibility
            line = line.replace('==', '>=')
        reqs.append(line)
    return reqs

setup(
    name='price_mfg_solver',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=read_requirements(relax_versions=True),  # relaxed for setup
    python_requires='>=3.8',
)
