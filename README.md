## Local Python Environment Setup

To reproduce the project environment locally, the user should create a Python virtual environment and install the project dependencies.

Example:

```bash

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

```

## Registering the Jupyter Kernel

After creating and activating the virtual environment, register it as a Jupyter kernel:

```bash

python -m ipykernel install --user --name bdcc-env --display-name "Python (bdcc-env)"
```

Then, inside Jupyter or VS Code, select the kernel:

```text

Python (bdcc-env)

```