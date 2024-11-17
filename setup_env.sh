#!/bin/bash

# Set the name of your virtual environment
VENV_NAME="envv"

# Check if the virtual environment exists
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment '$VENV_NAME'..."
    python3 -m venv $VENV_NAME
else
    echo "Virtual environment '$VENV_NAME' already exists."
fi

# Activate the virtual environment
source $VENV_NAME/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the requirements
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

# Display installed versions
echo "Installed package versions:"
pip list

echo "Environment setup complete. Please try running your script again."