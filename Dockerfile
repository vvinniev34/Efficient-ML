# Use an appropriate base image
FROM python:3.8

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Command to run your application
CMD ["python", "train_model.py"]
