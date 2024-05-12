# Use an official Python runtime as a base image
FROM python:3.10
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of your application
COPY . /app

CMD ["python3", "pHPA.py"]