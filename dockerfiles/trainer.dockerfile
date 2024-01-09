FROM python:3.10-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# copy files over
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/

# set working directory
WORKDIR /

# install dependencies
RUN pip install -r requirements.txt --no-cache-dir

# set entrypoint
ENTRYPOINT ["python", "-u", "src/fruity/train.py"]