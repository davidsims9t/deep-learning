FROM python:3.6
COPY . /opt/app
WORKDIR /opt/app
RUN pip install -r requirements.txt
RUN useradd -ms /bin/bash admin
USER admin
CMD ["python", "cnn.py"]
