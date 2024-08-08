FROM python:3.12.5-bookworm
# Install git
RUN apt-get update && apt-get install -y git && apt-get clean
WORKDIR /usr/src/application
COPY requirements.txt ./
COPY IndicTransTokenizer/requirements.txt ./IndicTransTokenizer/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -r IndicTransTokenizer/requirements.txt
COPY . .
CMD ["uvicorn", "speedtest:app", "--host", "0.0.0.0", "--port", "5000" ,"--reload"]