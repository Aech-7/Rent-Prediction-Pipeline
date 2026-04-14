#Base image

FROM python:3.11-slim

#Set working directory

WORKDIR /app

COPY requirements.txt .
#Install depecdencies

RUN pip install -r requirements.txt

# Copy

COPY . .

#Expose port
EXPOSE 8501

#Run streamlit

CMD ["streamlit","run","app.py","--server.address=0.0.0.0"]
