from flask import Flask
from src.logger import logging
from src.exception import CustomException
import os, sys

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def index():
    try:
        raise Exception("Just testing my Custom Exceptiom")
    except Exception as e:
        err = CustomException(e,sys)
        logging.info(err.error_message)
        return "Welcome to Haadi Raza's app where you come first ;)"
        

if __name__ == '__main__':
    app.run(debug=True)