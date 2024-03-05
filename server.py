from iie import IIE
from flask import Flask,request, jsonify
from dataclasses import asdict

def launch(ie:IIE):
    server = Flask(import_name=__name__)
    @server.post("/")
    def post_():
        request_data = request.json
        prompt = request_data["prompt"]
        intent = ie.extract(prompt)
        response_data = asdict(intent)
        response = jsonify(response_data)
        return response
    
    server.run("0.0.0.0",port=3838)
    return server