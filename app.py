# from flask import Flask, request,abort
from flask_restful import Resource, Api
from werkzeug.utils import secure_filename
import os
from marshmallow import Schema, fields
from classes import *

app = Flask(__name__)
api = Api(app)

UPLOAD_FOLDER = r'C:\Users\91886\Desktop\intership\text_extraction\temp'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif'}
JSON_FILE = "data.json"
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class BarQuerySchema(Schema):
    key1 = fields.Str(required=True)
    # key2 = fields.Str(required=True)

schema = BarQuerySchema()

bfile = None

class Load(Resource):


    #  TODO flag if there are more than one aadhaar in an image or pdf
    def post(self):
        if 'file' not in request.files:
            return {'message': 'No file part'}, 400
        # global file
        file = request.files['file']
        if file.filename == '':
            return {'message': 'No selected file'}, 400

        # if not allowed_file(file.filename):
        #      return {'message': 'file format not supported','filename':str(file.filename)}
        #TODO check for the type in the backend not the frontend

        #     # filename = secure_filename(file.filename)
        #     # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #     extractor = AadhaarExtractor()
        #     # TODO protect against wrong password
        #     extractor.load(file)                                            #processes just single file
        #     print(extractor.to_json(JSON_FILE))
        #
        #     # print(txt)# work is here
        global bfile
        bfile = file.read()
        print(type(bfile))
        return {'message': 'File uploaded successfully','type':str(type(bfile))}, 200




    # def get(self):
    #     import json
    #     try:
    #         with open(JSON_FILE,'r') as f:
    #             wholedict = json.load(f)
    #             aadhaar_no = (request.args['key1'])
    #             # if errors:
    #             #     abort(400, str(errors))
    #             try:
    #                 info = wholedict[aadhaar_no]
    #             except KeyError :
    #                 return {"message":"This aadhaar number is not present in the database"}
    #
    #              # FIXED what if requested aadhaar number is not present in the database
    #
    #             return info
    #     except FileNotFoundError:
    #         return {'message':"No data in the database"}


class Getjson(Resource):
    def get(self):
        # global file
        # file.read()
        from tempfile import TemporaryFile
        file = TemporaryFile()
        file.write(bfile)
        print(type(file))
        if (file==None) :
            return {"message":"file not selected for jsonification"}
        extractor = AadhaarExtractor()

        print(type(file))
        extractor.load(file)
        json = extractor.to_json()
        return json
class Test(Resource):
    def get(self):
        return {'message':"success"}

api.add_resource(Getjson, '/getjson')
api.add_resource(Load, '/load')
api.add_resource(Test,"/test")

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
