import SimpleITK as sitk
import numpy as np
import os
import pandas as pd
from flask import Flask, send_file, jsonify, send_from_directory, make_response, request

app = Flask(__name__)
app.debug = True

al_fw_path = 'C:\\Users\\sb17\\MResProject\\DLTK\\contributions\\applications\\AL_framework\\applications\\'


@app.route("/list_applications", methods=['GET'])
def list_applications():
    apps = os.listdir(al_fw_path)
    cur_apps = []
    for i, ap in enumerate(apps):
        app_df = pd.read_csv(al_fw_path + ap + '\\app_data',
                               dtype=object,
                               keep_default_na=False,
                               na_values=[]).as_matrix()
        app_data = {
            'id': i,
            'name': app_df[0][0],
            'input': app_df[0][2],
            'output': app_df[0][3]
        }
        cur_apps.append(app_data)

    return jsonify(cur_apps)


# @app.route("/patch", methods=['GET'])
# def get_patch():
#     filename = "contributions/applications/AL_framework/datasets/ALout/test.nii.gz"
#     rv = send_file(filename)
#     rv.headers.set('Content-Type', 'application/gzip')
#     rv.headers.set('Content-Disposition', 'attachment', filename="filename.nii.gz")
#     return rv
#
#
# @app.route("/patch2", methods=['GET'])
# def get_patch2():
#     filename = "/contributions/applications/AL_framework/datasets/ALout/"
#     rv = send_from_directory(filename, "test.nii.gz")
#     rv.headers.set('Content-Type', 'application/gzip')
#     rv.headers.set('Content-Disposition', 'attachment', filename="filename.nii.gz")
#     return rv
#
#
# @app.route("/raw", methods=['GET'])
# def get_raw():
#     filename = "C:\\Users\\sb17\\MResProject\\DLTK\\contributions\\applications\\AL_framework\\datasets\\ALout\\patch_example.nii.gz"
#     im = sitk.GetArrayFromImage(sitk.ReadImage(filename))
#     response = jsonify({'data': im.tolist()})
#     return response
#
#
# @app.route("/seg", methods=['GET', 'POST', 'OPTIONS'])
# def get_seg():
#     print(request.method)
#     if request.method == 'GET':
#         filename = "C:\\Users\\sb17\\MResProject\\DLTK\\contributions\\applications\\AL_framework\\datasets\\ALout\\seg_example.nii.gz"
#         im = sitk.GetArrayFromImage(sitk.ReadImage(filename))
#         response = jsonify({'data': im.tolist()})
#         return response
#     if request.method == 'POST':
#         data = request.json
#         print(request)
#         print(data)
#         response = jsonify({'data': "Thanks for the patch!"})
#         return response
#
#     if request.method == 'OPTIONS':
#         print(request.json)
#         response = jsonify({'data': "Thanks for the OPTIONs patch!"})
#         return response
#
#     #return jsonify("Errorrrrrr")
#     return make_response()


@app.after_request
def add_header(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


app.run()
