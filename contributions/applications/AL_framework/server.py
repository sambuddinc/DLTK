import SimpleITK as sitk
import numpy as np
import os
import pandas as pd
from flask import Flask, send_file, jsonify, send_from_directory, make_response, request
from importlib import import_module
import threading
import json

app = Flask(__name__)
app.debug = True

# al_fw_path = 'C:\\Users\\sb17\\MResProject\\DLTK\\contributions\\applications\\AL_framework\\applications\\'
al_fw_path = os.path.join(os.path.dirname(__file__), 'applications')

def get_config_for_app(app_id):
    app_fn = os.path.join(os.path.dirname(__file__), 'applications', 'app' + str(app_id), 'app_config.json')
    with open(app_fn) as json_data:
        app_json = json.load(json_data)
    return app_json


def write_app_config(app_id, app_json):
    app_fn = os.path.join(os.path.dirname(__file__), 'applications', 'app' + str(app_id),  'app_config.json')
    with open(app_fn, 'w') as outfile:
        json.dump(app_json, outfile, indent=4)


@app.route("/list_applications", methods=['GET'])
def list_applications():
    apps = sorted(os.listdir(al_fw_path))
    cur_apps = []
    for i, ap in enumerate(apps):
        app_json = get_config_for_app(i)

        # app_df = pd.read_csv(al_fw_path + ap + '\\app_data',
        #                        dtype=object,
        #                        keep_default_na=False,
        #                        na_values=[]).as_matrix()
        app_data = {
            'id': i,
            'name': app_json['name'],
            'input': app_json['input_desc'],
            'output': app_json['output_desc']
            # 'name': app_df[0][0],
            # 'input': app_df[0][2],
            # 'output': app_df[0][3]
        }
        cur_apps.append(app_data)

    return jsonify(cur_apps)


@app.route("/app_info/<app_id>", methods=['GET'])
def get_app_info(app_id):
    info_df = pd.read_csv(al_fw_path + 'app' + str(app_id)+'\\app_data',
                          dtype=object,
                          keep_default_na=False,
                          na_values=[]).as_matrix()
    app_data = {
        'id': app_id,
        'name': info_df[0][0],
        'data path': info_df[0][1],
        'input': info_df[0][2],
        'output': info_df[0][3],
        'model exists': info_df[0][4],
        'model iteration': info_df[0][5],
        'retrain flag': info_df[0][6],
        'dp_for_update': info_df[0][7]
    }

    return jsonify(app_data)


@app.route("/init_app/<app_id>", methods=['GET'])
def init_app_for(app_id):
    print('init app for:')
    # Get app module and call init method in new thread:
    try:
        module_name = 'contributions.applications.AL_framework.applications.app' + str(app_id) + '.init_app'
        mod = import_module(module_name)
        met = vars(mod)['init_app_func']
        print('got func')
        res = met()
        if res == 2:
            return jsonify({'message': 'app is already initialised'})
        elif res == 1:
            return jsonify({'message': 'app is under initialisation/training, please be patient'})
        elif res == -1:
            return jsonify({'error': 'something went wrong nononononononon'})
    except ImportError as err:
        print(err)
        return jsonify({'error': err})


@app.route("/select_patches/<app_id>", methods=['GET'])
def select_patches_for(app_id):
    print('selectin patches')
    try:
        module_name = 'contributions.applications.AL_framework.applications.app' + str(app_id) + '.patch_selection'
        mod = import_module(module_name)
        met = vars(mod)['select_patches_func']
        res = met()
        print('got a result: ', res)
        if res == 2:
            return jsonify({'message': 'app is preparing patches to be annotated'})
        elif res == 1:
            return jsonify({'message': 'app is training, please try again later'})
        elif res == 3:
            # TODO: return the patches from folder
            return get_patch_batch_from(app_id)
            # return jsonify({'message': 'thanks for requesting some pathces, theyl;l be along shorttly',
            #                 'data': patches})
        elif res == 0:
            return jsonify({'message': 'no model exists for this app yet, please init using GET /init_app/<app_id>'})
        elif res == -1:
            return jsonify({'error': 'something not right here'})

    except ImportError as err:
        print(err)
        return jsonify({'error': err})


@app.route("/patch_batch/<app_id>/", methods=['GET'])
def get_patch_batch_from(app_id):
    app_json = get_config_for_app(app_id)

    patch_dir = os.path.join(al_fw_path, 'app' + str(app_id), 'data', 'active_patches')
    patches = []
    segs = []
    emsegs = []
    for i in range(app_json['batch_size']):
        raw_path = os.path.join(patch_dir, str(i) + '_patch.nii.gz')
        seg_path = os.path.join(patch_dir, str(i) + '_seg.nii.gz')
        emseg_path = os.path.join(patch_dir, str(i) + '_emseg.nii.gz')
        raw_sitk = sitk.ReadImage(raw_path)
        seg_sitk = sitk.ReadImage(seg_path)
        emseg_sitk = sitk.ReadImage(emseg_path)
        patches.append(sitk.GetArrayFromImage(raw_sitk).tolist())
        segs.append(sitk.GetArrayFromImage(seg_sitk).tolist())
        emsegs.append(sitk.GetArrayFromImage(emseg_sitk).tolist())

    return jsonify({
                'patch': patches,
                'seg': segs,
                'emseg': emsegs
            })


@app.route("/list_patches/<app_id>", methods=['GET'])
def list_patches_for(app_id):
    app_json = get_config_for_app(app_id)
    patches = list(range(app_json['batch_size']))
    return jsonify(patches)


@app.route("/patch/<app_id>/<patch_id>", methods=['GET'])
def get_patch_from(app_id, patch_id):
    patch_dir = os.path.join(al_fw_path, 'app' + str(app_id), 'data', 'active_patches')
    app_json = get_config_for_app(app_id)
    ims = []
    for i, im in enumerate(app_json['input_postfix']):
        raw_path = os.path.join(patch_dir, str(patch_id) + '_' + im)
        raw_sitk = sitk.ReadImage(raw_path)
        raw_sitk = sitk.RescaleIntensity(raw_sitk, 0, 255)
        raw_l = sitk.GetArrayFromImage(raw_sitk).tolist()
        ims.append(raw_l)

    seg_path = os.path.join(patch_dir, str(patch_id) + '_seg.nii.gz')
    emseg_path = os.path.join(patch_dir, str(patch_id) + '_emseg.nii.gz')

    seg_sitk = sitk.ReadImage(seg_path)
    emseg_sitk = sitk.ReadImage(emseg_path)
    em_l = np.unique(sitk.GetArrayFromImage(emseg_sitk).flatten())
    print(em_l)

    return jsonify({
                'patch': ims,
                'seg': sitk.GetArrayFromImage(seg_sitk).tolist(),
                'emseg': sitk.GetArrayFromImage(emseg_sitk).tolist()
            })


@app.route("/annotate_patch/<app_id>/<patch_id>", methods=['POST'])
def annontate_patch(app_id, patch_id):
    content = request.get_json()
    patch_arr = content['data']
    patch = np.array(patch_arr)
    patch = np.reshape(patch, [1, 64, 64])
    sitk_img = sitk.GetImageFromArray(patch)
    sitk_info = sitk.ReadImage(os.path.join(al_fw_path, 'app' + str(app_id), 'data', 'active_patches', str(patch_id) + '_seg.nii.gz'))
    sitk_img.CopyInformation(sitk_info)
    sitk.WriteImage(sitk_img, os.path.join(al_fw_path, 'app' + str(app_id), 'data', 'active_patches', str(patch_id) + '_anot.nii.gz'))
    return jsonify({
        'message': "Thanks For The Annotations!",
    })

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
@app.route("/raw", methods=['GET'])
def get_raw():
    filename = "C:\\Users\\sb17\\MResProject\\DLTK\\contributions\\applications\\AL_framework\\datasets\\ALout\\patch_example.nii.gz"
    im = sitk.GetArrayFromImage(sitk.ReadImage(filename))
    response = jsonify({'data': im.tolist()})
    return response
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
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response


app.run()
