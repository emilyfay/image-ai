# Code by Emily Fay, Sept-2016

import os.path
import random, string
import shutil
import subprocess
import werkzeug

import numpy as np
from flask import render_template, request, redirect, url_for, jsonify
from scipy import ndimage, misc, sparse

from Image_completer import app

landing_upload_folder = "/home/ubuntu/web_app_AWS/Image_completer/static/upload_landing/"
raw_upload_folder = "/home/ubuntu/web_app_AWS/Image_completer/static/raw_uploads/"
proc_upload_folder = "/home/ubuntu/web_app_AWS/Image_completer/static/proc_uploads/"
masked_folder = "/home/ubuntu/web_app_AWS/Image_completer/static/masked_images/"
filled_folder = "/home/ubuntu/web_app_AWS/Image_completer/static/filled_images/"
completed_folder = "/home/ubuntu/web_app_AWS/Image_completer/static/completed_images/"

folder_list = [landing_upload_folder, raw_upload_folder, proc_upload_folder, masked_folder, filled_folder,
               completed_folder]


def clean_dir(file_handle):
    if "DEMO" in file_handle:
        return "demo"
    else:
        print(file_handle)
        print('Deleting')
        for folder in folder_list:
            for the_file in os.listdir(folder):
                if file_handle in the_file:
                    if "DEMO" not in the_file:
                        file_path = os.path.join(folder, the_file)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                        except Exception as e:
                            print(e)
        return "cleaned directories"


allowed_extensions = set(['jpg', 'jpeg', 'gif', 'png',
                          'eps', 'raw', 'bmp',
                          'tif', 'tiff',
                          'JPG', 'JPEG', 'GIF', 'PNG',
                          'EPS', 'BMP',
                          'TIF', 'TIFF'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in allowed_extensions


def randword(length):
    return ''.join(random.choice(string.lowercase) for i in range(length))


@app.route('/')
@app.route('/home')
def home():
    r_filename = randword(8)+'.png'
    return render_template("drop_image.html", rand_filename = r_filename)


@app.route('/clean')
def clean():
    file = request.args.get("filename")
    clean_dir(file)
    return redirect('home')


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/image')
def display_image():
    filename = request.args.get("filename")
    return render_template("display_image.html", image_path="../static/raw_uploads/"+filename,
                           url_extend=url_for('extend', filename=filename),
                           url_mask=url_for('apply_mask', filename=filename))


# dropzone activates this
@app.route('/flask-upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        new_filename = request.form['new_filename']
        print(new_filename)
        if file and allowed_file(file.filename):
            filename = werkzeug.secure_filename(file.filename)
            file.save(os.path.join(landing_upload_folder, filename))
            return redirect(url_for('file_upload', filename=filename, new_filename = new_filename))
    return 'error'


@app.route('/file_upload')
def file_upload():
    filename = request.args.get("filename")
    new_filename = request.args.get("new_filename")
    orig_path = os.path.join(landing_upload_folder, filename)

    # Move file out of landing zone
    raw_path = os.path.join(raw_upload_folder, new_filename)
    shutil.copy(orig_path, raw_path)

    return jsonify(dict(filename=filename, path=raw_path))


@app.route('/random_mask')
def apply_mask():
    filename = request.args.get("filename")

    in_path = raw_upload_folder + filename
    masked_path = masked_folder + filename
    in_image = (ndimage.imread(in_path, mode='RGB').astype(float))

    im_x_size = in_image.shape[0]
    im_y_size = in_image.shape[1]

    mask = np.zeros([im_x_size, im_y_size, 3])
    not_mask = np.ones([im_x_size, im_y_size, 3])
    noise_mat = np.ones([im_x_size, im_y_size, 3]) * 120

    szN = 2 * np.floor(im_x_size / 64)

    x1 = int(im_x_size - np.floor(im_x_size / 4))
    x2 = int(im_x_size - szN)
    y1 = int(np.floor(im_y_size / 3))
    y2 = int(np.floor(im_y_size / 2))

    mask[x1:x2, y1:y2, :] = noise_mat[x1:x2, y1:y2, :]
    not_mask[x1:x2, y1:y2, :] = 0.0

    masked_image = np.add(np.multiply(in_image, not_mask), mask)
    misc.imsave(masked_path, masked_image)

    return render_template("display_masked_image.html", image_path="../static/masked_images/" + filename,
                           filename=filename)


@app.route('/extend')
def extend():
    filename = request.args.get("filename")
    return render_template("extend.html", image_path="../static/raw_uploads/" + filename, filename = filename)


@app.route('/image-complete')
def complete_image():
    filename = request.args.get("filename")
    origin_path = raw_upload_folder + filename
    filled_path = filled_folder + "AI" + filename
    original_im = ndimage.imread(origin_path, mode='RGB').astype(float)
    x_ = original_im.shape[0]
    y_ = original_im.shape[1]
    szN = 2 * np.floor(x_ / 64)

    x1 = int(x_ - np.floor(x_ / 4))
    x2 = int(x_ - szN)
    y1 = int(np.floor(y_ / 3))
    y2 = int(np.floor(y_ / 2))

    sub_script = "python /home/ubuntu/web_app_AWS/image_fill_deconv.py --image_path %s --out_path %s --mask_x1 %i --mask_x2 %i --mask_y1 %i --mask_y2 %i" % (
        origin_path, filled_path, int(x1 * 64 / x_), int(x2 * 64 / x_), int(y1 * 64 / y_), int(y2 * 64 / y_))
    print('subprocess')
    subprocess.call(sub_script, shell=True)

    big_not_mask = np.ones([x_, y_, 3])

    big_not_mask[x1:x2, y1:y2, :] = 0.0

    big_mask = 1 - big_not_mask

    filled_image = ndimage.imread(filled_path, mode='RGB').astype(float)

    big_filled = misc.imresize(filled_image, (x_, y_, 3))

    complete_im = np.add(np.multiply(original_im, big_not_mask), np.multiply(big_filled, big_mask))
    completed_path = completed_folder + filename
    misc.imsave(completed_path, complete_im)
    return render_template("display_completed_image.html", ds_image_path="../static/raw_uploads/" + filename,
                           image_path="../static/completed_images/" + filename, filename = filename)


@app.route('/extend-edge', methods=['POST'])
def extend_edge():

    if request.method == 'POST':
        edge = request.form['edge']
        filename = request.form['filename']
        return redirect(url_for('image_extend', edge=edge, filename=filename))
    return 'error'


@app.route('/image_extend')
def image_extend():
    edge = request.args.get("edge")
    filename = request.args.get("filename")

    in_path = raw_upload_folder + filename
    in_im = ndimage.imread(in_path, mode='RGB').astype(float)
    x_ = in_im.shape[0]
    y_ = in_im.shape[1]
    ex_x = int(np.ceil(x_ / 5))
    ex_y = int(np.ceil(y_ / 5))

    if edge == "T":
        # extend the top
        image_holder = np.random.normal(size=[x_ + ex_x, y_, 3], scale=1)
        big_not_mask = np.zeros([x_ + ex_x, y_, 3])
        image_holder[ex_x:, :, :] = in_im
        big_not_mask[ex_x + 1:, :, :] = 1.0
        x1 = 0
        x2 = 64.0 // 5.5
        y1 = 0
        y2 = 64
        x_f = x_ + ex_x
        y_f = y_
    elif edge == "L":
        # extend left edge
        image_holder = np.random.normal(size=[x_, y_ + ex_y, 3], scale=1)
        image_holder[:, ex_y:, :] = in_im
        big_not_mask = np.zeros([x_, y_ + ex_y, 3])
        big_not_mask[:, ex_y + 1:, :] = 1.0
        x1 = 0
        x2 = 64
        y1 = 0
        y2 = 64 // 5
        x_f = x_
        y_f = y_ + ex_y
    elif edge == "R":
        # extend right edge
        image_holder = np.random.normal(size=[x_, y_ + ex_y, 3], scale=1)
        image_holder[:, :y_, :] = in_im
        big_not_mask = np.zeros([x_, y_ + ex_y, 3])
        big_not_mask[:, :y_ - 1, :] = 1.0
        x1 = 0
        x2 = 64
        y1 = 64 - 64 // 5
        y2 = 64
        x_f = x_
        y_f = y_ + ex_y
    else:
        # extend the bottom
        image_holder = np.random.normal(size=[x_ + ex_x, y_, 3], scale=1)
        image_holder[:x_, :, :] = in_im
        big_not_mask = np.zeros([x_ + ex_x, y_, 3])
        big_not_mask[:x_ - 1, :, :] = 1.0
        x1 = 64 - 64 // 5
        x2 = 64
        y1 = 0
        y2 = 64
        x_f = x_ + ex_x
        y_f = y_

    origin_path = masked_folder + filename
    misc.imsave(origin_path, image_holder)

    filled_path = filled_folder + "AI" + filename
    sub_script = "python /home/ubuntu/web_app_AWS/image_fill_deconv.py --image_path %s --out_path %s --mask_x1 %i --mask_x2 %i --mask_y1 %i --mask_y2 %i" % (
        origin_path, filled_path, x1, x2, y1, y2)
    print('subprocess')
    subprocess.call(sub_script, shell=True)

    big_mask = 1 - big_not_mask
    rs_filled_path = filled_folder + "rsAI" + filename
    filled_image = ndimage.imread(filled_path, mode='RGB').astype(float)

    big_filled = misc.imresize(filled_image, (x_f, y_f, 3))
    misc.imsave(rs_filled_path, big_filled)
    masked_im = np.multiply(image_holder, big_not_mask)
    masked_path = masked_folder + "M" + filename
    misc.imsave(masked_path, masked_im)
    complete_im = np.add(np.multiply(image_holder, big_not_mask), np.multiply(big_filled, big_mask))
    completed_path = completed_folder + filename
    misc.imsave(completed_path, complete_im)
    return render_template("display_completed_image.html", ds_image_path="../static/masked_images/" + "M" + filename,
                           image_path="../static/completed_images/" + filename)

# set the secret key.
# app.secret_key = os.urandom(24)
