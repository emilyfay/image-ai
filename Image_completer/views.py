import os.path
import random, string
import shutil
import subprocess
import werkzeug

import numpy as np
from flask import render_template, request,redirect,url_for, session
from scipy import ndimage, misc

from Image_completer import app

landing_upload_folder = "/home/ubuntu/web_app_AWS/Image_completer/static/upload_landing/"
raw_upload_folder = "/home/ubuntu/web_app_AWS/Image_completer/static/raw_uploads/"
proc_upload_folder = "/home/ubuntu/web_app_AWS/Image_completer/static/proc_uploads/"
masked_folder = "/home/ubuntu/web_app_AWS/Image_completer/static/masked_images/"
filled_folder = "/home/ubuntu/web_app_AWS/Image_completer/static/filled_images/"
completed_folder = "/home/ubuntu/web_app_AWS/Image_completer/static/completed_images/"

folder_list = [landing_upload_folder,raw_upload_folder,proc_upload_folder,masked_folder,filled_folder, completed_folder]

def clean_dir():
    for folder in folder_list:
        for the_file in os.listdir(folder):
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

    return''.join(random.choice(string.lowercase) for i in range(length))

clean_dir()


@app.route('/')
@app.route('/home')
def home():
    clean_dir()
    session.clear()
    return render_template("drop_image.html")

# dropzone activates this
@app.route('/flask-upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = werkzeug.secure_filename(file.filename)
            file.save(os.path.join(landing_upload_folder, filename))
            return redirect(url_for('file_upload', filename = filename))
    return 'error'

@app.route('/file_upload')
def file_upload():
    filename = request.args.get("filename")
    orig_path = os.path.join(landing_upload_folder, filename)

    # Move file out of landing zone
    raw_path = os.path.join(raw_upload_folder, "raw"+filename)
    shutil.copy(orig_path, raw_path)

    session['image_file'] = "raw"+filename
    return 'uploaded'

@app.route('/image')
def display_image():
    filename = session.get('image_file',False)
    if filename==False:
        filename = "DEMO.png"
        orig_path = os.path.join(raw_upload_folder, filename)
        raw_path = os.path.join(raw_upload_folder, "raw"+filename)
        shutil.copy(orig_path, raw_path)

    r_filename = randword(6)+filename.strip('raw')
    session['r_image_file'] = r_filename

    return render_template("display_image.html", image_path="../static/raw_uploads/"+filename)


@app.route('/random_mask')
def apply_mask():
    filename = session.get('image_file', False)
    r_filename = session.get('r_image_file', False)
    if r_filename==False:
        return redirect(url_for('home'))
    if filename==False:
        filename = "DEMO.png"

    in_path = raw_upload_folder+filename
    proc_filename = r_filename
    masked_path = masked_folder+proc_filename
    in_image = (ndimage.imread(in_path, mode='RGB').astype(float))

    im_x_size = in_image.shape[0]
    im_y_size = in_image.shape[1]

    mask = np.zeros([im_x_size,im_y_size,3])
    not_mask = np.ones([im_x_size,im_y_size,3])
    noise_mat = np.ones([im_x_size,im_y_size,3])*120

    szN = np.floor(im_x_size/)

    x1 = im_x_size-np.floor(im_x_size/4.2)
    x2 = im_x_size-szN
    y1 = im_y_size-np.floor(im_y_size/4.2)
    y2 = im_y_size-szN

    mask[x1:x2,y1:y2,:] = noise_mat[x1:x2,y1:y2,:]
    not_mask[x1:x2,y1:y2,:] = 0.0

    masked_image = np.add(np.multiply(in_image,not_mask), mask)
    misc.imsave(masked_path, masked_image)

    session['mask_x1'] = x1
    session['mask_x2'] = x2
    session['mask_y1'] = y1
    session['mask_y2'] = y2

    return render_template("display_masked_image.html", image_path="../static/masked_images/"+proc_filename)

@app.route('/extend')
def extend():
    filename = session.get('image_file', False)
    r_filename = session.get('r_image_file', False)
    if r_filename==False:
        return redirect(url_for('home'))
    if filename==False:
        filename = "DEMO.png"

    in_path = raw_upload_folder+filename
    proc_filename = r_filename
    masked_path = masked_folder+proc_filename
    in_image = (ndimage.imread(in_path, mode='RGB').astype(float))

    im_x_size = in_image.shape[0]
    im_y_size = in_image.shape[1]
    extend_sz = 20*(np.floor(im_x_size/64))

    image_holder = np.ones([im_x_size,im_y_size+extend_sz,3])*120
    image_holder[:im_x_size,extend_sz:im_y_size+extend_sz,:] = in_image

    x1 = 0
    x2 = (im_x_size)*(np.floor(im_x_size/64))
    y1 = 0
    y2 = 20*(np.floor(im_y_size/64))

    misc.imsave(masked_path, image_holder)

    session['mask_x1'] = x1
    session['mask_x2'] = x2
    session['mask_y1'] = y1
    session['mask_y2'] = y2

    return render_template("display_extended_image.html", image_path="../static/masked_images/"+proc_filename)



@app.route('/image-complete')
def complete_image():

    x1 = session.get('mask_x1')
    x2 = session.get('mask_x2')
    y1 = session.get('mask_y1')
    y2 = session.get('mask_y2')

    print('here')
    filename = session.get('image_file', False)
    r_filename = session.get('r_image_file', False)
    if r_filename==False:
        return redirect(url_for('home'))
    if filename==False:
        filename = "DEMO.png"
    print('here')

    filename = filename.strip('raw')
    origin_path = raw_upload_folder+"raw"+filename
    filled_path = filled_folder+"AI"+r_filename
    original_im = ndimage.imread(origin_path, mode='RGB').astype(float)
    x_ = original_im.shape[0]
    y_ = original_im.shape[1]
    sub_script = "python /home/ubuntu/web_app_AWS/image_fill_deconv.py --image_path %s --out_path %s --mask_x1 %i --mask_x2 %i --mask_y1 %i --mask_y2 %i" %(origin_path,filled_path,int(x1*64/x_),int(x2*64/x_),int(y1*64/y_),int(y2*64/y_))
    print('subprocess')
    subprocess.call(sub_script, shell = True)


    big_not_mask = np.ones([x_,y_,3])

    big_not_mask[x1:x2, y1:y2,:] = 0.0

    big_mask = 1-big_not_mask

    filled_image = ndimage.imread(filled_path, mode='RGB').astype(float)

    big_filled = misc.imresize(filled_image,(x_,y_,3))

    complete_im = np.add(np.multiply(original_im, big_not_mask),np.multiply(big_filled,big_mask))
    completed_path = completed_folder+r_filename
    misc.imsave(completed_path, complete_im)
    return render_template("display_completed_image.html", ds_image_path = "../static/filled_images/"+"AI"+r_filename,image_path="../static/completed_images/"+r_filename)

@app.route('/image-extend')
def extend_image():

    x1 = session.get('mask_x1')
    x2 = session.get('mask_x2')
    y1 = session.get('mask_y1')
    y2 = session.get('mask_y2')

    r_filename = session.get('r_image_file', False)
    if r_filename==False:
        return redirect(url_for('home'))

    origin_path = masked_folder+r_filename
    filled_path = filled_folder+"AI"+r_filename
    sub_script = "python /home/ubuntu/web_app_AWS/image_fill_deconv.py --image_path %s --out_path %s --mask_x1 %i --mask_x2 %i --mask_y1 %i --mask_y2 %i" %(origin_path,filled_path,x1,x2,y1,y2)
    print('subprocess')
    subprocess.call(sub_script, shell = True)

    original_im = ndimage.imread(origin_path, mode='RGB').astype(float)
    x_ = original_im.shape[0]
    y_ = original_im.shape[1]
    big_not_mask = np.ones([x_,y_,3])

    big_not_mask[x1:x2, y1:y2,:] = 0.0

    big_mask = 1-big_not_mask

    filled_image = ndimage.imread(filled_path, mode='RGB').astype(float)

    big_filled = misc.imresize(filled_image,(x_,y_,3))

    complete_im = np.add(np.multiply(original_im, big_not_mask),np.multiply(big_filled,big_mask))
    completed_path = completed_folder+r_filename
    misc.imsave(completed_path, complete_im)
    return render_template("display_completed_image.html", ds_image_path = "../static/filled_images/"+"AI"+r_filename,image_path="../static/completed_images/"+r_filename)


# set the secret key.
app.secret_key = os.urandom(24)
