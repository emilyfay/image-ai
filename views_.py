import os.path
import random
import shutil

import numpy as np
import tensorflow as tf
import werkzeug
from flask import render_template, request,redirect,url_for, session
from scipy import ndimage, misc

from Image_completer import app
from fill_deconv import DCGAN
from utils import imsave_single

landing_upload_folder = "Image_completer/static/upload_landing/"
raw_upload_folder = "Image_completer/static/raw_uploads/"
proc_upload_folder = "Image_completer/static/proc_uploads/"
masked_folder = "Image_completer/static/masked_images/"
completed_folder = "Image_completer/static/completed_images/"

folder_list = [landing_upload_folder,raw_upload_folder,proc_upload_folder,masked_folder,completed_folder]
for folder in folder_list:
    for the_file in os.listdir(folder):
        if "test" not in the_file:
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)


allowed_extensions = set(['jpg', 'jpeg', 'gif', 'png',
                          'eps', 'raw', 'bmp',
                          'tif', 'tiff',
                          'JPG', 'JPEG', 'GIF', 'PNG',
                          'EPS', 'BMP',
                          'TIF', 'TIFF'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in allowed_extensions

@app.route('/')
@app.route('/home')
def home():
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
        filename='rawtest.png'
    return render_template("display_image.html", image_path="../static/raw_uploads/"+filename)


@app.route('/random_mask')
def apply_mask():
    filename = session.get('image_file', False)
    if filename==False:
        filename='rawtest.png'

    in_path = raw_upload_folder+filename
    proc_filename = filename.strip('raw')
    proc_path = proc_upload_folder+proc_filename
    masked_path = masked_folder+proc_filename
    in_image = ndimage.imread(in_path, mode='RGB').astype(float)

    im_size = 64
    pixel_depth = 255.0
    new_image = (misc.imresize(in_image, (im_size,im_size,3))-pixel_depth/2) / pixel_depth
    misc.imsave(proc_path, new_image)

    n = random.randint(1,6)
    mask = np.zeros([64,64,3])
    not_mask = np.ones([64,64,3])
    noise_mat = np.random.normal(size=[64,64,3], scale = 0.05)
    l = 8
    if n==1:
        mask[:, 0:l, :] = noise_mat[:, 0:l, :]
        not_mask[:, 0:l, :] = 0.0
    elif n==2:
        mask[0:l, :,:] = noise_mat[0:l, :, :]
        not_mask[0:l, :,:] = 0.0
    elif n==3:
        mask[-l:, :,:] = noise_mat[-l:, :, :]
        not_mask[-l:, :,:] = 0.0
    elif n==4:
        mask[:,-l:,] = noise_mat[:,-l:,]
        not_mask[:,-l:,] = 0.0
    else:
        mask[8:20, 8:20,:] = noise_mat[8:20, 8:20,:]
        not_mask[8:20, 8:20,:] = 0.0

    masked_image = np.add(np.multiply(new_image,not_mask), mask)
    misc.imsave(masked_path, masked_image)
    print(n)
    session['n_mask'] = n

    return render_template("display_masked_image.html", image_path="../static/masked_images/"+proc_filename)

@app.route('/image-complete')
def complete_image():
    n = session.get('n_mask')
    print(n)
    filename = session.get('image_file', False)
    if filename==False:
        filename='rawtest.png'
    filename = filename.strip('raw')
    origin_path = raw_upload_folder+"raw"+filename
    original_im = ndimage.imread(origin_path, mode='RGB').astype(float)
    x_ = original_im.shape[0]
    y_ = original_im.shape[1]
    big_not_mask = np.ones([x_,y_,3])
    lx = int(8*np.ceil(x_/64))
    ly = int(8*np.ceil(y_/64))
    mx = int(8*np.floor(x_/64))
    my = int(8*np.floor(y_/64))
    nx = int(20*np.ceil(x_/64))
    ny = int(20*np.ceil(y_/64))
    if n==1:
        big_not_mask[:, 0:ly, :] = 0.0
    elif n==2:
        big_not_mask[0:lx, :,:] = 0.0
    elif n==3:
        big_not_mask[-lx:, :,:] = 0.0
    elif n==4:
        big_not_mask[:,-ly:,] = 0.0
    else:
        big_not_mask[mx:nx, my:ny,:] = 0.0

    big_mask = 1-big_not_mask

    image_path = masked_folder+filename
    masked_image = ndimage.imread(image_path, mode='RGB')
    print('got image')

    '''
    Tensorflow code
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        dcgan = DCGAN(sess, image_size=64, batch_size=1,
                      is_crop=False, checkpoint_dir="../final_checkpoint_deconv")
        filled_image = dcgan.infill(masked_image)

    imsave_single(filled_image, ("Image_completer/static/completed_images/out_%s" %filename))
    print('saved completed image as %s' %filename)

    big_filled = misc.imresize(filled_image,(x_,y_,3), interp = 'lanczos')

    complete_im = np.add(np.multiply(original_im, big_not_mask),np.multiply(big_filled,big_mask))
    completed_path = completed_folder+filename
    misc.imsave(completed_path, complete_im)
    return render_template("display_completed_image.html", image_path="../static/completed_images/"+filename)


# set the secret key.
app.secret_key = os.urandom(24)