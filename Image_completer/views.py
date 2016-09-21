from flask import render_template, request,redirect,url_for,send_from_directory, jsonify, session
from Image_completer import app
import werkzeug
import os.path
import shutil
from scipy import ndimage, misc
import random
import numpy as np
from PIL import Image

landing_upload_folder = "Image_completer/static/upload_landing/"
raw_upload_folder = "Image_completer/static/raw_uploads/"
proc_upload_folder = "Image_completer/static/proc_uploads/"
masked_folder = "Image_completer/static/masked_images/"
completed_folder = "Image_completer/static/completed_images/"

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
    in_path = proc_upload_folder+filename
    origin_path = raw_upload_folder+"raw"+filename
    image = ndimage.imread(in_path, mode='RGB').astype(float)
    original_im = ndimage.imread(origin_path, mode='RGB').astype(float)
    x_ = original_im.shape[0]
    y_ = original_im.shape[1]
    big_not_mask = np.ones([x_,y_,3])
    lx = 8*np.ceil(x_/64)
    ly = 8*np.ceil(y_/64)
    mx = 8*np.floor(x_/64)
    my = 8*np.floor(y_/64)
    nx = 20*np.ceil(x_/64)
    ny = 20*np.ceil(y_/64)
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

    big_image = misc.imresize(image,(x_,y_,3), interp = 'lanczos')

    complete_im = np.add(np.multiply(original_im, big_not_mask),np.multiply(big_image,big_mask))
    completed_path = completed_folder+filename
    misc.imsave(completed_path, complete_im)
    return render_template("display_completed_image.html", image_path="../static/completed_images/"+filename)


# set the secret key.
app.secret_key = os.urandom(24)