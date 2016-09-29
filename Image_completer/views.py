import os.path
import random, string
import shutil
import subprocess
import werkzeug
import gc

import numpy as np
from flask import render_template, request,redirect,url_for, session
from scipy import ndimage, misc, sparse
from PIL import Image
import pyamg

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
            if the_file != "DEMO.png":
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

@app.route('/about')
def about():
    return render_template("about.html")

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

@app.route('/random_mask')
def apply_mask():
    filename = session.get('image_file', False)
    if filename==False:
        filename = "DEMO.png"

    orig_path = os.path.join(raw_upload_folder, filename)
    r_filename = randword(6)+filename
    raw_path = os.path.join(raw_upload_folder, r_filename)
    shutil.copy(orig_path, raw_path)

    session['r_image_file'] = r_filename

    in_path = raw_upload_folder+r_filename
    proc_filename = r_filename
    masked_path = masked_folder+proc_filename
    in_image = (ndimage.imread(in_path, mode='RGB').astype(float))

    im_x_size = in_image.shape[0]
    im_y_size = in_image.shape[1]

    mask = np.zeros([im_x_size,im_y_size,3])
    not_mask = np.ones([im_x_size,im_y_size,3])
    noise_mat = np.ones([im_x_size,im_y_size,3])*120

    szN = 2*np.floor(im_x_size/64)

    x1 = im_x_size-np.floor(im_x_size/4)
    x2 = im_x_size-szN
    y1 = np.floor(im_y_size/3.5)
    y2 = np.floor(im_y_size/2)

    mask[x1:x2,y1:y2,:] = noise_mat[x1:x2,y1:y2,:]
    not_mask[x1:x2,y1:y2,:] = 0.0

    masked_image = np.add(np.multiply(in_image,not_mask), mask)
    misc.imsave(masked_path, masked_image)

    session['mask_x1'] = x1
    session['mask_x2'] = x2
    session['mask_y1'] = y1
    session['mask_y2'] = y2

    return render_template("display_masked_image.html", image_path="../static/masked_images/"+r_filename)

@app.route('/extend')
def extend():
    filename = session.get('image_file',False)

    if filename==False:
        filename = "DEMO.png"

    orig_path = os.path.join(raw_upload_folder, filename)
    r_filename = randword(6)+filename
    raw_path = os.path.join(raw_upload_folder, r_filename)
    shutil.copy(orig_path, raw_path)

    session['r_image_file'] = r_filename
    gc.collect()

    return render_template("extend.html", image_path="../static/raw_uploads/"+r_filename)



@app.route('/image-complete')
def complete_image():

    x1 = int(session.get('mask_x1'))
    x2 = int(session.get('mask_x2'))
    y1 = int(session.get('mask_y1'))
    y2 = int(session.get('mask_y2'))

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

    blended_path = filled_folder+"b"+r_filename
    target = np.asarray(Image.open(origin_path).resize((64,64),resample=Image.LANCZOS))
    source = np.asarray(Image.open(filled_path))
    mask = np.zeros((64,64))
    mask[x1:x2,y1:y2] = 1.0

    mask.flags.writeable = True
    target.flags.writeable = True
    source.flags.writeable = True

    small_blended = blend(target,source,mask,offset=(0,0))
    misc.imsave(blended_path,small_blended)
    big_blended = misc.imresize(small_blended,(x_,y_,3))
    big_blended = big_blended[:,:,:3]

    complete_im = np.add(np.multiply(original_im, big_not_mask),np.multiply(big_blended,big_mask))
    completed_path = completed_folder+r_filename
    misc.imsave(completed_path, complete_im)
    return render_template("display_completed_image.html", ds_image_path = "../static/masked_images/"+r_filename,image_path="../static/completed_images/"+r_filename)

@app.route('/extend-edge', methods=['POST'])
def extend_edge():
    if request.method == 'POST':
        edge = request.form['edge']
        return redirect(url_for('image_extend', edge = edge))
    return 'error'

@app.route('/image_extend')
def image_extend():
    edge = request.args.get("edge")

    r_filename = session.get('r_image_file', False)
    if r_filename==False:
        return redirect(url_for('home'))

    in_path = raw_upload_folder+r_filename
    in_im = ndimage.imread(in_path, mode='RGB').astype(float)
    x_ = in_im.shape[0]
    y_ = in_im.shape[1]
    ex_x = int(np.ceil(x_/5))
    ex_y = int(np.ceil(y_/5))

    if edge == "T":
        # extend the top
        image_holder = np.random.normal(size=[x_+ex_x,y_,3], scale = 0.01)
        big_not_mask = np.zeros([x_+ex_x,y_,3])
        image_holder[ex_x:,:,:] = in_im
        big_not_mask[ex_x+1:,:,:] = 1.0
        x1 = 0
        x2 = int(64.0/5)
        y1 = 0
        y2 = 64
        x_f = x_+ex_x
        y_f = y_
    elif edge == "L":
        # extend left edge
        image_holder = np.random.normal(size=[x_,y_+ex_y,3], scale = 0.01)
        image_holder[:,ex_y:,:] = in_im
        big_not_mask = np.zeros([x_,y_+ex_y,3])
        big_not_mask[:,ex_y+1:,:] = 1.0
        x1 = 0
        x2 = 64
        y1 = 0
        y2 = int(64/5)
        x_f = x_
        y_f = y_+ex_y
    elif edge == "R":
        # extend right edge
        image_holder = np.random.normal(size=[x_,y_+ex_y,3], scale = 0.01)
        image_holder[:,:y_,:] = in_im
        big_not_mask = np.zeros([x_,y_+ex_y,3])
        big_not_mask[:,:y_-1,:] = 1.0
        x1 = 0
        x2 = 64
        y1 = 64-int(64/5)
        y2 = 64
        x_f = x_
        y_f = y_+ex_y
    else:
        # extend the bottom
        image_holder = np.random.normal(size=[x_+ex_x,y_,3], scale = 0.01)
        image_holder[:x_,:,:] = in_im
        big_not_mask = np.zeros([x_+ex_x,y_,3])
        big_not_mask[:x_-1,:,:] = 1.0
        x1 = 64-int(64/5)
        x2 = 64
        y1 = 0
        y2 = 64
        x_f = x_+ex_x
        y_f = y_

    origin_path = masked_folder+r_filename
    misc.imsave(origin_path, image_holder)

    filled_path = filled_folder+"AI"+r_filename
    sub_script = "python /home/ubuntu/web_app_AWS/image_fill_deconv.py --image_path %s --out_path %s --mask_x1 %i --mask_x2 %i --mask_y1 %i --mask_y2 %i" %(origin_path,filled_path,x1,x2,y1,y2)
    print('subprocess')
    subprocess.call(sub_script, shell = True)

    big_mask = 1-big_not_mask

    masked_im = np.multiply(image_holder, big_not_mask)
    masked_path = masked_folder+"M"+r_filename
    misc.imsave(masked_path, masked_im)

    blended_path = filled_folder+"b"+r_filename
    target = np.asarray(Image.open(masked_path).resize((64,64),resample=Image.LANCZOS))
    source = np.asarray(Image.open(filled_path))
    mask = np.zeros((64,64))
    mask[x1:x2,y1:y2] = 1.0

    mask.flags.writeable = True
    target.flags.writeable = True
    source.flags.writeable = True

    small_blended = blend(target,source,mask,offset=(0,0))
    misc.imsave(blended_path,small_blended)
    big_blended = misc.imresize(small_blended,(x_f,y_f,3))
    complete_im = np.add(np.multiply(image_holder, big_not_mask),np.multiply(big_blended,big_mask))
    completed_path = completed_folder+r_filename
    misc.imsave(completed_path, complete_im)

    return render_template("display_completed_image.html", ds_image_path="../static/masked_images/"+"M"+r_filename,image_path="../static/completed_images/"+r_filename)


# set the secret key.
app.secret_key = os.urandom(24)

# function to apply poisson blending, from github.com/parosky
def blend(img_target, img_source, img_mask, offset=(0, 0)):
    # compute regions to be blended
    region_source = (
            0,
            0,
            64,
            64)
    region_target = (
            0,
            0,
            64,
            64)
    region_size = (64,64)#region_source[2]-region_source[0], region_source[3]-region_source[1])

    # clip and normalize mask image
    #img_mask = img_mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
    img_mask[img_mask==0] = False
    img_mask[img_mask!=False] = True

    # create coefficient matrix
    A = sparse.identity(np.prod(region_size), format='lil')

    for y in range(64):
        for x in range(64):
            if img_mask[y,x]:
                index = x+y*region_size[1]
                A[index, index] = 4
                if index+1 < np.prod(region_size):
                    A[index, index+1] = -1
                if index-1 >= 0:
                    A[index, index-1] = -1
                if index+region_size[1] < np.prod(region_size):
                    A[index, index+region_size[1]] = -1
                if index-region_size[1] >= 0:
                    A[index, index-region_size[1]] = -1
    A = A.tocsr()

    # create poisson matrix for b
    P = pyamg.gallery.poisson(img_mask.shape)

    # for each layer (ex. RGB)
    for num_layer in range(3):
        # get subimages
        t = img_target[:,:,num_layer]
        s = img_source[:, : ,num_layer]
        t = t.flatten()
        s = s.flatten()

        # create b
        b = P * s
        for y in range(region_size[0]):
            for x in range(region_size[1]):
                if not img_mask[y,x]:
                    index = x+y*region_size[1]
                    b[index] = t[index]
        print('here')
        # solve Ax = b
        x = pyamg.solve(A,b,verb=False,tol=1e-10)

        # assign x to target image
        x = np.reshape(x, region_size)
        x[x>255] = 255
        x[x<0] = 0
        x = np.array(x, img_target.dtype)
        img_target[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer] = x

    return img_target

