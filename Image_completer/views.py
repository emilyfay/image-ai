from flask import render_template, request,redirect,url_for,send_from_directory, jsonify
from Image_completer import app
import werkzeug
import os.path
import shutil
import subprocess

landing_upload_folder = "Image_completer/static/upload_landing"
raw_upload_folder = "Image_completer/raw_uploads"
proc_upload_folder = "Image_completer/proc_uploads"

# TODO: check to make sure all these can work
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

@app.route('/image')
def display_image():
    return render_template("display_image.html")


# dropzone activates this
@app.route('/flask-upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = werkzeug.secure_filename(file.filename)
            file.save(os.path.join(landing_upload_folder, filename))
            return redirect(url_for('file_upload', filename = filename))
    return

@app.route('/file_upload')
def file_upload():
    filename = request.args.get("filename")
    base, ext = os.path.splitext(filename)
    print('hello! \n')
    orig_path = os.path.join(landing_upload_folder, filename)

    # Move file out of landing zone
    raw_path = os.path.join(raw_upload_folder, "raw"+filename)

    shutil.copy(orig_path, raw_path)

    # Downsample and crop
    #
    # Might want to resize rather than crop so that I actually have
    # all the pixels contributing.
    # Force image format to png.
    new_ext = '.png'
    new_filename = "raw" + base + new_ext
    new_path = os.path.join(proc_upload_folder, new_filename)
    # TODO: change this when I figure out how to set area to extend
   # downsize = ['convert', # use convert not mogrify to not overwrite orig
    #       raw_path, # input fn
     #      '-resize', '128x128^', # ^ => min size
      #     '-gravity', 'center',  # do crop in center of photo
       #    '-crop', '150x150+0+0', # crop to 150x150 square
        #   '-auto-orient', # orient the photo
         #  new_path] # output fn
    #subprocess.call(downsize)

    # now work with image in new_path to run analysis
    # run completion model
    # return / generate new image

    return jsonify(dict(filename=new_filename))