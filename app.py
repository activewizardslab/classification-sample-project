# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os.path
import re
import sys
import tarfile
import os
import requests
import shutil
import datetime
from collections import OrderedDict
import numpy as np
from six.moves import urllib

import tensorflow as tf

from flask import Flask
from flask import render_template, request, jsonify
from werkzeug.utils import secure_filename
###############################################################################################


images_name = range(1, 11)
error_msg = "The file was not selected"
ALLOWED_EXTENSIONS = ['jpg']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join('static', 'uploads')


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

###############################################################################################
# Remove all images uploaded 24 hours ago 
dir_to_search = os.path.join(BASE_DIR, UPLOAD_FOLDER)
for dirpath, dirnames, filenames in os.walk(dir_to_search):
   for file in filenames:
      curpath = os.path.join(dirpath, file)
      file_modified = datetime.datetime.fromtimestamp(os.path.getmtime(curpath))
      if datetime.datetime.now() - file_modified > datetime.timedelta(hours=24):
          os.remove(curpath)

# Create /static/uploads folder if it does not exist
if not os.path.exists(dir_to_search):
    os.makedirs(dir_to_search)

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



###############################################################################################
########################################  TensorFlow  #########################################
#################################### IMAGE CLASSIFICATION  ####################################
###############################################################################################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
	'model_dir', os.path.join(BASE_DIR, "imagenet"), 
	"""Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt."""
)
tf.app.flags.DEFINE_string('image_file', '', """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 10, """Display this many predictions.""")

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""
    def __init__(self, label_lookup_path=None, uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = os.path.join(FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """Loads a human readable English name for each softmax node.
	    Args:
	      label_lookup_path: string UID to integer node ID.
	      uid_lookup_path: string UID to human-readable string.
	    Returns:
	      dict from integer node ID to human-readable string.
	    """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)
        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string
        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]
        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name
        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    with tf.gfile.FastGFile(os.path.join(FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def run_inference_on_image(image):
    """Runs inference on an image. """
    outputs = {}
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = tf.gfile.FastGFile(image, 'rb').read()
    # Creates graph from saved GraphDef.
    create_graph()
    with tf.Session() as sess:
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across 1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048 float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)
        # Creates node ID --> English string lookup.
        node_lookup = NodeLookup()
        top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            outputs[human_string] = predictions[node_id]
        labels = list(OrderedDict(sorted(outputs.items(), key=lambda t: t[1], reverse=True)).keys())
        scores = list(OrderedDict(sorted(outputs.items(), key=lambda t: t[1], reverse=True)).values())
        scores = list(map(float, scores))
        scores = [round(i, 3) for i in scores]
        for score in scores:
          if score < 0.005:
              scores.pop()
              labels.pop()
    return labels, scores


def maybe_download_and_extract():
    """Download and extract model tar file."""
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        statinfo = os.stat(filepath)
        print('\nSuccesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def main(_):
    maybe_download_and_extract()
    app.run(port=8080, debug=True)


###############################################################################################
########################################  TensorFlow  #########################################
############################### HANDWRITTEN DIGIT CLASSIFICATION  #############################
###############################################################################################
# Multilayer Convolutional Network
def model(x, keep_prob):
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # First Convolutional Layer
    x_image = tf.reshape(x, [-1,28,28,1])
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second Convolutional Layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Densely Connected Layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout Layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    return y, [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]

sys.path.append('mnist')
x = tf.placeholder("float", [None, 784])
sess = tf.Session()

with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2, variables = model(x, keep_prob)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/convolutional.ckpt")

def convolutional(input):
    return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()


###############################################################################################
###########################################  VIEWS  ###########################################
###############################################################################################
@app.route("/")
@app.route("/image")
def hello():
	return render_template("image.html", images=images_name)

@app.route("/digit")
def digit():
	return render_template("digit.html")


@app.route("/upload", methods = ['GET', 'POST'])
def upload():
	if request.method == 'POST':
		if 'file' not in request.files:
			return jsonify(error=error_msg)
		file = request.files['file']
		if file.filename == '':
			return jsonify(error=error_msg)
		if not allowed_file(file.filename):
			return jsonify(error="Not supported format of the uploaded file. Please upload one of the following: " + ", ".join(ALLOWED_EXTENSIONS))
		if file:
			filename = secure_filename(file.filename)
			file.save(os.path.join(BASE_DIR, app.config['UPLOAD_FOLDER'], filename))
			labels, scores = run_inference_on_image(os.path.join(BASE_DIR, app.config['UPLOAD_FOLDER'], filename))
			return jsonify(filename=filename, labels=labels, scores=scores)
	else:
		return jsonify(error=error_msg)

@app.route("/link", methods = ['GET', 'POST'])
def link():
	if request.method == 'POST':
		url = request.json['url']
		response = requests.get(url, stream=True)
		url = "".join(list(filter(lambda x: x.isalnum() or x == ".", os.path.basename(url))))
		filename = os.path.join(BASE_DIR, app.config['UPLOAD_FOLDER'], url)
		if '.jpg' not in filename:
			return jsonify(error="Not supported format of the uploaded file. Please upload one of the following: " + ", ".join(ALLOWED_EXTENSIONS))
		with open(filename, 'wb') as out_file:
			shutil.copyfileobj(response.raw, out_file)
		labels, scores = run_inference_on_image(filename)
		return jsonify(filename=os.path.join(app.config['UPLOAD_FOLDER'], url), labels=labels, scores=scores)

@app.route("/test", methods = ['GET', 'POST'])
def test():
	img = os.path.join(BASE_DIR, "static", "images", request.args.get('img'))
	labels, scores = run_inference_on_image(img)
	return jsonify(filename=img, labels=labels, scores=scores)

@app.route("/mnist", methods = ['GET', 'POST'])
def mnist():
	input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
	if len(input[input != 0.0]) == 0: 
		return jsonify(output="Write a digit at first!")
	return jsonify(output=list(map(lambda x: round(x*100, 3), convolutional(input))))

if __name__ == "__main__":
    tf.app.run()
