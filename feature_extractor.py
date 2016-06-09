#from http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb
import numpy as np
import matplotlib.pyplot as plt
import time
import caffe

caffe_root = '../../'  
model_def = caffe_root + 'examples/_temp/Bottom_up_13k_deploy.prototxt'
model_weights = caffe_root + 'examples/_temp/bvlc_googlenet_bottomup_12988_trainval.caffemodel'
label_name='prob'
inputfile='temp.txt';
outputfile='output.txt'
batchsize=10
j=-1

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

import sys
sys.path.insert(0, caffe_root + 'python')

caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(10,        # batch size
                          3,         # 3-channel (BGR) images
                          224, 224)  # image size is 227x227

t0 = time.time()
with open(inputfile, 'r') as reader:
        with open(outputfile, 'w') as writer:
            writer.truncate()
	    for image_path in reader:
		j=j+1
		i=j%10
		image_path = image_path.strip()
	        image = caffe.io.load_image(image_path)			
		transformed_image = transformer.preprocess('data', image)
		net.blobs['data'].data[i,...] = transformed_image
		if (j>1 and i==batchsize-1):
			output = net.forward()		
			for m in range(0,batchsize):
				output_prob = output['prob'][m]  # the output probability vector for the first image in the batch
				print 'predicted class is:', output_prob.argmax()
				#feat = net.blobs[label_name].data[m,:5].reshape(1,-1)
				feat = net.blobs[label_name].data[m].reshape(1,-1)
				np.savetxt(writer, feat, fmt='%.8g')
print time.time() - t0, "seconds wall time"
