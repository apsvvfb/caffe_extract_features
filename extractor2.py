import numpy as np
import matplotlib.pyplot as plt
import time
import caffe
import os 
import sys

caffe_root = '../../'  
model_def = caffe_root + 'examples/_temp/Bottom_up_13k_deploy.prototxt'
model_weights = caffe_root + 'examples/_temp/bvlc_googlenet_bottomup_12988_trainval.caffemodel'
label_name='prob'

inputpath=sys.argv[1]
outpath=sys.argv[2]
log=sys.argv[3]
suffix=".png"


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

net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          224, 224)  # image size is 224x224

t0 = time.time()
for root, videopath, files in os.walk(inputpath):
   for videos in videopath:
        videopath=os.path.join(videos)
        outfile=outpath + '/' + videopath 
	if not os.path.exists(outfile):
            os.mknod(outfile)   
        with open(outfile, 'w') as writer:
            writer.truncate()
	    for subroot, subdirs, framenames in os.walk(os.path.join(root,videopath)):
		for frame in framenames:
		    if os.path.splitext(frame)[1] == suffix:
		       framepath=os.path.join(subroot,frame)
	               image = caffe.io.load_image(framepath)			
		       transformed_image = transformer.preprocess('data', image)
		       net.blobs['data'].data[0,...] = transformed_image
		       output = net.forward()		
		       feat = net.blobs[label_name].data[0].reshape(1,-1)
		       np.savetxt(writer, feat, fmt='%.8g')
t1=time.time() - t0

logfile=os.path.join(outpath,log)
if not os.path.exists(logfile):
      os.mknod(logfile)   
with open(logfile, 'w') as f:
      f.write(str(t1)+'seconds wall time')
