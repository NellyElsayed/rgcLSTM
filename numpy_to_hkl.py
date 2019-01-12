
import numpy as np
import hickle as hkl
import matplotlib as plt
from sklearn import preprocessing as p
mnist_size=10000
LENGTH_OF_VID =20*4000#20*50#20*10000##50#20*10000#20*mnist_size # For later experiments, modify size as necessary20*1000
IM_SZ_WID = 64
IM_SZ_HGT=64
VIDEO_CHANNELS = 1


train_video=np.load("mnist_test_seq.npy")


def video_generator():
    in_video = np.empty([LENGTH_OF_VID  , IM_SZ_HGT, IM_SZ_WID, VIDEO_CHANNELS], dtype=np.float32)#LENGTH_OF_VID was 1st attribute
  #  in_video[:,:,:,:] = 1 # set background of 1st channel to white
    
  #  train_video = np.zeros((n_frames, row, col, 1), dtype=np.float)
    
  #  for i in range(0,LENGTH_OF_VID):
    count =0
    for i in range (6001, 4000+6000):#):#10000):
           gray_frame= train_video[:,i,:,:]
         #  gray_frame = np.multiply((1/255.0), gray_frame)
           gray_frame = np.expand_dims(gray_frame, axis=3)
           in_video[0+count:count+20,:,:,:]= np.copy(gray_frame)
           count=count+20
    return in_video
 

def save_as_hickle():
    in_video = video_generator()
    num_frames = in_video.shape[0]
    print(in_video.shape)
    source_string = ["mnist"]*num_frames
    # dump data to file
    print( in_video.shape)
    hkl.dump(in_video, 'kitti_data/4000_mnist_data1.hkl', mode='w')
    # dump names to file
    hkl.dump(source_string, 'kitti_data/4000_mnist_sources1.hkl', mode='w')
    
    
save_as_hickle()