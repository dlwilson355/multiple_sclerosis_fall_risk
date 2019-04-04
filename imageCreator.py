# this file generates images from the sequences of data

from data_reader import DataReader
from PIL import Image
import numpy as np

MASTER_FILEPATH = "D:\\deep learning dataset\\MS Fall Study" # replace this with the filepath to the "MS Fall Study" directory
SEGMENT_SIZE = 40 # the number of sequential data measurements that are part of each "segment" of x data
SEGMENTS_PER_PATIENT = 10 # the number of segments to get from each patient

reader = DataReader(MASTER_FILEPATH)
data = reader.get_segmented_data(SEGMENT_SIZE, SEGMENTS_PER_PATIENT)

width = 21
height = SEGMENT_SIZE
img = Image.fromarray(data[0][0], 'L')
img.show()
img.save('example.png')