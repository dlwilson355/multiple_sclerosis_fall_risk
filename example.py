# a quick example showing the new method of reading data by getting segments of data which we will then use to perform classification on the patients

from data_reader import DataReader

MASTER_FILEPATH = "D:\\deep learning dataset\\MS Fall Study" # replace this with your filepath to the "MS Fall Study" directory
SEGMENT_SIZE = 40 # this variable represents the number of sequential data measurements that are part of each "segment" of x data
SEGMENTS_PER_PATIENT = 10 # this variable represents the number of segments to get from each patient

reader = DataReader(MASTER_FILEPATH)
data = reader.get_segmented_data(SEGMENT_SIZE, SEGMENTS_PER_PATIENT)

print(data[0])
print(data[1])