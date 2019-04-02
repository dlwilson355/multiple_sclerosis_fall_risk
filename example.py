# a quick example showing the new method of reading data by getting segments of data which we will then use to perform classification on the patients

from data_reader import DataReader

reader = DataReader("D:\\deep learning dataset\\MS Fall Study")
data = reader.get_segmented_data(40, 10)

print(data[0])
print(data[1])