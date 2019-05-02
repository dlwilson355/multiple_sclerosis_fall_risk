import sys, getopt
import numpy as np
import csv

def main(argv):
    filename = None
    try:
        opts, args = getopt.getopt(argv,"f:")
    except getopt.GetoptError:
        return
    for opt, arg in opts:
        if opt == '-f':
            filename = arg
    if None == filename:
        return
    text_file = open(filename, "r") 
    lines = text_file.readlines()
    data = []
    for line in lines:
        if "val_loss" in line:
            s = line.find('val_loss:') + 10
            t = line[s:].split('- val_acc:')
            val_loss = float(t[0])
            s = line.find('val_acc:') + 9
            t = line[s:].split('\n')
            val_acc = float(t[0])
            data.append((val_loss, val_acc))

    csv_name = filename.split('.')[0] + '.csv'
    csv_file = open(csv_name,'w+')
    csv_file.write('epoch')
    csv_file.write(',')
    csv_file.write('val_loss')
    csv_file.write(',')
    csv_file.write('val_acc')
    csv_file.write('\n')
    csv_file.flush()
    for i, d in enumerate(data):
        csv_file.write(str(i))
        csv_file.write(',')
        csv_file.write(str(d[0]))
        csv_file.write(',')
        csv_file.write(str(d[1]))
        csv_file.write('\n')
    csv_file.flush()
    print('output',csv_name)


if __name__ == "__main__":  
    main(sys.argv[1:])

