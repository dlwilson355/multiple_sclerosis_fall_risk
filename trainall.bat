python network.py -f "F:\MS Fall Study"  -t Simple -1 1792 -2 28 -e 60 -h 100 > report_simple.txt
python network.py -f "F:\MS Fall Study"  -t VGG16 -1 224 -2 224 -e 60 > report_vgg16.txt
python network.py -f "F:\MS Fall Study"  -t GRU -1 1792 -2 28 -e 60 -h 100 > report_gru.txt
python network.py -f "F:\MS Fall Study"  -t VGGBN -1 224 -2 224 -e 60 > report_vggbn.txt
python network.py -f "F:\MS Fall Study"  -t LSTM -1 1792 -2 28 -e 60 -h 100 > report_lstm.txt
python network.py -f "F:\MS Fall Study"  -t ResNet -1 224 -2 224 -e 60 > report_resnet.txt

