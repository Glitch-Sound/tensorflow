import argparse 
import csv
import glob
import tensorflow as tf
import os

parser = argparse.ArgumentParser()
parser.add_argument('arg1') 
args = parser.parse_args()

root = args.arg1
root.rstrip('/')
root += '/'

dict_label = {}
with open('labels.csv') as f:
    reader = csv.reader(f)
    list_row = [row for row in reader]
    for target_row in list_row:
        dict_label[target_row[0]] = int(target_row[1].strip(' '))

list_dir = [os.path.basename(p.rstrip(os.sep)) for p
    in glob.glob(os.path.join(root, '*' + os.sep), recursive=True)]

with tf.io.TFRecordWriter('image.tfrecord') as w:
    for target_dir in list_dir:
        list_img = [i for i in glob.glob(root + target_dir + '/*.jpg')]
        for target_img in list_img:
            print(target_img)

            with tf.io.gfile.GFile(target_img, 'rb') as f:
                img_data = f.read()
                caption = target_img.encode('utf-8')

                features = tf.train.Features(feature={
                    'image_data' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_data])),
                    'caption'    : tf.train.Feature(bytes_list=tf.train.BytesList(value=[caption])),
                    'label'      : tf.train.Feature(int64_list=tf.train.Int64List(value=[dict_label[target_dir]]))
                })

                example = tf.train.Example(features=features)
                w.write(example.SerializeToString())
