# my_tfrecords
存储为tfrecords和读取tfrecords


运行方法：

python create_tfrecord.py --dataset_dir='/dataset' --num_shards=2 --tfrecord_filename='name'

#Example: python create_tfrecord.py --dataset_dir=/path/to/flowers --tfrecord_filename=flowers

文件夹的格式：

dataset\
      ...jpg
      ...jpg
      ...jpg
        ....jpg
            ....jpg
            ....jpg
      
      
    
