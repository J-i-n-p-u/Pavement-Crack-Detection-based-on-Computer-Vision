import argparse
import os 
import shutil
from random import shuffle

def split_data(dataset_dir, train_ratio, test_ratio, output_dir_name):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir")
    parser.add_argument("--train",type=int)
    parser.add_argument("--test",type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    
    # python src/utils/split_dataset.py --dir= --train=80 --test=20 --output=data/datasets/cracks_splitted8020

    # args.dir = 'wsp_dataset/'
    # args.train = 80
    # args.test = 20
    # args.output = 'wsp_cracks_splitted8020'
    
    args.dir = dataset_dir
    args.train = train_ratio
    args.test = test_ratio
    args.output = output_dir_name
    
    print("Processing directory: %s" % args.dir)
    print("Splitting %d%% training set and %d%% test set" % (args.train, args.test)  )
    print("Generating output in directory: %s" % args.output  )

    try:

        # Create output directory
        os.mkdir(args.output)
        os.mkdir(args.output+"/train_set/")
        os.mkdir(args.output+"/test_set/")

        # Count classes
        classes_directories = os.listdir(args.dir) 
        print("Found %d classes." % len(classes_directories) )
        for d in classes_directories:
            os.mkdir(args.output+"/train_set/"+d)
            os.mkdir(args.output+"/test_set/"+d)
            path, dirs, files = next(os.walk(args.dir+"/"+d))
            file_count = len(files)
            print("Found %d files for class %s" % (file_count, d) ) 
            files_train_count = int(file_count * (args.train/100.0))
            files_test_count = int(file_count * (args.test/100.0))
            print("Using %d for training and %d for testing" % (files_train_count, files_test_count) ) 
            all_files = os.listdir(args.dir+"/"+d)
            shuffle(all_files)
            train_file_list = all_files[:files_train_count]
            test_file_list = all_files[files_train_count:]
            for f in train_file_list:
                shutil.copy(args.dir+"/"+d+"/"+f, args.output+"/train_set/"+d)
            for f in test_file_list:
                shutil.copy(args.dir+"/"+d+"/"+f, args.output+"/test_set/"+d)                
    except OSError as ex:
        print("Exception: %s. Aborting." % ex)
    
    
