import sys
import argparse
import os
import shutil

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def main(argv=None):
    parser = argparse.ArgumentParser(description='Utility to prepare dataset for training')
    parser.add_argument('--in_dir', dest='in_dir', action='store') 
    parser.add_argument('--out_dir', dest='out_dir', action='store')
    parser.add_argument('--perc_train', type=float,dest='perc_train', action='store')

    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    perc_train = args.perc_train

    # creates folder tree
    create_dir(out_dir)
    create_dir(os.path.join(out_dir,'train/cats'))
    create_dir(os.path.join(out_dir,'train/dogs'))
    create_dir(os.path.join(out_dir,'val/cats'))
    create_dir(os.path.join(out_dir,'val/dogs'))
    create_dir(os.path.join(out_dir,'test/cats_and_dogs'))

    # split Kaggle training set into training and validation set
    train_dir = os.path.join(in_dir,'train')
    files = os.listdir(train_dir)
    for index,filename in enumerate(files):
        filename_split = filename.split('.')    
        if filename_split[0] == 'cat':
            if int(filename_split[1]) < len(files)/2*perc_train:
                shutil.copy2(os.path.join(train_dir,filename),os.path.join(out_dir,'train/cats'))
                print( 'copying ' + filename )
            else:
                shutil.copy2(os.path.join(train_dir,filename),os.path.join(out_dir,'val/cats'))
                print( 'copying ' + filename )
        else:
            if int(filename_split[1]) < len(files)/2*perc_train:
                shutil.copy2(os.path.join(train_dir,filename),os.path.join(out_dir,'train/dogs'))
                print( 'copying ' + filename )
            else:
                shutil.copy2(os.path.join(train_dir,filename),os.path.join(out_dir,'val/dogs'))
                print( 'copying ' + filename )

    # copies Kaggle test set into our dataset folder
    test_dir = os.path.join(in_dir,'test')
    files = os.listdir(test_dir)
    for index,filename in enumerate(files):
        shutil.copy2(os.path.join(test_dir,filename),os.path.join(out_dir,'test/cats_and_dogs'))
        print( 'copying ' + filename )


if __name__ == "__main__":
    main(sys.argv)


