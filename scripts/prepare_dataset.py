import sys
import argparse
import os
import shutil

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def main(argv=None):
    parser = argparse.ArgumentParser(description='Utility to create list of images')
    parser.add_argument('--in_dir', dest='in_dir', action='store') 
    parser.add_argument('--out_dir', dest='out_dir', action='store')
    parser.add_argument('--perc_train', type=float,dest='perc_train', action='store')

    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    perc_train = args.perc_train

    # create folder tree
    create_dir(out_dir)
    create_dir(os.path.join(out_dir,'train/cats'))
    create_dir(os.path.join(out_dir,'train/dogs'))
    create_dir(os.path.join(out_dir,'val/cats'))
    create_dir(os.path.join(out_dir,'val/dogs'))
    create_dir(os.path.join(out_dir,'test/cats_and_dogs'))


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

    test_dir = os.path.join(in_dir,'test')
    files = os.listdir(test_dir)
    for index,filename in enumerate(files):
        shutil.copy2(os.path.join(test_dir,filename),os.path.join(out_dir,'test/cats_and_dogs'))
        print( 'copying ' + filename )


#    if not os.path.exists(os.path.join(out_path,'test/test')):
#        shutil.copytree(os.path.join(out_path,'test'),os.path.join(out_path,'test/test'))
#        print 'copying test folder'

#    path = os.path.dirname(direct)
#    basename = os.path.basename(path)
#    files = os.listdir(direct)
#    files.sort()
#    with open(os.path.join(out_path, basename+'.txt'), 'wt') as f:
#        for index, file in enumerate(files):
#            filename = file.split('.')[0]
#            print('   File: {}'.format(filename))
#            if index!=(len(files)-1):
#                f.write('{}\n'.format(filename))
#            else:
#                f.write('{}'.format(filename))
#
#    print('File saved as {}'.format(os.path.join(out_path, basename+'.txt')))

if __name__ == "__main__":
    main(sys.argv)


