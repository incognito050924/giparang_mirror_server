import os

def rmdir_recursive(path):
    os.removedirs(path)

def rmdir(path):
    os.rmdir(path)

def mkdir_recursive(path):
    os.makedirs(path)

def mkdir(path):
    os.mkdir(path)

def rm(path):
    os.remove(path)

def is_exist(path):
    print(os.access(path, os.F_OK))

def is_readable(path):
    print(os.access(path, os.R_OK))

def is_writable(path):
    print(os.access(path, os.W_OK))

def is_executable(path):
    print(os.access(path, os.X_OK))

def list_dir_recursive(top_path, topdown=True):
    depth = 0
    for path, dirs, files in os.walk(top=top_path, topdown=topdown):
        depth += 1
        print('Depth: %d\nDirectory Count: %d,\tFile Count: %d' % (depth, len(dirs), len(files)))
        print('Path: ', path)
        print('Directories: ', dirs)
        print('Files: ', files, '\n')