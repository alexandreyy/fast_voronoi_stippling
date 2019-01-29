'''
Created on 05/04/2015

@author: Alexandre Yukio Yamashita
'''

from glob import glob
import os


class Files:
    '''
    List of file paths.
    '''
    def __init__(self, path, all = False):
        if not all:
            self.paths = glob(path + '*.jpeg')
            self.paths.extend(glob(path + '*.png'))
            self.paths.extend(glob(path + '*.bmp'))
            self.paths.extend(glob(path + '*.jpg'))
            self.paths.extend(glob(path + '*.gif'))
            self.paths.extend(glob(path + '*.pgm'))
            self.paths.extend(glob(path + '*.PGM'))
            self.paths.extend(glob(path + '*.JPG'))
            self.paths.extend(glob(path + '*.JPEG'))
            self.paths.extend(glob(path + '*.PNG'))
            self.paths.extend(glob(path + '*.BMP'))
            self.paths.extend(glob(path + '*.GIF'))
        else:
            self.paths = glob(path + '*')
            
    def remove(self):
        '''
        Remove all files.
        '''
        for f in self.paths:
            print f
            os.remove(f)