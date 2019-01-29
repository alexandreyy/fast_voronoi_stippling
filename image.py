'''
Created on 05/04/2015

@author: Alexandre Yukio Yamashita
'''

from argparse import ArgumentParser
import cv2
import os
import numpy as np
from logger import Logger


class Image:
    '''
    Read image in rgb. 
    '''
    _logger = Logger()

    data = None  # The image data.

    def __init__(self, file_path = None, image = None):
        # Create image from matrix.
        if image is not None:
            self._set_image_data(image)

        # Load image if user specified file path.
        elif file_path is not None:
            # Check if file exist.
            if os.path.isfile(file_path):
                # File exists.
                file_path = os.path.abspath(file_path)

                # Load image
                self._logger.log(Logger.INFO, "Loading image " + file_path)
                image_data = cv2.imread(file_path)
                self.file_path = file_path

                # If image is in bgr, convert it to rgb.
                if len(image_data.shape) == 3:
                    # Image is in bgr.

                    # Convert image to rgb.
                    self._logger.log(Logger.INFO, "Converting image to RGB.")
                    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

                self._set_image_data(image_data)
            else:
                # File does not exist.
                self._logger.log(Logger.ERROR, "File '" + file_path + "' does not exist.")


    def _set_image_data(self, data):
        '''
        Set image data.
        '''
        # Check if image is in rgb or gray scale
        if len(data.shape) == 3:
            # Image is in rgb.
            self.height, self.width, self.channels = data.shape
        else:
            # Image is in gray scale.
            self.height, self.width = data.shape
            self.channels = 1

        self.data = data


    def equalize(self, image = None):
        '''
        Equalize image.
        '''
        if image is None and self.data is None:
            self._logger.log(Logger.ERROR, "There is no data to equalize image.")
        elif image is not None:
            self._set_image_data(image)

        if self.data is not None:
            # Convert to gray scale if it image is in rgb.
            if len(self.data.shape) == 3:
                self._logger.log(Logger.DEBUG, "We need to convert image to gray scale before equalization.")
                self.convert_to_gray(self.data)

            # Equalize image.
            self._logger.log(Logger.INFO, "Equalizing image.")
            self.data = cv2.equalizeHist(self.data)

        return self.data


    def binarize(self, image = None):
        '''
        Binarize image.
        '''
        if image is None and self.data is None:
            self._logger.log(Logger.ERROR, "There is no data to binarize image.")
        elif image is not None:
            self._set_image_data(image)

        if self.data is not None:
            # Convert to gray scale if it image is in rgb.
            if len(self.data.shape) == 3:
                self._logger.log(Logger.DEBUG, "We need to convert image to gray scale before binarizing image.")
                self.convert_to_gray(self.data)

            # Equalize image.
            self._logger.log(Logger.INFO, "Binarizing image.")
            self.data = cv2.medianBlur(self.data, 5)
            self.data = cv2.adaptiveThreshold(self.data, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
            cv2.THRESH_BINARY, 11, 2)

        return self.data


    def convert_to_gray(self, image = None):
        '''
        Convert rgb to gray.
        '''
        if image is None and self.data is None:
            self._logger.log(Logger.ERROR, "There is no data to convert image.")
        elif image is not None:
            self._set_image_data(image)

        if self.data is not None:
            # Convert image only if it is in rgb.
            if len(self.data.shape) == 3:
                # self._logger.log(Logger.INFO, "Converting image to gray scale.")
                self.data = cv2.cvtColor(self.data, cv2.COLOR_RGB2GRAY)
            else:
                self._logger.log(Logger.INFO, "Image is already in gray scale.")

            self.channels = 1

        return self.data


    def plot(self, image = None):
        '''
        Plot image.
        '''
        if image is None and self.data is None:
            self._logger.log(Logger.ERROR, "There is no data to plot image.")
        elif image is not None:
            self._set_image_data(image)

        # Display image if we have data for it.
        if self.data is not None:
            # Convert image to BGR, if it is in rgb.
            if self.channels == 3:
                image = cv2.cvtColor(self.data, cv2.COLOR_RGB2BGR)
            else:
                image = self.data

            # Plot image.
            self._logger.log(Logger.INFO, "Plotting image.")
            cv2.imshow("Image", image)
            cv2.waitKey()


    def resize(self, width, height, image = None):
        '''
        Resize image.
        '''
        if image is None and self.data is None:
            self._logger.log(Logger.ERROR, "There is no data to resize image.")
        elif image is not None:
            self._set_image_data(image)

        self._logger.log(Logger.INFO, "Resizing image to: width = " + str(width) + " height = " + str(height))
        resized = cv2.resize(self.data, (width, height), interpolation = cv2.INTER_AREA)

        return Image(image = resized)


    def crop(self, origin, end, image = None):
        '''
        Crop image.
        '''
        if image is None and self.data is None:
            self._logger.log(Logger.ERROR, "There is no data to crop image.")
        elif image is not None:
            self._set_image_data(image)

        if self.data is not None:
            # Correct parameters.
            if origin.x >= self.width:
                origin.x = self.width - 1
            elif origin.x < 0:
                origin.x = 0

            if end.x >= self.width:
                end.x = self.width - 1
            elif end.x < 0:
                end.x = 0

            if origin.y >= self.height:
                origin.y = self.height - 1
            elif origin.y < 0:
                origin.y = 0

            if end.y >= self.height:
                end.y = self.height - 1
            elif end.y < 0:
                end.y = 0

            if origin.x > end.x:
                change = end.x
                end.x = origin.x
                origin.x = change

            if origin.y > end.y:
                change = end.y
                end.y = origin.y
                origin.y = change

            self._logger.log(Logger.INFO, "Cropping image. Origin: (%d, %d) End: (%d, %d)" \
                % (origin.x, origin.y, end.x, end.y))
            return Image(image = self.data[origin.y:end.y, origin.x:end.x])


    def save(self, file_path = None, image = None):
        '''
        Save image in file path.
        '''
        if file_path is None and self.file_path is None:
            self._logger.log(Logger.ERROR, "There is no file path to save image.")
        elif file_path is not None:
            self.file_path = file_path

        if image is None and self.data is None:
            self._logger.log(Logger.ERROR, "There is no data to save image.")
        elif image is not None:
            self._set_image_data(image)

        if self.file_path is not None and self.data is not None:
            image = self.data

            if len(self.data.shape) == 3:
                image = cv2.cvtColor(self.data, cv2.COLOR_RGB2BGR)

            self._logger.log(Logger.INFO, "Saving image in " + self.file_path)
            cv2.imwrite(self.file_path, image);


if __name__ == '__main__':
    # Parses args.
    parser = ArgumentParser(description = 'Load and plot image.')
    parser.add_argument('file_path', help = 'image file path')
    args = vars(parser.parse_args())

    # Load and plot image.
    image = Image(args["file_path"])
    image.plot()
