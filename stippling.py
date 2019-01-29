'''
Created on 07/06/2015

@author: Alexandre Yukio Yamashita
'''
import os
import random
from multiprocessing.dummy import Pool as ThreadPool

from image import Image

from files import Files
from logger import Logger
from voronoi import Voronoi
import numpy as np


class Stippling:
    '''
    Compute stippling.
    '''
    _logger = Logger()
    image = None
    pixels_by_gray_level = None
    total_points = 0
    max_level = 256
    pixel_distribution = None
    points = None
    max_iterations = 0
    alpha = 2.0
    voronoi = None


    def __init__(self, image = None, total_points = None, max_iterations = None, alpha = None, path = None):
        if path is None:
            self.total_points = total_points
            self.max_iterations = max_iterations
            self.alpha = alpha
        else:
            self.load(path)

        self.stippled_image = Image(image = np.ones(image.data.shape) * (self.max_level - 1))
        self.image = image


    def generate(self):
        '''
        Generate stippled image.
        '''
        self._logger.log(Logger.INFO, "Generating stippled image.")
        self.voronoi = Voronoi(self.points, self.image.width, self.image.height)
        self.pre_compute()

        for index_iteration in range(self.max_iterations):
            self._logger.log(Logger.INFO, str(index_iteration) + " iteration: moving points to new positions.")
            self.plot("test/image_" + str(index_iteration) + ".png")
            self.move_points()


    def get_image(self):
        self.voronoi.compute_region_map(self.points)
        return self.voronoi.get_image()


    def pre_compute(self):
        '''
        Pre compute stippled image.
        '''
        self._logger.log(Logger.INFO, "Pre-computing stippled image.")
        total = 0
        self.points = []
        self.get_pixels_by_gray_level()
        self.compute_pixels_distribution()

        # Pre-compute stippled image.
        for index_level in range(self.max_level):
            index_pixel = 0
            total_pixels = int(self.pixel_distribution[index_level])
            random.shuffle(self.pixels_by_gray_level[index_level])

            # Set points for each level.
            for _ in range(total_pixels):
                y = self.pixels_by_gray_level[index_level][index_pixel][0]
                x = self.pixels_by_gray_level[index_level][index_pixel][1]
                self.points.append((y, x))
                total += 1
                index_pixel += 1


    def get_pixels_by_gray_level(self):
        '''
        Get pixels by gray level.
        '''
        self.pixels_by_gray_level = [[] for _ in range(self.max_level)]

        # Set pixels by gray level.
        for y in range(self.image.height):
            for x in range(self.image.width):
                self.pixels_by_gray_level[self.image.data[y][x]].append((y, x))

        return self.pixels_by_gray_level


    def compute_pixels_distribution(self):
        '''
        Compute pixels distribution.
        '''
        # Compute how much pixels goes to each level.
        self.pixel_distribution = np.zeros(self.max_level)

        index_level = 0
        total_points = self.total_points
        total_pixels = self.image.width * self.image.height

        if total_points > total_pixels:
            total_points = total_pixels

        # While there are points to distribute.
        while total_points > 0:
            # Calculate total of pixels to distribute in level.
            total_pixels_in_level = len(self.pixels_by_gray_level[index_level])
            distributed_pixels_in_level = total_pixels_in_level * 1.0 / total_pixels
            distributed_pixels_in_level *= total_points
            distributed_pixels_in_level *= (self.max_level - index_level) ** self.alpha * 1.0 / self.max_level ** self.alpha
            distributed_pixels_in_level = int(distributed_pixels_in_level)

            if distributed_pixels_in_level == 0:
                distributed_pixels_in_level = 1

            # Add pixels to level.
            self.pixel_distribution[index_level] += distributed_pixels_in_level

            if self.pixel_distribution[index_level] > total_pixels_in_level:
                distributed_pixels_in_level -= self.pixel_distribution[index_level] - total_pixels_in_level
                self.pixel_distribution[index_level] = total_pixels_in_level

            total_points -= distributed_pixels_in_level

            # Get next level.
            if index_level == 255:
                index_level = 0
            else:
                index_level += 1


    def plot(self, save_path = None):
        '''
        Plot stippled image.
        '''
        import matplotlib.pyplot as plt

        x = [point[1] for point in self.points]
        y = [self.image.height - point[0] for point in self.points]
        fig = plt.figure(frameon = False)
        plt.axis('off')
        ax = fig.add_subplot(1, 1, 1)
        plt.xlim(0, self.stippled_image.width)
        plt.ylim(0, self.stippled_image.height)
        plt.scatter(x, y, s = 1)
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        if save_path != None:
            plt.savefig(save_path, bbox_inches = extent)
            plt.clf()
            plt.close('all')
        else:
            plt.show()


    def move_points(self):
        '''
        Move points to new positions.
        '''
        self.voronoi.compute_region_map(self.points)

        self.x_total_density_x = np.zeros(self.total_points)
        self.y_total_density_y = np.zeros(self.total_points)
        self.sum_x = np.zeros(self.total_points)
        self.sum_y = np.zeros(self.total_points)
        self.total_density = np.zeros(self.total_points)
        density = (1 - self.image.data * 1.0 / self.max_level) ** self.alpha

        # Compute integrals to calculate new centroids.
        for y in range(self.image.height):
            for x in range(self.image.width):
                self.total_density[self.voronoi.region_map[y, x]] += density[y, x]
                self.x_total_density_x[self.voronoi.region_map[y, x]] += x * density[y, x]
                self.y_total_density_y[self.voronoi.region_map[y, x]] += y * density[y, x]
                self.sum_x[self.voronoi.region_map[y, x]] += x
                self.sum_y[self.voronoi.region_map[y, x]] += y


        # Compute new centroids.
        self.points = []

        for index_region in range(self.total_points):
            self.compute_new_centers(index_region)


    def compute_new_centers(self, index_region):
        '''
        Compute new centers.
        '''
        if self.total_density[index_region] > 0:
            x = int(self.x_total_density_x[index_region] / self.total_density[index_region] + 0.5)
            y = int(self.y_total_density_y[index_region] / self.total_density[index_region] + 0.5)
        elif self.sum_x[index_region] > 0 and self.sum_y[index_region] > 0:
            x = int(self.x_total_density_x[index_region] / self.sum_x[index_region] + 0.5)
            y = int(self.y_total_density_y[index_region] / self.sum_y[index_region] + 0.5)
        else:
            x = np.random.random() * self.image.width - 1
            y = np.random.random() * self.image.height - 1

        if x > self.image.width - 1:
            x = self.image.width - 1
        elif x < 0:
            x = 0

        if y > self.image.height - 1:
            y = self.image.height - 1
        elif y < 0:
            y = 0

        self.points.append([y, x])


    def save(self, path):
        '''
        Save stippling result.
        '''

        np.savez_compressed(path, points = np.array(self.points))


    def load(self, path):
        '''
        Load stippling result.
        '''

        compressed_file = np.load(path)
        self.points = compressed_file["points"].tolist()


def compute_stippling(path):
    features_path = path.replace("/pre_processed/", "/features/");
    features_path = os.path.splitext(features_path)[0] + ".npz"
    image = Image(path)
    image = image.resize(200, 200)
    image.convert_to_gray()

    # Compute stippled image and plot it.
    stippling = Stippling(image, 10000, 15, 2)
    stippling.generate()
    stippling.save(features_path)
    # stippling.plot()
    print features_path


def compute_stippling_of_images_in_folder(path):
    '''
    Compute stippling.
    '''

    image_paths = Files(path)
    for path in image_paths.paths:
        compute_stippling(path)


if __name__ == '__main__':
    image = Image("resources/lena.png")
    image = image.resize(200, 200)
    image.convert_to_gray()
    stippling = Stippling(image, 10000, 15, 2)
    stippling.generate()

