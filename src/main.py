import argparse

import image
import k_means_compressor as kmcomp


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', default='../images/bird_uncompressed.png', dest='img_file',
        help='File location for image to be compressed (default is ../images/bird_uncompressed.png)')
    parser.add_argument('--num-centroids', default=20, type=int, dest='num_centroids',
        help='The number of centroids to use in the K-Means clustering algorithm (default is 20)')
    parser.add_argument('--iters', default=50, type=int, dest='iters',
        help='The number of iterations to run the K-Means clustering algorithm (default is 50')
    args = parser.parse_args()
    img_file = args.img_file
    num_centroids = args.num_centroids
    iters = args.iters

    img = image.Image('../images/bird_uncompressed.png')
    compressed_img = kmcomp.compress_image(img, num_centroids, iters)


if __name__ == '__main__':
    main()

