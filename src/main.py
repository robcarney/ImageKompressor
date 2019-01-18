import argparse

import image
import k_means_compressor as kmcomp


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', default='../images/bird_uncompressed.png', dest='img_file',
        help='File location for image to be compressed (default is ../images/bird_uncompressed.png)')
    parser.add_argument('--out', default='../images/compressed_img.png', dest='out_file',
        help='File location for compressed image to be stored (default is ../images/compressed_img.png)')
    parser.add_argument('--num-centroids', default=40, type=int, dest='num_centroids',
        help='The number of centroids to use in the K-Means clustering algorithm (default is 40)')
    parser.add_argument('--iters', default=100, type=int, dest='iters',
        help='The number of iterations to run the K-Means clustering algorithm (default is 100')
    args = parser.parse_args()
    img_file = args.img_file
    out_file = args.out_file
    num_centroids = args.num_centroids
    iters = args.iters

    img = image.Image(image_path='../images/bird_uncompressed.png')
    compressed_img = kmcomp.compress_image(img, num_centroids, iters)
    compressed_img.save(out_file)


if __name__ == '__main__':
    main()

