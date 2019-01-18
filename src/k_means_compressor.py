import numpy as np
import pyopencl as cl

import image


def compress_image(img, num_centroids, iters):
    """compress_image compresses an image, given as an image.Image,
            using the K-Means clustering algorithm.
    
    Args:
        img_data: image.Image to be compressed
    
    Returns:
        New image.Image which has been compressed
    """
    # Get OpenCL context and queue
    context, queue = setup_opencl()
    mf = cl.mem_flags

    # Gather image data
    img_data = img.raw_data(image.ImageDataFormat.FLATTENED_NORMALIZED)
    img_dims = img.shape()

    # Create buffers
    imgBuffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img_data)
    
    centroids = np.random.random_sample((num_centroids * 4)).astype(np.float32)
    centroidsBuffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR, 
        hostbuf=centroids)

    indices = np.zeros((img_dims[0] * img_dims[1],)).astype(np.int32)
    indicesBuffer = cl.Buffer(context, cl.mem_flags.READ_WRITE, 
        size=indices.itemsize * img_dims[0] * img_dims[1])
    
    # Load and compile the kernel
    build_ops = ["-D NUM_CENTROIDS={0}".format(num_centroids), "-D IMG_WIDTH={0}".format(img_dims[1])]
    program = cl.Program(context, open('kernels/image_old.cl').read()).build(options=build_ops)
    
    # Get the kernel and set the arguments
    kernel = cl.Kernel(program, 'FindClosestCentroid')
    kernel.set_arg(0, imgBuffer)
    kernel.set_arg(1, centroidsBuffer)
    kernel.set_arg(2, indicesBuffer)

    for iter in range(iters):
        cl.enqueue_nd_range_kernel(queue, kernel, (img_dims[0], img_dims[1]), None)
        cl.enqueue_copy(queue, indices, indicesBuffer, is_blocking=True)
        
        indexCounts = [0] * num_centroids
        indexTotals = np.zeros((num_centroids, 3))
        for i in range(0, len(indices)):
            idx = indices[i]
            indexCounts[idx] += 1
            indexTotals[idx][0] += img_data[3*i]
            indexTotals[idx][1] += img_data[3*i + 1]
            indexTotals[idx][2] += img_data[3*i + 2]
        for i in range(num_centroids):
            count = indexCounts[i]
            if (count == 0):
                continue
            else:
                total = indexTotals[i]
                centroids[i*3] = total[0] / count
                centroids[i*3+1] = total[1] / count
                centroids[i*3+2] = total[2] / count
        cl.enqueue_copy(queue, centroidsBuffer, centroids, is_blocking=True)
    compressed_img = np.zeros(img_dims)
    for x in range(img_dims[1]):
        for y in range(img_dims[0]):
            img_idx = img_dims[1] * y + x
            centroids_idx = indices[img_idx]
            compressed_img[y][x][0] = int(centroids[3*centroids_idx] * 256)
            compressed_img[y][x][1] = int(centroids[3*centroids_idx+1] * 256)
            compressed_img[y][x][2] = int(centroids[3*centroids_idx+2] * 256)
    return compressed_img



def setup_opencl():
    """setup_opencl sets up the platform and devices to create a OpenCL context and queue
            to be used.
    Returns:
        OpenCL context and queue to be used
    """
    platforms = cl.get_platforms()
    platform = platforms[0]
    devices = platform.get_devices(cl.device_type.GPU)
    device = devices[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context, device)
    return context, queue