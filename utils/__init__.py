import math


def get_grid_dims(shp, block_size):
    return math.ceil(shp[0] / block_size), math.ceil(shp[1] / block_size)


class KernelCaller:
    def __getitem__(self, item):
        kernel, image, block_size = item
        g_dims = get_grid_dims(image.shape, block_size)

        def call_kernel(*args):
            kernel[g_dims, (block_size, block_size)](*args)

        return call_kernel

call_kernel = KernelCaller()

# def call_kernel(image, kernel, block_size, *args):
#     g_dims = get_grid_dims(image.shape, block_size)
#     kernel[g_dims, (block_size, block_size)](*args)
