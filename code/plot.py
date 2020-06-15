import matplotlib.pyplot as plt

def minmax(x):
    return (x - x.min()) / (x.max() - x.min())

def plot_images(src_image, target_image, shifted_image, filename=None, cmap='Greys_r'):
    plt.figure(figsize=(16, 10))
    ax = plt.subplot(2, 3, 1)
    ax.imshow(src_image, cmap=cmap)
    ax.set_title('Source image')

    ax = plt.subplot(2, 3, 2)
    ax.imshow(target_image, cmap=cmap)
    ax.set_title('Target image')

    ax = plt.subplot(2, 3, 3)
    ax.imshow(minmax(src_image - target_image), cmap=cmap)
    ax.set_title('Difference (Src - Target)')

    ax = plt.subplot(2, 3, 4)
    ax.imshow(src_image, cmap=cmap)
    ax.set_title('Source image')

    ax = plt.subplot(2, 3, 5)
    ax.imshow(shifted_image, cmap=cmap)
    ax.set_title('Shifted image')

    ax = plt.subplot(2, 3, 6)
    ax.imshow(minmax(src_image - shifted_image), cmap=cmap)
    ax.set_title('Difference (Src - Shifted)')

    if filename != None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()
