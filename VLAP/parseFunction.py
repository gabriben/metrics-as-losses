import tensorflow as tf
import globals

def parseFunction(filename, label):
    """Function that returns a tuple of normalized image array and labels array.
    Args:
        filename: string representing path to image
        label: 0/1 one-dimensional array of size N_LABELS
    """
    # Read an image from a file
    image_string = tf.io.read_file(filename)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_jpeg(image_string, channels=globals.CHANNELS)
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [globals.IMG_SIZE, globals.IMG_SIZE])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0

    return image_normalized, label
