import pyspark


spark = pyspark.sql.SparkSession \
        .builder \
        .appName("Image Preprocessing Pipeline") \
        .getOrCreate()


def load_images(filenames_pattern, train_size=0.):
    """
    Using Spark new built-in data source for images,
    we want to load data into Dataframes in order to pass
    them in a computing statistics pipeline.

    Args:
    filenames_pattern : A string representing path pattern of each image
    train_size : float number representing the train size 
    (use Dataframe.split([train_size, 1 - train_size]))
    """
    image_df = spark.read.format("image").load(filenames_pattern)
    return image_df

def per_pixel_mean(dataframe):
    """
    Compute the mean of each pixel over the entire dataset.

    """
    pass

def per_pixel_stddev(dataframe):
    """
    Compute the stddev of each pixel over the entire dataset.

    """
    pass

def per_channel_mean(dataframe):
    """
    Compute the mean of each channelf for every image.

    """
    pass

def per_channel_stddev(dataframe):
    """
    Compute the stddev of each channel for every image.

    """
    pass


a = load_images("D:/MURA-v1.1/train/*/*/*/*.png") 