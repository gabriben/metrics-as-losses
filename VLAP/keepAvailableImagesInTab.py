import pandas as pd
import os

def keepAvailableImagesInTab(imageDir, labelDir):

    images = pd.DataFrame(os.listdir(imageDir), columns = ["imdbId"])
    images.replace(".jpg", "", regex = True, inplace = True)
    images = images[images['imdbId'].astype(str).str.isdigit()]

    movies = pd.read_csv(labelDir, encoding="ISO-8859-1")

    movies['imdbId'] = movies['imdbId'].astype('int')
    images['imdbId'] = images['imdbId'].astype('int')

    movies = pd.merge(movies, images, on = "imdbId", how = "inner")
    movies['imdbId'] = movies['imdbId'].astype('object')

    return movies
