def keepAvailableImagesInTable(imageDir, labelDir):
    i = pd.DataFrame(os.listdir(imageDir), columns = ["imdbId"])
    i.replace(".jpg", "", regex = True, inplace = True)
    i = i[i['imdbId'].astype(str).str.isdigit()]

    movies = pd.read_csv(labelDir, encoding="ISO-8859-1")

    movies['imdbId'] = movies['imdbId'].astype('int')
    i['imdbId'] = i['imdbId'].astype('int')

    movies = pd.merge(movies, d, on = "imdbId", how = "inner")
    movies['imdbId'] = movies['imdbId'].astype('object')

    return movies
