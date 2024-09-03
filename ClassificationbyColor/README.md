# Classification by Color

This approach aims to classify images into clusters based on their color characteristics:

The primary goal of both methods is to categorize images by analyzing their color features, though they differ in their specific techniques for clustering and comparison.


## -- colorCluster.ipynb --

This code processes images to classify them based on their average color into three clusters: red, green, and blue.

Loads images and converts them to RGB format.
Calculates the average color by sampling 100 random pixels from the image and computing the mean of their RGB values.
Clustering by Color by comparing results using Euclidean distance.
Outputs a dictionary that maps each cluster to the images that belong to it.

## -- colorKmeansCluster.ipynb --

This code uses K-Means clustering to group images based on the RGB values of sampled pixels from each image.

Loads images and converts them to RGB format.
Samples pixels by dividing the image into a grid and taking one pixel per grid cell.
Performs K-Means clustering on all the sampled pixels from all images to find three centroids (representing clusters).
Calculates the mean color of the sampled pixels and assigns the image to the closest cluster based on the centroids found by K-Means.
Outputs the RGB values of the centroids and shows which cluster each image is closest to.