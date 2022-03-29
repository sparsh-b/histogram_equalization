# Histogram Equalization
- Histogram Equalization is used to enhance the contrast of the image.
- The naive way of Histogram Equalizing the color images is to apply it on each of Red, Blue & Green channels separately.
- But it disturbs the color balance of the image as suggested by the wikipedia page. https://en.wikipedia.org/wiki/Histogram_equalization#Of_color_images 
- Therefore, separating the intensity information (by converting the image to HSV color space) from color imformation & applying histogram equalization is considered better.
- For the considered image, HSV equalization gave better contrast enhancement than RGB equalization.