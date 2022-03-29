import numpy as np
import cv2
import plotly.graph_objects as go
from hist_eq import hist_equalize

img = cv2.imread('adaptive_hist_data-20220320T151318Z-001/adaptive_hist_data/0000000000.png')
#cv2.imshow('original_img', img)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cell_size = 200
print('cell_size:', cell_size)

for row in range(0, hsv_img.shape[0], cell_size):
    first_row_cell = row
    if (hsv_img.shape[0] - row <= cell_size) and (hsv_img.shape[0] - row != 1):
        last_row_cell = hsv_img.shape[0]
    else:
        last_row_cell = row + cell_size
    for col in range(0, hsv_img.shape[1], cell_size):
        first_col_cell = col
        if (hsv_img.shape[1] - col <= cell_size) and (hsv_img.shape[1] - col != 1):
            last_col_cell = hsv_img.shape[1]
        else:
            last_col_cell = col + cell_size
        mapping = hist_equalize(hsv_img[first_row_cell:last_row_cell, first_col_cell:last_col_cell, 2])
        for i in range(first_row_cell, last_row_cell):
            for j in range(first_col_cell, last_col_cell):
                hsv_img[i,j,2] = mapping[hsv_img[i,j,2]]

print('Standard Deviation of the HSV adaptive histogram equalized scene:', np.std(hsv_img[:,:,2]))
unique_vals, counts = np.unique(hsv_img[:,:,2], return_counts=True)
fig = go.Figure()
fig.add_trace(go.Scatter(x=unique_vals, y=counts))
fig.update_layout(title='adapt_hist_eq', title_x=0.5, xaxis_title='pixel value', yaxis_title='counts', font=dict(size=28))
fig.show()

bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
#cv2.imshow('hsv_adapt_eq', bgr_img)
cv2.imwrite('hsv_adapt_eq.jpg', bgr_img)
cv2.waitKey(0)

print('Adaptive histogram Equalized image is saved to current directory as: hsv_adapt_eq.jpg')