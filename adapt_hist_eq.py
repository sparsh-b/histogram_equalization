import os
import glob
import numpy as np
import cv2
import plotly.graph_objects as go
from hist_eq import hist_equalize

cell_size = 200
print('cell_size used for Adaptive Histogram Equalization:', cell_size)

def adapt_hist_eq(img_path='adaptive_hist_data-20220320T151318Z-001/adaptive_hist_data/0000000000.png', flag=0):
    img = cv2.imread(img_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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

    bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    if flag==0:
        cv2.imwrite('adapt_hist_eq.jpg', bgr_img)
        print('Standard Deviation of the adaptive histogram equalized scene for 1st frame:', np.std(hsv_img[:,:,2]))
        unique_vals, counts = np.unique(hsv_img[:,:,2], return_counts=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=unique_vals, y=counts))
        fig.update_layout(title='adapt_hist_eq', title_x=0.5, xaxis_title='pixel value', yaxis_title='counts', font=dict(size=28))
        fig.show()
    else:
        if not os.path.exists(os.path.join('.','results','adapt_hist_eq')):
            os.makedirs(os.path.join('.','results','adapt_hist_eq'))
        cv2.imwrite(os.path.join('.','results','adapt_hist_eq',img_path.split('/')[-1]), bgr_img)

def adapt_hist_eq_all():
    imgs = glob.glob('adaptive_hist_data-20220320T151318Z-001/adaptive_hist_data/*.png')
    for img in imgs:
        adapt_hist_eq(img, 1)

if __name__ == '__main__':
    adapt_hist_eq()
    adapt_hist_eq_all()

    print('Adaptive histogram Equalized image for 1st frame is saved to current directory as: adapt_hist_eq.jpg')
    print('Intensities of Adaptive histogram equalized scene for 1st frame are displayed in browser!')
    print('\nAdaptive histogram Equalized images for the entire dataset is stored at `./results/adapt_hist_eq`!')