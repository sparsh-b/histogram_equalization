import glob
import os
import numpy as np
import cv2
import plotly.graph_objects as go

def hist_equalize(v):
    unique_vals, counts = np.unique(v, return_counts=True)
    counts = counts/np.sum(counts)
    vals_counts = dict(zip(unique_vals, counts))

    cdf = {}
    for i in range(256):
        if i in vals_counts:
            if i==0:
                cdf[i] = vals_counts[i]
                continue
            cdf[i] = cdf[i-1] + vals_counts[i]
        else:
            if i==0:
                cdf[i] = 0
                continue
            cdf[i] = cdf[i-1]
    cdf_x = np.array([i for i in cdf.keys()])
    cdf_y = np.array([i for i in cdf.values()])

    histEq_x = cdf_x
    histEq_y = (cdf_x*cdf_y)
    histEq_y = np.around(histEq_y, 0).astype(int)
    mapping = dict(zip(histEq_x, histEq_y))
    return mapping

def hsv_hist_equalize(img_path='adaptive_hist_data-20220320T151318Z-001/adaptive_hist_data/0000000000.png', flag=0):
    img = cv2.imread(img_path)
    cv2.imshow('original_img', img)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv_img[:,:,2]
    if flag==0:
        print('Standard Deviation of the original scene for 0th frame:', np.std(v))
        unique_vals, counts = np.unique(v, return_counts=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=unique_vals, y=counts))
        fig.update_layout(title='original_distn', title_x=0.5, xaxis_title='pixel value', yaxis_title='counts', font=dict(size=28))
        fig.show()

    mapping = hist_equalize(v)

    for row in range(hsv_img.shape[0]):
        for col in range(hsv_img.shape[1]):
            hsv_img[row, col, 2] = mapping[hsv_img[row, col, 2]]
    bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)    

    if flag==0:
        cv2.imwrite('hsv_equalization.jpg', bgr_img)
        print('Standard Deviation of the HSV equalized scene for 0th frame:', np.std(hsv_img[:,:,2]))
        unique_vals, counts = np.unique(hsv_img[:,:,2], return_counts=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=unique_vals, y=counts))
        fig.update_layout(title='hist_equalization', title_x=0.5, xaxis_title='pixel value', yaxis_title='counts', font=dict(size=28))
        fig.show()
    else:
        if not os.path.exists(os.path.join('.','results','hist_eq')):
            os.makedirs(os.path.join('.','results','hist_eq'))
        cv2.imwrite(os.path.join('.','results','hist_eq',img_path.split('/')[-1]), bgr_img)
    

def rgb_hist_equalize():
    img = cv2.imread('adaptive_hist_data-20220320T151318Z-001/adaptive_hist_data/0000000000.png')
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    mapping_b = hist_equalize(b)
    mapping_g = hist_equalize(g)
    mapping_r = hist_equalize(r)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            img[row, col, 0] = mapping_b[img[row, col, 0]]
            img[row, col, 1] = mapping_g[img[row, col, 1]]
            img[row, col, 2] = mapping_r[img[row, col, 2]]
    #cv2.imshow('bgr_equalization', img)
    cv2.imwrite('bgr_equalization.jpg', img)
    hsv_img_rgb_eq = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_rgb_eq = hsv_img_rgb_eq[:,:,2]
    print('Standard Deviation of the RGB equalized scene for 0th frame:', np.std(v_rgb_eq))
    unique_vals, counts = np.unique(v_rgb_eq, return_counts=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=unique_vals, y=counts))
    fig.update_layout(title='rgb_equalization', title_x=0.5, xaxis_title='pixel value', yaxis_title='counts', font=dict(size=28))
    fig.show()
    #cv2.waitKey(0)

def hist_equalize_all():
    imgs = glob.glob('adaptive_hist_data-20220320T151318Z-001/adaptive_hist_data/*.png')
    for img in imgs:
        hsv_hist_equalize(img, 1)

if __name__ == '__main__':
    hsv_hist_equalize()
    rgb_hist_equalize()
    hist_equalize_all()
    print('\nrgb vs hsv equalized outputs for 1st frame are written to current directory!')
    print('Intensities of original, rgb & hsv equalized scenes for 1st frame are displayed in browser!')
    print('Results for the entire datset are written to `./results/hist_eq`!')