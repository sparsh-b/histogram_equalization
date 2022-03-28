import numpy as np
import cv2
import plotly.graph_objects as go

img = cv2.imread('adaptive_hist_data-20220320T151318Z-001/adaptive_hist_data/0000000000.png')
cv2.imshow('original_img', img)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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

v = hsv_img[:,:,2]
mapping = hist_equalize(v)

for row in range(hsv_img.shape[0]):
    for col in range(hsv_img.shape[1]):
        hsv_img[row, col, 2] = mapping[hsv_img[row, col, 2]]
unique_vals, counts = np.unique(hsv_img[:,:,2], return_counts=True)
bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
fig = go.Figure()
fig.add_trace(go.Scatter(x=unique_vals, y=counts))
fig.update_layout(title='hsv_equalization', title_x=0.5)
fig.show()
cv2.imshow('hsv_equalization', bgr_img)
cv2.imwrite('hsv_equalization.jpg', bgr_img)

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
cv2.imshow('bgr_equalization', img)
cv2.imwrite('bgr_equalization.jpg', img)
hsv_img_rgb_eq = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
v_rgb_eq = hsv_img_rgb_eq[:,:,2]
unique_vals, counts = np.unique(v_rgb_eq, return_counts=True)
fig = go.Figure()
fig.add_trace(go.Scatter(x=unique_vals, y=counts))
fig.update_layout(title='rgb_equalization', title_x=0.5)
fig.show()

cv2.waitKey(0)