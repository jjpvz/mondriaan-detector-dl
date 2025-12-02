import cv2 as cv
import numpy as np

def order_quad_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)

def get_quadrilateral_from_contour(contour):
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.02 * peri, True)
    
    if len(approx) == 4:
        return order_quad_points(approx.reshape(-1, 2))

    rect = cv.minAreaRect(contour.reshape(-1,1,2))
    box = cv.boxPoints(rect)

    return order_quad_points(box)

def warp_image(cropped_img):
    gray_img = cv.cvtColor(cropped_img, cv.COLOR_RGB2GRAY)
    contour_in_cropped = getLargestContour(gray_img)
    src = get_quadrilateral_from_contour(contour_in_cropped)  # TL,TR,BR,BL

    (tl, tr, br, bl) = src
    w_top  = np.linalg.norm(tr - tl)
    w_bot  = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl)
    h_right= np.linalg.norm(br - tr)
    maxW = int(round(max(w_top, w_bot)))
    maxH = int(round(max(h_left, h_right)))

    dst = np.array([
        [0,     0],
        [maxW-1,0],
        [maxW-1,maxH-1],
        [0,     maxH-1]
    ], dtype=np.float32)

    M = cv.getPerspectiveTransform(src, dst)

    warped = cv.warpPerspective(cropped_img, M, (maxW, maxH),
                                flags=cv.INTER_LINEAR,
                                borderMode=cv.BORDER_REPLICATE)
    return warped

def getLargestContour(img_BW):
    contours, _ = cv.findContours(img_BW.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv.contourArea)

    return np.squeeze(contour)

def getContourExtremes(contour):
    left = contour[contour[:, 0].argmin()]
    right = contour[contour[:, 0].argmax()]
    top = contour[contour[:, 1].argmin()]
    bottom = contour[contour[:, 1].argmax()]

    return np.array((left, right, top, bottom))

def mask_feature_color(img_rgb, h_range, s_min, v_min, invert=False):
    hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
    masks = []

    for (h1, h2) in h_range:
        lower = (h1, s_min, v_min)
        upper = (h2, 255, 255)
        masks.append(cv.inRange(hsv, lower, upper))

    mask = masks[0]

    for m in masks[1:]:
        mask = cv.bitwise_or(mask, m)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7,7))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  kernel, iterations=1)
    
    if invert:
        mask = cv.bitwise_not(mask)

    return mask

def preprocess_image(img_rgb):
    mask = mask_feature_color(img_rgb, [(40, 95)], 50, 50, True)
    masked_img = cv.bitwise_and(img_rgb, img_rgb, mask=mask)

    gray = cv.cvtColor(masked_img, cv.COLOR_RGB2GRAY)
    contour = getLargestContour(gray)
    left, right, top, bottom = getContourExtremes(contour)
    x_min, x_max = left[0], right[0]
    y_min, y_max = top[1], bottom[1]
    cropped_img = masked_img[y_min:y_max, x_min:x_max]

    warped_img = warp_image(cropped_img)
    
    return warped_img