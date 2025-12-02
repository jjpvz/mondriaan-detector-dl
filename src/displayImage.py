import cv2 as cv

def display_image_cv(img, title="Image", auto_close_ms=None):
    brg_img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    img = brg_img

    cv.namedWindow(title, cv.WINDOW_NORMAL)
    cv.resizeWindow(title, 800, 600)
    cv.imshow(title, img)

    if auto_close_ms is not None:
        cv.waitKey(auto_close_ms)
    else:
        cv.waitKey(0)
        
    cv.destroyAllWindows()