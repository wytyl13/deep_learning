import cv2

def hist_color_image(input_image_path, write_image_path, imshow_image = False):
    image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(image)
    # enhanced_b = cv2.equalizeHist(b)
    enhanced_g = cv2.equalizeHist(g)
    # enhanced_r = cv2.equalizeHist(r)
    # image_enhance = cv2.merge((enhanced_b, enhanced_g, enhanced_r))
    image_enhance = cv2.merge((b, enhanced_g, r))
    cv2.imwrite(write_image_path, image_enhance)
    if imshow_image:
        cv2.imshow("original image", image)
        cv2.imshow("enhancement image", image_enhance)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



def hist_gray_image(input_image_path, write_image_path, imshow_image = False):
    image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_image = cv2.equalizeHist(image_gray)
    cv2.imwrite(write_image_path, enhanced_image)
    if imshow_image:
        cv2.imshow("gray image", image_gray)
        cv2.imshow("enhance gray image", enhanced_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # hist_gray_image("c:/users/80521/desktop/image/14.jpg", "c:/users/80521/desktop/test1.jpg")
    hist_color_image("c:/users/80521/desktop/image/14.jpg", "c:/users/80521/desktop/test2.jpg")
