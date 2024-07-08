import cv2

paths = ['angry.png', 'disgusted.png', 'fearful.png', 'happy.png', 'neutral.png', 'sad.png', 'surprised.png']

def canny(image):
    image_display = image
    cv2.imshow('grayscale_image', image_display)
    canny_edges = cv2.Canny(image, threshold1=150, threshold2=400)
    cv2.namedWindow('canny_edges')   
    cv2.moveWindow('canny_edges', 513,0)  
    image = cv2.resize(canny_edges, (510, 510))
    cv2.imshow('canny_edges', image)
    cv2.waitKey(0)

for path in paths:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    canny(image)