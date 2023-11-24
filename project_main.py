import cv2 as cv 

# Start a video capture, using device's camera
cap = cv.VideoCapture(0)

# Check if video file opened successfully
if (cap.isOpened() == False):    
  print("Error opening video stream or file")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print("Frame width: " , frame_width)
print("Frame height: " , frame_height)

cnt = 1
# Read until video is completed
while(cap.isOpened()):    
  # Capture frame-by-frame    
  ret, frame = cap.read()    
  if ret == False:        
    break       
    # Display the frame    
  cv.imshow('frame',frame)
  key = cv.waitKey(25)
  if key == ord('q'):
    break  

# Release the video capture 
cap.release()

# Close all the frames
cv.destroyAllWindows()
