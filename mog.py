import cv2

flag = True;
#"TrafficVideo.avi"
videoObject = cv2.VideoCapture(0);

# define display window name

windowName = "Live Camera Input"; # window name
windowNameFGP = "Foreground Probabiity"; # window name

if (videoObject.isOpened):
    
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL);
    cv2.namedWindow(windowNameFGP, cv2.WINDOW_NORMAL);

    mog = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=600, detectShadows=False);

    while (flag):

        # if video file successfully open then read frame from video

        if (videoObject.isOpened):
            ret, frame = videoObject.read();

            # when we reach the end of the video (file) exit cleanly

            if (ret == 0):
                flag = False;
                continue;

        # add current frame to background model and retrieve current foreground objects

        foreground = mog.apply(frame);

        # display images - input and original

        cv2.imshow(windowName,frame);
        cv2.imshow(windowNameFGP,foreground);

        key = cv2.waitKey(40) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

        # e.g. if user presses "x" then exit

        if (key == ord('x')):
            flag = False;

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.");
