# Smart-Curve-Text-Alignment-Enhancer
Running
To process the entire image
Run the Main_App.py  : streamlit run Main_App.py 
Both functions exhibit comparable performance, with no discernible advantage in either. The primary distinction lies in their operational scope: dewarp.py operates across the entire image, whereas tight_dewarp.py specifically tracks the leftmost and rightmost black pixels within Otsu's threshold image, concentrating its efforts within that identified range.

Steps
Load Image :
Original image

Convert from RGB to Grayscale :
Output image

Apply Otsu's Thresholding Method, Erosion and then Dilation :
Original image

Calculate curve using Generalized Additive Model :
Output image

Final Image :
Output image

Greek Text Example
Input Image :
Output image

Output Image :
Output image

Rectification
Input Image :
Output image

Semi-processed Image :
Output image

Output Image :
Output image
