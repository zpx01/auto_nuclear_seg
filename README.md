# Automated Nucleus Semantic Segmentation  

## Summary
In this project, I utilized 2D nuclear divergent image data provided by Kaggle to train a 2D U-Net CNN that would automate nuclear semantic segmentation. The 2D U-Net is developed using the TensorFlow-based Keras API. For the script, I developed a data pipeline to go through all image files in an efficient manner, the actual 2D U-Net CNN with some addons such as callback APIs, and a small snippet of code to plot the training statistics (accuracy + loss). Additionally, I have included code to display the outputted predictions of the 2D U-Net and to compare it with the raw image. On average, the accuracies attained were above 96% with an IoU of 0.76 or higher. The max IoU value attained on any training was 0.894. 


Feel free to tinker with the code and let me know if there are any changes that could potentially boost the accuracy of the model. 
