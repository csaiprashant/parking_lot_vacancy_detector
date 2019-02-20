# Parking Lot Vacancy Detector

## Introduction
A public infrastructure has various parking lots which get completely occupied very often and the public visiting the infrastructure spend too much time looking for a parking space, unaware that the parking lot is completely occupied. They would like to implement an automated solution to convey this information by displaying the number of available parking spaces at the entrance to the parking lot. These parking lots are overlooked be surveillance cameras. The task is to leverage them to detect and count the number of empty parking spots.
To acheive this, we need to train a classifier that can detect cars. In this project we train two such cascade classifiers - one using HAAR features and the other using LBP features. The project goes into great detail on how to generate positive training examples by parsing the surveillance camera images and their respective ground truth files. We use OpenCV's [Cascade Classifier Training](https://docs.opencv.org/2.4/doc/user_guide/ug_traincascade.html) utilities to train our cascade classifier. Finally we measure the accuracy of our classifier against the ground truth.

## Example
![Sample Example](https://github.com/csaiprashant/parking_lot_vacancy_detector/blob/master/example.png)
## Files in this Repository
- haar10kcascade.xml - The HAAR cascade classifier.
- haar10000.txt - Console output of training a HAAR cascade classifier using 10000 positive examples.
- lbp50kcascade.xml - The LBP cascade classifier.
- lbp50000.txt - Console output of training an LBP cascade classifier using 50000 positive examples.
- parse_positives.py - Python script to parse ground truth XML files and generate positive training examples.
- projectreport.pdf - Project Report.
- test_model.py - Python script to apply trained classifier to an image and measure its accuracy and F-score.

### For full report, please refer [Project Report](https://github.com/csaiprashant/academic_projects/blob/master/Parking%20Lot%20Vacancy%20Detector/projectreport.pdf)
