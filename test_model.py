import os
import sys
import cv2
import argparse
import xml.etree.ElementTree


def testing(img, xmlName, detector, scaleFactor, minNeighbors):

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cars = detector.detectMultiScale(gray, scaleFactor, minNeighbors, maxSize=(80, 80), minSize=(20, 20))
	for (x, y, w, h) in cars:
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

	takenParkingSpots = 0
	emptyParkingSpots = 0
	truePositives = 0
	falsePositives = 0

	e = xml.etree.ElementTree.parse(xmlName).getroot()
	for space in e.findall('space'):
		ID = space.get('id')
		occupied = (space.get('occupied') == '1')
		if occupied:
			takenParkingSpots = takenParkingSpots + 1
		else:
			emptyParkingSpots = emptyParkingSpots + 1
		vertices = []
		for point in space.iter('point'):
			x = point.attrib.get('x') 
			y = point.attrib.get('y')
			vertices.append((x, y))
		xCoordinates = [int(x) for (x, y) in vertices]
		yCoordinates = [int(y) for (x, y) in vertices]
		xCenter = (min(xCoordinates) + max(xCoordinates)) / 2
		yCenter = (min(yCoordinates) + max(yCoordinates)) / 2
		for (cx,cy,w,h) in cars:
			xCondition = cx < xCenter and xCenter < (cx + w)
			yCondition = cy < yCenter and yCenter < (cy + h)
			if xCondition and yCondition and occupied:
				truePositives = truePositives + 1
				break
			elif xCondition and yCondition and not occupied:
				falsePositives = falsePositives + 1
				break

	falseNegatives = takenParkingSpots - truePositives
	trueNegatives = emptyParkingSpots - falsePositives

	accuracy = ((truePositives + trueNegatives) / (takenParkingSpots + emptyParkingSpots)) * 100

	print("takenParkingSpots = ", takenParkingSpots)
	print("emptyParkingSpots = ", emptyParkingSpots)
	print("=====================================")
	print("truePositives = ", truePositives)
	print("falsePositives = ", falsePositives)
	print("trueNegatives = ", trueNegatives)
	print("falseNegatives = ", falseNegatives)
	print("=====================================")
	print("accuracy = ", accuracy)
	print("=====================================")

	return(img)


def main():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("-img", "--imgPath", dest="imgPath", help="Specify the path to image", required=True)
	parser.add_argument("-fType", "--featureType", dest="featureType", 
		help="Specify the feature type of the classifier to be used (HAAR or LBP)", required=True,
		type=str, choices = ['HAAR','LBP'])
	parser.add_argument("-sF", "--scaleFactor", dest="scaleFactor", 
		help="Specify the scale factor to be used for detection (use float values between 1.02 and 2)", type=float, required=True)
	parser.add_argument("-mN", "--minimumNeighbors", dest="minNeighbors",
		help="Specify minimum neighbours present for individual detection to be considered (use values between 10 and 30)", 
		type = int, required = True)
	
	args = parser.parse_args()

	img = cv2.imread(args.imgPath)
	if img is None:
		print("Path to the image is incorrect")
		sys.exit()
	else:
		outputFileName = os.path.splitext(args.imgPath)[0] + args.featureType + "result.jpg"
		xmlName = os.path.splitext(args.imgPath)[0] + ".xml"
		print("=====================================")
		print(args.imgPath)
		print("=====================================")

	if args.featureType == 'HAAR':
		detector = cv2.CascadeClassifier("haar10kcascade.xml")
		result = testing(img, xmlName, detector, args.scaleFactor, args.minNeighbors)
		cv2.imwrite(outputFileName, result)
		
	if args.featureType == 'LBP':
		detector = cv2.CascadeClassifier("lbp50kcascade.xml")
		result = testing(img, xmlName, detector, args.scaleFactor, args.minNeighbors)
		cv2.imwrite(outputFileName, result)


if __name__ == "__main__":
	main()
