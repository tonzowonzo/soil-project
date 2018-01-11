# soil-project
project for soil classification - includes all functions used for image manipulation

The aim of this project is to classify different types of soil based on image recognition via CNN's.

The data used for the project comes from google images and http://www.fao.org/soils-portal/soil-survey/soil-maps-and-databases/harmonized-world-soil-database-v12/en/ is used for the random forest classifier.

Goal: To make a computer vision soil classifier that makes use of both images and input data.
	- Also find certain horizons in the soil
	- Also create a simple GUI for it (started)
	- Allow to run off not just images but data input too, both the data and the
	  soil image will vote 50% each for the highest % chance of what soil type it is. 
	  If there is no data input we will only use the values from the image. Else we can
	  use only certain values to trim the probabilities of a soil being a particular type 
	  (if I can find info on this).
	- Each piece of data added will tweak the percentiles, with 1 data point added ie pH 
	  the influence of the random forest classifier will be very low (5%) when all data points
	  are added the influence will be much higher (95%). This will allow an ensemble method to
	  be most accurate based on the information given.
Steps:

1. Gather labelled images and data from soil databases
	- Create a script to gather images (labelled) and data (Done)
	- Gather own images (Working on this)
	- Label the horizons in the image

2. Preprocess the images and data for input into a CNN combined algorithm
	- Trim images to same sizes and resolutions (Done)
	- Homogenise any data (Done)
	- Resample images and data

3. Build the model

4. Run the model

5. Make the model work with open cv from laptops camera

6. Make the model work with raspberry pi's camera
