How to run this program: HW1.py


Assume HW1.py is inside /directory

/directory should contain: 3 folders and HW1.py as following


/directory ------ HW1.py
	   |
	   |------ image (folder)
	   |
	   |------ trimap (folder)
	   |
	   |------ background

You should 

1.Put all the images into image folder with format .png

2.Put all the trimaps into trimap folder with format .png, be aware to have the same file name as the corresponding image.

3.Put all the background images into background folder with format.png


then run the code, the results will be stored in ./results folder(will be generated if there is no one) like this form, I use an example to explain


if we have 2 background b1.png and b2.png along with 3 images with their trimap img1.png, img2.png, img3.png

then we shall do composition on (b1 , img1) , (b1, img2) .... (b2, img3)

that is, we do composition on each img with these two background

the results will then be stored in the results folder in this form

for (b1,img1) composition, it will be stored as ./results/b1/img1.png, where b1 is a sub-directory of ./results

further (b1,img2) will be stored as ./results/b1/img2.png

.
.
.

until (b2,img3) be stored as ./results/b2/img3.png , then all the computation and compositions are done



After the execution, you can then go to the ./results folder to get all the composite images




 
