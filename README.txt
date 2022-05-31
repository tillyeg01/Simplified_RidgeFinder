This is my attempt to begin to make the RidgeFinding process cleaner and more streamlined. It's still very much a work in progress and I will continue to iterate through it.

Now, to run the actual RidgeFinder, just go into RunRF.py, change the directory to one which contains your images to be analysed (they must be .fits files). 
Then, you can cycle through those images with fitted ridgelines (displaying in both linear and logarithmic space) by uncommenting the cycleimages block and playing with the parameters (or look at a single image by uncommenting the make_plot block). Once the desired parameters are found, comment this out and uncomment the getridges block and it will run the analysis on the whole directory. To save, uncomment the Ridges.to_csv block and input the desired save location. 

CycleThrough.cycleimages: scans through each image, fitting ridges to each one, for the purpose of evaluating the parameters

CycleThrough.make_plot: looks at a specified image

GetRidges_Linked.getridges: creates two dataframes of the ridges found and the linked ridges found in the images of the directory

