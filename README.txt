This is my attempt to begin to make the RidgeFinding process cleaner and more streamlined. It's still very much a work in progress, but it works.

Files Within:

RFFunctions contains all of the functions (and more!) for finding the ridgelines in images.
CycleThrough contains functions which allow you to cycle through images within a given directory with their ridgelines overlaid on top.
GetRidges_Linked contains functions which grab all of the ridges from all of the images within a given directory.
TrackIsolation contains functions to isolate tracks inside of an image and snip the image to their size (this only works for tiffs right now). 
RunRF contains some example code for using this. 


Parameters which will be important for most of the algorithms:

Sigma: float: sigma for derivative determination ~> Supposedly related to track width
lthresh: float: tracks with a response lower than this are rejected (0 accepts all)
uthresh: float: tracks with a response higher than this are rejected (0 accepts all)
minlen: int:  minimum track length accepted
linkthresh: int: maximum distance to link ridges (in the case that there are multiple ridgelines on a given track)
logim: Boolean: which allows you to analyse an image in linear space (False) or log space (True)
