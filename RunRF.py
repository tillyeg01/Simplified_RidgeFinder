directory = "C:\\Users\\tilly\\Documents\\Alphas\\Test_Alphas_Austin\\"
# directory = "C:\\Users\\tilly\\Documents\\Simulations\\50torrCF4_5.204keV\\5.204 keV\\"

SIGMA = 3.6 #sigma for derivative determination ~> Related to track width
lthresh = 0.3 #tracks with a response lower than this are rejected (0 accepts all)
uthresh = 0 #tracks with a response higher than this are rejected (0 accepts all)
minlen = 20 #minimum track length accepted
linkthresh = 11 #maximum distance to be linked
logim = False




from CycleThrough import cycleimages,make_plot, er_plot
cycleimages(directory,SIGMA,lthresh,uthresh,minlen,linkthresh,logim)
# make_plot(0,directory,SIGMA,lthresh,uthresh,minlen,linkthresh,logim)

# er_plot(0,directory)



# from GetRidges_Linked import getridges
# Ridges, Linked_Ridges = getridges(directory,SIGMA,lthresh,uthresh,minlen,logim=logim,linked_lines=True, linkthresh=linkthresh)
# Linked_Ridges.to_csv("C:\\Users\\tilly\\OneDrive - University of New Mexico\\Desktop\\Linked_Ridges_test.csv",index=False)


