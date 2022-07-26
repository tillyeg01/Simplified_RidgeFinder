# directory = "C:\\Users\\tilly\\Documents\\Alphas\\Test_Alphas_Austin\\"
# directory = "C:\\Users\\tilly\\Documents\\Simulations\\50torrCF4_5.204keV\\5.204 keV\\"
# directory = "C:\\Users\\tilly\\Documents\\Alphas\\Central crossing\\45 CF4 + 4 CS2 gain 1000\\"
directory = "C:\\Users\\tilly\\Documents\\Fe-55 Data\\2022.June.11 - CS2 smoothed tracks\\sigma is 4 by 3 pixels\\"
# directory = "C:\\Users\\tilly\\Documents\\Fe-55 Data\\2022.June.11 - CS2 smoothed tracks\\sigma is 1 pixel\\"
# directory = "C:\\Users\\tilly\\Documents\\Fe-55 Data\\2022.June.10 - CS2 tracks\\"


SIGMA = 2.5 #sigma for derivative determination ~> Related to track width
lthresh = 4 #tracks with a response lower than this are rejected (0 accepts all)
uthresh = 0 #tracks with a response higher than this are rejected (0 accepts all)
minlen = 9 #minimum track length accepted
linkthresh = 10 #maximum distance to be linked
logim = False


from CycleThrough import cycleimages,make_plot,make_plot_2, er_plot
# cycleimages(directory,
#             SIGMA,
#             lthresh,
#             uthresh,
#             minlen,
#             linkthresh,
#             logim)
make_plot_2(5,directory,SIGMA,
            lthresh,
            uthresh,
            minlen,
            linkthresh,
            logim)
# make_plot(3,directory,
#           SIGMA,
#           lthresh,
#           uthresh,
#           minlen,
#           linkthresh,
#           logim)

# er_plot(1,directory)



from GetRidges_Linked import getridges
# Ridges, Linked_Ridges = getridges(directory,
#                                   SIGMA,
#                                   lthresh,
#                                   uthresh,
#                                   minlen,
#                                   logim=logim,
#                                   linked_lines=True, 
#                                   linkthresh=linkthresh)

# Linked_Ridges.to_csv("C:\\Users\\tilly\\OneDrive - University of New Mexico\\Desktop\\Linked_Ridges_test.csv",index=False)


