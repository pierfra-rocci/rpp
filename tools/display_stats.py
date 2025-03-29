from aspylib import astro
import numpy as np

#------ inputs ------
Folder = "D:/Astronomy/221110_Pier-Francesco/HAT-52P-b/"
Imagelist = [Folder + "Exoplanet-HAT-P-52-b-22-11-10_R_2x2_30,000secs00000010_solved.fits",
             Folder + "Exoplanet-HAT-P-52-b-22-11-10_R_2x2_40,000secs00000012_solved.fits"]

generic_name = Folder + "Exoplanet-HAT-P-52-b-22-11-10_R_2x2_"

sizes = [[2,8,40],
         [2.5,8,40],
         [3,8,40],
         [3.5,8,40],
         [4,8,40],
         [4.5,8,40],
         [5,8,40]]

#--- loads and displays image ---
data = astro.get_imagedata(Imagelist)
# astro.display(data)
# data2 = astro.scaling_Bspline(data, [100.2, 20.4], 1.5, 3)
# fig, ax = astro.display(data, show=False) # type: ignore

#--- display image ---
# while 1:
#     print(" ")
#     xy = astro.select(data, color='r')
#     print(xy)
#     x, y = xy[0]
#     param = astro.fit_gauss_circular([x-10, y-10], data[x-10:x+11, y-10:y+11])
#     print("floor (in ADU)   =", param[1])
#     print("height (in ADU)  =", param[2])
#     print("x0 (in pixels)   =", param[3])
#     print("y0 (in pixels)   =", param[4])
#     print("fwhm (in pixels) =", param[5])

# xy = astro.select(data, color='r') # type: ignore


#--- displays image stats and keyword header ---
for i in range(len(Imagelist)):
    median_values = astro.get_median(data[i,:,:])
    date_obs = astro.get_headervalues([Imagelist[i]], "DATE-OBS")
    print(Imagelist[i],
          "DATE-OBS=", date_obs[0],
          "min=", np.min(data[i,:,:].flatten()),
          "mean=", np.mean(data[i,:,:].flatten()),
          "median=", median_values[0],
          "max=", np.max(data[i,:,:].flatten()))

print(np.shape(data))

# (image number, vertical dimension, horizontal dimension)
# col_10_to_20 = images[:,10:20,:]
# data2 = astro.mirror_vert(data)
# data2 = astro.mirror_horiz(data)
# data2 = astro.rot_90(data)

# #--- loads image data and headers ---
# data = astro.get_imagedata(Image_in)
# headers = astro.get_headers(Image_in)

# #--- adds offset to data ---
# data = data + 100.0

# #--- saves modified image ---
# astro.save_imagelist(data, headers, Image_out)




