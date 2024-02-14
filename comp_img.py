import skimage as ski
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy as sci
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import (
    exposure, util
)
plot_dir="/home/mehmet/Documents/jena lectures/231016_thirdsemester/Computational Imaging/plots"
def planewise_correlation(picture_array,no_of_images,main_plane="False"):
    corrmat=np.ones(no_of_images)
    midpos=len(picture_array)//2
    
    if main_plane!="False":
        midpos=main_plane
    print("main plane",midpos)
    print("tot images",no_of_images)
    #max_corr = np.sum(sci.signal.correlate2d(picture_array[15],picture_array[15]))
    #min_corr = np.sum(sci.signal.correlate2d(picture_array[15],picture_array[0]))
    #max_corr = sci.signal.correlate2d(picture_array[midpos],picture_array[mıdpos])[149,149]
    #min_corr = sci.signal.correlate2d(picture_array[midpos],picture_array[0])[149,149]
    coef2=np.sqrt(np.sum(picture_array[midpos]*picture_array[midpos]))
    #corr_min,corr_max = min_corr,max_corr
    for ix in range(no_of_images):
        coef1=np.sqrt(np.sum(picture_array[ix]*picture_array[ix]))
        correlation2 = np.sum(picture_array[midpos]*picture_array[ix])/(coef1*coef2)
        corrmat[ix]=correlation2
    return corrmat

def order_pictures(adress,im_dir,shape):
    ordered=[]
    
    for ix in enumerate(im_dir):
        if ".tiff" in ix[1]:
            ind=ix[1].index(".tiff")
            key=int(ix[1][:ind])
            ordered.append(key)
    or_im=np.ones(shape=(len(ordered),shape[0],shape[1]))
    ordered=sorted(ordered)
    for ord in range(len(ordered)):
        for ix in enumerate(im_dir):
            if str(ordered[ord]) in ix[1]:
                #print("read this image",ix[1])
                or_im[ord,:,:]=ski.io.imread(adress+"/"+ix[1],as_gray=True)
    return or_im

##get images from directort FİG 2
dir1=os.getcwd()
dir2=os.listdir(dir1)
dir3 = os.listdir(dir1+"/"+dir2[3])
dir31=os.listdir(dir1+"/"+dir2[3]+"/"+dir3[-1])
dir32=os.listdir(dir1+"/"+dir2[3]+"/"+dir3[0])
psfdir=os.listdir(dir1+"/"+dir2[3]+"/"+dir3[-1]+"/"+dir31[0])
print(dir31,dir32)
print(psfdir)
psfdirnum=[]
for ix in psfdir:
    num=int(ix[:-5])
    psfdirnum.append(num)
print(psfdirnum)
sort_index=np.argsort(psfdirnum)
str1=dir1+"/"+dir2[3]+"/"+dir3[-1]+"/"+dir31[0]+"/"+psfdir[-1]
#print(str1)
im2=ski.io.imread(str1)
print(im2.shape)
picture_array=np.ones((len(psfdir),im2.shape[0],im2.shape[1]))
for number,order in enumerate(sort_index):
    str1=dir1+"/"+dir2[3]+"/"+dir3[-1]+"/"+dir31[0]+"/"+psfdir[order]
    #print(str1)
    im2=ski.io.imread(str1,as_gray=True)
    #print(im2.dtype)
    plt.imshow(im2)
    plt.show()
    picture_array[number,:,:]=im2
### find dir FIG 3
print(dir3)
print(dir32)
spsfdir=os.listdir(dir1+"/"+dir2[3]+"/"+dir3[0]+"/"+dir32[1])#speckle point spread functions
#print(spsfdir)
## saturated spsf images reading
saturated_dir=os.listdir(dir1+"/"+dir2[3]+"/"+dir3[0]+"/"+dir32[1]+"/"+spsfdir[1])
nonsaturated_dir=os.listdir(dir1+"/"+dir2[3]+"/"+dir3[0]+"/"+dir32[1]+"/"+spsfdir[2])
#print(saturated_dir,nonsaturated_dir)

sample_im=ski.io.imread(dir1+"/"+dir2[3]+"/"+dir3[0]+"/"+dir32[1]+"/"+spsfdir[1]+"/"+saturated_dir[1])

#nonsaturated

sample_im_2=ski.io.imread(dir1+"/"+dir2[3]+"/"+dir3[0]+"/"+dir32[1]+"/"+spsfdir[2]+"/"+nonsaturated_dir[1])


## generating stack of images
saturated_stack_spsf = order_pictures(dir1+"/"+dir2[3]+"/"+dir3[0]+"/"+dir32[1]+"/"+spsfdir[1],saturated_dir,sample_im.shape)
nonsaturated_stack_spsf = order_pictures(dir1+"/"+dir2[3]+"/"+dir3[0]+"/"+dir32[1]+"/"+spsfdir[2],nonsaturated_dir,sample_im_2.shape)
#print(saturated_stack_spsf[0],saturated_stack_spsf[0].dtype)

### correlation saturated
corr_saturated=planewise_correlation(saturated_stack_spsf,len(saturated_stack_spsf),main_plane=15)
#print(corr_saturated)


### plotting saturated spsf

print("calc",abs(1-corr_saturated[np.where(corr_saturated>1)]))
corr_saturated=np.abs(corr_saturated)
corr_saturated[np.where(corr_saturated>1)] -= 1
#print(corr_saturated)
norm_saturated=(corr_saturated[:]-np.min(corr_saturated))/(np.max(corr_saturated)-np.min(corr_saturated))### important
plt.plot(corr_saturated)
plt.title("saturated correlation")
plt.show()
plt.plot(norm_saturated)#normalized cross correlation plot
plt.show()

##non saturated
### correlation non saturated
corr_nonsaturated=planewise_correlation(nonsaturated_stack_spsf,len(nonsaturated_stack_spsf),main_plane=15)
print(corr_nonsaturated)
#print("calc",abs(1-corr_saturated[np.where(corr_nonsaturated>1)]))
corr_nonsaturated=np.abs(corr_nonsaturated)
corr_nonsaturated[np.where(corr_nonsaturated>1)] -= 1
norm_nonsaturated=(corr_nonsaturated[:]-np.min(corr_nonsaturated))/(np.max(corr_nonsaturated)-np.min(corr_nonsaturated))#important
##plotting non saturated correlation
#print(corr_nonsaturated)
plt.plot(corr_nonsaturated)
plt.title("nonsaturated correlation")
plt.show()
plt.plot(norm_nonsaturated)#normalized cross correlation plot
plt.show()
## comparison plot
plt.plot((corr_nonsaturated[:]-np.min(corr_nonsaturated))/(np.max(corr_nonsaturated)-np.min(corr_nonsaturated)),"*-r",label="non-saturated")#normalized cross correlation plot
plt.plot((corr_saturated[:]-np.min(corr_saturated))/(np.max(corr_saturated)-np.min(corr_saturated)),"o-b",label="saturated")
plt.legend()
plt.gcf().savefig(plot_dir+"/"+"saturated vs nonsaturated correlation plots.png",dpi=400)
plt.show()
plt.plot(norm_saturated[:16],"r",label="saturated")
plt.plot(norm_nonsaturated[:16],"b",label="non saturated")
dist=np.arange(0,len(norm_saturated[:16]),1,dtype=np.float32)*0.2
plt.xticks(np.arange(0,len(norm_saturated[:16]),1),labels=dist,rotation=45)
plt.xlabel("distance in transverse plane(um)")
plt.legend()
plt.show()
#print(np.arange(0,len(norm_saturated[:16]),1)*0.2,np.arange(0,len(norm_saturated[:16]),1)*0.2)


#### gaussian fit

from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

x = np.arange(0,31,1)*0.2
## take left wing of normalized correlation results
y1 = np.concatenate((norm_saturated[:16],norm_saturated[:15][::-1]))
y2 = np.concatenate((norm_nonsaturated[:16],norm_nonsaturated[:15][::-1]))

n = len(x)                          #the number of data
mean = 3                  #note this correction
sigma1 = sum(y1*(x-mean)**2)/n        #note this correction
sigma2 = sum(y2*(x-mean)**2)/n        #note this correction
def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))
## just take all points to calculate the fitting parameters
popt1,pcov1 = curve_fit(gaus,x,y1,p0=[1,mean,sigma1])
popt2,pcov2 = curve_fit(gaus,x,y2,p0=[1,mean,sigma2])

x2=x[12:19]
n2=len(x2)
y3=y1[12:19]
#print(x2,y3)
mean2=3
## take few points to calcultae fitting parameters for saturated correlation
sigma3=sum(y3*(x2-mean2)**2)/n2
popt3,pcov3 = curve_fit(gaus,x2,y3,p0=[1,mean2,sigma3])
#plt.plot(x,gaus(x,*popt3),'go:',label='fit non saturated')

### plotting the gaussian fits of the normalized correlation plots
plt.plot(x,y1,'b+:',label='data saturated')
plt.plot(x,y2,'r+:',label='data non saturated')
#plt.plot(x,gaus(x,*popt1),'ro:',label='fit saturated')
plt.plot(x,gaus(x,*popt2),'bo:',label='fit non saturated')
plt.plot(x,gaus(x,*popt3),'ro:',label='fit saturated')
plt.legend()
plt.title('Fig. 3 - Fit for Time Constant')
plt.xlabel('Distance in transverse axis(z) [$\mu$m]')
plt.ylabel("Correlation [arb. u.]")
### calculating the fwhm ratio between saturated and non saturated
#print("fwhm",popt1,popt2,popt3)
saturated_fwhm,nonsaturated_fwhm = np.abs(popt3[-1]),popt2[-1]
#print("saturated fwhm",np.abs(popt3[-1]),"nonsaturated fwhm",popt2[-1])
ratio=nonsaturated_fwhm/saturated_fwhm##### Very IMPORTANT
print("ratio",ratio,np.sqrt(2))
#plt.gcf().savefig(plot_dir+"/"+"Fig3_correlation_comperison_"+str(ratio)+"_fwhm_ratio.png",dpi=400)
plt.show()

x = np.arange(0,25,1)*0.2
## take left wing of normalized correlation results
y1 = norm_saturated
y2 = norm_nonsaturated

n = len(x)                          #the number of data
mean = 3                  #note this correction
sigma1 = sum(y1*(x-mean)**2)/n        #note this correction
sigma2 = sum(y2*(x-mean)**2)/n        #note this correction
def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))
## just take all points to calculate the fitting parameters
popt1,pcov1 = curve_fit(gaus,x,y1,p0=[1,mean,sigma1])
popt2,pcov2 = curve_fit(gaus,x,y2,p0=[1,mean,sigma2])

x2=x[12:19]
n2=len(x2)
y3=y1[12:19]
#print(x2,y3)
mean2=3
## take few points to calcultae fitting parameters for saturated correlation
sigma3=sum(y3*(x2-mean2)**2)/n2
popt3,pcov3 = curve_fit(gaus,x2,y3,p0=[1,mean2,sigma3])
#plt.plot(x,gaus(x,*popt3),'go:',label='fit non saturated')

### plotting the gaussian fits of the normalized correlation plots
plt.plot(x,y1,'b+:',label='data saturated')
plt.plot(x,y2,'r+:',label='data non saturated')
#plt.plot(x,gaus(x,*popt1),'ro:',label='fit saturated')
plt.plot(x,gaus(x,*popt2),'bo:',label='fit non saturated')
plt.plot(x,gaus(x,*popt3),'ro:',label='fit saturated')
plt.legend()
plt.title('Fig. 3 - Fit for Time Constant')
plt.xlabel('Distance in transverse axis(z) [$\mu$m]')
plt.ylabel("Correlation [arb. u.]")
### calculating the fwhm ratio between saturated and non saturated
#print("fwhm",popt1,popt2,popt3)
saturated_fwhm,nonsaturated_fwhm = np.abs(popt3[-1]),popt2[-1]
#print("saturated fwhm",np.abs(popt3[-1]),"nonsaturated fwhm",popt2[-1])
ratio=nonsaturated_fwhm/saturated_fwhm##### Very IMPORTANT
print("ratio",ratio,np.sqrt(2))
plt.gcf().savefig(plot_dir+"/"+"Fig3_correlation_comperison_"+str(ratio)+"_fwhm_ratio.png",dpi=400)
plt.show()

### Generate 3D Stack of PSF like in fig2
from mpl_toolkits.mplot3d import Axes3D
print(picture_array.shape)
print(picture_array[:,:,75])
plt.imshow(picture_array[:,:,75])
plt.show()
plt.imshow(picture_array[:,75,:])
plt.show()
print(saturated_stack_spsf.shape)
plt.imshow(nonsaturated_stack_spsf[:,:,50])
plt.show()
plt.imshow(nonsaturated_stack_spsf[:,50,:])
plt.show()
### 3d heat plot tryout
def do_every3d_thing(data):
    def show_plane(ax, plane, cmap="gray", title=None):
        ax.imshow(plane, cmap=cmap)
        ax.axis("off")
    
        if title:
            ax.set_title(title)
    def display(im3d, cmap='hot', step=2):
        data_montage = util.montage(im3d[::step], padding_width=4, fill=np.nan)
        _, ax = plt.subplots(figsize=(16, 14))
        ax.imshow(data_montage, cmap=cmap)
        ax.set_axis_off()
    
    
    display(data)
    
    
    def slice_in_3D(ax, i):
        # From https://stackoverflow.com/questions/44881885/python-draw-3d-cube
        Z = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [1, 1, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [1, 0, 1],
                      [1, 1, 1],
                      [0, 1, 1]])
    
        Z = Z * data.shape
        r = [-1, 1]
        X, Y = np.meshgrid(r, r)
    
        # Plot vertices
        ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])
    
        # List sides' polygons of figure
        verts = [[Z[0], Z[1], Z[2], Z[3]],
                 [Z[4], Z[5], Z[6], Z[7]],
                 [Z[0], Z[1], Z[5], Z[4]],
                 [Z[2], Z[3], Z[7], Z[6]],
                 [Z[1], Z[2], Z[6], Z[5]],
                 [Z[4], Z[7], Z[3], Z[0]],
                 [Z[2], Z[3], Z[7], Z[6]]]
    
        # Plot sides
        ax.add_collection3d(
            Poly3DCollection(
                verts,
                facecolors=(0, 1, 1, 0.25),
                linewidths=1,
                edgecolors="darkblue"
            )
        )
    
        verts = np.array([[[0, 0, 0],
                           [0, 0, 1],
                           [0, 1, 1],
                           [0, 1, 0]]])
        verts = verts * (60, 256, 256)
        verts += [i, 0, 0]
    
        ax.add_collection3d(
            Poly3DCollection(
                verts,
                facecolors="magenta",
                linewidths=1,
                edgecolors="black"
            )
        )
    
        ax.set_xlabel("plane")
        ax.set_xlim(0, 100)
        ax.set_ylabel("row")
        ax.set_zlabel("col")
    
        # Autoscale plot axes
        scaling = np.array([getattr(ax,
                                    f'get_{dim}lim')() for dim in "xyz"])
        ax.auto_scale_xyz(* [[np.min(scaling), np.max(scaling)]] * 3)
    
    
    def explore_slices(data, cmap="gray"):
        from ipywidgets import interact
        N = len(data)
    
        @interact(plane=(0, N - 1))
        def display_slice(plane=34):
            fig, ax = plt.subplots(figsize=(20, 5))
    
            ax_3D = fig.add_subplot(133, projection="3d")
    
            show_plane(ax, data[plane], title=f'Plane {plane}', cmap=cmap)
            slice_in_3D(ax_3D, plane)
    
            plt.show()
    
        return display_slice
    
    
    explore_slices(data)
#do_every3d_thing(deconvolved_stack_im)
str_lym=dir1+"/"+dir2[3]+"/"+dir3[0]+"/"+dir32[0]
print(str_lym)
print(dir3[0])
print(dir32)
lymosome_dir=os.listdir(str_lym)
print(lymosome_dir)
for ix in lymosome_dir:
    if ".tiff" in ix:
        nonsaturated_lym=ski.io.imread(str_lym+"/"+ix,as_gray=True)
        plt.imshow(nonsaturated_lym)
        plt.show()
print(dir31)
print(dir1+"/"+dir2[3]+"/"+dir3[-1]+"/"+dir31[-1])
fig2_speckle = ski.io.imread(dir1+"/"+dir2[3]+"/"+dir3[-1]+"/"+dir31[-1])
plt.imshow(fig2_speckle)
plt.show()

####Find the fluoresence image of dye
print(dir32)
speckle_image_str = dir31[-1]
str_spek=dir1+"/"+dir2[3]+"/"+dir3[-1]+"/"+speckle_image_str
print(str_spek)
lymosome_str = dir1+"/"+dir2[3]+"/"+dir3[0]+"/"+dir32[0]
lymosome_dir = os.listdir(lymosome_str)
print(lymosome_dir)
lymosome_names =[116,117,118,119,120]
lymosome_im=np.ones((len(lymosome_names),170,170))
for ix in enumerate(lymosome_names):
    for ly in lymosome_dir:
        #print(ly)
        ind1= ly.index(".tif")
        if str(ix[1]) == ly[:ind1]:
            print(ly)
            lymo_im=ski.io.imread(lymosome_str+"/"+ly)
            lymosome_im[ix[0],:,:] = lymo_im
            #print(lymo_im.shape)
#print(lymosome_im)
speckle_image=ski.io.imread(str_spek)
psf_im=picture_array[0,:,:,]
### plot the two images for wiener deconvolution
print("speckle image",str_spek)
plt.imshow(speckle_image)
plt.show()
plt.imshow(saturated_stack_spsf[0,:,:,])
#### Wiener Deconvolution
def wiener_deconvolution(speckle_image,picture_array,deconvolve_dims):
    
    deconvolved_stack_im=np.ones((len(picture_array),deconvolve_dims[0],deconvolve_dims[1]))
    for ix in range(len(picture_array)):
        psf_im=picture_array[ix,:,:,]
        #rng=rng = np.random.default_rng()
        """
        conv_speckle_im=sci.signal.convolve2d(speckle_image, psf_im, 'same')
        conv_speckle_im += 0.1 * conv_speckle_im.std() * rng.standard_normal(conv_speckle_im.shape)
        deconvolved, _ = ski.restoration.unsupervised_wiener(conv_speckle_im, psf_im)
        """
        deconvolved =ski.restoration.wiener(speckle_image, psf_im,balance=0.01)
        ###plotting
        
        deconvolved_stack_im[ix,:,:]=deconvolved
    print("scale color by this",deconvolved_stack_im.min())
    for ress in range(len(picture_array)):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5),
                               sharex=True, sharey=True)
    
        plt.viridis()
    
        ax[0].imshow(psf_im, vmin=psf_im.min(), vmax=psf_im.max())
        ax[0].axis('off')
        ax[0].set_title('Data')
    
        ax[1].imshow(deconvolved_stack_im[ress,:,:])
        ax[1].axis('off')
        ax[1].set_title('Self tuned restoration')
        #fig.colorbar()
        fig.tight_layout()
    
        plt.show()
    return deconvolved_stack_im
#deconvolution of spsf
print(len(picture_array))
normalized_picture_array=np.zeros((len(picture_array),picture_array[0].shape[0],picture_array[0].shape[1]))
for ix in range(len(picture_array)):
    normalized_picture_array[ix,:,:] = picture_array[ix]/np.sum(picture_array[ix])
deconvolved_stack_im = wiener_deconvolution(speckle_image/np.sum(speckle_image),normalized_picture_array,[300,300])
#padded_deconvolced_stack_im = wiener_deconvolution(speckle_image,padded_picture_array,[300,300])
#deconvolution of saturated and unsaturated images
"""diff_lymos_saturated=[]
diff_lymos_nonsaturated=[]
for ix in range(len(lymosome_im)):
    print(ix)
    deconvolved_stack_from_saturated_spsf = wiener_deconvolution(lymosome_im[ix],saturated_stack_spsf,[170,170])
    deconvolved_stack_from_nonsaturated_spsf = wiener_deconvolution(lymosome_im[ix],nonsaturated_stack_spsf,[170,170])
    diff_lymos_saturated.append(deconvolved_stack_from_saturated_spsf)
    diff_lymos_nonsaturated.append(deconvolved_stack_from_nonsaturated_spsf)"""
for ix in enumerate(deconvolved_stack_im):
    plt.imshow(ix[1],cmap="gray")
    plt.title("position at transverse axis "+str(ix[0]))
    plt.gcf().savefig(plot_dir+"/"+"Fig2_reconstruction_plane_num"+str(ix[0])+"_.png",dpi=400)
    plt.show()
diff_lymos_saturated=[]
diff_lymos_nonsaturated=[]
shape_2=lymosome_im.shape
shape_1 = saturated_stack_spsf.shape
print(saturated_stack_spsf.shape[0],shape_1[0])
print(shape_1)
normalized_saturated_stack_spsf = np.zeros((shape_1[0],shape_1[1],shape_1[2]))
padded_nonsaturated_stack_spsf = np.zeros((shape_1[0],shape_2[1],shape_2[2]))
new_shape1 = [int((shape_2[1]-shape_1[1])/2),int((shape_2[2]-shape_1[2])/2)]
for ix in range(len(saturated_stack_spsf)):
    normalized_saturated_stack_spsf[ix,:,:] = saturated_stack_spsf[ix]/np.sum(saturated_stack_spsf[ix])
#    padded_nonsaturated_stack_spsf[ix,:,:] = np.pad(nonsaturated_stack_spsf[ix],(new_shape1[0],new_shape1[1]),mode="constant",constant_values=(0,0))
#for ix in range(len(lymosome_im)):
#    print(ix)
deconvolved_stack_from_saturated_spsf = wiener_deconvolution(lymosome_im[-1]/np.sum(lymosome_im[-1]),normalized_saturated_stack_spsf,[170,170])
    #deconvolved_stack_from_nonsaturated_spsf = wiener_deconvolution(lymosome_im[ix],nonsaturated_stack_spsf,[170,170])
diff_lymos_saturated.append(deconvolved_stack_from_saturated_spsf)
    #diff_lymos_nonsaturated.append(deconvolved_stack_from_nonsaturated_spsf)

# Simple Implementation of FISTA by using SPSF(Speckle Point Spread Function)
#first LSQ implementation
shape_3 =point_scan.shape
save_dir="/home/mehmet/Documents/jena lectures/231016_thirdsemester/Computational Imaging/plots/FISTA/"
print(shape_3)
lsq_res=np.zeros((len(saturated_stack_spsf),shape_3[1],shape_3[2]))
ft_spsf_stack=np.zeros((len(saturated_stack_spsf),shape_3[1],shape_3[2]))
ft_speckle_image = np.fft.fftshift(np.fft.fft2(point_scan[-1]))
padding_width =int((shape_3[1]-shape_1[1])*0.5)
print(padding_width)
for ix in range(len(point_scan)):
    #you can't jsut take psf because their dimensions are not enough
    #you need to pad them
    ft_speckle_image = np.fft.fftshift(np.fft.fft2(point_scan[ix]))
    padded_normalized_saturated_stack_spsf=np.pad(normalized_saturated_stack_spsf[ix],(padding_width,padding_width),"constant",constant_values=(0,0))
    ft_spsf=np.fft.fftshift(np.fft.fft2(padded_normalized_saturated_stack_spsf))
    print("shape2",ft_spsf.shape)
    lsq_res_sub = np.divide((ft_spsf.conj().T*ft_speckle_image),(ft_spsf.conj().T*ft_spsf+1e-10))
    lsq_res[ix,:,:] = lsq_res_sub
    ft_spsf_stack[ix,:,:]=ft_spsf
    plt.imshow(np.abs(np.fft.ifftshift(np.fft.ifft2(lsq_res_sub))))
    plt.title("plane by plane LSQ")
    plt.show()
    #lasso
    lambd=0.1
    lasso_sol=np.copy(lsq_res_sub)
    for ite in range(10):
        gamma=1/np.sqrt(np.abs(lasso_sol)+1e-10)
        gamma_mat = gamma*gamma
        lasso_sol_new = np.divide((ft_spsf.conj().T*ft_speckle_image),(ft_spsf.conj().T*ft_spsf+np.eye(shape_3[1])*lambd*gamma_mat+1e-10))
        lasso_sol=lasso_sol_new
    sol_of_lasso = np.fft.ifftshift(np.fft.ifft2(lasso_sol))
    #plt.imshow(np.abs(lasso_sol))
    #plt.title("lasso_sol")
    #plt.show()
    plt.imshow(np.abs(sol_of_lasso))
    plt.title("deconvolved with lasso")
    plt.gcf().savefig(save_dir+"fista_point_scan_"+str(ix)+"saturated_spsf.png",dpi=400)
    plt.show()
    plt.imshow(point_scan[ix])
    plt.title("image")
    plt.gcf().savefig(save_dir+"fista_point_scan_"+str(ix)+"image.png",dpi=400)
    plt.show()
    plt.imshow(padded_normalized_saturated_stack_spsf)
    plt.title("spsf")
    plt.gcf().savefig(save_dir+"fista_point_scan_"+str(ix)+"spsf.png",dpi=400)
    plt.show()


new_arr = np.ones((10,20,20))
#print(new_arr[0])
new_arr_pad=np.pad(new_arr[0],(5,5),mode="constant",constant_values=(0,0))
#print(new_arr_pad)
plt.imshow(new_arr_pad)
pre_shape=picture_array[0].shape
last_shape=speckle_image.shape
#print(last_shape)
padded_picture_array = np.zeros((len(picture_array),last_shape[0],last_shape[1]))
new_shape = [int((last_shape[0]-pre_shape[0])/2),int((last_shape[1]-pre_shape[1])/2)]
#print(np.iinfo(np.int32).min)
print(picture_array.dtype)
for ix in range(len(picture_array)):
    padded_picture_array[ix,:,:] = np.pad(picture_array[ix],(new_shape[0],new_shape[1]),mode="constant",constant_values=(0,0))
    plt.imshow(padded_picture_array[ix])
    plt.show()

#try to deconvolve the 3e by using point scaning psf and point scan images
print(lymosome_str)
print(lymosome_dir)
ps_pos =[i for i in range(len(lymosome_dir)) if "point_scanning" in lymosome_dir[i]]
point_scan=ski.io.imread(lymosome_str+"/"+lymosome_dir[ps_pos[0]])
psf_point_scan = ski.io.imread(lymosome_str+"/"+lymosome_dir[ps_pos[1]])
construction=np.zeros((len(point_scan[0]),point_scan[0].shape[0],point_scan[0].shape[0]))
for ix in enumerate(point_scan):
    stack=ski.restoration.wiener(ix[1]/np.sum(ix[1]),psf_point_scan[ix[0]]/np.sum(psf_point_scan[ix[0]]),balance=0.2)
    construction[ix[0],:,:] = stack
    
    plt.imshow(stack/np.sum(stack))
    plt.gcf().savefig(save_dir+"point_scan_deconvolution_"+str(ix[0])+".png",dpi=400)
    #plt.colorbar()
    plt.show()
    plt.imshow(psf_point_scan[ix[0]]/np.sum(psf_point_scan[ix[0]]))
    plt.gcf().savefig(save_dir+"point_scan_psf_"+str(ix[0])+".png",dpi=400)
    #plt.colorbar()
    plt.show()
    plt.imshow(ix[1]/np.sum(ix[1]))
    plt.gcf().savefig(save_dir+"point_scan_"+str(ix[0])+".png",dpi=400)
    #plt.colorbar()
    plt.show()
