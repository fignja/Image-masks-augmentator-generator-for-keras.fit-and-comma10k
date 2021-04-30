import numpy as np
import cv2
import random

#this class can be used with keras.fit for comma ai 10k dataset augmentation
class ImageMaskAUGGenerator:
	def __init__(self, imgslist, maskslist, batchSize=5, 
				img_width=96, img_height=64, rot_angle=5, mask_ratio=0.25, zoom_ratio=2.0, horisontal_flip=True, interior_crop=True, 
				 LAB=False, GREY=False, GaussBlur=False, mean_extraction=False, equalize=False, interi=cv2.INTER_AREA,interm=cv2.INTER_NEAREST):
		
		#if LAB=False and GREY=False then BGR returned
		# store the returning batch size, preprocessing and data augmentation parameters,
		# whether or not the labels should be binarized, along with
		
		self.batchSize = batchSize
		#list of images and corresponding masks names
		self.imgslist = imgslist
		self.maskslist = maskslist
		
	
		# store the target image width, height that will be returned
		
		self.img_width = img_width
		self.img_height = img_height
		# returned mask to image ratio.  0.5 means that mask will be two times smaller then image size
		self.mask_ratio=mask_ratio
		
		#maximum zoom during augmentation zoom_ratio=1 mean that no zoom allowed
		# zoom_ratio=2 mean than image can be randomly croped by sides on some part of it size, but only if self.img_height/len(image)<1
		self.zoom_ratio=zoom_ratio

		#add random rotate for image and mask
		self.rot_angle=rot_angle
		
		#activates random horisontal flip for image and mask
		self.horisontal_flip=horisontal_flip
		#car interior class is removed from images,masks
		self.interior_crop=interior_crop


		#return image in lab
		self.LAB = LAB
		#return image in grey
		self.GREY = GREY
		#return blured image
		self.GaussBlur=GaussBlur
		
		#mean extraction for nn
		self.mean_extraction = mean_extraction
		#equalize light histogram image profile
		self.equalize = equalize


		 # interpolation method used when resizing image and mask
		self.interi = interi
		self.interm = interm








	def Mean(self, image):
		# split the image into its respective Red, Green, and Blue
		# channels
		# subtract the means for each channel
		# merge the channels back together and return the image		
		(B, G, R) = cv2.split(image.astype("float32"))
		
		R -= np.mean(R)
		G -= np.mean(G)
		B -= np.mean(B)
		
		return cv2.merge([B, G, R])

	def ColorTransform(self, image):
		#transform image to required color scheme, adds blur, image=image/255  and etc.
		if self.GREY:
			image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
			if self.equalize:
				image=cv2.equalizeHist(image)
		else:
			if self.LAB or self.equalize:
				LABimage=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
				if self.equalize:
					channels=cv2.split(LABimage)
					channels[0]=cv2.equalizeHist(channels[0])
					image=cv2.merge(channels)
				if not self.LAB:
					image=cv2.cvtColor(image,cv2.COLOR_LAB2BGR)
			
		
		if self.GaussBlur:
			image= cv2.GaussianBlur(image,(3,3),cv2.BORDER_DEFAULT)

		
		image=image/255

		if self.mean_extraction:
			image=self.Mean(image)
		
		return image






	def generator(self, passes=np.inf):
		# generator loop over the dataset
		while True:
			# initialize the list of loaded images and masks
			
			imagesbatch=[]
			masksbatch=[]


			for counter in np.arange(0, self.batchSize):
				# get names list of random images and corresponding masks to load and load them
				
				random_image_number=random.randint(0, len(self.imgslist)-1)
				imagesbatch.append(cv2.imread(self.imgslist[random_image_number]))
				masksbatch.append(cv2.imread(self.maskslist[random_image_number]))
								

			# initialize the list of processed images
			procImages = []
			procMasks = []
			# loop over the images
			for image,mask in zip(imagesbatch,masksbatch):


				# load and process every image and mask with iprocessor (and makes augmentation)
				image,mask,bool = self.iprocess(image,mask)
				# if image mask still contains car interior after crop then load another image and mask
				while(bool and self.interior_crop):
					ar=random.randint(0, len(self.imgslist)-1)
					image=cv2.imread(self.imgslist[ar])
					mask=cv2.imread(self.maskslist[ar])
					image,mask,bool = self.iprocess(image,mask)
									
				image=self.ColorTransform(image)
				# update the array of processed images and masks
				procImages.append(image)
				procMasks.append(mask)
			# update the images array to be the processed
			# images
			imagesbatch = np.array(procImages).astype('float32')
			masksbatch = np.array(procMasks).astype('float32')
			
				
			# yield a tuple of images and masks
			yield(imagesbatch, masksbatch)
				


	def iprocess(self,image,mask):



		#cv2.imshow("loaded image", image)
		#key = cv2.waitKey(0)	
		#cv2.imshow("loaded mask", mask)
		#key = cv2.waitKey(0)





		#crop car interior if bool enabled
		if self.interior_crop:
			image,mask=self.img_mask_interior_crop( image, mask)



		#cv2.imshow("croped interior image", image)
		#key = cv2.waitKey(0)	
		#cv2.imshow("croped interior mask", mask)
		#key = cv2.waitKey(0)



		
		#angle rotation activation if self.rot_angle>0
		if self.rot_angle>0:
			
			image,mask=self.angle_img_mask_aug( image, mask)


		#cv2.imshow("rotate image <± rot_angle", image)
		#key = cv2.waitKey(0)	
		#cv2.imshow("rotate mask <± rot_angle", mask)
		#key = cv2.waitKey(0)


		
		#crop/zoom augmentation part
		if self.zoom_ratio>1:
			crop_ratioh=(len(image)/self.img_height)
			crop_ratiow=(len(image[0])/self.img_width)


			crop_ratio=min(crop_ratioh,crop_ratiow)
			crop_ratioh=(crop_ratioh/crop_ratio-1)*crop_ratio
			crop_ratiow=(crop_ratiow/crop_ratio-1)*crop_ratio
			if crop_ratio>1 and self.zoom_ratio>1:
				crop_ratio=min(crop_ratio,self.zoom_ratio)-1
				
				crop_ratio=crop_ratio*random.random()
				crop_ratioh+=crop_ratio
				crop_ratiow+=crop_ratio




			#random but narrowed horisontal crop distribution for image and mask ( hor = randomy from 0.3 to 0.7 (on one side) and (1-hor)= from 0.7 to 0.3 on another) 
			hor=0.3+0.4*random.random()
			#random vertical crop distribution for image and mask crop from 0 to 1
			ver=random.random()
		
			#random crop with zoom_ratio apply for image and mask
			croptop=int(self.img_height*crop_ratioh*ver)
			cropleft=int(self.img_width*crop_ratiow*hor)
			cropbottom=int(self.img_height*crop_ratioh*(1-ver))
			cropright=int(self.img_width*crop_ratiow*(1-hor))		


			image=image[max(0,croptop):min(image.shape[0]-cropbottom,image.shape[0]), max(0,cropleft):min(image.shape[1]-cropright,image.shape[1])]
			mask=mask[max(0,croptop):min(mask.shape[0]-cropbottom,mask.shape[0]), max(0,cropleft):min(mask.shape[1]-cropright,mask.shape[1])]
		


		#cv2.imshow("random zoom/crop image", image)
		#key = cv2.waitKey(0)	
		#cv2.imshow("random zoom/crop mask", mask)
		#key = cv2.waitKey(0)





		#this part of code used for better nn results presentation
		mask=self.mask_recolor(mask)		


		#cv2.imshow("recolored mask", mask)
		#key = cv2.waitKey(0)


		# resize image and mask to the final size
		image=cv2.resize(image, (self.img_width , self.img_height),interpolation=self.interi)
		mask=cv2.resize(mask, (int(self.img_width*self.mask_ratio), int(self.img_height*self.mask_ratio)),interpolation=self.interi)







		#fix mask pixel classes colors
		mask=self.mask_border_fix(mask)	
		

		
		#cv2.imshow("ROIimage", image)
		#key = cv2.waitKey(0)	
		#cv2.imshow("ROImask", cv2.resize(mask, (int(self.width), int(self.height)),interpolation=self.interi))
		#key = cv2.waitKey(0)
		

		#last mask check for car interior parts on it if interior_crop=True. If it was not fully cropped from wirings or car wipers bool=True and other image will be loaded in generator images batch
		
		bool=False
		if self.interior_crop:
			interior_mask=np.zeros_like(mask[:,:,0])
			interior_mask[(mask[:,:,0]==0)&(mask[:,:,1]==255)&(mask[:,:,2]==0)]=1
			if interior_mask.any():
				bool=True



		#random horizontal flip
		if self.horisontal_flip:
			if random.randint(0, 1):
				image=cv2.flip(image,1)
				mask=cv2.flip(mask,1)
		
		
		
		#cv2.imshow("final image", image)
		#key = cv2.waitKey(0)	
		#cv2.imshow("final mask", mask)
		#key = cv2.waitKey(0)

	

		#mask to class labels encoder for nn mask_labeled.shape=[maskk,maskw,[road,marks,not driveble,cars,car interior]]

		#if interior_crop=true then mask_labeled.shape=[maskk,maskw,[road,marks,not driveble,cars]]
		mask_labeled=self.mask_to_label_encoder(mask)




		#print(maskkat.shape)
		#transform mask to image
		mask_back=self.labels_to_mask_encoder(mask_labeled)

		#cv2.imshow("mask back after labels", mask)
		#key = cv2.waitKey(0)









		return (image,mask_labeled,bool)



	def mask_recolor(self, mask):
		#for better video nn visualisation and simpler borders classification after mask resize
		#road is black
		mask[(mask[:,:,2]==64),:]=[0,0,0]
		
		#marks are white
		mask[(mask[:,:,2]==255),:]=[255,255,255]
		
		#no drive parts is blue
		mask[(mask[:,:,2]==128),:]=[255,0,0]	
		#car interiour is red
		mask[(mask[:,:,2]==204),:]=[0,0,255]
		
		
		
		



		#outside cars is green
		
		mask[(mask[:,:,0]==102),:]=[0,255,0]
		
		return mask





	def img_mask_interior_crop(self, image, mask):
		croptop=0
		cropbottom=0
		cropleft=0
		cropright=0

		for pixnum in range(0,int(len(image)/2)):

			if (mask[pixnum,int(len(image[0])/2),2] != 204) and (mask[pixnum,int(len(image[0])/3),2] != 204) and (mask[pixnum,int(len(image[0])*2/3),2] != 204):
				break
			croptop=pixnum


		for pixnum in range((len(image)-1),int(len(image)/2),-1):
			if (mask[pixnum,int(len(image[0])/2),2] != 204) and (mask[pixnum,int(len(image[0])/3),2] != 204) and (mask[pixnum,int(len(image[0])*2/3),2] != 204):
				break
			cropbottom+=1
		
		
		for pixnum in range(0,int(len(image[0])/2)):
			if (mask[int(len(image)/2),pixnum,2] != 204) and (mask[int(len(image)/3),pixnum,2] != 204) and (mask[int(len(image)*2/3),pixnum,2] != 204):
				break
			cropleft=pixnum

		for pixnum in range(int(len(image[0])-1),int(len(image[0])/2),-1):
			if (mask[int(len(image)/2),pixnum,2] != 204) and (mask[int(len(image)/3),pixnum,2] != 204) and (mask[int(len(image)*2/3),pixnum,2] != 204):
				break
			
			cropright+=1
		
		#fast working crutch
		if croptop>0 or cropbottom>0  or cropleft>0 or cropright>0  :
			croptop+=5
			cropbottom+=5
			cropleft+=5
			cropright+=5		

		#if crop is too big no crop happens
		#image will be removed with interior check in iprocess later
		if ((croptop+cropbottom)>(len(image)/2) or (cropleft+cropright)>(len(image[0])/2)) :
			croptop=0
			cropbottom=0
			cropleft=0
			cropright=0
			



		image=image[max(0,croptop+1):min(image.shape[0]-cropbottom-1,image.shape[0]), max(0,cropleft+1):min(image.shape[1]-cropright-1,image.shape[1])]
		
		mask=mask[max(0,croptop+1):min(mask.shape[0]-cropbottom-1,mask.shape[0]), max(0,cropleft+1):min(mask.shape[1]-cropright-1,mask.shape[1])]
		
		
		return image,mask

	
	def mask_border_fix(self, mask):	
		#return fixed colors for classes after mask resize
		mask[(mask[:,:,0]>50)&(mask[:,:,1]>50)&(mask[:,:,2]>50),:]=[255,255,255]
		mask[(mask[:,:,0]<230)&(mask[:,:,1]<230)&(mask[:,:,2]>50),:]=[0,0,255]
		#mask[(mask[:,:,0]<230)&(mask[:,:,1]>30)&(mask[:,:,2]<230),:]=[0,255,0]
		mask[(mask[:,:,0]>50)&(mask[:,:,1]<230)&(mask[:,:,2]<230),:]=[255,0,0]
		mask[(mask[:,:,0]<255)&(mask[:,:,1]<255)&(mask[:,:,2]<255),:]=[0,0,0]
		return mask
	

	def mask_to_label_encoder(self, mask):
		maskkat1=np.zeros_like(mask[:,:,0])
		maskkat1[(mask[:,:,0]==0)&(mask[:,:,1]==0)&(mask[:,:,2]==0)]=1
				
		

		maskkat2=np.zeros_like(mask[:,:,0])
		maskkat2[(mask[:,:,0]==255)&(mask[:,:,1]==255)&(mask[:,:,2]==255)]=1

		maskkat3=np.zeros_like(mask[:,:,0])
		maskkat3[(mask[:,:,0]==255)&(mask[:,:,1]==0)&(mask[:,:,2]==0)]=1
		
		
		maskkat4=np.zeros_like(mask[:,:,0])		
		maskkat4[(mask[:,:,0]==0)&(mask[:,:,1]==255)&(mask[:,:,2]==0)]=1

		maskkat5=np.zeros_like(mask[:,:,0])
		maskkat5[(mask[:,:,0]==0)&(mask[:,:,1]==0)&(mask[:,:,2]==255)]=1

		if self.interior_crop:
			maskkat=np.stack([maskkat1,maskkat2,maskkat3,maskkat4], axis = 2)
		else:
			maskkat=np.stack([maskkat1,maskkat2,maskkat3,maskkat4,maskkat5], axis = 2)
		return maskkat


	def labels_to_mask_encoder(self,mask_labeled):
		slovar={0:[0,0,0],1:[255,255,255],2:[255,0,0],3:[0,255,0],4:[0,0,255]}
		pix_class=np.argmax(mask_labeled,axis=2)
		pix_class=np.reshape(pix_class,(len(mask_labeled),len(mask_labeled[0])))
		colored_mask=np.zeros((len(mask_labeled),len(mask_labeled[0]),3))

		for i in range(0,len(mask_labeled)):
			for j in range(0,len(mask_labeled[0])):
				colored_mask[i,j,:]=slovar[int(pix_class[i,j])]
		

		#cv2.imshow("mask2img", cv2.resize(mask, (int(self.width), int(self.height)),interpolation=self.interi))
		#key = cv2.waitKey(0)






		return colored_mask

	def angle_img_mask_aug(self, image, mask):
		#turn image and mask on random angle
		angle=int((random.random()-0.5)*2*self.rot_angle)
		rows=len(image)
		cols=len(image[0])
		M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),angle,1)
		image = cv2.warpAffine(image,M,(cols,rows),flags = self.interm)
		mask = cv2.warpAffine(mask,M,(cols,rows),flags = self.interm)
	
		angle=abs(angle)
		#crop black triangles
		croptop=angle*10
		cropbottom=angle*10
		cropleft=angle*9
		cropright=angle*9
		


		
		image=image[max(0,croptop):min(image.shape[0]-cropbottom,image.shape[0]),max(0,cropleft):min(image.shape[1]-cropright,image.shape[1])]
		mask=mask[max(0,croptop):min(mask.shape[0]-cropbottom,mask.shape[0]),max(0,cropleft):min(mask.shape[1]-cropright,mask.shape[1])]
		return image,mask








