import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import transform
import tifffile

plt.rcParams['figure.figsize'] = (16, 9) # What is this parameter?

class post_process():
    
    #Two gloabal helper functions#
    global _get_normal_distribution
    global _generate_random_bkg
    
    
    def __init__(self, image_path, file_num, defect_list):
        
        #Input parameters of this class:
        #string: image_path, the path of you images and labels
        #int: file_num
        #list: defect_list, list of str, includes the defect labels you need
        
        self.image_path = image_path
        self.file_num = file_num
        self.defect_list = defect_list
        
        
    def _get_normal_distribution(param):
        '''
        Get positive/negative normal distribution from a tuple
        Imput: tuple: (mean, std)
        return: float: normal_distribution_number
        '''
        num = np.random.normal(param[0],param[1])
        pos = np.random.rand()
        
        if pos>0.5:
            return num
        else:
            return -num
    
    
    def read_image_and_label(self):
        
        '''
        Put image and its defect into one image stack for following process
        self.image_stack: np.array shape = (num_image, x,y,num_defects+1)
        '''
        
        all_files = os.listdir(self.image_path)
        image_list = []
        for image_file in all_files:
            if image_file[:5] == 'Image':
                image_list.append(image_file)
        img = plt.imread(self.image_path+image_list[0])
        #get image size
        x,y = np.shape(img)
        image_stacks = np.zeros((self.file_num,x,y,len(self.defect_list)+1))
        
        for i in range(self.file_num):
            image_name = image_list[i]
            temp_image = plt.imread(self.image_path+image_name)
            image_stacks[i,:,:,0] = temp_image
            for j in range(len(self.defect_list)):
                temp_defect_name = self.defect_list[j]+'_'+image_name[5:]
                temp_defect_image = plt.imread(self.image_path+temp_defect_name)
                image_stacks[i,:,:,j+1] = temp_defect_image
        self.image_stacks = image_stacks.copy()
    
    def add_horizontal_sheer(self, sheer_rate):
        
        """
        Add horizental sheer to simulate horizental sample draft during taking the image in STEM
        sheer_rate = (sheer_mean,sheer_std) in guassian distribution
        single image has a linear sheer, while in different images, sheer rates are different
        """
        num_image, x, y, num_defects = np.shape(self.image_stacks)
        for i in range(num_image):
            temp_stack = self.image_stacks[i,:,:,:]
            shear = _get_normal_distribution(sheer_rate)
            afine_tf = transform.AffineTransform(shear = shear)
            modified = transform.warp(temp_stack, inverse_map=afine_tf)
            self.image_stacks[i,:,:,:] = modified
        
        
        
    def add_vertical_constrain(self, constrain_rate):
        '''
        Add vertical constrain to simulate vertical sample draft during taking the image in STEM
        constrain_rate = (constrain_mean,constrain_std) in guassian distribution
        single image has a constant constrain rate, while in different images, constrain rate are different
        '''
        
        num_image, x, y, num_defects = np.shape(self.image_stacks)
        for i in range(num_image):
            temp_stack = self.image_stacks[i,:,:,:]
            constrain = _get_normal_distribution(constrain_rate)+1
            afine_tf = transform.AffineTransform(scale = (1,constrain))
            modified = transform.warp(temp_stack, inverse_map=afine_tf)
            self.image_stacks[i,:,:,:] = modified
    
    
    def rotate(self, degree):
        
        '''
        rotate the images into the same degree of real images, if there are multiple orientations in real images, 
        you can genreate different degrees in training set
        '''
        
        num_images,x,y,num_defects = np.shape(self.image_stacks)
        for i in range(num_images):
            for j in range(num_defects):
                 self.image_stacks[i,:,:,j] = transform.rotate(self.image_stacks[i,:,:,j],degree)
        pass
    
    def crop(self, target_x, target_y):
        
        '''
        The raw images is a bit larger because the sheer and rotation will cause some blank in the image 
        Crop the center image to shape(target_x, target_y) and avoid the blank
        '''
        
        num_image, x, y, num_defects = np.shape(self.image_stacks)
        cropped_x = abs(x-target_x)//2
        cropped_y = abs(y-target_y)//2
        self.image_stacks = self.image_stacks[:,cropped_x:-cropped_x,cropped_y:-cropped_y,:]
        pass
    
    def change_brightness_and_contrast(self, target_mean, target_std):
        
        '''
        Change the B/C of simulate images to the real image
        You can get the target_mean and target_std from a real image
        
        '''
        
        num_images,x,y,num_defects = np.shape(self.image_stacks)
        for i in range(num_images):
            temp_image = self.image_stacks[i,:,:,0]
            temp_image *= target_std/temp_std
            temp_mean = np.mean(temp_image)
            temp_image += target_mean-temp_mean
            self.image_stacks[i,:,:,0] = temp_image
        sel = self.image_stacks<0
        self.image_stacks[sel] = 0
        sel1 = self.image_stacks>255
        self.image_stacks[sel1] = 255

    def add_gaussian_noise(self, gaussian_rate):
        
        '''
        add gaussian noise in the image
        '''
        
        for i in range(self.file_num):
            self.image_stacks[i,:,:,0] += np.random.normal(gaussian_rate[0],gaussian_rate[1],np.shape(self.image_stacks[i,:,:,0]))
        sel = self.image_stacks<0
        self.image_stacks[sel] = 0
        sel1 = self.image_stacks>255
        self.image_stacks[sel1] = 255
        
    def _generate_random_bkg(bkg_stack):
        
        num, x, y = np.shape(bkg_stack)
        random_weight = np.random.rand(9)
        random_weight = random_weight/np.sum(random_weight)
        bkg_image = np.zeros((x,y))
        for i in range(num):
            bkg_image += random_weight[i]*bkg_stack[i,:,:]
        return bkg_image
            
    
    
    def add_bkg(self, bkg_stack):
        
        '''
        Add random, ununiform background on the images,
        you can extract bkg through high pass filter of STEM image
        '''
        
        _,x,y,_ = np.shape(self.image_stacks)
        for i in range(self.file_num):
            bkg_image = _generate_random_bkg(bkg_stack)
            self.image_stacks[i,:,:,0] += bkg_image[:x,:y]
        pass
    

    
    def save_as_image(self, save_path, rot):
        
        '''
        Save images stacks as image
        '''
        if os.path.exists(save_path) == False:
            os.makedirs(save_path)
        num_images,x,y,num_defects = np.shape(self.image_stacks)

        for i in range(num_images):
            temp_path = save_path+str(rot)+'deg_{}/'.format(i)
            os.makedirs(temp_path)
            # replace 'input.tiff' = 'training_image{}'.format(i)+'.tiff'
            tifffile.imsave(temp_path+'input.tiff',self.image_stacks[i,:,:,0].astype('uint8'))
            for j in range(len(defect_list)):
                # replace 'label_'+defect_list[j]+'.tiff' = 'image{}_label'.format(i)+defect_list[j]+'.tiff'
                tifffile.imsave(temp_path+'label_'+defect_list[j]+'.tiff',self.image_stacks[i,:,:,j+1].astype('uint8'))
    

    def alt_save_as_image(self, save_path):

        '''
        Temporary method to save image
        '''
        if os.path.exists(save_path) == False:
            os.makedirs(save_path)
        num_images, x, y, num_defects = np.shape(self.image_stacks)

        for i in range(num_images):
            temp_path = save_path
            # os.makedirs(temp_path)
            tifffile.imsave(temp_path+'input.tiff', self.image_stacks[i,:,:,0].astype('uint8'))
            for j in range(len(defect_list)):
                tifffile.imsave(temp_path+'label_'+defect_list[j]+'.tiff', self.image_stacks[i,:,:,j+1].astype('uint8'))


    
    def save_as_npy_files(self, save_path):
        
        '''
        Save images files as .npy files, it is easy to load into deep learning code
        '''
        
        if os.path.exists(save_path) == False:
            os.makedirs(save_path)
        
        np.save(save_path+'images_files.npy',self.image_stacks[:,:,:,0])
        for i in range(len(self.defect_list)):
            np.save(save_path+defect_list[i]+'_files.npy',self.image_stacks[:,:,:,i+1])	


# defect list
defect_list = ['1vacancy', '2vacancy']

# Return image data from TIFF file(s) as numpy array
# bkg_stack = tifffile.imread('./Bkg_stack.tif')

# operation

# Alternative image path (absolute)
# path = 'home/sychoi/ResUNet_final/data/MoS2/simulated/20201019_Run1_Images'
# path = 'home/sychoi/git_file_trans/STEM-Deep-Learning/20201019_Run2_Images/'


path = '/data/aims/sychoi/git_file_trans/STEM-Deep-Learning/New_Images/'

dir1 = 'a74_52mrad_30pA_16us/'
file_num = 2

path1 = path + dir1
Mypostprocess = post_process(path1, file_num, defect_list)
Mypostprocess.read_image_and_label()
Mypostprocess.add_horizontal_sheer((0.05, 0.025))
Mypostprocess.add_vertical_constrain((0.05, 0.025))

rot = 8
Mypostprocess.rotate(rot)
# Get the bc from real image
# Mypostprocess.change_brightness_and_contrast(self, target_mean, target_std)
# bkg_stack = tiffile.imread('./Bkg_stack.tif')
# Mypostprocess.add_bkg()
Mypostprocess.crop(512, 512)
Mypostprocess.add_gaussian_noise((0, 20))

saveDir = path + 'pa74_52mrad_30pA_16us/'
Mypostprocess.save_as_image(saveDir, rot)


'''
for i in range(15):

    startDir = '../../data/MoS2/train_set2/'
    startPath = os.path.join(startDir + str(i) + '/')

    home = '/home/sychoi/20201022_trainset/'
    path = os.path.join(home + str(i) + '/')

    Mypostprocess = post_process(image_path=startPath, file_num=1, defect_list = defect_list) 
    Mypostprocess.read_image_and_label()
    Mypostprocess.add_horizontal_sheer((0.05, 0.025))
    Mypostprocess.add_vertical_constrain((0.05, 0.025))
    Mypostprocess.crop(400, 400) # original file size = (450, 457) -> (256, 256)
    Mypostprocess.add_gaussian_noise((0, 20))
    Mypostprocess.alt_save_as_image(path)
'''


'''
numDir = 16
startDir = '../../data/MoS2/train_set2/'
home = '/home/sychoi/20201022_trainset/'
for i in range(numDir):
    # Debugging
    path = os.path.join(startDir + str(i) + '/')
    print("Current directory is: ")
    print(path)

    Mypostprocess = post_process(image_path=path, file_num=1, defect_list = defect_list)
    
    # Debug
    print("Reading image and labels...")

    Mypostprocess.read_image_and_label()
    Mypostprocess.add_horizontal_sheer((0.05, 0.025))
    Mypostprocess.add_vertical_constrain((0.05, 0.025))
    # Mypostprocess.rotate(5)
    Mypostprocess.crop(400, 400)
    Mypostprocess.add_gaussian_noise((0, 20))
    # When I have the filters ready, use the below
    # bkg_stack = tiffile.imread('./Bkg_stack.tif')
    # Mypostprocess.add_bkg(bkg_stack)
    pathSave = os.path.join(home + str(i))
    Mypostprocess.alt_save_as_image(pathSave)
    # pathSaveNpy = os.path.join(path + '/saved_npy_files/')
    # Mypostprocess.save_as_npy_files(pathSaveNpy)
'''
