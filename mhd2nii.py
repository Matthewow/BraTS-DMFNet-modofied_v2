import SimpleITK as sitk


def convert(pathin, pathout, number):
    img = sitk.ReadImage(pathin)
    sitk.WriteImage(img, pathout + str(number) +'.nii.gz')


pathIn = 'data/TrainingData/Case'
pathOut = 'data/TrainingData2nii/Case'

for i in range(10):
    path = pathIn + '0' + str(i) + '.mhd'
    convert(path, pathOut, i)

for i in range(10, 50):
    path = pathIn + str(i) + '.mhd'
    convert(path, pathOut, i)
