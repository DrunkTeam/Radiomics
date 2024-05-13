import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import nibabel as nib

#show slices from NIFTI
def show_in_row(image_path, mask_path):

    # Get the number of slices in the image
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    num_slices_image = image.GetSize()[2]
    num_slices_mask = mask.GetSize()[2]

    # Create a plot with multiple subplots
    fig, axs = plt.subplots(1, num_slices_image, figsize=(20, 20))

    # Loop over each slice in the image
    for i in range(num_slices_image):
        # Extract the slice from the image
        slice_image = sitk.GetArrayFromImage(image)[i, :, :]

        # Display the slice in the subplot
        axs[i].imshow(slice_image, cmap='gray')
        axs[i].axis('off')


    fig, axs = plt.subplots(1, num_slices_image, figsize=(20, 20))
    for i in range(num_slices_mask):
        # Extract the slice from the image
        slice_mask = sitk.GetArrayFromImage(mask)[i, :, :]

        # Display the slice in the subplot
        axs[i].imshow(slice_mask, cmap='gray')
        axs[i].axis('off')
    #fig.savefig("MRI-mask.png")
    # Show the plot
    plt.show()
