def removeComments(inputFileName, outputFileName):

    input = open(inputFileName, "r")
    output = open(outputFileName, "w")

    output.write(input.readline())

    for line in input:
        if not line.lstrip().startswith("# In["):
            output.write(line)

    input.close()
    output.close()

if __name__ == "__main__":
    removeComments('02_07_2_unet_masked_6_colour_crop_facies.py','02_07_3_unet_masked_6_colour_crop_facies.py')
    removeComments('02_07_2_unet_masked_6_colour_crop_facies.py', '02_07_4_unet_masked_6_colour_crop_facies.py')
