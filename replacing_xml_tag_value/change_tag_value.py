#*************** THINGS TO DO BEFORE RUNNING THIS CODE OR FILE ****************
#* first, saperate the annotation's files in a directory		      *
#* then change the location there in os.chdir(path)			      *
#* add the value to modified and its replacing value			      *
#* run the code								      *
#*************** THINGS TO DO BEFORE RUNNING THIS CODE OR FILE ****************

# importing necessary modules
import xml.etree.ElementTree as ET
import os, time

# start time to count execution time
start_time = time.time()

# change this path where you have kept your annotations files
os.chdir("/media/dinesh/Disk_Data/Rams_WFH/Mahindra_Old/train85/annotation_85/hood_latch_2")

# get list from xml contained folder
files = os.listdir()

# blank list to count total modified files
file_count = []

# use for loop to access all files
for file in files:
# exemption handling to avoid file not found error
    try:
	# validate files with extention
        if file.endswith('.xml'):
	# parsing files one by one 
            tree = ET.parse(file)
	# get root of the file
            root = tree.getroot()
	# find object tag and iterate over it
            for i in root.iter('object'):
	# iterating over name tag in xml file
                for sub in i.iter('name'):
	# validate the exist value or text of name tag and change the value/ text of <name> tag
                    if sub.text == 'pto_guard_inside warning_sticker':
                        sub.text = 'pto_guard_inside_warning_sticker'
	# using write method to make changes in xml file
            tree.write(file)
            print("Modified",file)
# handle exemption or error of file not found
    except(FileNotFoundError):
        break
# adding file name to calculate total modified files
    file_count.append("Modified "+str(file))

# printing total modified files length
print("\nTotal modified files :",len(file_count))
    
# end time of execution
end_time = time.time()

# find the difference
exe_time = round((end_time-start_time)/60,2)

# print the execution time of code 
print("\nExecution_time :", exe_time)

