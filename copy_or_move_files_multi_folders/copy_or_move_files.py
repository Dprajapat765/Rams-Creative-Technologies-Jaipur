#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:19:25 2020

@author: Dinesh
"""
#importing necessary modules
import shutil, os, time
from tqdm import tqdm # see progress of loop

# source folder
source = "/home/dinesh/Documents/Py_Codes_Challenges/My_folder/copy_or_move_files_multi_folders/files"

# getting all the files in list
src_dir = os.listdir(source)

# change the path of directory
os.chdir(source)

# destination folder
dest_dir = "/home/dinesh/Documents/Py_Codes_Challenges/My_folder/copy_or_move_files_multi_folders/dest_dir"
if os.path.exists(dest_dir):
	print("Destination Folder Found!")
else:
	os.mkdir(dest_dir)
	print("Destination folder created successfully\n")
# blank list to count the number of folders
total_folder = []

# user input to skip files of the directory
user_input = int(input("enter the number to be skipped between next file:"))

# user input for copying files from the directory
user_input_2 = int(input("How files you want to move (press 0 to stop program): "))

# start time of code
start_time =  time.time()

# use loop to create and move files from source folders to destination folders
for file_name in tqdm(src_dir):  
    # add name to new folders
    full_path = os.path.join(dest_dir,file_name+"_2")
    
    # validate if folder already exists
    if os.path.exists(full_path):
        print("{} folder already exist!".format(full_path[62:]))
    else:
        os.mkdir(full_path)
        print("\n{} folder created successfully!".format(full_path[62:]))

    # get all the files of the directory 
    access_files = os.listdir(os.path.join(source, file_name))

    # getting the path of the directory 
    files_path = os.path.join(source, file_name)

    # using exemption handling for moving the files
    try: 
        # now access the files using the loop 
        for file in access_files:
            # skip the files in the directory
            new_dir = access_files[0::user_input]

            # empty list for moving limited files
            limited_files = []
            # using loop to transfer limited files 
            for i in new_dir:
                if len(limited_files) != user_input_2:
                    limited_files.append(i)

            # input of files less than 0 exit the program
            if user_input_2 <0 or user_input_2 == 0:
                exit()
            # now move the files
            else:
                [shutil.move(os.path.join(files_path,str(x)),full_path) for x in limited_files]
                # print total files moved in directory
                print("{} -\n total files copied: {}".format(full_path[62:],len(limited_files)))
    # handle error 
    except:
        pass

    # add all path to list
    total_folder.append(full_path)

# end time
end_time = time.time()

# count execution time
total_time = round((end_time - start_time)/60,4)

# print total numbers of folders created
print("\nTotal folders created : ",(len(total_folder)))

# print the execution time
print("\n execution time :{} seconds".format(total_time))
