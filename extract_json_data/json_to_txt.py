#importing necessary modules
import csv,os,json,time
import pandas as pd

# user input to access 
input_path = input("enter the full path to access json files contained folder:")

# storing starting time in variable
start_time = time.time()

# change the path of directory
ch_path = os.chdir(input_path)
# get current working directory path
c_path = str(os.getcwd()+"/")

# now get all the json files in a list
json_files = os.listdir(ch_path)
# assign class name to a variable
c_id = 1
# use dict to store img name and their cordinates
dict1 = {}

# get access to all the files with for loop and read and write into files
for i in json_files:
    # validate the path of the file
    if i.endswith('.jpg.json'):
        # read the data from a json file using json.load method
        with open(i,"r") as json_file:
            new_file = json.load(json_file)
            # now access the json by key names 
            for x in new_file['objects']:                              
                # now use dictionary to sotre in key, value method
                if c_path+i[:-5] in dict1:                    
                    dict1[c_path+i[:-5]]+= str(x['points']['exterior'][0][0:][0])+","+str(x['points']['exterior'][0][0:][1])+","+str(x['points']['exterior'][1][0:][0])+","+str(x['points']['exterior'][1][0:][1])+","+str(c_id)+" "
                else:                    
                    dict1[c_path+i[:-5]] = str(x['points']['exterior'][0][0:][0])+","+str(x['points']['exterior'][0][0:][1])+","+str(x['points']['exterior'][1][0:][0])+","+str(x['points']['exterior'][1][0:][1])+","+str(c_id)+" "

# create a output file with write method                 
output_file = open("../output.txt","w")     

# now store key,values in file with write method  
for name,cord in dict1.items():
    output_file.write(name+" "+cord+"\n")

# close the file
output_file.close()


# end time of execution
end_time = time.time()
# find the execution time
ex_time = end_time-start_time
# print execution times 
print("execution time(in sec):",ex_time)

