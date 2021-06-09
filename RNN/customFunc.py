import numpy as np


def inputRawData(msg, purpose):
   print(msg)
   usernum = input("User number: ")
   daynum = input("Day number: ")

   a = np.loadtxt(open("../Raw Data Files/User " + usernum + "/emotIOn_U" + usernum + "_D" + daynum + ".csv"), delimiter = ",", skiprows = 1) #Take Data from file on src path.

   addmore = True

   while addmore:
       noans = True
       cont = "y"
       while noans:
           cont = input("Add more data to the " + purpose + "? [y/n]: ")
           if cont[0] == "y" or cont[0] == "Y" or cont[0] == "n" or cont[0] == "N":
               noans = False
               if cont[0] == "n" or cont[0] == "N":
                   addmore = False
           else:
               print("Invalid answer. Try again.")
       if addmore == False:
           continue
       usernum = input("User number: ")
       daynum = input("Day number: ")
       a1 = np.loadtxt(open("../Raw Data Files/User " + usernum + "/emotIOn_U" + usernum + "_D" + daynum + ".csv"), delimiter = ",", skiprows = 1) #Take Data from file on src path.
       a0 = a
       a = np.concatenate((a0, a1))

   return a
