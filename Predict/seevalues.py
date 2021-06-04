import numpy as np                                                              #Array handling
import matplotlib.pyplot as plt                                                 #Plotting
import math



usernum = input("User number: ")
daynum = input("Day number: ")

a = np.loadtxt(open("../Raw Data Files/User " + usernum + "/emotIOn_U" + usernum + "_D" + daynum + ".csv"), delimiter = ",", skiprows = 1) #Take Data from file on src path.

addmore = True

while addmore:
    noans = True
    cont = "y"
    while noans:
        cont = input("Add more data to the graphs? [y/n]: ")
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


paramlist=['X Accel','Y Accel','Z Accel','Temperature','Light','Humidity','Proximity','Pressure','Altitude','DewPoint']

plt.rcParams.update({'font.size': 25})
for param in range(a.shape[1] - 1):
    noans = True
    cont = "y"
    while noans:
        cont = input("See " + paramlist[param] + " graph? [y/n]: ")
        if cont[0] == "y" or cont[0] == "Y" or cont[0] == "n" or cont[0] == "N":
            noans = False
        else:
            print("Invalid answer. Try again.")
    if cont[0] == "n" or cont[0] == "N":
        continue
    plt.figure(param + 1)
    parammin = a[0, param]
    parammax = a[0, param]
    showprogress = 21
    progress = 0
    lastprogress = 0
    progressbar = '['
    for _ in range(showprogress - 1):
        progressbar = progressbar + '.'
    progressbar = list(progressbar)
    print('\r' + "".join(progressbar) + ']', end = ' ')
    for times in range(a.shape[0]):
        progress = math.floor(times / (a.shape[0]) * showprogress)
        if progress > lastprogress:
           lastprogress = progress
           progressbar[progress] = '*'
           print('\r' + "".join(progressbar), end = ']')
        if a[times, param] < parammin:
            parammin = a[times, param]
        if a[times, param] > parammax:
            parammax = a[times, param]
        hr = times * 1.5 / 3600
        if a[times, 10] == 0:
            plt.plot(hr, a[times, param], 'ko')
        elif a[times, 10] == 1:
            plt.plot(hr, a[times, param], 'bo')
        elif a[times, 10] == 2:
            plt.plot(hr, a[times, param], 'go')
        elif a[times, 10] == 3:
            plt.plot(hr, a[times, param], 'ro')
        else:
            plt.plot(hr, a[times, param], 'yo')
    print('\r' + "".join(progressbar))
    plt.axis([0, a.shape[0] * 1.5 / 3600, parammin, parammax])
    plt.title('User ' + usernum + ': D' + daynum)
    #plt.legend(['N','HH','LH','LL','HL'])
    plt.ylabel(paramlist[param])
    plt.xlabel('Time (hours)')
    plt.show()


"""
plt.rcParams.update({'font.size': 25})
for param in range(a.shape[1] - 1):
    noans = True
    cont = "y"
    while noans:
        cont = input("See " + paramlist[param] + " graph? [y/n]: ")
        if cont[0] == "y" or cont[0] == "Y" or cont[0] == "n" or cont[0] == "N":
            noans = False
        else:
            print("Invalid answer. Try again.")
    if cont[0] == "n" or cont[0] == "N":
        continue
    plt.figure(param + 1)
    parammin = a[0, param]
    parammax = a[0, param]
    showprogress = 21
    progress = 0
    lastprogress = 0
    progressbar = '['
    for _ in range(showprogress - 1):
        progressbar = progressbar + '.'
    progressbar = list(progressbar)
    print('\r' + "".join(progressbar) + ']', end = ' ')
    for times in range(a.shape[0]-1):
        progress = math.floor(times / (a.shape[0]-1) * showprogress)
        if progress > lastprogress:
           lastprogress = progress
           progressbar[progress] = '*'
           print('\r' + "".join(progressbar), end = ']')
        if a[times+1, param] < parammin:
            parammin = a[times+1, param]
        if a[times+1, param] > parammax:
            parammax = a[times+1, param]
        hr = times * 1.5 / 3600
        if a[times+1, 10] == 0:
            plt.plot([hr, hr+1.5/3600], [a[times, param], a[times+1,param]], 'k-')
        elif a[times+1, 10] == 1:
            plt.plot([hr, hr+1.5/3600], [a[times, param], a[times+1,param]], 'b-')
        elif a[times+1, 10] == 2:
            plt.plot([hr, hr+1.5/3600], [a[times, param], a[times+1,param]], 'g-')
        elif a[times+1, 10] == 3:
            plt.plot([hr, hr+1.5/3600], [a[times, param], a[times+1,param]], 'r-')
        else:
            plt.plot([hr, hr+1.5/3600], [a[times, param], a[times+1,param]], 'y-')
    print('\r' + "".join(progressbar))
    plt.axis([0, a.shape[0] * 1.5 / 3600, parammin, parammax])
    plt.title('User ' + usernum + ': D' + daynum)
    #plt.legend(['N','HH','LH','LL','HL'])
    plt.ylabel(paramlist[param])
    plt.xlabel('Time (hours)')
    plt.show()

"""
