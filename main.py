############################################# IMPORTING ################################################
# import tkinter as tk
# from tkinter import ttk
# from tkinter import messagebox as mess
# import tkinter.simpledialog as tsd
# import cv2
# import os
# import csv
# import numpy as np
# from PIL import Image
# import pandas as pd
# import datetime
# import time
import csv
import datetime
import os
import time
import tkinter as tk
from tkinter import messagebox as mess
from tkinter import simpledialog as tsd
from tkinter import ttk

import cv2
import numpy as np
import pandas as pd
from PIL import Image

__all__ = [cv2]

############################################# FUNCTIONS ################################################
global new, old, master, nnew


# Checks whether there is a specified path or not.

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


##################################################################################
# Shows time on the screen.
def tick():
    time_string = time.strftime('%H:%M')
    clock.config(text=time_string)
    clock.after(200, tick)


###################################################################################
# Contact details
def contact():
    mess._show(title='Contact us', message="Please contact us on : 'shyamanthulasaipavan@gmail.com' ")


###################################################################################
#Haar Cascade is a machine learning-based approach Detecting objects using Haar Cascade Classifier.
def check_haarcascadefile():
    xists = os.path.isfile("haarcascade_frontalface_default.xml")
    if xists:
        pass
    else:
        mess._show(title='Some file missing', message='Please contact us for help')
        window.destroy()


###################################################################################
#Saving of passwords
def save_pass():
    global key
    assure_path_exists("TrainingImageLabel/")
    exists1 = os.path.isfile("TrainingImageLabel/psd.txt")
    if exists1:
        #Storing entered password into psd.txt file
        tf = open("TrainingImageLabel/psd.txt", "r")
        key = tf.read()
    else:
        #If Entered Password is found.
        master.destroy()
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        if new_pas is None:
            #If newely entered password is null
            mess._show(title='No Password Entered', message='Password not set!! Please try again')
        else:
            #If newely entered password is not null
            tf = open("TrainingImageLabel/psd.txt", "w")
            tf.write(new_pas)
            mess._show(title='Password Registered', message='New password was registered successfully!!')
            return
    op = (old.get())
    newp = (new.get())
    nnewp = (nnew.get())
    if op == key:
        if newp == nnewp:
            txf = open("TrainingImageLabel/psd.txt", "w")
            txf.write(newp)
        else:
            mess._show(title='Error', message='Confirm new password again!!!')
            return
    else:
        mess._show(title='Wrong Password', message='Please enter correct old password.')
        return
    mess._show(title='Password Changed', message='Password changed successfully!!')
    master.destroy()


###################################################################################
#Changing of password to new password.
def change_pass():
    maaster = tk.Tk()
    maaster.geometry("400x160")
    maaster.resizable(False, False)
    maaster.title("Change Password")
    maaster.configure(background="white")
    lbl4 = tk.Label(maaster, text='    Enter Old Password', bg='white', font=('times', 12, ' bold '))
    lbl4.place(x=10, y=10)
    global old
    old = tk.Entry(maaster, width=25, fg="black", relief='solid', font=('times', 12, ' bold '), show='*')
    old.place(x=180, y=10)
    lbl5 = tk.Label(maaster, text='   Enter New Password', bg='white', font=('times', 12, ' bold '))
    lbl5.place(x=10, y=45)
    global new
    new = tk.Entry(maaster, width=25, fg="black", relief='solid', font=('times', 12, ' bold '), show='*')
    new.place(x=180, y=45)
    lbl6 = tk.Label(maaster, text='Confirm New Password', bg='white', font=('times', 12, ' bold '))
    lbl6.place(x=10, y=80)
    global nnew
    nnew = tk.Entry(maaster, width=25, fg="black", relief='solid', font=('times', 12, ' bold '), show='*')
    nnew.place(x=180, y=80)
    cancel = tk.Button(maaster, text="Cancel", command=maaster.destroy, fg="black", bg="red", height=1, width=25,
                       activebackground="white", font=('times', 10, ' bold '))
    cancel.place(x=200, y=120)
    save1 = tk.Button(maaster, text="Save", command=save_pass, fg="black", bg="#3ece48", height=1, width=25,
                      activebackground="white", font=('times', 10, ' bold '))
    save1.place(x=10, y=120)
    maaster.mainloop()


#####################################################################################

def psw():
    global key
    assure_path_exists("TrainingImageLabel/")
    exists1 = os.path.isfile("TrainingImageLabel/psd.txt")
    if exists1:
        tf = open("TrainingImageLabel/psd.txt", "r")
        key = tf.read()
    else:
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        if new_pas is None:
            mess._show(title='No Password Entered', message='Password not set!! Please try again')
        else:
            tf = open("TrainingImageLabel/psd.txt", "w")
            tf.write(new_pas)
            mess._show(title='Password Registered', message='New password was registered successfully!!')
            return
    password = tsd.askstring('Password', 'Enter Password', show='*')
    if password == key:
        TrainImages()
    elif password is None:
        pass
    else:
        mess._show(title='Wrong Password', message='You have entered wrong password')


######################################################################################

def clear():
    txt.delete(0, 'end')
    rres = "1)Take Images  >>>  2)Save Profile"
    message1.configure(text=rres)


def clear2():
    txt2.delete(0, 'end')
    rres = "1)Take Images  >>>  2)Save Profile"
    message1.configure(text=rres)

    ######################################################################################

    """ This method Take Images is a function used for
    creating the sample of the images which is used for
    training the model. It takes 60 Images of every new user."""


def TakeImages():
    check_haarcascadefile()
    columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
    #Checking whether there is a required path to proceed or not
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")
    serial = 0
    xists = os.path.isfile("StudentDetails/StudentDetails.csv")
    if xists:
        """If there is required path to the StudentDetails.csv then save the serial no, id, name"""
        with open("StudentDetails/StudentDetails.csv", 'r') as csvf:
            reador1 = csv.reader(csvf)
            for l in reador1:
                serial = serial + 1
        serial = (serial // 2)
        csvf.close()
    else:
        with open("StudentDetails/StudentDetails.csv", 'a+') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(columns)
            serial = 1
        csvf.close()
    # Both ID and Name is used for recognising the Image
    Id = (txt.get())
    name = (txt2.get())
    # Checking if the ID is numeric and name is Alphabetical
    if (name.isalpha()) or (' ' in name):
        """ Opening the primary camera if you want to access 
         the secondary camera you can mention the number 
         as 1 inside the parenthesis"""
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        # Initializing the sample number(No. of images) as 0
        sampleNum = 0
        while True:
            # Reading the video captures by camera frame by frame
            ret, img = cam.read()
            """Converting the image into grayscale as most of the the processing is done in gray scale format """
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            # For creating a rectangle around the image
            for (x, y, w, h) in faces:
                """Specifying the coordinates of the image as well as color and thickness of the rectangle 
                incrementing sample number for each image """
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # incrementing sample number
                sampleNum = sampleNum + 1
                # saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage/ " + name + "." + str(serial) + "." + Id + '.' + str(sampleNum) + ".jpg",
                            gray[y:y + h, x:x + w])
                # display the frame
                cv2.imshow('Taking Images', img)
            # wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum > 100:
                break
        cam.release()
        cv2.destroyAllWindows()
        # Displaying message for the user
        rres = "Images Taken for ID : " + Id
        # Creating the entry for the user in a csv file
        row = [serial, '', Id, '', name]
        with open('StudentDetails/StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message1.configure(text=rres)
    else:
        if not name.isalpha():
            rres = "Enter Correct name"
            message.configure(text=rres)


########################################################################################

# Training the images saved in training image folder
def TrainImages():
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel/")
    """Local Binary Pattern Histogram is an Face Recognizer 
    algorithm inside OpenCV module used for training the image dataset """
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    # creating detector for faces
    detector = cv2.CascadeClassifier(harcascadePath)
    # Saving the detected faces in variables
    faces, ID = getImagesAndLabels("TrainingImage")
    """Saving the trained faces and their respective ID's in a model named as "trainer.yml"."""
    try:
        recognizer.train(faces, np.array(ID))
    except:
        mess._show(title='No Registrations', message='Please Register someone first!!!')
        return
    recognizer.save("TrainingImageLabel/Trainer.yml")
    result = "Profile Saved Successfully"
    message1.configure(text=result)
    message.configure(text='Total Registrations till now  : ' + str(ID[0]))


############################################################################################3

def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empth face list
    faces = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(ID)
    return faces, Ids


###########################################################################################
# For testing phase
def TrackImages():
    global attendance, df
    check_haarcascadefile()
    #Checking whether there exists proper path or not
    assure_path_exists("Attendance/")
    assure_path_exists("StudentDetails/")
    for k in tv.get_children():
        tv.delete(k)
    # msg = ''
    i = 0
    # j = 0
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer() ###pip install
    # opencv-contrib-python
    # Reading the trained model
    exists3 = os.path.isfile("TrainingImageLabel/Trainer.yml")
    if exists3:
        recognizer.read("TrainingImageLabel/Trainer.yml")
    else:
        mess._show(title='Data Missing', message='Please click on Save Profile to reset data!!')
        return
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    #Taking Images through web camera
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', '', 'Name', '', 'Date', '', 'Time']
    exists1 = os.path.isfile("StudentDetails/StudentDetails.csv")
    if exists1:
        df = pd.read_csv("StudentDetails/StudentDetails.csv")
    else:
        mess._show(title='Details Missing', message='Students details are missing, please check!')
        cam.release()
        cv2.destroyAllWindows()
        window.destroy()
    while True:
        ret, im = cam.read()
        """Converting the image into grayscale as most of the the processing is done in gray scale format """
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        """It converts the images in different sizes (decreases by 1.3 times) and 5 specifies the number of times scaling happens """
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        # For creating a rectangle around the image
        for (x, y, w, h) in faces:
            """Specifying the coordinates of the image as well as color and thickness of the rectangle incrementing sample number for each image """
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 50:
                times = time.time()
                dti = datetime.datetime.fromtimestamp(times).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(times).strftime('%H:%M:%S')
                aa = df.loc[df['SERIAL NO.'] == serial]['NAME'].values
                ID = df.loc[df['SERIAL NO.'] == serial]['ID'].values
                ID = str(ID)
                ID = ID[1:-1]
                bb = str(aa)
                bb = bb[2:-2]
                attendance = [str(ID), '', bb, '', str(dti), '', str(timeStamp)]

            else:
                Id = 'Unknown'
                bb = str(Id)
            cv2.putText(im, str(bb), (x, y + h), font, 1, (255, 255, 255), 2)
        cv2.imshow('Taking Attendance', im)
        if cv2.waitKey(1) == ord('q'):
            break
    times = time.time()
    dti = datetime.datetime.fromtimestamp(times).strftime('%d-%m-%Y')
    xists = os.path.isfile("Attendance/Attendance_" + dti + ".csv")
    if xists:
        with open("Attendance/Attendance_" + dti + ".csv", 'a+') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(attendance)
        csvf.close()
    else:
        with open("Attendance/Attendance_" + dti + ".csv", 'a+') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(col_names)
            writer.writerow(attendance)
        csvf.close()
    with open("Attendance/Attendance_" + dti + ".csv", 'r') as csvf:
        reador1 = csv.reader(csvf)
        for lines in reador1:
            i = i + 1
            if i > 1:
                if i % 2 != 0:
                    iidd = str(lines[0]) + '   '
                    tv.insert('', 0, text=iidd, values=(str(lines[2]), str(lines[4]), str(lines[6])))
    csvf.close()
    cam.release()
    cv2.destroyAllWindows()


######################################## USED STUFFS ############################################

key = ''

tis = time.time()
dt = datetime.datetime.fromtimestamp(tis).strftime('%d-%m-%Y')
day, month, year = dt.split("-")

mont = {'01': '01',
        '02': '02',
        '03': '03',
        '04': '04',
        '05': '05',
        '06': '06',
        '07': '07',
        '08': '08',
        '09': '09',
        '10': '10',
        '11': '11',
        '12': '12'
        }

######################################## GUI FRONT-END ###########################################

window = tk.Tk()
window.geometry("1280x1024")
window.resizable(True, False)
window.title("Attendance System")
window.configure(background='#355454')
# For Already Registered frame1
frame1 = tk.Frame(window, bg="white")
frame1.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.80)
# For New Registration frame2
frame2 = tk.Frame(window, bg="white")
frame2.place(relx=0.51, rely=0.17, relwidth=0.38, relheight=0.80)

message3 = tk.Label(window, text="Facial Recognition Based Attendance System", fg="white", bg="#355454", width=60,
                    height=1, font=('times', 29, ' bold '))
message3.place(x=10, y=10, relwidth=1)

frame3 = tk.Frame(window, bg="white")
frame3.place(relx=0.52, rely=0.09, relwidth=0.09, relheight=0.07)

frame4 = tk.Frame(window, bg="white")
frame4.place(relx=0.36, rely=0.09, relwidth=0.16, relheight=0.07)

datef = tk.Label(frame4, text=day + "-" + mont[month] + "-" + year + "      |", fg="orange", bg="#262523", width=60,
                 height=1, font=('times', 22, ' bold '))
datef.pack(fill='both', expand=3)

clock = tk.Label(frame3, fg="orange", bg="#262523", width=55, height=1, font=('times', 22, ' bold '))
clock.pack(fill='both', expand=1)
tick()

head2 = tk.Label(frame2, text="                       For New Registrations                       ", fg="white",
                 bg="black", font=('times', 17, ' bold '))
head2.grid(row=0, column=0)

head1 = tk.Label(frame1, text="                       For Already Registered                       ", fg="white",
                 bg="black", font=('times', 17, ' bold '))
head1.place(x=0, y=0)
# Registration frame
lbl = tk.Label(frame2, text="Enter ID", width=20, height=1, anchor="center", fg="black", bg="white", font=('times', 17, ' bold '))
lbl.place(x=80, y=55)

txt = tk.Entry(frame2, width=32, fg="black", bg="#e1f2f2", highlightcolor="#00aeff", highlightthickness=3,
               font=('times', 15, ' bold '))
txt.place(x=30, y=88)

lbl2 = tk.Label(frame2, text="Enter Name", width=20, fg="black", bg="white", font=('times', 17, ' bold '))
lbl2.place(x=80, y=140)

txt2 = tk.Entry(frame2, width=32, fg="black", bg="#e1f2f2", highlightcolor="#00aeff", highlightthickness=3,
                font=('times', 15, ' bold '))
txt2.place(x=30, y=173)

message1 = tk.Label(frame2, text="1)Take Images  >>>  2)Save Profile", bg="white", fg="black", width=39, height=1,
                    activebackground="yellow", font=('times', 15, ' bold '))
message1.place(x=7, y=230)

message = tk.Label(frame2, text="", bg="white", fg="black", width=39, height=1, activebackground="yellow",
                   font=('times', 16, ' bold '))
message.place(x=7, y=450)

lbl3 = tk.Label(frame1, text="Attendance", width=20, fg="black", bg="white", height=1, font=('times', 17, ' bold '))
lbl3.place(x=100, y=115)
# Display total registrations
res = 0
exists = os.path.isfile("StudentDetails/StudentDetails.csv")
if exists:
    with open("StudentDetails/StudentDetails.csv", 'r') as csvFile1:
        reader1 = csv.reader(csvFile1)
        for l in reader1:
            res = res + 1
    res = (res // 2) - 1
    csvFile1.close()
else:
    res = 0
message.configure(text='Total Registrations till now  : ' + str(res))

##################### MENUBAR #################################

menubar = tk.Menu(window, relief='ridge')
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label='Change Password', command=change_pass)
filemenu.add_command(label='Contact Us', command=contact)
filemenu.add_command(label='Exit', command=window.destroy)
menubar.add_cascade(label='Help', font=('times', 29, ' bold '), menu=filemenu)

################## TREEVIEW ATTENDANCE TABLE ####################

tv = ttk.Treeview(frame1, height=13, columns=('name', 'date', 'time'))
tv.column('#0', width=82)
tv.column('name', width=130)
tv.column('date', width=133)
tv.column('time', width=133)
tv.grid(row=2, column=0, padx=(0, 0), pady=(150, 0), columnspan=4)
tv.heading('#0', text='ID')
tv.heading('name', text='NAME')
tv.heading('date', text='DATE')
tv.heading('time', text='TIME')

###################### SCROLLBAR ################################

scroll = ttk.Scrollbar(frame1, orient='vertical', command=tv.yview)
scroll.grid(row=2, column=4, padx=(0, 100), pady=(150, 0), sticky='ns')
tv.configure(yscrollcommand=scroll.set)

###################### BUTTONS ##################################

clearButton = tk.Button(frame2, text="Clear", command=clear, fg="black", bg="#ea2a2a", width=11,
                        activebackground="white", font=('times', 11, ' bold '))
clearButton.place(x=335, y=86)
clearButton2 = tk.Button(frame2, text="Clear", command=clear2, fg="black", bg="#ea2a2a", width=11,
                         activebackground="white", font=('times', 11, ' bold '))
clearButton2.place(x=335, y=172)
takeImg = tk.Button(frame2, text="Take Images", command=TakeImages, fg="black", bg="#00aeff", width=34, height=1,
                    activebackground="white", font=('times', 15, ' bold '))
takeImg.place(x=30, y=300)
trainImg = tk.Button(frame2, text="Save Profile", command=psw, fg="black", bg="#00aeff", width=34, height=1,
                     activebackground="white", font=('times', 15, ' bold '))
trainImg.place(x=30, y=380)
trackImg = tk.Button(frame1, text="Take Attendance", command=TrackImages, fg="black", bg="#00aeff", width=35, height=1,
                     activebackground="white", font=('times', 15, ' bold '))
trackImg.place(x=30, y=50)
quitWindow = tk.Button(frame1, text="Quit", command=window.destroy, fg="black", bg="red", width=35, height=1,
                       activebackground="white", font=('times', 15, ' bold '))
quitWindow.place(x=30, y=450)

##################### END ######################################

window.configure(menu=menubar)
window.mainloop()

####################################################################################################
