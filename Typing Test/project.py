import tkinter as tk
from tkinter import *
import time
import threading
import random
import sqlite3
from PIL import ImageTk,Image
import tkinter.messagebox


cnt = sqlite3.connect("typedb.dp")
try:
        cnt.execute('''
        CREATE TABLE ADVTYPEDB(
        NAME TEXT,
        WPM REAL
        );
        ''')
        cnt.execute('''
        CREATE TABLE TYPEDB(
        NAME TEXT,
        WPM REAL 
        );
        ''')
except:
    pass

        
class TypeSpeedGUI:
    
    def __init__(self):

        self.flag = 0                                               #0 for standard and 1 for advanced

        
        self.root=tk.Tk()
        self.root.title("Speed Typing Test")
        self.root.geometry("1920x600")
        #self.root.configure(background="purple")
        self.bg = PhotoImage(file = "backgrounds.png")
        label = Label(self.root,image=self.bg)
        label.place(x=0,y=0)
 
        self.texts=open("texts.txt","r").read().split("\n")
        self.advtexts = open("advtexts.txt","r").read().split("\n")

        self.frame=tk.Frame(self.root, background = "black")
        self.prev=random.choice(self.texts)
        self.curr=self.prev
        self.sample_label=tk.Label(self.frame,text=self.prev,font=("Helvetica",18),bg="black",fg="yellow")
        self.sample_label.grid(row=0,column=0,columnspan=2,padx=2,pady=5)

        self.input_entry=tk.Entry(self.frame,width=40,font=("Helvetica",24))
        self.input_entry.grid(row=1,column=0,columnspan=2,padx=5,pady=10)
        self.input_entry.bind("<KeyRelease >",self.start)

        
        self.speed_label=tk.Label(self.frame,text="Time: 0.00 seconds\nSpeed: \n0.00 CPS\n0.00 CPM\n0.00 WPS\n0.00 WPM\n",font=("Helvetica",18), bg="black", fg="cyan")
        self.speed_label.grid(row=2,column=0,columnspan=2,padx=5,pady=10)

        self.reset_button=tk.Button(self.frame,text="Reset",command=self.reset,font=("Helvetica",20),background="black",foreground="orange",activeforeground="green")
        self.reset_button.grid(row=3,column=1,columnspan=2,padx=5,pady=10)
        self.reset_button.bind("<Enter>", self.on_enter)
        self.reset_button.bind("<Leave>", self.on_leave)

        self.advbutton=tk.Button(self.frame,text="Advanced",command=self.advance,font=("Helvetica",20),background="black",foreground="orange",activeforeground="green")
        self.advbutton.grid(row=3,column=0,columnspan=2,padx=5,pady=10)
        self.advbutton.bind("<Enter>", self.on_enter_adv)
        self.advbutton.bind("<Leave>", self.on_leave_adv)

        self.frame.pack(expand=True)

        self.counter=0
        self.running=False
        self.name=tk.StringVar()
        self.name.set("Enter your name")
        
        self.root.mainloop()

    def advance(self):
        #self.sample_label.config(text=random.choice(self.advtexts))
        #self.advbutton.config(text="standard")
        butlabel = self.advbutton.cget('text')
        if butlabel == "Advanced":
            self.flag = 1
        else:
            self.flag = 0
            
        if self.flag == 0:
            self.sample_label.config(text=random.choice(self.texts))
            self.advbutton.config(text="Advanced")
        if self.flag == 1:
            self.sample_label.config(text=random.choice(self.advtexts))
            self.advbutton.config(text="Standard")

        #reset function
        self.running=False
        self.counter=0
        self.speed_label.config(text="Time: 0.00 seconds\nSpeed: \n0.00 CPS\n0.00 CPM\n0.00 WPS\n0.00 WPM\n")
        if self.flag == 0:
            self.sample_label.config(text=random.choice(self.texts))
            self.input_entry.delete(0,tk.END)
            try:
                if self.leadbut.winfo_exists()==1:
                        self.leadbut.destroy()
            except:
                pass
                
            try:
                self.nameEntry.destroy()
                self.namebut.destroy()
            except:
                pass
        if self.flag == 1:
            self.sample_label.config(text=random.choice(self.advtexts))
            self.input_entry.delete(0,tk.END)
            try:
                if self.leadbut.winfo_exists()==1:
                        self.leadbut.destroy()
            except:
                pass
                
            try:
                self.sample_label1.destroy()
                self.nameEntry.destroy()
                self.namebut.destroy()
            except:
                pass       
        
    def on_enter(self,e):
        self.reset_button['background'] = 'green'

    def on_leave(self,e):
        self.reset_button['background'] = 'black'

    def on_enter_adv(self,e):
        self.advbutton['background'] = 'green'

    def on_leave_adv(self,e):
        self.advbutton['background'] = 'black'
        
    def compleadInput(self):
        self.lead.title("Leaderboard")
        self.lead.geometry("800x600")
        self.lead.allleadbut.destroy()
        if(self.flag==0):
            cursor=cnt.execute('''SELECT * FROM TYPEDB ORDER BY WPM DESC;''')
        else:
            cursor=cnt.execute('''SELECT * FROM ADVTYPEDB ORDER BY WPM DESC;''')
            
        self.e=tk.Entry(self.lead,width=10,bg='black',fg='orange',font=('Arial',16,'bold'))
        self.e.grid(row=0,column=0)
        self.e.insert(tk.END,"S.NO")
        self.e=tk.Entry(self.lead,width=20,bg='black',fg='orange',font=('Arial',16,'bold'))
        self.e.grid(row=0,column=1)
        self.e.insert(tk.END,"NAME")
        self.e=tk.Entry(self.lead,width=20,bg='black',fg='orange',font=('Arial',16,'bold'))
        self.e.grid(row=0,column=2)
        self.e.insert(tk.END,"TYPING SPEED(WPM)")
        for ind,i in enumerate(cursor):
            if ind>9:
                self.e=tk.Entry(self.lead,width=10,bg='#0d0447',fg='white',font=('Arial',16,'bold'))
                self.e.grid(row=ind+1,column=0)
                self.e.insert(tk.END,ind+1)
                for ind2,j in enumerate(i):
            #print(i[0],i[1])
                    self.e=tk.Entry(self.lead,width=20,bg='#0d0447',fg='white',font=('Arial',16,'bold'))
                    self.e.grid(row=ind+1,column=ind2+1)
                    if ind2==1:
                        j=round(float(j),2)
                    self.e.insert(tk.END,j)
        
    def leadInput(self):
        self.leadbut.destroy()
        self.lead=tk.Tk()
        self.lead.title("TOP10Leaderboard") 
        self.lead.geometry("600x400")
        if(self.flag==0):
            cursor=cnt.execute('''SELECT * FROM TYPEDB ORDER BY WPM DESC LIMIT 10;''')
        else:
            cursor=cnt.execute('''SELECT * FROM ADVTYPEDB ORDER BY WPM DESC LIMIT 10;''')
            
        self.e=tk.Entry(self.lead,width=10,bg='black',fg='orange',font=('Arial',16,'bold'))
        self.e.grid(row=0,column=0)
        self.e.insert(tk.END,"S.NO")
        self.e=tk.Entry(self.lead,width=20,bg='black',fg='orange',font=('Arial',16,'bold'))
        self.e.grid(row=0,column=1)
        self.e.insert(tk.END,"NAME")
        self.e=tk.Entry(self.lead,width=20,bg='black',fg='orange',font=('Arial',16,'bold'))
        self.e.grid(row=0,column=2)
        self.e.insert(tk.END,"TYPING SPEED(WPM)")
        for ind,i in enumerate(cursor):
            self.e=tk.Entry(self.lead,width=10,bg='#0d0447',fg='white',font=('Arial',16,'bold'))
            self.e.grid(row=ind+1,column=0)
            self.e.insert(tk.END,ind+1)
            for ind2,j in enumerate(i):
            #print(i[0],i[1])
                self.e=tk.Entry(self.lead,width=20,bg='#0d0447',fg='white',font=('Arial',16,'bold'))
                self.e.grid(row=ind+1,column=ind2+1)
                if ind2==1:
                    j=round(float(j),2)
                self.e.insert(tk.END,j)
        self.lead.allleadbut=tk.Button(self.lead, text = 'CompleteLeaderboard',font=("Helvetica",18),command = self.compleadInput,background="black",foreground="orange",activeforeground="green")
        self.lead.allleadbut.place(x=300,y=350)
        
                
    def checkname(self):
            self.named = self.name.get()
            if self.named == "":
                    tkinter.messagebox.showwarning(title="Warning", message="Please Enter a Name")
            else:
                    self.printInput()


            

    def printInput(self):
            self.namedb=self.name.get()
            #print(self.namedb)
            if(self.flag==0):
                cursor=cnt.execute('''SELECT * FROM TYPEDB;''')
            else:
                cursor=cnt.execute('''SELECT * FROM ADVTYPEDB;''')
            flag1=0
            for i in cursor:
                if(i[0]==self.namedb):
                    flag1=1
                    mx=max(i[1],self.wpm)
                    params=(mx,i[0])
                    if(self.flag==0):
                        cnt.execute("UPDATE TYPEDB SET WPM=(?) WHERE NAME=(?)",params)
                    else:
                        cnt.execute("UPDATE ADVTYPEDB SET WPM=(?) WHERE NAME=(?)",params)
                    break
                    
            if(flag1==0):
                params = (self.namedb,self.wpm)
                if(self.flag==0):
                    cnt.execute("INSERT INTO TYPEDB VALUES(?,?)",params)
                else:
                    cnt.execute("INSERT INTO ADVTYPEDB VALUES(?,?)",params)
            cnt.commit()
            
            self.sample_label1.destroy()
            self.nameEntry.destroy()
            self.namebut.destroy()
            self.leadbut=tk.Button(self.root, text = 'Leaderboard',font=("Helvetica",18),command = self.leadInput,background="black",foreground="orange",activeforeground="green")
            self.leadbut.pack(side = 'top',padx=5,pady=10)
                
                
    def start(self,event):
        if not self.running:
            if not event.keycode in [16,17,18]:
                self.running=True
                t=threading.Thread(target=self.time_thread)
                t.start()
        if not self.sample_label.cget('text').startswith(self.input_entry.get()):
            self.input_entry.config(fg="red")
        else:
            self.input_entry.config(fg="black")
        if self.input_entry.get()==self.sample_label.cget('text'):
            self.running=False
            self.input_entry.config(fg="green")
            try:
                if self.namebut.winfo_exists()==0:
                    self.name.set("")
                    self.namebut= tk.Button(self.root, text = 'Submit !',font=("Helvetica",18),command = self.checkname,background="black",foreground="orange",activeforeground="green")
                    self.namebut.pack(side = 'top',padx=10,pady=20)
                    self.sample_label1=tk.Label(self.root,text="Enter your name:",font=("Helvetica",18),bg="Black",fg="#d0fa37")
                    self.sample_label1.place(x=525,y=500)
                    self.nameEntry= tk.Entry(self.root, width=30, font=("Arial",18,""),textvariable=self.name,bg="white",fg="red")
                    self.nameEntry.place(x=750,y=500)
            except:
                self.name.set("")
                self.namebut= tk.Button(self.root, text = 'Submit',font=("Helvetica",18),command = self.checkname,background="black",foreground="orange",activeforeground="green")
                self.namebut.pack(side = 'top',padx=10,pady=20)
                self.sample_label1=tk.Label(self.root,text="Enter your name:",font=("Helvetica",18),bg="Black",fg="#d0fa37")
                self.sample_label1.place(x=525,y=500)
                self.nameEntry= tk.Entry(self.root, width=30, font=("Arial",18,""),textvariable=self.name,bg="white",fg="red")
                self.nameEntry.place(x=750,y=500)


    def time_thread(self):
        while self.running:
            time.sleep(0.1)
            self.counter+=0.1
            cps=(len(self.input_entry.get())/self.counter)
            cpm=cps*60
            wps=len(self.input_entry.get().split(" "))/self.counter
            self.wpm=wps*60
            self.speed_label.config(text=f"Time: {self.counter:.2f} seconds\nSpeed:\n{cps:.2f} CPS\n{cpm:.2f} CPM\n{wps:.2f} WPS\n {self.wpm:.2f} WPM\n")

    def reset(self):
        self.running=False
        self.counter=0
        self.speed_label.config(text="Time: 0.00 seconds\nSpeed: \n0.00 CPS\n0.00 CPM\n0.00 WPS\n0.00 WPM\n")
        while self.curr==self.prev:
                if self.flag == 0:
                    self.curr=random.choice(self.texts)
                else:
                    self.curr=random.choice(self.advtexts)
        self.prev=self.curr           
        self.sample_label.config(text=self.curr)
        self.input_entry.delete(0,tk.END)
        try:
            if self.leadbut.winfo_exists()==1:
                self.leadbut.destroy()
        except:
            pass
                
        try:
            self.sample_label1.destroy()
            self.nameEntry.destroy()
            self.namebut.destroy()
        except:
            pass
       
            

TypeSpeedGUI()
