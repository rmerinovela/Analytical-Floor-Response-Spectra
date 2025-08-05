import numpy as np
import os
from matplotlib import pyplot as plt
import glob
import matplotlib as mpl
import tkinter as tk
import webbrowser
import datetime
from PIL import Image,ImageTk
import random
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import csv
import pandas as pd
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  
NavigationToolbar2Tk) 
  
mpl.rc('font',family='Times New Roman')


g = 9.805

def validate_entry(text):
    return (text.isnumeric() or text=='.')


def plot_PO(Say,Dy,Du): 
  
    # the figure that will contain the plot 
    fig = Figure(figsize = (5, 5), 
                 dpi = 100) 

    # adding the subplot 
    plot1 = fig.add_subplot(111) 
    fig.set_layout_engine('tight')
    # plotting the graph 
    plot1.plot((0,Dy),(0,Say),color='k')
    plot1.plot((Dy,Du),(Say,Say),color='k')

    plot1.set_xlabel(r'$D$ [m]')
    plot1.set_ylabel(r'$S_{ay}$ [g]')
    
    plot1.set_xlim(0)
    plot1.set_ylim(0)
    
    plot1.set_facecolor('#DEE2E6')
    plot1.grid(color= 'black', ls= '--', lw= 0.5)
  
    # creating the Tkinter canvas 
    # containing the Matplotlib figure 
    canvas = FigureCanvasTkAgg(fig, 
                               master = tab1)   
    canvas.draw() 
  
    # placing the canvas on the Tkinter window 
    canvas.get_tk_widget().grid(column=10,row=0,rowspan=30) 
  
    # creating the Matplotlib toolbar 
    toolbar = NavigationToolbar2Tk(canvas, 
                                   tab1,pack_toolbar=False) 
    toolbar.update() 
  
    # placing the toolbar on the Tkinter window 
    canvas.get_tk_widget().grid(column=10,row=0,rowspan=30) 
    
class Structure:
    def __init__(self,Say,Dy,Du,nst):
        self.Say = float(Say)
        self.Dy = float(Dy)
        self.Du = float(Du)
        self.nst = int(nst)
        self.delta = np.zeros(int(self.nst))   
        if(nst<=2):
            for i in range(len(self.delta)):
                self.delta[i] = (1/self.nst)*(i+1)
        else:
            for i in range(len(self.delta)):
                self.delta[i] = (4/3)*((i+1)/self.nst)*(1-0.25*((i+1)/self.nst))
    def GammaE(self):
        md = 0
        md2 = 0
        for i in range(len(self.delta)):
            md = md+self.delta[i]
            md2 = md2+self.delta[i]**2
            
        return md/md2
    def Ty(self):
        return float(2*np.pi*(self.Dy/(self.Say*g))**0.5)

def getInput_PO():
    
    global struct1
    global totn
      
    Say=Say_entry.get()
    Dy=Dy_entry.get()
    Du=Du_entry.get()

    nst=int(nst_entry.get())
    
    struct1 = Structure(Say,Dy,Du,nst) 
    plot_PO(struct1.Say,struct1.Dy,struct1.Du)
    
    tabControl.tab(tab2,state='normal')
    
def getInput_CP():
    
    global TC
    global TD
    
    TC=float(TC_entry.get())
    TD=float(TD_entry.get())
    
    if(tabGM_en == 1):
        tabControl.tab(tab3,state='normal')
    
    
    
def getInput_FRS():
    
    global fn
    global ksiNS
      
    fn=int(fn_entry.get())
    ksiNS=float(ksi_entry.get())
    
    FRS_Est(struct1,spectrum,solCS,fn,ksiNS,TC,TD)
    

def open_csv_file():
    
    global spectrum
    global tabGM_en
    
    file_path = filedialog.askopenfilename(title="Open CSV File", filetypes=[("CSV files", "*.csv")])
    if file_path:
        spect = pd.read_csv(file_path).values
        spectrum = GroundSpectrum(spect)
        display_csv_data(file_path)
    tabGM_en = 1

def display_csv_data(file_path):
    try:
        with open(file_path, 'r', newline='') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)  # Read the header row
            tree.delete(*tree.get_children())  # Clear the current data
            tree.column('1',width=120,stretch=False)
            tree.column('2',width=120,stretch=False)
            tree.column('3',width=120,stretch=False)
            tree.grid(column=0,row=1,rowspan=20)
            tree.heading("1", text="T (s)")  # Set header text for the "Name" column
            tree.heading("2", text="Sa (g)")    # Set header text for the "Age" column
            tree.heading("3", text="Sd (m)")  # Set header text for the "City" column


            for row in csv_reader:
                tree.insert("", "end", values=row)
                
            ttk.Label(tab2,text=f"CSV file loaded: {file_path}").grid(column=0,row=30,columnspan = 5,sticky=tk.W)  

            #status_label.config(text=f"CSV file loaded: {file_path}")
            
            plot_GS(spectrum)
          

    except Exception as e:
        status_label.config(text=f"Error: {str(e)}")

        
def plot_GS(spectrum): 
  
    # the figure that will contain the plot 
    fig = Figure(figsize = (5, 5), 
                 dpi = 100) 
    fig.set_layout_engine('tight')

    # adding the subplot 
    plot1 = fig.add_subplot(311) 
  
    # plotting the graph 
    plot1.plot(spectrum.Gspectrum[:,0],spectrum.Gspectrum[:,1])

    plot1.set_xlabel(r'$T$ [s]')
    plot1.set_ylabel(r'$S_{AG}$ [g]')
    
    plot1.set_xlim(0,2)
    plot1.set_ylim(0)
    plot1.set_facecolor('#DEE2E6')
    plot1.grid(color= 'black', ls= '--', lw= 0.5)
    
    plot1 = fig.add_subplot(312) 
    
    
  
    # plotting the graph 
    plot1.plot(spectrum.Gspectrum[:,0],spectrum.Gspectrum[:,2])

    plot1.set_xlabel(r'$T$ [s]')
    plot1.set_ylabel(r'$S_{DG}$ [m]')
    
    plot1.set_xlim(0,10)
    plot1.set_ylim(0)
    plot1.set_facecolor('#DEE2E6')
    plot1.grid(color= 'black', ls= '--', lw= 0.5)
    
    plot1 = fig.add_subplot(313) 
    
  
    # plotting the graph 
    plot1.plot(spectrum.Gspectrum[:,2],spectrum.Gspectrum[:,1])

    plot1.set_xlabel(r'$S_{DG}$ [m]')
    plot1.set_ylabel(r'$S_{AG}$ [g]')
    
    plot1.set_xlim(0)
    plot1.set_ylim(0)
    plot1.set_facecolor('#DEE2E6')
    plot1.grid(color= 'black', ls= '--', lw= 0.5)
  
    # creating the Tkinter canvas 
    # containing the Matplotlib figure 
    canvas = FigureCanvasTkAgg(fig, 
                               master = tab2)   
    canvas.draw() 
  
    # placing the canvas on the Tkinter window 
    canvas.get_tk_widget().grid(column=6,row=0,rowspan=30) 
  
    # creating the Matplotlib toolbar 
    toolbar = NavigationToolbar2Tk(canvas, 
                                   tab2,pack_toolbar=False) 
    toolbar.update() 
  
    # placing the toolbar on the Tkinter window 
    canvas.get_tk_widget().grid(column=6,row=0,rowspan=30) 
    
    plot_CS(struct1,spectrum,1)
            
        

class GroundSpectrum:
    def __init__(self,Gspectrum):
        self.Gspectrum = Gspectrum
        

def plot_CS(structure,spectrum,eta):
    
    fig = Figure(figsize = (5, 5), 
                 dpi = 100) 
    fig.set_layout_engine('tight')

    # adding the subplot 
    plot1 = fig.add_subplot(111) 
  
    # plotting the graph 
    plot1.plot(eta*spectrum.Gspectrum[:,2],eta*spectrum.Gspectrum[:,1])
    plot1.plot((0,struct1.Dy),(0,struct1.Say),color='k')
    plot1.plot((struct1.Dy,struct1.Du),(struct1.Say,struct1.Say),color='k')

    plot1.set_xlabel(r'$S_{DG}$ [m]')
    plot1.set_ylabel(r'$S_{AG}$ [g]')
    
    plot1.set_xlim(0)
    plot1.set_ylim(0)
    plot1.set_facecolor('#DEE2E6')
    plot1.grid(color= 'black', ls= '--', lw= 0.5)
    
  
    # creating the Tkinter canvas 
    # containing the Matplotlib figure 
    canvas = FigureCanvasTkAgg(fig, 
                               master = tab3)   
    canvas.draw() 
  
    # placing the canvas on the Tkinter window 
    canvas.get_tk_widget().grid(column=6,row=0,rowspan=30) 
  
    # creating the Matplotlib toolbar 
    toolbar = NavigationToolbar2Tk(canvas, 
                                   tab3,pack_toolbar=False) 
    toolbar.update() 
  
    # placing the toolbar on the Tkinter window 
    canvas.get_tk_widget().grid(column=6,row=0,rowspan=30) 
    
    
            
        
def CapacitySpectrum(struct,spectrum):
    
    global solCS
    global fn_entry
    
    ind = abs(struct.Ty()-spectrum.Gspectrum[:,0]).argmin()

    if(spectrum.Gspectrum[ind,2]<=struct.Dy):
        #elastic supporting structure
        d1 = spectrum.Gspectrum[ind,2]
        Sa1 = spectrum.Gspectrum[ind,1]
        eta =1
        mu = d1/struct.Dy
        solCS = sol_CS(1,d1,Sa1,1)
        
    tol = 0.01
    if(spectrum.Gspectrum[ind,2]>struct.Dy):

        for i in range(0,1001):
            d = struct.Dy+(struct.Du-struct.Dy)*i/1001
            mu = d/struct.Dy
            ksi = 0.05+0.565*(mu-1)/(mu*np.pi)
            eta = (0.07/(0.02+ksi))**0.5

            iny = abs(struct.Dy-eta*spectrum.Gspectrum[:,2]).argmin()
            ina = abs(struct.Say-eta*spectrum.Gspectrum[iny:,1]).argmin()+iny
            Sdcr = eta*spectrum.Gspectrum[ina,2]

            if(abs(Sdcr-d)<=tol):
                d1 = d
                Sa1 = struct.Say
                solCS = sol_CS(mu,d1,Sa1,eta)
                break   
                
    plot_CS(struct,spectrum,eta)
    
    answer = "Solution: mu = {m},  d = {d},  Sa = {a}".format(m=round(mu,2),d=round(d1,3),a=round(Sa1,3))
    ttk.Label(tab3,text = answer).grid(column=6,row=32,sticky=tk.EW)  
    tabControl.tab(tab4,state='normal')
    current_value_fn = tk.StringVar(value=1)
    fn_entry = ttk.Spinbox(tab4, from_=1, to=struct.nst, textvariable=current_value_fn, wrap=True)
    fn_entry.grid(column=1,row=10, padx=5, pady=5)
    
class sol_CS:
    def __init__(self,mu,d1,Say,eta):
        self.mu = float(mu)
        self.d1 = float(d1)
        self.Say = float(Say)
        self.eta = float(eta)
        
class FloorSpectrum:
    def __init__(self,T,Saf,Sdf,ksiNS,nf):
        self.T = T
        self.Saf = Saf
        self.Sdf = Sdf
        self.ksiNS = ksiNS
        self.nf = nf
                
def FRS_Est(struct,spectrum,solCS,ns,ksiNS,TC,TD):
    
    global floorspect
    Ty = struct.Ty()
    
    if(struct.nst==1):
        Tn = np.zeros(1)
        Tn[0] = 0.84*Ty
    elif(struct.nst==2):
        Tn = np.zeros(2)
        Tn[0] = 0.84*Ty
        Tn[1] = 0.84*Ty/4
    elif(struct.nst==3):
        Tn = np.zeros(3)
        Tn[0] = 0.84*Ty
        Tn[1] = 0.84*Ty/4
        Tn[2] = 0.84*Ty/7
    else:
        Tn = np.zeros(4)
        Tn[0] = 0.84*Ty
        Tn[1] = 0.84*Ty/4
        Tn[2] = 0.84*Ty/7
        Tn[3] = 0.84*Ty/9
        
    
    
    PGD = spectrum.Gspectrum[-1,2]
    SdTD =max(spectrum.Gspectrum[:,2])
    SaTD = SdTD/g*(4*np.pi**2)/TD**2
    T = spectrum.Gspectrum[:,0]
    Sam = spectrum.Gspectrum[:,1]
    Sdm = spectrum.Gspectrum[:,2]
    
    if(solCS.d1<struct.Dy):
        mu = 1
    else:
        mu = solCS.d1/struct.Dy
        
        
    if(struct.nst>1):
        Sa2 = Sam[abs(T-(0.84*Ty/4)).argmin()]
        Sd2 = Sdm[abs(T-(0.84*Ty/4)).argmin()]
        if(struct.nst>2):
            Sa3 = Sam[abs(T-(0.84*Ty/7)).argmin()]
            Sd3 = Sdm[abs(T-(0.84*Ty/7)).argmin()]
            if(struct.nst>3):
                Sa4 = Sam[abs(T-(0.84*Ty/9)).argmin()]
                Sd4 = Sdm[abs(T-(0.84*Ty/9)).argmin()]
    
    a = np.zeros(len(Tn))
    d = np.zeros(len(Tn))
    
    if(struct.nst==1):
        a[0] = struct.GammaE()*struct.delta[ns-1]*solCS.Say
        d[0] = struct.GammaE()*struct.delta[ns-1]*solCS.d1
    elif(struct.nst==2):
        a[0] = struct.GammaE()*struct.delta[ns-1]*solCS.Say
        a[1] = 0.426*abs(np.sin(4.71*(ns/struct.nst)))*Sa2
        d[0] = struct.GammaE()*struct.delta[ns-1]*solCS.d1
        d[1] = 0.426*abs(np.sin(4.71*(ns/4)))*Sd2
    elif(struct.nst==3):
        a[0] = struct.GammaE()*struct.delta[ns-1]*solCS.Say
        a[1] = 0.426*abs(np.sin(4.71*(ns/struct.nst)))*Sa2
        a[2] = 0.254*abs(np.sin(7.85*(ns/struct.nst)))*Sa3
        d[0] = struct.GammaE()*struct.delta[ns-1]*solCS.d1
        d[1] = 0.426*abs(np.sin(4.71*(ns/4)))*Sd2
        d[2] = 0.254*abs(np.sin(7.85*(ns/4)))*Sd3
    else:
        a[0] = struct.GammaE()*struct.delta[ns-1]*solCS.Say
        a[1] = 0.426*abs(np.sin(4.71*(ns/struct.nst)))*Sa2
        a[2] = 0.254*abs(np.sin(7.85*(ns/struct.nst)))*Sa3
        a[3] = 0.202*abs(np.sin(10.99*(ns/struct.nst)))*Sa4
        d[0] = struct.GammaE()*struct.delta[ns-1]*solCS.d1
        d[1] = 0.426*abs(np.sin(4.71*(ns/4)))*Sd2
        d[2] = 0.254*abs(np.sin(7.85*(ns/4)))*Sd3
        d[3] = 0.202*abs(np.sin(10.99*(ns/4)))*Sd4
    

    Te = np.zeros(len(Tn))
    if(mu==1):
        if(struct.nst==1):
            Te[0] = 0.84*Ty
        elif(struct.nst==2):
            Te[0] = 0.84*Ty
            Te[1] = 0.84*Ty/3
        elif(struct.nst==3):
            Te[0] = 0.84*Ty
            Te[1] = 0.84*Ty/3
            Te[2] = 0.84*Ty/5
        else:
            Te[0] = 0.84*Ty
            Te[1] = 0.84*Ty/3
            Te[2] = 0.84*Ty/5
            Te[3] = 0.84*Ty/7
            
    elif(mu>1):
        if(struct.nst==1):
            Te[0] = 0.84*Ty*((1+(mu)**0.5+mu)/3)**0.5
        elif(struct.nst==2):
            Te[0] = 0.84*Ty*((1+(mu)**0.5+mu)/3)**0.5
            Te[1] = 0.84*Ty/3*((1+(mu)**0.5+mu)/3)**0.5
        elif(struct.nst==3):
            Te[0] = 0.84*Ty*((1+(mu)**0.5+mu)/3)**0.5
            Te[1] = 0.84*Ty/3*((1+(mu)**0.5+mu)/3)**0.5
            Te[2] = 0.84*Ty/5
        else:
            Te[0] = 0.84*Ty*((1+(mu)**0.5+mu)/3)**0.5
            Te[1] = 0.84*Ty/3*((1+(mu)**0.5+mu)/3)**0.5
            Te[2] = 0.84*Ty/5
            Te[3] = 0.84*Ty/7
        
        
    DAF = np.zeros(len(Tn))
    for i in range(len(Tn)):     
        if(Tn[i]/TC == 0):
            DAF[i] = 2.5*(0.1/(0.05+ksiNS))**0.5;
        elif(Tn[i]/TC > 0 and Tn[i]/TC <=0.2):
            DAF[i] = 2.5*(0.1/(0.05+ksiNS))**0.5 + (1/(ksiNS)**0.5 - 2.5*(0.1/(0.05+ksiNS))**0.5)/0.2*(Tn[i]/TC);
        else:
            DAF[i] = 1/(ksiNS)**0.5;
            
    SAF = np.zeros(len(Tn))
    SDF = np.zeros(len(Tn))
    
    if(struct.nst==1):
        SAF[0] = a[0]*DAF[0]
        SDF[0] = SAF[0]*g*(Te[0]/(2*np.pi))**2
    elif(struct.nst==2):
        SAF[0] = a[0]*DAF[0]
        SDF[0] = SAF[0]*g*(Te[0]/(2*np.pi))**2
        SAF[1] = a[1]*DAF[1]/(mu)**0.4
        SDF[1] = SAF[1]*g*(Te[1]/(2*np.pi))**2
    elif(struct.nst==3):
        SAF[0] = a[0]*DAF[0]
        SDF[0] = SAF[0]*g*(Te[0]/(2*np.pi))**2
        SAF[1] = a[1]*DAF[1]/(mu)**0.4
        SDF[1] = SAF[1]*g*(Te[1]/(2*np.pi))**2
        SAF[2] = a[2]*DAF[2]/(mu)**0.2
        SDF[2] = SAF[2]*g*(Te[2]/(2*np.pi))**2
    else:
        SAF[0] = a[0]*DAF[0]
        SDF[0] = SAF[0]*g*(Te[0]/(2*np.pi))**2
        SAF[1] = a[1]*DAF[1]/(mu)**0.4
        SDF[1] = SAF[1]*g*(Te[1]/(2*np.pi))**2
        SAF[2] = a[2]*DAF[2]/(mu)**0.2
        SDF[2] = SAF[2]*g*(Te[2]/(2*np.pi))**2
        SAF[3] = a[3]*DAF[3]/(mu)**0.0
        SDF[3] = SAF[3]*g*(Te[3]/(2*np.pi))**2
    
      
    dabs = np.zeros(len(Tn))  
    if(SDF[0]>SdTD):
        dabs[0] = (d[0]**2+PGD**2)**0.5
    else:
        dabs[0] = d[0]
    
    if(struct.nst>1):
        for i in range(1,len(Tn)):
            dabs[i] = d[i]

    #Compute modal contributions to floor spectrum
    Safm = np.zeros((len(T),len(Tn)));
    Sdfm = np.zeros((len(T),len(Tn)));

    for j in range (len(Tn)):
        for i in range(len(T)):
            if(T[i]<=Tn[j]):
                Safm[i,j] = (T[i]/Tn[j])**2*(SAF[j]-a[j])+a[j]
                Sdfm[i,j] = Safm[i,j]*g*(T[i]**2/(4*np.pi**2)) 
            elif(T[i]>Tn[j] and T[i]<=Te[j]):
                Safm[i,j] = SAF[j]
                Sdfm[i,j] = Safm[i,j]*g*(T[i]**2)/(4*np.pi**2)
            elif(T[i]>Te[j]):
                if(mu>1 and j==0 and SDF[j]<SdTD):
                    if(T[i]<TD):  
                        Sdfm[i,j] = SDF[j]+(SdTD-SDF[j])/(TD-Te[j])*(T[i]-Te[j])
                        Safm[i,j] = Sdfm[i,j]*((4*np.pi**2)/T[i]**2)/g 
                    else:
                        Sdfm[i,j] = SdTD
                        Safm[i,j] = Sdfm[i,j]*((4*np.pi**2)/T[i]**2)/g 
                else:
                    Sdfm[i,j] = dabs[j] + (Te[j]/T[i])**2*(SDF[j]-dabs[j])
                    Safm[i,j] = Sdfm[i,j]*((4*np.pi**2)/T[i]**2)/g 
       
             
    #Combine modal contributions to floor response spectrum
    
    if(struct.nst>1):
        Saf = np.sqrt(np.sum(np.multiply(Safm,Safm),axis=1))
        Sdf = np.sqrt(np.sum(np.multiply(Sdfm,Sdfm),axis=1))
    else:
        Saf = Safm
        Sdf = Sdfm

    nslim = round(struct.nst/2);
        
    if ns<nslim:
        for i in range(len(T)):
            Saf[i]=max(Saf[i],Sam[i]*(0.07/(0.02+ksiNS))**0.5);
            Sdf[i]=max(Sdf[i],Sdm[i]*(0.07/(0.02+ksiNS))**0.5);
    
    for i in range(len(T)):
        if(T[i]>Te[0]):
            Saf[i]=max(Saf[i],Sam[i]*(0.07/(0.02+ksiNS))**0.5);
            Sdf[i]=max(Sdf[i],Sdm[i]*(0.07/(0.02+ksiNS))**0.5);
            
            
    floorspect = FloorSpectrum(T,Saf,Sdf,ksiNS,ns)
    plot_FS(floorspect)
    
def plot_FS(floorspectrum): 
  
    # the figure that will contain the plot 
    fig = Figure(figsize = (5, 5), 
                 dpi = 100) 
    fig.set_layout_engine('tight')

    # adding the subplot 
    plot1 = fig.add_subplot(311) 
  
    # plotting the graph 
    plot1.plot(floorspectrum.T,floorspectrum.Saf)

    plot1.set_xlabel(r'$T$ [s]')
    plot1.set_ylabel(r'$S_{AF}$ [g]')
    
    plot1.set_xlim(0,2)
    plot1.set_ylim(0)
    plot1.set_facecolor('#DEE2E6')
    plot1.grid(color= 'black', ls= '--', lw= 0.5)
    
    plot1 = fig.add_subplot(312) 
  
    # plotting the graph 
    plot1.plot(floorspectrum.T,floorspectrum.Sdf)

    plot1.set_xlabel(r'$T$ [s]')
    plot1.set_ylabel(r'$S_{DF}$ [m]')
    
    plot1.set_xlim(0,10)
    plot1.set_ylim(0)
    plot1.set_facecolor('#DEE2E6')
    plot1.grid(color= 'black', ls= '--', lw= 0.5)
    
    plot1 = fig.add_subplot(313) 
  
    # plotting the graph 
    plot1.plot(floorspectrum.Sdf,floorspectrum.Saf)

    plot1.set_xlabel(r'$S_{DF}$ [m]')
    plot1.set_ylabel(r'$S_{AF}$ [g]')
    
    plot1.set_xlim(0)
    plot1.set_ylim(0)
    plot1.set_facecolor('#DEE2E6')
    plot1.grid(color= 'black', ls= '--', lw= 0.5)
  
    # creating the Tkinter canvas 
    # containing the Matplotlib figure 
    canvas = FigureCanvasTkAgg(fig, 
                               master = tab4)   
    canvas.draw() 
  
    # placing the canvas on the Tkinter window 
    canvas.get_tk_widget().grid(column=6,row=0,rowspan=30) 
  
    # creating the Matplotlib toolbar 
    toolbar = NavigationToolbar2Tk(canvas, 
                                   tab4,pack_toolbar=False) 
    toolbar.update() 
  
    # placing the toolbar on the Tkinter window 
    canvas.get_tk_widget().grid(column=6,row=0,rowspan=30)
    
    
def write_csv_FRS(T,Saf,Sdf,ksiNS,nf,nst):

    fn = 'FloorSpectra_Damp'+str(ksiNS*100)+'_Floor'+str(nf)+'of'+str(nst)+'.csv'

    with open(fn, 'w', newline='') as csvfile:
        # Creating a CSV writer object
        csvwriter = csv.writer(csvfile)
        header = ['T (s)','SAF (g)','SDF (m)']
        csvwriter.writerow(header)
        for j in range(len(T)):
            row = [T[j],Saf[j],Sdf[j]]
            csvwriter.writerow(row)
            
        ttk.Label(tab4,text=f"Data exported to: {fn}").grid(column=0,row=30,columnspan = 5,sticky=tk.W)
                
root = tk.Tk() 
root.title("FRS Estimator") 
root.geometry("1000x600") 

for i in range(35):
    root.rowconfigure(i, weight=1)

for j in range(20):
    root.columnconfigure(i, weight=1)


tabControl = ttk.Notebook(root) 
  
tab1 = ttk.Frame(tabControl) 
tab2 = ttk.Frame(tabControl)
tab3 = ttk.Frame(tabControl)
tab4 = ttk.Frame(tabControl)
  
tabControl.add(tab1, text ='Acceleration-displacement Capacity Curve') 
tabControl.add(tab2, text ='Ground Response Spectrum',state='disabled') 
tabControl.add(tab3, text ='Capacity Spectrum Method',state='disabled') 
tabControl.add(tab4, text ='Floor Response Spectrum',state='disabled') 
tabControl.pack(expand = 1, fill ="both") 
  
ttk.Label(tab1,text = 'Say').grid(column=0,row=10,sticky=tk.E)  
ttk.Label(tab1,text = 'Dy').grid(column=0,row=11,sticky=tk.E)
ttk.Label(tab1,text = 'Du').grid(column=0,row=12,sticky=tk.E)
ttk.Label(tab1,text = 'Number of storeys').grid(column=0,row=15,sticky=tk.E)

ttk.Label(tab1,text = 'g').grid(column=2,row=10,sticky=tk.W) 
ttk.Label(tab1,text = 'm').grid(column=2,row=11,sticky=tk.W) 
ttk.Label(tab1,text = 'm').grid(column=2,row=12,sticky=tk.W) 


ttk.Label(tab1,text = '     ').grid(column=3,row=10,sticky=tk.W) 
ttk.Label(tab1,text = '     ').grid(column=3,row=11,sticky=tk.W) 
ttk.Label(tab1,text = '     ').grid(column=3,row=12,sticky=tk.W) 

ttk.Label(tab1,text = '     ').grid(column=3,row=16,sticky=tk.W) 

Say_entry = ttk.Entry(tab1,validate="key",
                        validatecommand=(root.register(validate_entry), "%S"))
Say_entry.grid(column=1,row=10, padx=5, pady=5)

Dy_entry = ttk.Entry(tab1,validate="key",
                        validatecommand=(root.register(validate_entry), "%S"))
Dy_entry.grid(column=1,row=11,padx=5, pady=5) 

Du_entry = ttk.Entry(tab1,validate="key",
                        validatecommand=(root.register(validate_entry), "%S"))
Du_entry.grid(column=1,row=12,padx=5, pady=5) 

current_value = tk.StringVar(value=2)
nst_entry = ttk.Spinbox(tab1,from_=2,to=8,textvariable=current_value,wrap=True)




#ttk.Entry(tab1,validate="key",
                        #validatecommand=(root.register(validate_entry), "%S"))
nst_entry.grid(column=1,row=15,sticky=tk.EW,padx=5, pady=5) 

plot_PO(0,0,0)
struct1 = Structure(0,0,0,1) 

submit_button = Button(master = tab1,  
                     command = getInput_PO, 
                     height = 2,  
                     width = 20, 
                     text = "Input Structural Data") 

# place the button  
# in main window 
submit_button.grid(column=1,row=20,sticky=tk.N)


tabGM_en = 0
open_button = tk.Button(tab2, text="Open CSV File", command=open_csv_file)
open_button.grid(column=1,row=0)

ttk.Label(tab2,text = 'TC').grid(column=0,row=23,sticky=tk.E)  
ttk.Label(tab2,text = 'TD').grid(column=0,row=24,sticky=tk.E)

ttk.Label(tab2,text = 's').grid(column=2,row=23,sticky=tk.W)  
ttk.Label(tab2,text = 's').grid(column=2,row=24,sticky=tk.W)

TC_entry = ttk.Entry(tab2, validate="key",
                        validatecommand=(root.register(validate_entry), "%S"))
TC_entry.grid(column=1,row=23)
TD_entry = ttk.Entry(tab2, validate="key",
                        validatecommand=(root.register(validate_entry), "%S"))
TD_entry.grid(column=1,row=24)

tree = ttk.Treeview(tab2, columns=("1", "2", "3"), show="headings", selectmode ='browse',height = 15)
tree.column('1',width=120)
tree.column('2',width=120)
tree.column('3',width=120)
tree.grid(column=0,row=1,rowspan=20,columnspan=5)
tree.heading("1", text="T (s)")  
tree.heading("2", text="Sa (g)")   
tree.heading("3", text="Sd (m)")  

status_label = tk.Label(tab2, text="", padx=20, pady=10)
status_label.grid(column=0,row=30)

verscrlbar = ttk.Scrollbar(tab2, 
                       orient ="vertical", 
                       command = tree.yview)

#verscrlbar.pack(side ='right', fill ='x')
verscrlbar.grid(column=5, row=1, rowspan=20,sticky=tk.NS)
 
# Configuring treeview
tree.configure(yscrollcommand = verscrlbar.set)

zspct = np.zeros((1000,3))

zspctO = GroundSpectrum(zspct)

plot_GS(zspctO)

inputCP_button = Button(master = tab2,  
                     command = getInput_CP, 
                     height = 2,  
                     width = 40, 
                     text = "Input Corner Periods") 

inputCP_button.grid(column=1,row=26,sticky=tk.N)

run_button = Button(master = tab3,  
                     command = lambda:CapacitySpectrum(struct1,spectrum), 
                     height = 2,  
                     width = 30, 
                     text = "Run Capacity Spectrum Method") 

run_button.grid(column=0,row=14,sticky=tk.N)

ttk.Label(tab4,text = 'Floor Number').grid(column=0,row=10,sticky=tk.E)  
ttk.Label(tab4,text = 'Non-structural damping ratio').grid(column=0,row=11,sticky=tk.E)

#fn_entry = ttk.Entry(tab4)



ksi_entry = ttk.Entry(tab4, validate="key",
                        validatecommand=(root.register(validate_entry), "%S"))
ksi_entry.grid(column=1,row=11, padx=5, pady=5)



est_button = Button(master = tab4,  
                     command = getInput_FRS, 
                     height = 2,  
                     width = 30, 
                     text = "Estimate Floor Response Spectrum") 

est_button.grid(column=0,row=13,columnspan=2)

save_button = Button(master = tab4,  
                     command = lambda:write_csv_FRS(floorspect.T,floorspect.Saf,floorspect.Sdf,floorspect.ksiNS,floorspect.nf,struct1.nst), 
                     height = 2,  
                     width = 30, 
                     text = "Export to csv") 

save_button.grid(column=0,row=14,columnspan=2)

fz = FloorSpectrum(zspctO.Gspectrum[:,0],zspctO.Gspectrum[:,1],zspctO.Gspectrum[:,2],0,0)

plot_FS(fz)

root.mainloop()   