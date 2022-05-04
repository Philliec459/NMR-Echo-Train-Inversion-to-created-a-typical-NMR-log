#!/usr/bin/env python
# coding: utf-8

# # NMR Echo Train processing and creation of an NMR log:

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 12:33:54 2020

@author: craig
"""
#get_ipython().run_line_magic('matplotlib', 'inline')


from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ## Read in NMR T2 bin data and create an echo train:

# In[2]:


df = pd.read_csv('nmr.csv')
#print(df.head())


# ## Define the SciPy curve_fit functions, Level Spacing, Stacking Levels and Noise:

# In[3]:



def func(x,p1,p2,p3,p4,p5,p6,p7,p8):
    return (p1*np.exp(-x/4)+p2*np.exp(-x/8)+p3*np.exp(-x/16)+p4*np.exp(-x/32)+p5*np.exp(-x/64)+p6*np.exp(-x/128)+p7*np.exp(-x/256)+p8*np.exp(-x/512))

def func_2bin(x,p1,p2,t21,t22):
    return (p1*np.exp(-x/t21)+p2*np.exp(-x/t22))


'''
    ---------------------------------------------------------------------------
    This is where we want to stack the data for a less noisy echotrain
    
    Set Level Spacing, Stacking levels and Noise below
      
      
            *** Stack_levels from 1 (no stacking) to 8 only stacked levels *** 
    ---------------------------------------------------------------------------
'''
level_spacing = .5 #well level spacing and this model works best on 0.5 data???
stack_levels  = 3   #odd numbers are geat, even changes level spacing in new depth offset of 0.25 ' 
noise = 1
'''
    ---------------------------------------------------------------------------
             *** Stack_levels from 1 (no stacking) to 8 only stacked levels *** 
    ---------------------------------------------------------------------------
'''



T2 =[2,4,8,16,32,64,128,256,512,1024,2048]


# # Create Echo Trains and then SciPy curve_fit(???):

# In[4]:



deptharray=[]
mphiarray=[]    
mbviarray=[]
mffiarray=[]    
echo=[]
ystack=[]
T21=[]
T22 = []
#ystack2=[]
#ystackecho=[]

for index, row in df.iterrows():
    P0=0
    P1=row['P1']
    P2=row['P2']
    P3=row['P3']
    P4=row['P4']
    P5=row['P5']
    P6=row['P6']
    P7=row['P7']
    P8=row['P8']
    P9=0
    P10=0
    mphi=row['MPHI']
    mbvi=row['MBVI']
    depth=row['Depth']
    bins=[P0,P1,P2,P3,P4,P5,P6,P7,P8,P9,P10]


    #Define the data to be fit with some noise:
    '''
      This is the xdata for TE x number of echoes for 240ms of echo train data
    '''
    xdata = np.linspace(0, 240, 201)
 
    '''
      this curve fits would be the original fit of the echo train where we have bin porosity
    '''
    y = func(xdata,P1, P2, P3,  P4, P5, P6,  P7,  P8)
    #np.random.seed(1729)


    
    '''
      now add noise to the original echo train data to create a typical echo train
    '''

    y_noise  = noise * np.random.normal(size=xdata.size)
    ydata = y + y_noise


    
     
    '''
      Store Echo Trains in np array
    '''
    echo.append(ydata)
    #yecho = np.array(echo)
    
    
    
                 
    
    if index > stack_levels -1:
        
        if stack_levels == 1:           
            ystack=ydata
        else:
            for k in range(1,stack_levels): 
    
                if stack_levels == 2:
                    ystack = (echo[index-1]+echo[index])/stack_levels                 
                elif stack_levels == 3:
                    ystack = (echo[index-2]+echo[index-1]+echo[index])/stack_levels
                elif stack_levels ==4:
                    ystack = (echo[index-3]+echo[index-2]+echo[index-1]+echo[index])/stack_levels
                elif stack_levels == 5:
                    ystack = (echo[index-4]+echo[index-3]+echo[index-2]+echo[index-1]+echo[index])/stack_levels
                elif stack_levels == 6:
                    ystack = (echo[index-5]+echo[index-4]+echo[index-3]+echo[index-2]+echo[index-1]+echo[index])/stack_levels    
                elif stack_levels == 7:
                    ystack = (echo[index-6]+echo[index-5]+echo[index-4]+echo[index-3]+echo[index-2]+echo[index-1]+echo[index])/stack_levels    
                elif stack_levels == 8:
                    ystack = (echo[index-7]+echo[index-6]+echo[index-5]+echo[index-4]+echo[index-3]+echo[index-2]+echo[index-1]+echo[index])/stack_levels    
                    
                else:
                    print('Stack Levels out of bounds')
   
    
        '''
        -----------------------------------------------------------------------
          Optimization using curve_fit of SciPy: 
            
             Constrain the optimization to the region of 0 <= a <= 3, 0 <= b <= 1 and 0 <= c <= 0.5:
        
        -----------------------------------------------------------------------
        '''

        #def func_2bin(x,p1,p2,t21,t22):
        #    return (p1*np.exp(-x/t21)+p2*np.exp(-x/t22))
        #popt, pcov = curve_fit(func_2bin, xdata, ystack, method='trf' , bounds=(0.0, [30, 30, 32, 512]))
        popt_2bin, pcov_2bin = curve_fit(func_2bin, xdata, ystack, method='trf' , bounds=([0, 0, 1, 45], [30, 30, 22, 512]))

        y_2bin = func_2bin(xdata,popt_2bin[0], popt_2bin[1], popt_2bin[2],  popt_2bin[3])

        ######print('P1 =' , popt_2bin[0],'at T2 of ', popt_2bin[2],'ms', 'and P2 =' , popt_2bin[1],'at T2 of ', popt_2bin[3],'ms')

        if popt_2bin[0] + popt_2bin[1] > 0:
            T21.append(popt_2bin[2])
            T22.append(popt_2bin[3])




        #popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [20, 20, 20, 20, 20, 20, 20, 20]))
        #popt, pcov = curve_fit(func, xdata, ystack, method='trf', max_nfev=100,bounds=(0.05, [20, 20, 20, 20, 20, 20, 20, 20]))
        popt, pcov = curve_fit(func, xdata, ystack, method='trf', bounds=(0.05, [20, 20, 20, 20, 20, 20, 20, 20]))
        #popt, pcov = curve_fit(func_2bin, xdata, ystack, method='trf' )#, bounds=(0.0, [0.1, 0.1, 20, 0, 0, 0, 20, 0]))

        ''' Calculate BVI and MPHI doe 33ms T2 Cutoff'''
        mbviarray.append(popt[0]+popt[1]+popt[2]+0.044*popt[3])    
        mphiarray.append(popt[0]+popt[1]+popt[2]+popt[3]+popt[4]+popt[5]+popt[6]+popt[7])
        mffiarray.append((1-0.044)*popt[3]+popt[4]+popt[5]+popt[6]+popt[7])


        # if stack_levels==1:
        #     deptharray.append(depth)
        # else:
        #     deptharray.append(depth - (stack_levels-1)/2 + ((stack_levels-1)/2)*level_spacing)
        
        
        #this works great of 0.5 feet level spacing and gets depth perfect
        #deptharray.append(depth - (stack_levels-1)/2 + ((stack_levels-1)/2)*level_spacing)
        #the above with 0 noise, 1' rlev and 7 level stacking ends up with depth 3' deep  
      
        
        #ended up 0.75 too deep with 1' rlev. Not bad, but why
        #deptharray.append(depth - (stack_levels-1)/2 + ((stack_levels-1)/2)*0.25)
 
        #this is perfect for 1' rlev, but 1.5' too shallow for 0.5 rlev with 7 level stack Why?
        #0.5' rlev with 3 stacked levels is 0.5' shallow
        #with 0.5' rlev and 5 level stack was 1' shallow 
        #deptharray.append(depth - (stack_levels-1)/2 )
   
    
        # this works for 0.5' rlev
        #deptharray.append(depth - (stack_levels-1)/2 + 0.25*(stack_levels-1))
        #but 1' rlev is 1' to deep
 
    
        # for the lack of anything else, we will use this for now. 

        # if level_spacing == 0.25:
        #     deptharray.append(depth - (stack_levels-1)/2 + 0.375*(stack_levels-1))
        # elif level_spacing == 0.5:
        #     deptharray.append(depth - (stack_levels-1)/2 + 0.25*(stack_levels-1))
        # elif level_spacing==1:
        #     deptharray.append(depth - (stack_levels-1)/2 )

        #multiplier = -0.5*level_spacing + 0.5
        multiplier = (1 - level_spacing)/2
        deptharray.append(depth - (stack_levels-1)/2 + multiplier*(stack_levels-1))
        
        #deptharray.append(depth - (stack_levels-1)/2 )

        newbins = [        P0,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],P9,P10]
        newbins_extra = [0,P0,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],P9,P10]
        ######print(deptharray)
        ######print(mphiarray)

        T2       =[  2,4,8,16,32,64,128,256,512,1024,2048]
        T2_extra =[1,2,4,8,16,32,64,128,256,512,1024,2048]

        
        def moving_average(x, w):
            return np.convolve(x, np.ones(w), 'valid') / w
        

        newbins_m = moving_average(newbins_extra,2)
        T2_m = moving_average(T2_extra,2)
        #print(len(newbins_m,len(T2_m))
        #print(T2_m)
        #print(newbins_m)
       
        
        '''
        Put moving average bin porosities back on the proper T2 times using numpy interpolation 
        '''
        x  = T2
        xp = T2_m
        fp = newbins_m
        newbins_m_int     = np.interp(x,xp,fp)
        
        #sum of arrays to see if equal:
        newbins_m_int_sum = np.sum(newbins_m_int)
        newbins_sum       = np.sum(newbins)
        bins_sum = np.sum(bins)
        
        #print(newbins_m_int ,'Ratio of newbin_int_sum vs. solved bin porosities =', round(newbins_m_int_sum/newbins_sum,2))
        print('Index =',index, ', new bins interpolated =', newbins_m_int ,', Newbin_int_sum  =', round(newbins_m_int_sum,2),', MPHI_stack_fit=',round((popt[0]+popt[1]+popt[2]+popt[3]+popt[4]+popt[5]+popt[6]+popt[7]),2),', MPHI =',round(bins_sum,2))
        
        #print(newbins_m_int ,'Ratio of newbin_int_sum vs. solved bin porosities =', round(newbins_m_int_sum/newbins_sum,2))
        #print(newbins_m_int ,'Newbin_int_sum  =', round(newbins_m_int_sum,2),', MPHI=',round((popt[0]+popt[1]+popt[2]+popt[3]+popt[4]+popt[5]+popt[6]+popt[7]),2))
        
        
        
        
        '''
        Plot Echo Train, T2 Dist and NMR Log data
        '''
        for p in range(1):
            p=0
            
            fig, ax = plt.subplots(1,3,figsize=(15,4))      
            updated_x = xdata
            #updated_y = ydata
            updated_y = ystack
            # not work updated_y=ystackecho[index]
            # works -- updated_y = echo[index]
            # works too - updated_y = yecho[index]
            updated_yfit = func(xdata, *popt)
            ax[0].plot(updated_x,updated_y    ,c='green',label='Echo Train')
            ax[0].plot(updated_x,updated_yfit ,c='red',linewidth=3, label='Fit')
            #ax[0].plot(updated_x,y_2bin ,c='black',linewidth=2, label='Fit 2 Bin')

            ax[0].set_xlim(0,240)
            ax[0].set_ylim(0,30)
            ax[0].set_ylabel('Amplitude (V/V)')
            ax[0].set_xlabel('Time (ms)')
            ax[0].set_title('NMR Echo Train')
            ax[0].legend(loc='upper right')
            ax[0].grid()
            
            #ax.figure(2)
            #ax[1].semilogx(T2,bins,c='green', linewidth=3,label='Original Bins')
            #ax[1].semilogx(T2,newbins,c='red', linewidth=1, label='New Bins')
            #ax[1].semilogx(T2_m,newbins_m,c='blue', linewidth=3, label='RA Bins')
            ax[1].semilogx(T2,newbins_m_int,c='red', linewidth=3, label='RA Int Bins')
            
            ax[1].semilogx(33,0.0 ,'k*',label='33ms Cutoff')
            #print('P1 =' , popt_2bin[0],'at', popt_2bin[2],'ms', 'and P2 =' , popt_2bin[1],'at', popt_2bin[3],'ms')
            #ax[1].semilogx(popt_2bin[2],popt_2bin[0], 'b*')
            #ax[1].semilogx(popt_2bin[3],popt_2bin[1], 'r*')
    
            ax[1].set_xlim(2,1024)
            ax[1].set_ylim(0,15)
            ax[1].set_ylabel('Bin Porosity (V/V)')
            ax[1].set_xlabel('T2 (ms)')
            ax[1].set_title('Echo Train Inversion for T2 Distribution')
            ax[1].axvspan(2,33,alpha=0.2,color='blue',label='BVI')
            ax[1].axvspan(33,1024,alpha=0.5,color='yellow',label='FFI')
            ax[1].legend()
            ax[1].grid()
            
            ax[2].plot(mphiarray,deptharray    ,c='red',label='NMR Porosity')
            ax[2].plot(mbviarray,deptharray ,c='blue',linewidth=2)
            ax[2].set_xlim(50,0)
            ax[2].set_ylim(7202,7177)
            #ax[2].set_ylim(12.5,0)
            ax[2].set_ylabel('Depth)')
            ax[2].set_xlabel('Porosity')
            ax[2].set_title('NMR Log using 33ms T2 Cutoff')
            
            ax[2].fill_betweenx(deptharray, mphiarray, 0,  color='yellow', alpha=0.9, label='FFI')   
            ax[2].fill_betweenx(deptharray, mbviarray, 0,  color='blue'  , alpha=0.9, label='BVI')    
            ax[2].legend(loc='upper left')
            ax[2].grid()
              
            plt.draw()  
            updated_x = xdata
            updated_y = ystack
            updated_yfit = func(xdata, *popt)
            plt.pause(0.25)
            #fig.clear()
            plt.close(fig)
            #fig.clf()
                 
            



a=10**(np.mean(np.log10(T21)))
b=10**(np.mean(np.log10(T22)))

print()
#print('log Mean of T21 =',round(a,1),'and log Mean of T22 =',round(b,1))
print()


plt.hist(T21,label='T21')
plt.hist(T22,label='T22')
plt.hist(a,label='lm of T21')
plt.hist(b,label='lm of T22')
plt.xlim(2,1024)
plt.grid()
plt.legend()
plt.xscale('log')
#plt.show()

plt.scatter(T21, mphiarray,label='T21')
#plt.scatter(0,a,'r*')
plt.scatter(T22, mphiarray,label='T22')
plt.xlim(2,1024)
#plt.scatter(0,b,'k*')
#plt.yscale('log',basey=10)
#plt.xscale('log')
plt.xscale('log')
plt.legend()
plt.grid()
#plt.show()
#print(T22)



