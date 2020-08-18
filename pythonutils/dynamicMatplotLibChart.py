### DYNAMIC CHART in  JUPYTER NOTEBOOK

fig,ax = plt.subplots(1,1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(0,10)
ax.set_ylim(0,1000)

xData=[]
yData=[]

for curVal in range(100):
    xData.append(curVal)
    yData.append(np.random.randint(1000))
    if(ax.lines):
        if(curVal%10==0 and curVal > 1):
            ax.set_xlim(0,curVal)
        if(np.min(yData)<0):
            ax.set_ylim(np.min(yData),np.max(yData))
        else:
            ax.set_ylim(0,np.max(yData))
        for line in ax.lines:
            line.set_xdata(xData)
            line.set_ydata(yData)
            time.sleep(1)
    else:
        ax.plot(xData,yData,color='b')
    fig.canvas.draw()    
