# -*- coding: utf-8 -*-
#  ======================== data for deep ======================================
idtype= []


with open("file_ID.csv", 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
            idtype.append(row)
csvfile.close()



data=[]
with open("dataxy.csv", 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
            data.append(row)
csvfile.close()



data_new=[]
for i in range(0,len(data)):
  p0row=data[i]
  if p0row['sensor']!='m052' and p0row['sensor'] != 'd017' and  p0row['sensor']!='d018'  and  p0row['sensor']!='i001'  and  p0row['sensor']!='i002'  and  p0row['sensor']!='i003'  and  p0row['sensor']!='i004' and  p0row['sensor']!='i005' and  p0row['sensor']!='i006' and p0row['sensor']!='i007' and  p0row['sensor']!='i008' and  p0row['sensor']!='i009' and  p0row['sensor']!='i010' :
    data_new.append(p0row)


data_source=data_new


#========================= make directory ======================================
saveFolder_trajectory= './img_trajectory/'
if not os.path.exists(saveFolder_trajectory):
    # Create a new directory because it does not exist 
    os.makedirs(saveFolder_trajectory)

saveFolder_speed= './img_speed/'
if not os.path.exists(saveFolder_speed):
    os.makedirs(saveFolder_speed)

counter=0

section_all=[]
section_all_2=[]
section_start_end=[]
section_1=[]
section_2=[]
count_num=1
all_name=[]
summary_speed=[]
summary_acceleration=[]
TH=60


for row in idtype:
  print('person number'+str(count_num))
  id_num=float(row.get('id'))
  id_num_str=row.get('id')

  id_type=float(row.get('type'))
  
  
  id_NUM=id_num_str
  type_NUM=id_type


  id=id_num
  temp=[]
  for j in range(0,len(data_source)):
    p0row=data_source[j]
    if int(p0row['patient'])==id:
      temp.append(p0row)

  data=temp

  #==============read csv file as input(act matrix)==============================
  # ========================== save time in seconds in time_s ====================
  for i in range (0,len(data)):
    point=data[i:i+1]
    for p1Row in point:
      time_1 = p1Row.get('time')
      t_hour_1=float(time_1[0:2])
      t_min_1=float(time_1[3:5])
      t_sec_1=float(time_1[6:10])
      time_sec_1=t_hour_1*3600+t_min_1*60+t_sec_1 
    p1Row['time_s']=time_sec_1


  #=====================================================================================
  # find sensor with high speed for person
  def high_speed(data):
    x=[]
    y=[]
    index=[]
    data_temp=[]
    for i in range(0,len(data)):
      g=data[i]
      x_temp=g['X']
      y_temp=g['Y']
      x.append(x_temp)
      y.append(y_temp)
      index.append(i)
      
    distance=[]
    time=[]
    speed=[]
    speed.append(0)
    for k in range(0,len(x)-1):
      g1= data[k]
      g2=data[k+1]
      x1 =float(g1['X'])
      y1= float(g1['Y'])
      x2 =float(g2['X'])
      y2= float(g2['Y'])
      time1=float(g1['time_s'])
      time2=float(g2['time_s'])

      dist_temp=(x2-x1)**2+(y2-y1)**2
      dist=math.sqrt(dist_temp)
      time_tmep=time2-time1
      distance.append(dist)
      time.append(time_tmep)

      if distance[k]==0 and time[k]==0:
        speed_temp=0
      elif distance[k]!=0 and time[k]==0:
          speed_temp=0
      else:
        speed_temp=distance[k]/time[k]
      speed.append(speed_temp)

    for L in range(0,len(speed)):
      if speed[L]==np.inf:
        speed[L][0]==0
      
    #define threshold for speed more than 15m/s
    for tt in range(0,len(speed)):
      gg=data[tt]
      if speed[tt]>=15:
        gg['speed']=speed[tt]
      else: 
        gg['speed']=speed[tt]
        data_temp.append(gg)


    x=[];
    y=[];
    index=[]
    data=data_temp
    return data


  #=====================================================================================
  # ==================== delete by distance more than 5 meter ====================
  def delete_by_dist(data):
    count=0
    data_temp=[]
    data_temp.append(data[0])
    x1=float(data[count]['X'])
    y1=float(data[count]['Y'])

    x2=float(data[count+1]['X'])
    y2=float(data[count+1]['Y'])
    while count< len(data)-2:

      dist=math.sqrt((x2-x1)**2+(y2-y1)**2)
      if dist<=5:
        data_temp.append(data[count+1])
        count=count+1

        x1=float(data[count]['X'])
        y1=float(data[count]['Y'])

        x2=float(data[count+1]['X'])
        y2=float(data[count+1]['Y'])
      else:

        count=count+1
        x2=float(data[count+1]['X'])
        y2=float(data[count+1]['Y'])

    data=data_temp 
    return data 

#===============================================================================
  size_delete_speed=1
  size_delete_dist=0
  while size_delete_dist!=size_delete_speed:
    data=delete_by_dist(data)
    size_delete_dist=len(data)

    data=high_speed(data)
    size_delete_speed=len(data)    


#===============================================================================
  def delete_bytime(data):
    data_temp=[]
    data_temp.append(data[0])


    for i in range (0,len(data)-1):
      point_1=data[i:i+1]
      for p1Row in point_1:
        time_1 = p1Row.get('time_s')

      point_2=data[i+1:i+2]
      for p2Row in point_2:
        time_2 = p2Row.get('time_s')
        
      time_diff=time_2-time_1



      if time_diff >0:
        data_temp.append(data[i+1:i+2][0])
    return data_temp
#===============================================================================
  len_1=0
  len_2=1
  while len_1!=len_2:
    len_1=len(data)
    data=delete_bytime(data)
    len_2=len(data)


#=======================================================================================
#=======================================================================================
#=======================================================================================
# ====================== deltax and delatay calculation=========================

  for i in range(0,len(data)-1):
    g1= data[i]
    g2=data[i+1]


    x1 =float(g1['X'])
    y1= float(g1['Y'])

    x2 =float(g2['X'])
    y2= float(g2['Y'])


    g1['deltax']=x2-x1
    g1['deltay']=y2-y1


  # ===========================define section ====================================
  #-------------------- define episode for every 60 seconds ---------------------
  count=1
  section_1=[]
  section_2=[]
  section_1.append(0)
  for i in range(len(data)):
    data_point_1=data[i:i+1]
    data_point_2=data[i+1:i+2]

    for p1Row in data_point_1:
      time_1 = p1Row.get('time')
      t_hour_1=float(time_1[0:2])
      t_min_1=float(time_1[3:5])
      t_sec_1=float(time_1[6:10])
      time_sec_1=t_hour_1*3600+t_min_1*60+t_sec_1 

    for p2Row in data_point_2:
      time_2 = p2Row.get('time')
      t_hour_2=float(time_2[0:2])
      t_min_2=float(time_2[3:5])
      t_sec_2=float(time_2[6:10])
      time_sec_2=t_hour_2*3600+t_min_2*60+t_sec_2
    
    time_diff=time_sec_2-time_sec_1


    if time_diff>TH:

      section_1.append(i+1)
      section_2.append(i)
      count=count+1
      
    elif time_diff<=TH  :
      p1Row['section']=count



        
  section_2.append(len(data)-1)

  # -----------------delete section with which start and end are  equal---------


  section_1_temp=[]
  section_2_temp=[]
  for i in range(0, len(section_1)):
    if section_1[i]!=section_2[i]:
      section_1_temp.append(section_1[i])
      section_2_temp.append(section_2[i])

  section_1=section_1_temp
  section_2=section_2_temp
  print(len(section_1))


  # delete section with less than 3 step length

  section_1_temp=[]
  section_2_temp=[]
  for i in range(0, len(section_1)):
    if section_2[i]-section_1[i]> 3:
      section_1_temp.append(section_1[i])
      section_2_temp.append(section_2[i])

  section_1=section_1_temp
  section_2=section_2_temp
  print(len(section_1))  


  for i in range(0,len(data)-2):
  #------------------------------------- calculate angle  ---------------------------------------------
    data_point_1=data[i:i+1]
    data_point_2=data[i+1:i+2]
    data_point_3=data[i+2:i+3]
    for p1Row in data_point_1:
      x1=float(p1Row.get('X'))
      y1=float(p1Row.get('Y'))
      time_1=p1Row.get('time_s')

    for p2Row in data_point_2:
      x2=float(p2Row.get('X'))
      y2=float(p2Row.get('Y'))
      time_2=p2Row.get('time_s')

    for p3Row in data_point_3:
      x3=float(p3Row.get('X'))
      y3=float(p3Row.get('Y'))
      time_3=p3Row.get('time_s')

    if (x2-x1)!=0:
      First_vec_angle=math.degrees(np.arctan((y2-y1)/(x2-x1)))
    else:
      First_vec_angle=90

    if (x3-x1)!=0:
      Second_vec_angle=math.degrees(np.arctan((y3-y1)/(x3-x1)))
    else:
      Second_vec_angle=90

    Angle=abs(Second_vec_angle-First_vec_angle)
    p1Row['Angle']=Angle
    data[len(data)-1]['Angle']=0
    data[len(data)-2]['Angle']=0


    # ---------------------SPEED AND ACCELERATION --------------------------------
    delta_x_first=x2-x1
    delta_y_first=y2-y1

    delta_x_second=x3-x2
    delta_y_second=y3-y2
    
    D_first=math.sqrt(delta_x_first**2+delta_y_first**2)
    D_second=math.sqrt(delta_x_second**2+delta_y_second**2)

    time_diff_fist=time_2-time_1
    time_diff_second=time_3-time_2
    
    speed_first=D_first/time_diff_fist
    speed_second=D_second/time_diff_second
    summary_speed.append(speed_first)

    acceleration=(speed_second-speed_first)/time_diff_fist
    summary_acceleration.append(acceleration)

    for p1Row in data_point_1:
      p1Row['speed']=speed_first


  # ================================ create plot ================================
  # ========================== create x and y file for every section =============
  # ===========================================================================

  for iiii in range(0,len(section_1)): 
    XX=[]
    YY=[]
    XY_circle_unique=[]
    XY_circle=[]
    for j in range(section_1[iiii],section_2[iiii]+1):
      data_point_1=data[j:j+1]

      for p1Row in data_point_1:
        x1=2*float(p1Row.get('X'))
        y1=2*float(p1Row.get('Y'))
        XX.append(x1)
        YY.append(y1)

    X_circle=[]
    Y_circle=[]
    XY_circle=[]
    a=[]

    for ii in range(0,len(XX)-2):

      line1 = LineString([(XX[ii], YY[ii]), (XX[ii+1], YY[ii+1])])
      for jj in range(ii+2,len(XX)-1):
        line2 = LineString([(XX[jj], YY[jj]), (XX[jj+1], YY[jj+1])])
        a=(line1.intersection(line2))
        if a:
          if a.type!='LineString':
            X_circle.append(a.x)
            
            Y_circle.append(a.y)
            a=[a.x,a.y]
            XY_circle.append(a)

    XY_circle=np.asarray(XY_circle)
    #Y_circle=np.asarray(Y_circle)

    if len(XY_circle)>=1:
      XY_circle_unique= np.unique(XY_circle, axis=0)
      idx=[]
      for m in range(0,len(XY_circle_unique)):
        row_x=XY_circle_unique[m][0]
        row_y=XY_circle_unique[m][1]
        count=0
        for n in range(0,len(XY_circle)):
          row_x_temp=XY_circle[n][0]
          row_y_temp=XY_circle[n][1]
          if row_x==row_x_temp and row_y==row_y_temp:
            count=count+1
        idx.append(count)
      


    # Create a Figure
    fig = plt.figure(figsize=(2*10,2*13), dpi=5)

    # Set up Axes
    ax = fig.add_subplot(111)
    ax.set_xlim([0, 2*10])
    ax.set_ylim([0, 2*13])
    plt.axis('off')

    # Scatter the data
    ax.plot(XX,YY,'-',linewidth=10)
    if len(XY_circle_unique)>=1:
      for kk in range(0,len(XY_circle_unique)):
        circle1 = plt.Circle((XY_circle_unique[kk][0],XY_circle_unique[kk][1]),0.5,color='r')
        ax.add_artist(circle1)


    plt.close()
    

    # Show the plot
    fig.show('off')
    #plt.figure(figsize=(26,12),)
    

    fig.savefig(saveFolder_trajectory + str(counter)+'_'+data[section_1[iiii]]['time']+'_'+data[section_2[iiii]]['time'] + '.png')


  # ================================ create plot ================================
  # ========================== create speed image================================
  # ===========================================================================


    XX=[]
    YY=[]
    SS=[] # speed
    XY_circle_unique=[]
    XY_circle=[]
    for j in range(section_1[iiii],section_2[iiii]+1):
      data_point_1=data[j:j+1]

      for p1Row in data_point_1:
        x1=2*float(p1Row.get('X'))
        y1=2*float(p1Row.get('Y'))
        s1=float(p1Row.get('speed'))
        XX.append(x1)
        YY.append(y1)
        SS.append(s1)



    # Create a Figure
    fig = plt.figure(figsize=(2*10,2*13), dpi=5)

    # Set up Axes
    ax = fig.add_subplot(111)
    ax.set_xlim([0, 2*10])
    ax.set_ylim([0, 2*13])
    plt.axis('off')

    # Scatter the data
    
    for k in range(0,len(XX)-1):
      V=SS[k]
      
      XX_temp=[XX[k],XX[k+1]]
      YY_temp=[YY[k],YY[k+1]]

      if V>=0 and V<2:
        c='purple'
      elif V>=2 and V<4:
        c='violet'
      elif V>=4 and V<6:
        c='blue'
      elif V>=6 and V<8:
        c='cyan'
      elif V>=8 and V<10:
        c='green'
      elif V>=10 and V<12:
        c='yellow'
      elif V>=12 and V<14:
        c='orange'
      elif V>=14 :
        c='red' 
      ax.plot(XX_temp,YY_temp,lw=10,color=c)
    

    plt.close()
    fig.show('off')

    

    fig.savefig(saveFolder_speed + str(counter)+'_'+data[section_1[iiii]]['time']+'_'+data[section_2[iiii]]['time'] + '.png')


    aa=[ counter,sec_type[iiii],sec_id[iiii]]
    section_all_2.append(aa)


    counter=counter+1
    np.savetxt("section_all_2.csv", section_all_2, delimiter=",") 
  count_num=count_num+1