Y1func<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_out
  for (i in 1:dim(Y)[1]){
    count=0
    for (j in 1:dim(Y)[2]){
      if (X[i,9,j]==1){count=count+1}
      else{count=0}
      if (count==3){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}


Y1newfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_out
  for (i in 1:dim(Y)[1]){
    count=0
    for (j in 1:dim(Y)[2]){
      if (X[i,9,j]==1){count=count+1}
      else{count=0}
      if (count==2){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}

Y1newerfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_out
  for (i in 1:dim(Y)[1]){
    count=0
    for (j in 1:dim(Y)[2]){
      if (X[i,9,j]==1){count=count+1}
      else{count=0}
      if (count==4){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}

Y2func<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    count1=0
    count2=0
    mark=0
    outj=0
    for (j in 1:dim(Y)[2]){
      if (X[i,9,j]==1){count1=count1+1}
      else{count1=0}
      if (count1==2){
        mark=1
        outj=j
      }
    }
    if (mark==1 && outj<dim(Y)[2]){
      for (k in (outj+1):dim(Y)[2]){
        # print(c(i,k))
        if (X[i,9,k]==1){count2=count2+1}
        else{count2=0}
        if (count2==2){
          Y_output[i,dim(Y)[2]]=1
        } 
      }
    }
  }
  return(Y_output)}


Y3func<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    time3=c()
    time4=c()
    for (j in 1:dim(Y)[2]){
      if (X[i,8,j]==1){time3=c(time3,j)}
      if (X[i,9,j]==1){time4=c(time4,j)}
    }
    if (!(is.null(time3)|is.null(time4))){
      if (time3[1]<time4[1]){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}


Y3newfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    time3=c()
    time4=c()
    for (j in 1:dim(Y)[2]){
      if (X[i,8,j]==1){time3=c(time3,j)}
      if (X[i,9,j]==1){time4=c(time4,j)}
    }
    if (!(is.null(time3)|is.null(time4))){
      if (time3[1]<time4[1] && time3[1]<5){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}

Y3newerfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    time3=c()
    time4=c()
    for (j in 1:dim(Y)[2]){
      if (X[i,8,j]==1){time3=c(time3,j)}
      if (X[i,9,j]==1){time4=c(time4,j)}
    }
    if (!(is.null(time3)|is.null(time4))&&length(time3)>1){
      if (time3[1]<time4[1] && time3[2]<time4[1]){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}

Y3newestfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    time3=c()
    time4=c()
    for (j in 1:dim(Y)[2]){
      if (X[i,8,j]==1){time3=c(time3,j)}
      if (X[i,9,j]==1){time4=c(time4,j)}
    }
    if (!(is.null(time3)|is.null(time4))&&length(time3)>2){
      if (time3[1]<time4[1] && time3[2]<time4[1] && time3[3]<time4[1]){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}

Y3altfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    time3=c()
    time4=c()
    for (j in 1:dim(Y)[2]){
      if (X[i,8,j]==1){time3=c(time3,j)}
      if (X[i,9,j]==1){time4=c(time4,j)}
    }
    if (!(is.null(time3)|is.null(time4))&&length(time3)>3){
      if (time3[1]<time4[1] && time3[2]<time4[1] && time3[3]<time4[1] && time3[4]<time4[1]){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}

Y4func<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    time3=c()
    time4=c()
    for (j in 1:dim(Y)[2]){
      if (X[i,8,j]==1){time3=c(time3,j)}
      if (X[i,9,j]==1){time4=c(time4,j)}
    }
    if (!(is.null(time3)|is.null(time4))){
      test=0
      for (q in 1:length(time3)){if(any(abs(time4-time3[q])<5)){test=test+1}}
      if (test>0){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}


Y4newfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    time3=c()
    time4=c()
    for (j in 1:dim(Y)[2]){
      if (X[i,8,j]==1){time3=c(time3,j)}
      if (X[i,9,j]==1){time4=c(time4,j)}
    }
    if (!(is.null(time3)|is.null(time4))){
      test=0
      for (q in 1:length(time3)){if(any(abs(time4-time3[q])<1)){test=test+1}}
      if (test>0){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}


Y4newerfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    time3=c()
    time4=c()
    for (j in 1:dim(Y)[2]){
      if (X[i,8,j]==1){time3=c(time3,j)}
      if (X[i,9,j]==1){time4=c(time4,j)}
    }
    if (!(is.null(time3)|is.null(time4))){
      test=0
      for (q in 1:length(time3)){if(any(abs(time4-time3[q])<1)){test=test+1}}
      if (test>1){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}


Y4newestfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    time3=c()
    time4=c()
    for (j in 1:dim(Y)[2]){
      if (X[i,8,j]==1){time3=c(time3,j)}
      if (X[i,9,j]==1){time4=c(time4,j)}
    }
    if (!(is.null(time3)|is.null(time4))){
      test=0
      for (q in 1:length(time3)){if(any(abs(time4-time3[q])<5)){test=test+1}}
      if (test>3){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}

Y4altfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    time3=c()
    time4=c()
    for (j in 1:dim(Y)[2]){
      if (X[i,8,j]==1){time3=c(time3,j)}
      if (X[i,9,j]==1){time4=c(time4,j)}
    }
    if (!(is.null(time3)|is.null(time4))){
      test=0
      for (q in 1:length(time3)){if(any(abs(time4-time3[q])<5)){test=test+1}}
      if (test>2){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}

Y4altnewfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    time3=c()
    time4=c()
    for (j in 1:dim(Y)[2]){
      if (X[i,8,j]==1){time3=c(time3,j)}
      if (X[i,9,j]==1){time4=c(time4,j)}
    }
    if (!(is.null(time3)|is.null(time4))){
      test=0
      for (q in 1:length(time3)){if(any(abs(time4-time3[q])<5)){test=test+1}}
      if (test>1){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}

Y4alt2func<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    time3=c()
    time4=c()
    for (j in 1:dim(Y)[2]){
      if (X[i,8,j]==1){time3=c(time3,j)}
      if (X[i,9,j]==1){time4=c(time4,j)}
    }
    if (!(is.null(time3)|is.null(time4))){
      test=0
      for (q in 1:length(time3)){if(any(abs(time4-time3[q])<3)){test=test+1}}
      if (test>0){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}


Y4alt2newfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    time3=c()
    time4=c()
    for (j in 1:dim(Y)[2]){
      if (X[i,8,j]==1){time3=c(time3,j)}
      if (X[i,9,j]==1){time4=c(time4,j)}
    }
    if (!(is.null(time3)|is.null(time4))){
      test=0
      for (q in 1:length(time3)){if(any(abs(time4-time3[q])<3)){test=test+1}}
      if (test>1){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}

Y4alt2newerfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    time3=c()
    time4=c()
    for (j in 1:dim(Y)[2]){
      if (X[i,8,j]==1){time3=c(time3,j)}
      if (X[i,9,j]==1){time4=c(time4,j)}
    }
    if (!(is.null(time3)|is.null(time4))){
      test=0
      for (q in 1:length(time3)){if(any(abs(time4-time3[q])<3)){test=test+1}}
      if (test>2){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}

Y4alt2newestfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    time3=c()
    time4=c()
    for (j in 1:dim(Y)[2]){
      if (X[i,8,j]==1){time3=c(time3,j)}
      if (X[i,9,j]==1){time4=c(time4,j)}
    }
    if (!(is.null(time3)|is.null(time4))){
      test=0
      for (q in 1:length(time3)){if(any(abs(time4-time3[q])<3)){test=test+1}}
      if (test>3){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}

Y4alt3func<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    time3=c()
    time4=c()
    for (j in 1:dim(Y)[2]){
      if (X[i,8,j]==1){time3=c(time3,j)}
      if (X[i,9,j]==1){time4=c(time4,j)}
    }
    if (!(is.null(time3)|is.null(time4))){
      test=0
      for (q in 1:length(time3)){if(any(abs(time4-time3[q])<4)){test=test+1}}
      if (test>0){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}


Y4alt3newfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    time3=c()
    time4=c()
    for (j in 1:dim(Y)[2]){
      if (X[i,8,j]==1){time3=c(time3,j)}
      if (X[i,9,j]==1){time4=c(time4,j)}
    }
    if (!(is.null(time3)|is.null(time4))){
      test=0
      for (q in 1:length(time3)){if(any(abs(time4-time3[q])<4)){test=test+1}}
      if (test>1){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}

Y4alt3newerfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    time3=c()
    time4=c()
    for (j in 1:dim(Y)[2]){
      if (X[i,8,j]==1){time3=c(time3,j)}
      if (X[i,9,j]==1){time4=c(time4,j)}
    }
    if (!(is.null(time3)|is.null(time4))){
      test=0
      for (q in 1:length(time3)){if(any(abs(time4-time3[q])<4)){test=test+1}}
      if (test>2){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}

Y4alt3newestfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    time3=c()
    time4=c()
    for (j in 1:dim(Y)[2]){
      if (X[i,8,j]==1){time3=c(time3,j)}
      if (X[i,9,j]==1){time4=c(time4,j)}
    }
    if (!(is.null(time3)|is.null(time4))){
      test=0
      for (q in 1:length(time3)){if(any(abs(time4-time3[q])<4)){test=test+1}}
      if (test>3){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}


Y4alt4func<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    time3=c()
    time4=c()
    for (j in 1:dim(Y)[2]){
      if (X[i,8,j]==1){time3=c(time3,j)}
      if (X[i,9,j]==1){time4=c(time4,j)}
    }
    if (!(is.null(time3)|is.null(time4))){
      test=0
      for (q in 1:length(time3)){if(any(abs(time4-time3[q])<2)){test=test+1}}
      if (test>0){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}


Y4alt4newfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    time3=c()
    time4=c()
    for (j in 1:dim(Y)[2]){
      if (X[i,8,j]==1){time3=c(time3,j)}
      if (X[i,9,j]==1){time4=c(time4,j)}
    }
    if (!(is.null(time3)|is.null(time4))){
      test=0
      for (q in 1:length(time3)){if(any(abs(time4-time3[q])<2)){test=test+1}}
      if (test>1){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}

Y4alt4newerfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    time3=c()
    time4=c()
    for (j in 1:dim(Y)[2]){
      if (X[i,8,j]==1){time3=c(time3,j)}
      if (X[i,9,j]==1){time4=c(time4,j)}
    }
    if (!(is.null(time3)|is.null(time4))){
      test=0
      for (q in 1:length(time3)){if(any(abs(time4-time3[q])<2)){test=test+1}}
      if (test>2){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}

Y4alt4newestfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    time3=c()
    time4=c()
    for (j in 1:dim(Y)[2]){
      if (X[i,8,j]==1){time3=c(time3,j)}
      if (X[i,9,j]==1){time4=c(time4,j)}
    }
    if (!(is.null(time3)|is.null(time4))){
      test=0
      for (q in 1:length(time3)){if(any(abs(time4-time3[q])<2)){test=test+1}}
      if (test>3){Y_output[i,dim(Y)[2]]=1}
    }
  }
  return(Y_output)}

Y5func<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    if (sum(X[i,9,1:10])==0){Y_output[i,dim(Y)[2]]=1}
  }
  return(Y_output)}


Y5newfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    if (sum(X[i,9,1:20])==0){Y_output[i,dim(Y)[2]]=1}
  }
  return(Y_output)}

Y5newerfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    if (sum(X[i,9,1:30])==0){Y_output[i,dim(Y)[2]]=1}
  }
  return(Y_output)}

Y5altfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    if (sum(X[i,9,1:15])==0){Y_output[i,dim(Y)[2]]=1}
  }
  return(Y_output)}

Y5altnewfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    if (sum(X[i,9,1:5])==0){Y_output[i,dim(Y)[2]]=1}
  }
  return(Y_output)}

Y5altnewerfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    if (sum(X[i,9,1:25])==0){Y_output[i,dim(Y)[2]]=1}
  }
  return(Y_output)}

Y5altnewestfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    if (sum(X[i,9,1:35])==0){Y_output[i,dim(Y)[2]]=1}
  }
  return(Y_output)}


Y6func<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    if (sum(X[i,9,1:10])>3){Y_output[i,dim(Y)[2]]=1}
  }
  return(Y_output)}


Y6newfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    if (sum(X[i,9,1:10])>2){Y_output[i,dim(Y)[2]]=1}
  }
  return(Y_output)}

Y6newerfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    if (sum(X[i,9,1:10])>4){Y_output[i,dim(Y)[2]]=1}
  }
  return(Y_output)}

Y6newestfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    if (sum(X[i,9,1:5])>3){Y_output[i,dim(Y)[2]]=1}
  }
  return(Y_output)}

Y6altfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    if (sum(X[i,9,1:10])>1){Y_output[i,dim(Y)[2]]=1}
  }
  return(Y_output)}

Y6altnewfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    if (sum(X[i,9,1:10])>0){Y_output[i,dim(Y)[2]]=1}
  }
  return(Y_output)}

Y7func<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    count_todler=sum(X[i,9,1:4])
    count_child=sum(X[i,9,5:12])
    count_teen=sum(X[i,9,13:17])
    count_younga=sum(X[i,9,18:25])
    count_adult=sum(X[i,9,26:40])
    # print(c(count_todler,count_child,count_teen,count_younga,count_adult))
    # print(count_todler+count_child/2+count_teen/3+count_younga/5+count_adult/10)
    if ((count_todler+count_child/2+count_teen/3+count_younga/5+count_adult/10)>2.5){Y_output[i,dim(Y)[2]]=1}
  }
  return(Y_output)}


Y7newfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_out
  for (i in 1:dim(Y)[1]){
    count_todler=sum(X_out[i,9,1:4])
    count_child=sum(X_out[i,9,5:12])
    count_teen=sum(X_out[i,9,13:17])
    count_younga=sum(X_out[i,9,18:25])
    count_adult=sum(X_out[i,9,26:40])
    # print(c(count_todler,count_child,count_teen,count_younga,count_adult))
    # print(count_todler+count_child/2+count_teen/3+count_younga/5+count_adult/10)
    if ((count_todler+count_child/2+count_teen/3+count_younga/5+count_adult/10)>3){Y_output[i,dim(Y)[2]]=1}
  }
  return(Y_output)}

Y7newerfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_out
  for (i in 1:dim(Y)[1]){
    count_todler=sum(X_out[i,9,1:4])
    count_child=sum(X_out[i,9,5:12])
    count_teen=sum(X_out[i,9,13:17])
    count_younga=sum(X_out[i,9,18:25])
    count_adult=sum(X_out[i,9,26:40])
    # print(c(count_todler,count_child,count_teen,count_younga,count_adult))
    # print(count_todler+count_child/2+count_teen/3+count_younga/5+count_adult/10)
    if ((count_todler+count_child/2+count_teen/3+count_younga/5+count_adult/10)>4){Y_output[i,dim(Y)[2]]=1}
  }
  return(Y_output)}

Y7newestfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    count_todler=sum(X[i,9,1:4])
    count_child=sum(X[i,9,5:12])
    count_teen=sum(X[i,9,13:17])
    count_younga=sum(X[i,9,18:25])
    count_adult=sum(X[i,9,26:40])
    # print(c(count_todler,count_child,count_teen,count_younga,count_adult))
    # print(count_todler+count_child/2+count_teen/3+count_younga/5+count_adult/10)
    if ((count_todler+count_child/2+count_teen/3+count_younga/5+count_adult/10)>2){Y_output[i,dim(Y)[2]]=1}
  }
  return(Y_output)}

Y7altfunc<-function(X,Y){
  Y_output=array(0,dim(Y))
  ##make my own Y_output
  for (i in 1:dim(Y)[1]){
    count_todler=sum(X[i,9,1:4])
    count_child=sum(X[i,9,5:12])
    count_teen=sum(X[i,9,13:17])
    count_younga=sum(X[i,9,18:25])
    count_adult=sum(X[i,9,26:40])
    # print(c(count_todler,count_child,count_teen,count_younga,count_adult))
    # print(count_todler+count_child/2+count_teen/3+count_younga/5+count_adult/10)
    if ((count_todler+count_child/2+count_teen/3+count_younga/5+count_adult/10)>5){Y_output[i,dim(Y)[2]]=1}
  }
  return(Y_output)}


## Add stochasticity
addstoc<-function(Y,prob){
  Yreturn=array(dim=dim(Y))
  for (i in 1:nrow(Y)){
    for (j in 1:ncol(Y)){
      if (Y[i,j]==1){
        probs=prob
      } else {
        probs=1-prob
      }
      Yreturn[i,j]=rbern(1,probs)
    }
  }
  return(Yreturn)
}