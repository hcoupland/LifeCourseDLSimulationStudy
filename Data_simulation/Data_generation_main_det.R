library(simcausal)

num_var=9 #nocorrauto
Npeople=100000
Nt=39 ## real Nt will be this plus 1
set.seed(9)

D <- DAG.empty() + 
  # node("t1",  distr="rcat.b1", probs = rep(1/(Nt+1), Nt+1),replaceNAw0=TRUE) + ## binary covariate
  node("sex", distr="rcat.b1", prob=c(0.5,0.5),replaceNAw0=TRUE) + ## fem/mal
  node("ethnicity", distr="rcat.b1", prob=c(0.9,0.1),replaceNAw0=TRUE) + ## white/other of mother
  node("marriage", distr="rcat.b1", prob=c(0.75,0.25),replaceNAw0=TRUE) + ## married/single of mother
  node("education", distr="rcat.b1", prob=c(0.35,0.65),replaceNAw0=TRUE) + ## degree or a level/gcse of mother
  node("class", distr="rcat.b1", prob=c(0.81,0.19),replaceNAw0=TRUE) + ## nonmanual/manual of mother
  # node("matage", distr="rcat.b1", prob=c(0.37,0.39,0.24),replaceNAw0=TRUE) + ## 30+/25-29/<25 of mother
  # node("parity", distr="rcat.b1", prob=c(0.45,0.35,0.20),replaceNAw0=TRUE) + ## 0/1/2+ of mother
  # node("ethnicity2", distr="rcat.b1", prob=c(0.84,0.03,0.08,0.04,0.01),replaceNAw0=TRUE) + ## white/mixed/asian/black/other of mother
  # node("IMD", distr="rcat.b1", prob=rep(0.2,5),replaceNAw0=TRUE) + ## Most deprived 20%/More deprived 20-40%/Medium deprived 40-60%/Less deprived 20-40%/Least deprived 20%
  
  node("admissions",t=0:Nt, distr="rbern", prob=(1/(5*log(t+3))-0.015 ),replaceNAw0=TRUE) + ## general admissions
  node("admissions2",t=0:Nt, distr="rbern", prob=(0.04*((class-1)*0.28+1)*((education-1)*0.22+1)),replaceNAw0=TRUE) + ## something different affected by variables
  node("admissions3",t=0, distr="rbern", prob=((0.02 + sex/100)*((ethnicity-1)*0.17+1)+plogis(-5)),replaceNAw0=TRUE) + ## general admissions
  node("admissions3",t=1:Nt, distr="rbern", prob=((0.02 + sex/100)*((ethnicity-1)*0.17+1)+plogis(-5+admissions3[t-1])),replaceNAw0=TRUE) + ## general admissions
  node("admissions4",t=0, distr="rbern", prob=plogis(-2+admissions2[t]+admissions[t]),replaceNAw0=TRUE) + ## something different affected by variables
  node("admissions4",t=1, distr="rbern", prob=plogis(-2+((admissions2[t]+admissions2[t-1])>0)+((admissions[t]+admissions[t-1])>0)),replaceNAw0=TRUE) + ## something different affected by variables
  #node("admissions4",t=0:Nt, distr="rbern", prob=((0.04*((class-1)*0.28+1)*((ethnicity-1)*0.17+1)*((education-1)*0.22+1))+0.6*(1/(5*log(t+3))-0.03)),replaceNAw0=TRUE) + ## something different affected by variables
  node("admissions4",t=2, distr="rbern", prob=plogis(-2+((admissions2[t]+admissions2[t-1]+admissions2[t-2])>1)+((admissions[t]+admissions[t-1]+admissions[t-2])>1)),replaceNAw0=TRUE) +   
  node("admissions4",t=3:Nt, distr="rbern", prob=plogis(-2+((admissions2[t]+admissions2[t-1]+admissions2[t-2]+admissions2[t-3])>2)+((admissions[t]+admissions[t-1]+admissions[t-2]+admissions[t-3])>2)),replaceNAw0=TRUE) +   
  # node("x1",t=0, distr="rbern", prob=plogis(-1),replaceNAw0=TRUE) + ## illness in family
  # node("x2",t=0, distr="rbern", prob=plogis(-3+6*((x1[t])>0)),replaceNAw0=TRUE) + ## random accident
  
  
  # node("x1",t=1:Nt, distr="rbern", prob=plogis(-1.5+2*x1[t-1]),replaceNAw0=TRUE) +
  #  node("x2",t=1, distr="rbern", prob=plogis(-3+6*((x1[t]+x1[t-1])>1)),replaceNAw0=TRUE) +
  # node("x2",t=2:Nt, distr="rbern", prob=plogis(-3+6*((x1[t]+x1[t-1]+x1[t-2])>2)),replaceNAw0=TRUE) +
  node("Y", t=0:(Nt-1), distr="rbern", prob=plogis(-2), EFU=FALSE,replaceNAw0=TRUE) +
  node("Y", t=Nt, distr="rbern", prob=plogis(-1 ), EFU=FALSE,replaceNAw0=TRUE)
D <- set.DAG(D)
plotDAG(D)
dat.long <- sim(D,n=Npeople)


data=dat.long[-1]
data_out=array(dim=c(dim(data)[1],num_var,Nt+1))
Y_out=array(dim=c(dim(data)[1],Nt+1))

data_out[,1,]<-array(replicate(40,unlist(data[1])),dim=c(dim(data)[1],Nt+1))-1
data_out[,2,]<-array(replicate(40,unlist(data[2])),dim=c(dim(data)[1],Nt+1))-1
data_out[,3,]<-array(replicate(40,unlist(data[3])),dim=c(dim(data)[1],Nt+1))-1
data_out[,4,]<-array(replicate(40,unlist(data[4])),dim=c(dim(data)[1],Nt+1))-1
data_out[,5,]<-array(replicate(40,unlist(data[5])),dim=c(dim(data)[1],Nt+1))-1
data_out[,6,]<-array(unlist(data[seq(from=6,by=5,to=dim(data)[2])]),dim=c(dim(data)[1],Nt+1))
data_out[,7,]<-array(unlist(data[seq(from=7,by=5,to=dim(data)[2])]),dim=c(dim(data)[1],Nt+1))
data_out[,8,]<-array(unlist(data[seq(from=8,by=5,to=dim(data)[2])]),dim=c(dim(data)[1],Nt+1))
data_out[,9,]<-array(unlist(data[seq(from=9,by=5,to=dim(data)[2])]),dim=c(dim(data)[1],Nt+1))

Y_out<-array(unlist(data[seq(from=10,by=5,to=dim(data)[2])]),dim=c(dim(data)[1],Nt+1))


## converting to right format
# (people, features, time)

X_out=data_out#data_topy(dat.long,num_var,Nt)[1]
#Y_out=data_topy(dat.long,num_var,Nt)[2]


# 1)	Repeats: Y=1 if there are 3 /4 events in a row ever
# 2)	Repeats: Y=1 if there are ever 2/3 sets of 2/3 repeated events ever==
# 3)	Ordering: Y=1 if event x1 occurs before event x2
# 4)	Timing: Y=1 if event x1 occurs within 5 years of x2
# 5)	Critical period: Y=1 if no event occurs in the first 10 years
# 6)	Sensitive period: Y=1 if over 30% of first 10 years are 1s
# 7)	Sensitive period: Y=1 if many events in one period but still effected by number of events in other periods to a lesser extent

source("C:/Users/hlc17/Documents/DANLIFE/Simulations/Simulations/Data_simulation/DAG_realism_big_nobal_newer_funcs.R")
names=c("1","1new","1newer","2","3","3new","3newer","3newest","4","4new",
  "4newer","4newest","5","5new","5newer", "6","6new","6newer","6newest","7","7new","7newer")

# names=c("4alt","4altnew","4alt2","4alt2new","4alt2newer","4alt2newest","4alt3","4alt3new","4alt3newer","4alt3newest","4alt4","4alt4new","4alt4newer","4alt4newest","7newest")
# names=c("3alt","5alt","5altnew","5altnewer","5altnewest","6alt","6altnew","7alt")
# stoc=0.1

for (i in 1:length(names)){
  print(names[i])
  set.seed(9)
  Y_outcheck=eval(parse(text=(paste("Y",names[i],"func(X_out,Y_out)",sep=""))))#Y7newfunc(X_out,Y_out)
  
  name=paste("data_2real",names[i],"bigdet",sep="")
  
  # Yorg=Y_outcheck
  # 
  # ## okay so first get 5/#1s and then 5/#0s
  # num1s<-sum(Y_outcheck[,ncol(Y_outcheck)])
  # num0s<-Npeople-num1s
  # num10<-ceiling(stoc*num1s)
  # num01<-num10
  # which1<-which(Y_outcheck[,ncol(Y_outcheck)]==1)
  # which0<-which(Y_outcheck[,ncol(Y_outcheck)]==0)
  # 
  # #then sample
  # which10<-sample(which1,num10)
  # which01<-sample(which0,num01)
  # Y_outcheck[which10,ncol(Y_outcheck)]=0
  # Y_outcheck[which01,ncol(Y_outcheck)]=1
  
  Y_outH=Y_outcheck
  
  ## moving data to python
  library(reticulate)
  np = import("numpy")
  np$save(paste("C:/Users/hlc17/Documents/DANLIFE/Simulations/Simulations/Data_simulation/",name,"_X.npy",sep=""),r_to_py(X_out))
  #np$save(paste("DANLIFE/Simulations/",name,"_Y.npy",sep=""),r_to_py(Y_out))
 # np$save(paste("DANLIFE/Simulations/",name,"_Yorg.npy",sep=""),r_to_py(Yorg))
  np$save(paste("C:/Users/hlc17/Documents/DANLIFE/Simulations/Simulations/Data_simulation/",name,"_YH.npy",sep=""),r_to_py(Y_outH))
}

# print(names[i])
# set.seed(9)
# Y_outcheck=eval(parse(text=(paste("Y",names[i],"func(X_out,Y_out)",sep=""))))#Y7newfunc(X_out,Y_out)
# table(Y_outcheck[,ncol(Y_outcheck)])

