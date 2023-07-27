library(simcausal)

num_var=9
Npeople=100000
Nt=39
set.seed(9)

D <- DAG.empty() + 
  node("sex", distr="rcat.b1", prob=c(0.5,0.5),replaceNAw0=TRUE) + ## fem/mal
  node("ethnicity", distr="rcat.b1", prob=c(0.9,0.1),replaceNAw0=TRUE) + ## white/other of mother
  node("marriage", distr="rcat.b1", prob=c(0.75,0.25),replaceNAw0=TRUE) + ## married/single of mother
  node("education", distr="rcat.b1", prob=c(0.35,0.65),replaceNAw0=TRUE) + ## degree or a level/gcse of mother
  node("class", distr="rcat.b1", prob=c(0.81,0.19),replaceNAw0=TRUE) + ## nonmanual/manual of mother
  
  node("admissions",t=0:Nt, distr="rbern", prob=(1/(5*log(t+3))-0.015 ),replaceNAw0=TRUE) + ## general admissions
  node("admissions2",t=0:Nt, distr="rbern", prob=(0.04*((class-1)*0.28+1)*((education-1)*0.22+1)),replaceNAw0=TRUE) + ## something different affected by variables
  node("admissions3",t=0, distr="rbern", prob=((0.02 + sex/100)*((ethnicity-1)*0.17+1)+plogis(-5)),replaceNAw0=TRUE) + ## general admissions
  node("admissions3",t=1:Nt, distr="rbern", prob=((0.02 + sex/100)*((ethnicity-1)*0.17+1)+plogis(-5+admissions3[t-1])),replaceNAw0=TRUE) + ## general admissions
  node("admissions4",t=0, distr="rbern", prob=plogis(-2+admissions2[t]+admissions[t]),replaceNAw0=TRUE) + ## something different affected by variables
  node("admissions4",t=1, distr="rbern", prob=plogis(-2+((admissions2[t]+admissions2[t-1])>0)+((admissions[t]+admissions[t-1])>0)),replaceNAw0=TRUE) + ## something different affected by variables
  node("admissions4",t=2, distr="rbern", prob=plogis(-2+((admissions2[t]+admissions2[t-1]+admissions2[t-2])>1)+((admissions[t]+admissions[t-1]+admissions[t-2])>1)),replaceNAw0=TRUE) +   
  node("admissions4",t=3:Nt, distr="rbern", prob=plogis(-2+((admissions2[t]+admissions2[t-1]+admissions2[t-2]+admissions2[t-3])>2)+((admissions[t]+admissions[t-1]+admissions[t-2]+admissions[t-3])>2)),replaceNAw0=TRUE) +   

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

X_out=data_out

source("Data_generation_functions.R")


names=c("1","1newer","2","3alt","3newer","3newest","4alt2newer","4alt2newest","4newer","4newest","5newer", "6newer","7newer")
New_names<-as.character(1:length(names))


for (i in 1:length(names)){
  print(names[i])
  set.seed(9)
  Y_outcheck=eval(parse(text=(paste("Y",names[i],"func(X_out,Y_out)",sep=""))))
  
  name=paste("data",new_names[i],sep="")
  
  Y_outH=Y_outcheck
  
  ## moving data to python
  library(reticulate)
  np = import("numpy")
  np$save(paste("/home/DIDE/smishra/Simulations/input_data/",name,"_X.npy",sep=""),r_to_py(X_out))
  np$save(paste("/home/DIDE/smishra/Simulations/input_data/",name,"_YH.npy",sep=""),r_to_py(Y_outH))
}

