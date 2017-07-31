#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

int vecsize,codesize,datasize;
double **data,**encode,**decode,*x,*x1,*dx,*y,*xi,weight;


int getdata(char *datafile)
{
FILE *fp;
int d,i,j;
double norm;

fp=fopen(datafile,"r");
if(!fp)
	{
	fprintf(stderr,"datafile not found\n");
	return 0;
	}
	
fscanf(fp,"%d%d",&datasize,&vecsize);

data=malloc(datasize*sizeof(double*));
for(d=0;d<datasize;++d)
	{
	data[d]=malloc(vecsize*sizeof(double));
	
	norm=0.;
	for(i=0;i<vecsize;++i)
		{
		fscanf(fp,"%lf",&data[d][i]);
		norm+=data[d][i]*data[d][i];
		}
		
	norm=sqrt(norm);
	
	for(i=0;i<vecsize;++i)
		data[d][i]/=norm;
	}
	
fclose(fp);

x=malloc(vecsize*sizeof(double));
x1=malloc(vecsize*sizeof(double));
dx=malloc(vecsize*sizeof(double));

y=malloc(codesize*sizeof(double));
xi=malloc(codesize*sizeof(double));

encode=malloc(codesize*sizeof(double*));
for(j=0;j<codesize;++j)
	encode[j]=malloc(vecsize*sizeof(double));
	
decode=malloc(vecsize*sizeof(double*));
for(i=0;i<vecsize;++i)
	decode[i]=malloc(codesize*sizeof(double));

return 1;
}


void init()
{
int i,j;

for(i=0;i<vecsize;++i)
for(j=0;j<codesize;++j)
	{
	encode[j][i]=2.*((double)rand())/RAND_MAX-1.;
	decode[i][j]=0.;
	}
}


double learn(double *datavec)
{
double err,xinorm,ynorm,xiscale,dxscale;
int i,j;

for(i=0;i<vecsize;++i)
	x[i]=datavec[i];
	
for(j=0;j<codesize;++j)
	{
	y[j]=0.;
	for(i=0;i<vecsize;++i)
		y[j]+=encode[j][i]*x[i];
		
	if(y[j]<0.)
		{
		for(i=0;i<vecsize;++i)
			encode[j][i]-=y[j]*x[i];
			
		y[j]=0.;
		}
	}
	
err=0.;
for(i=0;i<vecsize;++i)
	{
	x1[i]=0.;
	for(j=0;j<codesize;++j)
		x1[i]+=decode[i][j]*y[j];
		
	dx[i]=x[i]-x1[i];
	
	err+=dx[i]*dx[i];
	}

xinorm=0.;
ynorm=0.;
for(j=0;j<codesize;++j)
	{
	xi[j]=0.;
	for(i=0;i<vecsize;++i)
		xi[j]+=decode[i][j]*dx[i];
		
	xinorm+=xi[j]*xi[j];
	
	ynorm+=y[j]*y[j];
	}
	
xiscale=xinorm/err+weight*ynorm;
for(j=0;j<codesize;++j)
	xi[j]/=xiscale;

dxscale=xiscale/weight;
for(i=0;i<vecsize;++i)
	dx[i]/=dxscale;

for(i=0;i<vecsize;++i)
for(j=0;j<codesize;++j)
	{
	encode[j][i]+=xi[j]*x[i];
	decode[i][j]+=dx[i]*y[j];
	}

for(i=0;i<vecsize;++i)
for(j=0;j<codesize;++j)
	if(decode[i][j]<0.)
		decode[i][j]=0.;
		
return sqrt(err/vecsize);
}


void printmodel(char *modelfile)
{
FILE *fp;
int i,j;

fp=fopen(modelfile,"w");

for(j=0;j<codesize;++j)
	{
	for(i=0;i<vecsize;++i)
		fprintf(fp,"%lf ",encode[j][i]);
	fprintf(fp,"\n");
	}
	
fprintf(fp,"\n");

for(j=0;j<codesize;++j)
	{
	for(i=0;i<vecsize;++i)
		fprintf(fp,"%lf ",decode[i][j]);
	fprintf(fp,"\n");
	}
	
fclose(fp);
}


int main(int argc,char* argv[])
{
FILE *fp;
char *datafile,*id,logfile[50],modelfile[50];
int trainsize,trainbatch,datacount,b;
double err,errmin;

if(argc==7)
	{
	datafile=argv[1];
	codesize=atoi(argv[2]);
	weight=atof(argv[3]);
	trainsize=atoi(argv[4]);
	trainbatch=atoi(argv[5]);
	id=argv[6];
	}
else
	{
	fprintf(stderr,"expected six arguments: datafile, codesize, weight, trainsize, trainbatch, id\n");
	return 1;
	}
	
strcpy(logfile,id);
strcat(logfile,".log");

strcpy(modelfile,id);
strcat(modelfile,".model");

if(!getdata(datafile))
	return 1;
	
srand(time(0));
init();

fp=fopen(logfile,"w");
fprintf(fp,"data: %s\n",datafile);
fprintf(fp,"codesize: %d  weight: %lf\n\n",codesize,weight);
fprintf(fp,"data count    recon error\n");
fclose(fp);
	
errmin=1.;
datacount=0;
do	{
	err=0.;
	for(b=0;b<trainbatch;++b)
		{
		err+=learn(data[datacount%datasize]);
		++datacount;
		}
	err/=trainbatch;
	
	fp=fopen(logfile,"a");
	fprintf(fp,"%10d%15.10lf\n",datacount,err);
	fclose(fp);
	
	if(err<errmin)
		{
		errmin=err;
		printmodel(modelfile);
		}
	}
while(datacount<trainsize);

fp=fopen(logfile,"a");
fprintf(fp,"\nmin error:%15.10lf\n",errmin);
fclose(fp);
	
return 0;
}