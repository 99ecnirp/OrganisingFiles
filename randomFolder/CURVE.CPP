#include<stdio.h>
#include<conio.h>
#include<graphics.h>
int main()
{
 int gdriver=DETECT,gmode;
 int x[5],y[5],t[100],t1[100],n,i,k;
 float j;
 printf("Enter the number of control points(4 or 5)\n");
 scanf("%d",&n);
if(n==4)
 {
  for(i=0;i<4;i++)
  {
    printf("x(%d)=",i+1);
    scanf("%d",&x[i]);
    printf("y(%d)=",i+1);
    scanf("%d",&y[i]);
  }
 }
 else if(n==5)
 {
  for(i=0;i<5;i++)
  {
    printf("x(%d)=",(i+2)/2);
    scanf("%d",&x[i]);
    printf("y(%d)=",(i+2)/2);
    scanf("%d",&y[i]);
  }
 }
 else
   printf("Enter a valid input\n");
  i=0;
for(k=0,j=0.0;j<=1.0;j=j+.01,k++)
{
 if(n==4)
 {
  t[k]=((1-j)*(1-j)*(1-j)*x[i])+(3*j*(1-j)*(1-j)*x[i+1])+(3*j*j*(1-j)*x[i+2])+(j*j*j*j*x[i+3]);
  t1[k]=((1-j)*(1-j)*(1-j)*y[i])+(3*j*(1-j)*(1-j)*y[i+1])+(3*j*j*(1-j)*y[i+2])+(j*j*j*j*y[i+3]);
 }
 else
 {
  t[k]=((1-j)*(1-j)*(1-j)*(1-j)*x[i])+(4*j*(1-j)*(1-j)*(1-j)*x[i+1])+(6*j*j*(1-j)*(1-j)*x[i+2])+(4*j*j*j*(1-j)*x[i+3])+(j*j*j*j*x[i+4]);
  t1[k]=((1-j)*(1-j)*(1-j)*(1-j)*y[i])+(4*j*(1-j)*(1-j)*(1-j)*y[i+1])+(6*j*j*(1-j)*(1-j)*y[i+2])+(4*j*j*j*(1-j)*y[i+3])+(j*j*j*j*y[i+4]);
 }
}
initgraph(&gdriver,&gmode,"");
 line(getmaxx()/2,0,getmaxx()/2,getmaxy());
 line(0,getmaxy()/2,getmaxx(),getmaxy()/2);
 for(i=0;i<100;i++)
 {
  putpixel(getmaxx()/2+t[i],getmaxy()/2-t1[i],getmaxcolor());
 }
getch();
closegraph();
return 0;
}



