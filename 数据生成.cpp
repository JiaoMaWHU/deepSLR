//数据生成
#include<iostream>
#include<cstdio>
#include <cstdlib> 
#include<cstring> 
#include <time.h> 
using namespace std;
int main(){
	int x;
    srand( (unsigned)time( NULL ) );
	freopen("imutrain.txt","w",stdout);
	for (int i=1;i<=100;i++)
	{
		x=rand()%200+200;
		for (int k=1;k<=x;k++)
		{
			for (int j=1;j<=26;j++) printf("%d ",rand()%20);
			printf("\n");
		}
		printf("\n");
	} 
	freopen("imutest.txt","w",stdout);
	for (int i=1;i<=100;i++)
	{
		x=rand()%200+200;
		for (int k=1;k<=x;k++)
		{
			for (int j=1;j<=26;j++) printf("%d ",rand()%20);
			printf("\n");
		}
		printf("\n");
	} 
	freopen("emgtrain.txt","w",stdout);
	for (int i=1;i<=100;i++)
	{
		x=rand()%200+200;
		for (int k=1;k<=x;k++)
		{
			for (int j=1;j<=64;j++) printf("%d ",rand()%20);
			printf("\n");
		}
		printf("\n");
	} 
	freopen("emgtest.txt","w",stdout);
	for (int i=1;i<=100;i++)
	{
		x=rand()%200+200;
		for (int k=1;k<=x;k++)
		{
			for (int j=1;j<=64;j++) printf("%d ",rand()%20);
			printf("\n");
		}
		printf("\n");
	} 
	freopen("y_train.txt","w",stdout);
	for (int i=1;i<=100;i++)
	{
		x=rand()%5+4;
		for (int k=1;k<=x;k++)
			printf("%d ",rand()%20);
		printf("\n");
	} 
	freopen("y_test.txt","w",stdout);
	for (int i=1;i<=100;i++)
	{
		x=rand()%5+4;
		for (int k=1;k<=x;k++)
			printf("%d ",rand()%20);
		printf("\n");
	} 
	fclose(stdout);
}
