
// -- Parentheses checker --
#include<stdio.h>
#include<string.h>
#include<stdbool.h>
#include"S_HEAD.h"

int main(){
  char exp[100];
  int i,flag=1;
  char *dataPtr;
  STACK *stack;
char *temp ;
  stack = createStack();
  printf("-- Parentheses Checker --\n");
  printf("Enter any algebraic expression : ");
  gets(exp);

    for(i=0;i<strlen(exp);i++){
      if(exp[i]=='('|| exp[i]=='{' || exp[i]=='['){
         dataPtr = (char*)malloc(sizeof(char));
        dataPtr = exp[i];
        pushStack(stack,dataPtr);
      }

      if(exp[i]=='}' || exp[i]==']' || exp[i]==')'){
            if(emptyStack(stack)) {printf("in1"); flag=0; }
            else{
                    temp = (char*)popStack(stack);
                 if(exp[i]=='}' &&  (temp=='[' || temp=='(') )
                    { flag=0;}
                 if(exp[i]==']' && (temp=='{' || temp=='('))
                        { flag=0;}
                 if(exp[i]==')'&& (temp=='{' || temp=='['))
                    {
                        flag=0;
                    }
            }
            }

      }
      if(flag==1) printf(" --- valid expression --- \n");
      else{
        printf(" --- Invalid expression --- \n");
      }

      return 0;
}
