#include <stdio.h>
int maxSubs(const int A[], int N) {
	int thisSum, maxSum, j;

	thisSum = maxSum = 0;
	for (j = 0; j < N; j++) {
		thisSum += A[j];
		if (thisSum > maxSum)
			maxSum = thisSum;
		else if (thisSum < 0)
			thisSum = 0;	
	}
	return maxSum;
}


int main()
{
	/* 我的第一个 C 程序 */
	printf("Hello, World! \n");
	int A[] = { 4, -3, 5, -2, -1, 2, 6, -2};
	int N = 8;
	printf(maxSubs(A[8], N));
	return 0;
}