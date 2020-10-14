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


// Definition for singly-linked list.
struct SinglyListNode {
	int val;
	SinglyListNode* next;
	SinglyListNode(int x) : val(x), next(NULL) {}
};

int main()
{
	/* 我的第一个 C 程序 */
	printf("Hello, World! \n");

	return 0;
}