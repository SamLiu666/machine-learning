include "sprue.mzn";
predicate my_all_different(array[int] of var int: x);
constraint load_python_module("../propagators/my_all_different.mzn.py");
int: expected_failures_A;
int: expected_failures_B;
int: expected_failures_C;
constraint trace_stdout(
"% ----------------------------------------------------\n"++
"%   Make sure to run your code in all-solutions mode   \n"++
"%   (\"User-defined behavior\", stop after 0 solutions)\n"++
"%   and switch on solver statistics!\n"++
"%\n"++
"%  The expected number of failures is\n"++
"%     \(5,expected_failures_A) for part A\n"++
"%     \(5,expected_failures_C) for part B\n"++
"% ----------------------------------------------------\n"
);