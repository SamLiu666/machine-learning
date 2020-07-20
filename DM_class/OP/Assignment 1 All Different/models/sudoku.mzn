%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FIT5220 Assignment 1
%
% Sudoku
%
% See the PDF file for instructions.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

include "../propagators/my_all_different.mzn";
% Square root of size of the Sudoku field
int: s;
int: n = s*s;

% The hints (the pre-filled cells)
% Any non-zero entry is a pre-filled cell
array[1..n,1..n] of int: hints;

% The variables for output
array[1..n*n] of var 1..n: xx;

% The Sudoku field
array[1..n,1..n] of var int: x=array2d(1..n,1..n,xx);

% Pre-fill all non-zero hints
constraint forall (i,j in 1..n)(if hints[i,j]>0 then x[i,j]=hints[i,j] endif);

% All rows, columns and 3x3 block must be all-different
constraint forall (i in 1..n) (my_all_different(x[i,..]));
constraint forall (i in 1..n) (my_all_different(x[..,i]));
constraint forall (i,j in 1..s)(my_all_different(p,q in 1..s)(x[s*(i-1)+p,s*(j-1)+q]));
