
oned_arr=[]
twod_arr=[]
temp=[]
in_H = int(input("Enter High:"))
in_row = int(input("Enter Row:"))
in_column = int(input("Enter Column:"))

i=1
while i <= in_column:
    while j < in_row:
        oned_arr.append(int(input('Enter number: ')))
    twod_arr.append(oned_arr)
    oned_arr=[]