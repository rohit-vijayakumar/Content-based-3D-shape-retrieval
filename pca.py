import numpy as np
import math
# a = np.array([1,2,3])

# b = np.multiply(a,1)
# a = 1 if 1 == 2 else "b"

# a = np.array([1,0,1])
# for x in range(len(a)) :
#     a[x] = ~a[x]
a = np.array([[3,-4,2]])
b = np.array([[3,-4,2]])
# b = np.array([1,2,3])
# print(b)
# print(np.linalg.norm(b)/2)
# print(np.sqrt(np.dot(b,b.T))/2)

max_val = 2240
tp = int(.48 * max_val)
fp  = int(.02* max_val)
fn = int(.0349* max_val)
tn = int(.4651* max_val)
print(tp+fp+fn+tn)
accuracy = (tp+tn)/(tp+fp+fn+tn)
print(tp,tn,fp,fn)
recall = tp/(tp+fp)
precision = tp/(tp+fn)
f1 = 2* ((precision * recall )/ (precision + recall))
print("precision",round(precision,4))
print("recall",round(recall,4))
print("f1",round(f1,4))
print("accuracy",round(accuracy,4))

print(27**(1/3))