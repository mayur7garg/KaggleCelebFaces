�� 
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.3.02v2.3.0-rc2-23-gb36436b0878��
~
Conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameConv_1/kernel
w
!Conv_1/kernel/Read/ReadVariableOpReadVariableOpConv_1/kernel*&
_output_shapes
: *
dtype0
n
Conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameConv_1/bias
g
Conv_1/bias/Read/ReadVariableOpReadVariableOpConv_1/bias*
_output_shapes
: *
dtype0
~
Conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_nameConv_2/kernel
w
!Conv_2/kernel/Read/ReadVariableOpReadVariableOpConv_2/kernel*&
_output_shapes
: @*
dtype0
n
Conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameConv_2/bias
g
Conv_2/bias/Read/ReadVariableOpReadVariableOpConv_2/bias*
_output_shapes
:@*
dtype0

Conv_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*
shared_nameConv_3/kernel
x
!Conv_3/kernel/Read/ReadVariableOpReadVariableOpConv_3/kernel*'
_output_shapes
:@�*
dtype0
o
Conv_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameConv_3/bias
h
Conv_3/bias/Read/ReadVariableOpReadVariableOpConv_3/bias*
_output_shapes	
:�*
dtype0
�
Conv_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*
shared_nameConv_4/kernel
y
!Conv_4/kernel/Read/ReadVariableOpReadVariableOpConv_4/kernel*(
_output_shapes
:��*
dtype0
o
Conv_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameConv_4/bias
h
Conv_4/bias/Read/ReadVariableOpReadVariableOpConv_4/bias*
_output_shapes	
:�*
dtype0
�
Conv_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*
shared_nameConv_5/kernel
y
!Conv_5/kernel/Read/ReadVariableOpReadVariableOpConv_5/kernel*(
_output_shapes
:��*
dtype0
o
Conv_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameConv_5/bias
h
Conv_5/bias/Read/ReadVariableOpReadVariableOpConv_5/bias*
_output_shapes	
:�*
dtype0
�
Conv_T_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameConv_T_1/kernel
}
#Conv_T_1/kernel/Read/ReadVariableOpReadVariableOpConv_T_1/kernel*(
_output_shapes
:��*
dtype0
s
Conv_T_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameConv_T_1/bias
l
!Conv_T_1/bias/Read/ReadVariableOpReadVariableOpConv_T_1/bias*
_output_shapes	
:�*
dtype0
�
Conv_T_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameConv_T_2/kernel
}
#Conv_T_2/kernel/Read/ReadVariableOpReadVariableOpConv_T_2/kernel*(
_output_shapes
:��*
dtype0
s
Conv_T_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameConv_T_2/bias
l
!Conv_T_2/bias/Read/ReadVariableOpReadVariableOpConv_T_2/bias*
_output_shapes	
:�*
dtype0
�
Conv_T_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�* 
shared_nameConv_T_3/kernel
|
#Conv_T_3/kernel/Read/ReadVariableOpReadVariableOpConv_T_3/kernel*'
_output_shapes
:@�*
dtype0
r
Conv_T_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameConv_T_3/bias
k
!Conv_T_3/bias/Read/ReadVariableOpReadVariableOpConv_T_3/bias*
_output_shapes
:@*
dtype0
�
Conv_T_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: �* 
shared_nameConv_T_4/kernel
|
#Conv_T_4/kernel/Read/ReadVariableOpReadVariableOpConv_T_4/kernel*'
_output_shapes
: �*
dtype0
r
Conv_T_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameConv_T_4/bias
k
!Conv_T_4/bias/Read/ReadVariableOpReadVariableOpConv_T_4/bias*
_output_shapes
: *
dtype0
�
Conv_T_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameConv_T_5/kernel
{
#Conv_T_5/kernel/Read/ReadVariableOpReadVariableOpConv_T_5/kernel*&
_output_shapes
: *
dtype0
r
Conv_T_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConv_T_5/bias
k
!Conv_T_5/bias/Read/ReadVariableOpReadVariableOpConv_T_5/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
Adam/Conv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/Conv_1/kernel/m
�
(Adam/Conv_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv_1/kernel/m*&
_output_shapes
: *
dtype0
|
Adam/Conv_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/Conv_1/bias/m
u
&Adam/Conv_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv_1/bias/m*
_output_shapes
: *
dtype0
�
Adam/Conv_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*%
shared_nameAdam/Conv_2/kernel/m
�
(Adam/Conv_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv_2/kernel/m*&
_output_shapes
: @*
dtype0
|
Adam/Conv_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/Conv_2/bias/m
u
&Adam/Conv_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv_2/bias/m*
_output_shapes
:@*
dtype0
�
Adam/Conv_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*%
shared_nameAdam/Conv_3/kernel/m
�
(Adam/Conv_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv_3/kernel/m*'
_output_shapes
:@�*
dtype0
}
Adam/Conv_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/Conv_3/bias/m
v
&Adam/Conv_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv_3/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/Conv_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*%
shared_nameAdam/Conv_4/kernel/m
�
(Adam/Conv_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv_4/kernel/m*(
_output_shapes
:��*
dtype0
}
Adam/Conv_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/Conv_4/bias/m
v
&Adam/Conv_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv_4/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/Conv_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*%
shared_nameAdam/Conv_5/kernel/m
�
(Adam/Conv_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv_5/kernel/m*(
_output_shapes
:��*
dtype0
}
Adam/Conv_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/Conv_5/bias/m
v
&Adam/Conv_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv_5/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/Conv_T_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/Conv_T_1/kernel/m
�
*Adam/Conv_T_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv_T_1/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/Conv_T_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/Conv_T_1/bias/m
z
(Adam/Conv_T_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv_T_1/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/Conv_T_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/Conv_T_2/kernel/m
�
*Adam/Conv_T_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv_T_2/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/Conv_T_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/Conv_T_2/bias/m
z
(Adam/Conv_T_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv_T_2/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/Conv_T_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*'
shared_nameAdam/Conv_T_3/kernel/m
�
*Adam/Conv_T_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv_T_3/kernel/m*'
_output_shapes
:@�*
dtype0
�
Adam/Conv_T_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/Conv_T_3/bias/m
y
(Adam/Conv_T_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv_T_3/bias/m*
_output_shapes
:@*
dtype0
�
Adam/Conv_T_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*'
shared_nameAdam/Conv_T_4/kernel/m
�
*Adam/Conv_T_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv_T_4/kernel/m*'
_output_shapes
: �*
dtype0
�
Adam/Conv_T_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/Conv_T_4/bias/m
y
(Adam/Conv_T_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv_T_4/bias/m*
_output_shapes
: *
dtype0
�
Adam/Conv_T_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/Conv_T_5/kernel/m
�
*Adam/Conv_T_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv_T_5/kernel/m*&
_output_shapes
: *
dtype0
�
Adam/Conv_T_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/Conv_T_5/bias/m
y
(Adam/Conv_T_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv_T_5/bias/m*
_output_shapes
:*
dtype0
�
Adam/Conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/Conv_1/kernel/v
�
(Adam/Conv_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv_1/kernel/v*&
_output_shapes
: *
dtype0
|
Adam/Conv_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/Conv_1/bias/v
u
&Adam/Conv_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv_1/bias/v*
_output_shapes
: *
dtype0
�
Adam/Conv_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*%
shared_nameAdam/Conv_2/kernel/v
�
(Adam/Conv_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv_2/kernel/v*&
_output_shapes
: @*
dtype0
|
Adam/Conv_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/Conv_2/bias/v
u
&Adam/Conv_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv_2/bias/v*
_output_shapes
:@*
dtype0
�
Adam/Conv_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*%
shared_nameAdam/Conv_3/kernel/v
�
(Adam/Conv_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv_3/kernel/v*'
_output_shapes
:@�*
dtype0
}
Adam/Conv_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/Conv_3/bias/v
v
&Adam/Conv_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv_3/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/Conv_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*%
shared_nameAdam/Conv_4/kernel/v
�
(Adam/Conv_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv_4/kernel/v*(
_output_shapes
:��*
dtype0
}
Adam/Conv_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/Conv_4/bias/v
v
&Adam/Conv_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv_4/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/Conv_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*%
shared_nameAdam/Conv_5/kernel/v
�
(Adam/Conv_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv_5/kernel/v*(
_output_shapes
:��*
dtype0
}
Adam/Conv_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/Conv_5/bias/v
v
&Adam/Conv_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv_5/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/Conv_T_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/Conv_T_1/kernel/v
�
*Adam/Conv_T_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv_T_1/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/Conv_T_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/Conv_T_1/bias/v
z
(Adam/Conv_T_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv_T_1/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/Conv_T_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/Conv_T_2/kernel/v
�
*Adam/Conv_T_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv_T_2/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/Conv_T_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/Conv_T_2/bias/v
z
(Adam/Conv_T_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv_T_2/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/Conv_T_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*'
shared_nameAdam/Conv_T_3/kernel/v
�
*Adam/Conv_T_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv_T_3/kernel/v*'
_output_shapes
:@�*
dtype0
�
Adam/Conv_T_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/Conv_T_3/bias/v
y
(Adam/Conv_T_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv_T_3/bias/v*
_output_shapes
:@*
dtype0
�
Adam/Conv_T_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*'
shared_nameAdam/Conv_T_4/kernel/v
�
*Adam/Conv_T_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv_T_4/kernel/v*'
_output_shapes
: �*
dtype0
�
Adam/Conv_T_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/Conv_T_4/bias/v
y
(Adam/Conv_T_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv_T_4/bias/v*
_output_shapes
: *
dtype0
�
Adam/Conv_T_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/Conv_T_5/kernel/v
�
*Adam/Conv_T_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv_T_5/kernel/v*&
_output_shapes
: *
dtype0
�
Adam/Conv_T_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/Conv_T_5/bias/v
y
(Adam/Conv_T_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv_T_5/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
layer-15
layer_with_weights-5
layer-16
layer-17
layer-18
layer_with_weights-6
layer-19
layer_with_weights-7
layer-20
layer-21
layer-22
layer_with_weights-8
layer-23
layer_with_weights-9
layer-24
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
 
R
 	variables
!trainable_variables
"regularization_losses
#	keras_api
x
$
activation

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
R
+	variables
,trainable_variables
-regularization_losses
.	keras_api
x
/
activation

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
R
6	variables
7trainable_variables
8regularization_losses
9	keras_api
R
:	variables
;trainable_variables
<regularization_losses
=	keras_api
x
>
activation

?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
R
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
x
I
activation

Jkernel
Kbias
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
R
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
R
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
x
X
activation

Ykernel
Zbias
[	variables
\trainable_variables
]regularization_losses
^	keras_api
R
_	variables
`trainable_variables
aregularization_losses
b	keras_api
R
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
x
g
activation

hkernel
ibias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
R
n	variables
otrainable_variables
pregularization_losses
q	keras_api
R
r	variables
strainable_variables
tregularization_losses
u	keras_api
x
v
activation

wkernel
xbias
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
|
}
activation

~kernel
bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api

�
activation
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate%m�&m�0m�1m�?m�@m�Jm�Km�Ym�Zm�hm�im�wm�xm�~m�m�	�m�	�m�	�m�	�m�%v�&v�0v�1v�?v�@v�Jv�Kv�Yv�Zv�hv�iv�wv�xv�~v�v�	�v�	�v�	�v�	�v�
�
%0
&1
02
13
?4
@5
J6
K7
Y8
Z9
h10
i11
w12
x13
~14
15
�16
�17
�18
�19
�
%0
&1
02
13
?4
@5
J6
K7
Y8
Z9
h10
i11
w12
x13
~14
15
�16
�17
�18
�19
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
	variables
trainable_variables
�layers
�non_trainable_variables
regularization_losses
 
 
 
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
 	variables
!trainable_variables
�layers
�non_trainable_variables
"regularization_losses
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
YW
VARIABLE_VALUEConv_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEConv_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1

%0
&1
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
'	variables
(trainable_variables
�layers
�non_trainable_variables
)regularization_losses
 
 
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
+	variables
,trainable_variables
�layers
�non_trainable_variables
-regularization_losses
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
YW
VARIABLE_VALUEConv_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEConv_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11

00
11
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
2	variables
3trainable_variables
�layers
�non_trainable_variables
4regularization_losses
 
 
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
6	variables
7trainable_variables
�layers
�non_trainable_variables
8regularization_losses
 
 
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
:	variables
;trainable_variables
�layers
�non_trainable_variables
<regularization_losses
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
YW
VARIABLE_VALUEConv_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEConv_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
@1

?0
@1
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
A	variables
Btrainable_variables
�layers
�non_trainable_variables
Cregularization_losses
 
 
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
E	variables
Ftrainable_variables
�layers
�non_trainable_variables
Gregularization_losses
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
YW
VARIABLE_VALUEConv_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEConv_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

J0
K1

J0
K1
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
L	variables
Mtrainable_variables
�layers
�non_trainable_variables
Nregularization_losses
 
 
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
P	variables
Qtrainable_variables
�layers
�non_trainable_variables
Rregularization_losses
 
 
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
T	variables
Utrainable_variables
�layers
�non_trainable_variables
Vregularization_losses
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
YW
VARIABLE_VALUEConv_5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEConv_5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

Y0
Z1

Y0
Z1
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
[	variables
\trainable_variables
�layers
�non_trainable_variables
]regularization_losses
 
 
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
_	variables
`trainable_variables
�layers
�non_trainable_variables
aregularization_losses
 
 
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
c	variables
dtrainable_variables
�layers
�non_trainable_variables
eregularization_losses
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
[Y
VARIABLE_VALUEConv_T_1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv_T_1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

h0
i1

h0
i1
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
j	variables
ktrainable_variables
�layers
�non_trainable_variables
lregularization_losses
 
 
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
n	variables
otrainable_variables
�layers
�non_trainable_variables
pregularization_losses
 
 
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
r	variables
strainable_variables
�layers
�non_trainable_variables
tregularization_losses
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
[Y
VARIABLE_VALUEConv_T_2/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv_T_2/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

w0
x1

w0
x1
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
y	variables
ztrainable_variables
�layers
�non_trainable_variables
{regularization_losses
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
[Y
VARIABLE_VALUEConv_T_3/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv_T_3/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

~0
1

~0
1
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
 
 
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
 
 
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
[Y
VARIABLE_VALUEConv_T_4/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv_T_4/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
[Y
VARIABLE_VALUEConv_T_5/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv_T_5/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 

�0
�1
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
 
 
 
 
 
 
 
 
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
 
 
 

$0
 
 
 
 
 
 
 
 
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
 
 
 

/0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
 
 
 

>0
 
 
 
 
 
 
 
 
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
 
 
 

I0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
 
 
 

X0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
 
 
 

g0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
 
 
 

v0
 
 
 
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
 
 
 

}0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
 
 
 

�0
 
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
|z
VARIABLE_VALUEAdam/Conv_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/Conv_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Conv_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/Conv_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Conv_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/Conv_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Conv_4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/Conv_4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Conv_5/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/Conv_5/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_T_1/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv_T_1/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_T_2/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv_T_2/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_T_3/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv_T_3/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_T_4/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv_T_4/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_T_5/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv_T_5/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Conv_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/Conv_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Conv_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/Conv_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Conv_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/Conv_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Conv_4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/Conv_4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Conv_5/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/Conv_5/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_T_1/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv_T_1/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_T_2/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv_T_2/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_T_3/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv_T_3/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_T_4/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv_T_4/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_T_5/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv_T_5/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_Img_Input_LayerPlaceholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
 serving_default_Mask_Input_LayerPlaceholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_Img_Input_Layer serving_default_Mask_Input_LayerConv_1/kernelConv_1/biasConv_2/kernelConv_2/biasConv_3/kernelConv_3/biasConv_4/kernelConv_4/biasConv_5/kernelConv_5/biasConv_T_1/kernelConv_T_1/biasConv_T_2/kernelConv_T_2/biasConv_T_3/kernelConv_T_3/biasConv_T_4/kernelConv_T_4/biasConv_T_5/kernelConv_T_5/bias*!
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference_signature_wrapper_28965
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!Conv_1/kernel/Read/ReadVariableOpConv_1/bias/Read/ReadVariableOp!Conv_2/kernel/Read/ReadVariableOpConv_2/bias/Read/ReadVariableOp!Conv_3/kernel/Read/ReadVariableOpConv_3/bias/Read/ReadVariableOp!Conv_4/kernel/Read/ReadVariableOpConv_4/bias/Read/ReadVariableOp!Conv_5/kernel/Read/ReadVariableOpConv_5/bias/Read/ReadVariableOp#Conv_T_1/kernel/Read/ReadVariableOp!Conv_T_1/bias/Read/ReadVariableOp#Conv_T_2/kernel/Read/ReadVariableOp!Conv_T_2/bias/Read/ReadVariableOp#Conv_T_3/kernel/Read/ReadVariableOp!Conv_T_3/bias/Read/ReadVariableOp#Conv_T_4/kernel/Read/ReadVariableOp!Conv_T_4/bias/Read/ReadVariableOp#Conv_T_5/kernel/Read/ReadVariableOp!Conv_T_5/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/Conv_1/kernel/m/Read/ReadVariableOp&Adam/Conv_1/bias/m/Read/ReadVariableOp(Adam/Conv_2/kernel/m/Read/ReadVariableOp&Adam/Conv_2/bias/m/Read/ReadVariableOp(Adam/Conv_3/kernel/m/Read/ReadVariableOp&Adam/Conv_3/bias/m/Read/ReadVariableOp(Adam/Conv_4/kernel/m/Read/ReadVariableOp&Adam/Conv_4/bias/m/Read/ReadVariableOp(Adam/Conv_5/kernel/m/Read/ReadVariableOp&Adam/Conv_5/bias/m/Read/ReadVariableOp*Adam/Conv_T_1/kernel/m/Read/ReadVariableOp(Adam/Conv_T_1/bias/m/Read/ReadVariableOp*Adam/Conv_T_2/kernel/m/Read/ReadVariableOp(Adam/Conv_T_2/bias/m/Read/ReadVariableOp*Adam/Conv_T_3/kernel/m/Read/ReadVariableOp(Adam/Conv_T_3/bias/m/Read/ReadVariableOp*Adam/Conv_T_4/kernel/m/Read/ReadVariableOp(Adam/Conv_T_4/bias/m/Read/ReadVariableOp*Adam/Conv_T_5/kernel/m/Read/ReadVariableOp(Adam/Conv_T_5/bias/m/Read/ReadVariableOp(Adam/Conv_1/kernel/v/Read/ReadVariableOp&Adam/Conv_1/bias/v/Read/ReadVariableOp(Adam/Conv_2/kernel/v/Read/ReadVariableOp&Adam/Conv_2/bias/v/Read/ReadVariableOp(Adam/Conv_3/kernel/v/Read/ReadVariableOp&Adam/Conv_3/bias/v/Read/ReadVariableOp(Adam/Conv_4/kernel/v/Read/ReadVariableOp&Adam/Conv_4/bias/v/Read/ReadVariableOp(Adam/Conv_5/kernel/v/Read/ReadVariableOp&Adam/Conv_5/bias/v/Read/ReadVariableOp*Adam/Conv_T_1/kernel/v/Read/ReadVariableOp(Adam/Conv_T_1/bias/v/Read/ReadVariableOp*Adam/Conv_T_2/kernel/v/Read/ReadVariableOp(Adam/Conv_T_2/bias/v/Read/ReadVariableOp*Adam/Conv_T_3/kernel/v/Read/ReadVariableOp(Adam/Conv_T_3/bias/v/Read/ReadVariableOp*Adam/Conv_T_4/kernel/v/Read/ReadVariableOp(Adam/Conv_T_4/bias/v/Read/ReadVariableOp*Adam/Conv_T_5/kernel/v/Read/ReadVariableOp(Adam/Conv_T_5/bias/v/Read/ReadVariableOpConst*R
TinK
I2G	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference__traced_save_30249
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConv_1/kernelConv_1/biasConv_2/kernelConv_2/biasConv_3/kernelConv_3/biasConv_4/kernelConv_4/biasConv_5/kernelConv_5/biasConv_T_1/kernelConv_T_1/biasConv_T_2/kernelConv_T_2/biasConv_T_3/kernelConv_T_3/biasConv_T_4/kernelConv_T_4/biasConv_T_5/kernelConv_T_5/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/Conv_1/kernel/mAdam/Conv_1/bias/mAdam/Conv_2/kernel/mAdam/Conv_2/bias/mAdam/Conv_3/kernel/mAdam/Conv_3/bias/mAdam/Conv_4/kernel/mAdam/Conv_4/bias/mAdam/Conv_5/kernel/mAdam/Conv_5/bias/mAdam/Conv_T_1/kernel/mAdam/Conv_T_1/bias/mAdam/Conv_T_2/kernel/mAdam/Conv_T_2/bias/mAdam/Conv_T_3/kernel/mAdam/Conv_T_3/bias/mAdam/Conv_T_4/kernel/mAdam/Conv_T_4/bias/mAdam/Conv_T_5/kernel/mAdam/Conv_T_5/bias/mAdam/Conv_1/kernel/vAdam/Conv_1/bias/vAdam/Conv_2/kernel/vAdam/Conv_2/bias/vAdam/Conv_3/kernel/vAdam/Conv_3/bias/vAdam/Conv_4/kernel/vAdam/Conv_4/bias/vAdam/Conv_5/kernel/vAdam/Conv_5/bias/vAdam/Conv_T_1/kernel/vAdam/Conv_T_1/bias/vAdam/Conv_T_2/kernel/vAdam/Conv_T_2/bias/vAdam/Conv_T_3/kernel/vAdam/Conv_T_3/bias/vAdam/Conv_T_4/kernel/vAdam/Conv_T_4/bias/vAdam/Conv_T_5/kernel/vAdam/Conv_T_5/bias/v*Q
TinJ
H2F*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_restore_30466��
�]
�
U__inference_Facial_Landmark_Completion_layer_call_and_return_conditional_losses_28866

inputs
inputs_1
conv_1_28803
conv_1_28805
conv_2_28809
conv_2_28811
conv_3_28816
conv_3_28818
conv_4_28822
conv_4_28824
conv_5_28829
conv_5_28831
conv_t_1_28836
conv_t_1_28838
conv_t_2_28843
conv_t_2_28845
conv_t_3_28848
conv_t_3_28850
conv_t_4_28855
conv_t_4_28857
conv_t_5_28860
conv_t_5_28862
identity��Conv_1/StatefulPartitionedCall�Conv_2/StatefulPartitionedCall�Conv_3/StatefulPartitionedCall�Conv_4/StatefulPartitionedCall�Conv_5/StatefulPartitionedCall� Conv_T_1/StatefulPartitionedCall� Conv_T_2/StatefulPartitionedCall� Conv_T_3/StatefulPartitionedCall� Conv_T_4/StatefulPartitionedCall� Conv_T_5/StatefulPartitionedCall�
#Concatenated_Inputs/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_Concatenated_Inputs_layer_call_and_return_conditional_losses_282102%
#Concatenated_Inputs/PartitionedCall�
Conv_1/StatefulPartitionedCallStatefulPartitionedCall,Concatenated_Inputs/PartitionedCall:output:0conv_1_28803conv_1_28805*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_Conv_1_layer_call_and_return_conditional_losses_282302 
Conv_1/StatefulPartitionedCall�
Max_Pool_1/PartitionedCallPartitionedCall'Conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������`P * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_Max_Pool_1_layer_call_and_return_conditional_losses_275312
Max_Pool_1/PartitionedCall�
Conv_2/StatefulPartitionedCallStatefulPartitionedCall#Max_Pool_1/PartitionedCall:output:0conv_2_28809conv_2_28811*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������`P@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_Conv_2_layer_call_and_return_conditional_losses_282582 
Conv_2/StatefulPartitionedCall�
Max_Pool_2/PartitionedCallPartitionedCall'Conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0(@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_Max_Pool_2_layer_call_and_return_conditional_losses_275432
Max_Pool_2/PartitionedCall�
SPD_1/PartitionedCallPartitionedCall#Max_Pool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0(@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_1_layer_call_and_return_conditional_losses_283022
SPD_1/PartitionedCall�
Conv_3/StatefulPartitionedCallStatefulPartitionedCallSPD_1/PartitionedCall:output:0conv_3_28816conv_3_28818*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������0(�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_Conv_3_layer_call_and_return_conditional_losses_283252 
Conv_3/StatefulPartitionedCall�
Max_Pool_3/PartitionedCallPartitionedCall'Conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_Max_Pool_3_layer_call_and_return_conditional_losses_276232
Max_Pool_3/PartitionedCall�
Conv_4/StatefulPartitionedCallStatefulPartitionedCall#Max_Pool_3/PartitionedCall:output:0conv_4_28822conv_4_28824*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_Conv_4_layer_call_and_return_conditional_losses_283532 
Conv_4/StatefulPartitionedCall�
Max_Pool_4/PartitionedCallPartitionedCall'Conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_Max_Pool_4_layer_call_and_return_conditional_losses_276352
Max_Pool_4/PartitionedCall�
SPD_2/PartitionedCallPartitionedCall#Max_Pool_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_2_layer_call_and_return_conditional_losses_283972
SPD_2/PartitionedCall�
Conv_5/StatefulPartitionedCallStatefulPartitionedCallSPD_2/PartitionedCall:output:0conv_5_28829conv_5_28831*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_Conv_5_layer_call_and_return_conditional_losses_284202 
Conv_5/StatefulPartitionedCall�
Max_Pool_5/PartitionedCallPartitionedCall'Conv_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_Max_Pool_5_layer_call_and_return_conditional_losses_277152
Max_Pool_5/PartitionedCall�
SPD_3/PartitionedCallPartitionedCall#Max_Pool_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_3_layer_call_and_return_conditional_losses_284642
SPD_3/PartitionedCall�
 Conv_T_1/StatefulPartitionedCallStatefulPartitionedCallSPD_3/PartitionedCall:output:0conv_t_1_28836conv_t_1_28838*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Conv_T_1_layer_call_and_return_conditional_losses_278362"
 Conv_T_1/StatefulPartitionedCall�
Concat_1/PartitionedCallPartitionedCall)Conv_T_1/StatefulPartitionedCall:output:0#Max_Pool_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Concat_1_layer_call_and_return_conditional_losses_284882
Concat_1/PartitionedCall�
SPD_4/PartitionedCallPartitionedCall!Concat_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_4_layer_call_and_return_conditional_losses_285242
SPD_4/PartitionedCall�
 Conv_T_2/StatefulPartitionedCallStatefulPartitionedCallSPD_4/PartitionedCall:output:0conv_t_2_28843conv_t_2_28845*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Conv_T_2_layer_call_and_return_conditional_losses_279612"
 Conv_T_2/StatefulPartitionedCall�
 Conv_T_3/StatefulPartitionedCallStatefulPartitionedCall)Conv_T_2/StatefulPartitionedCall:output:0conv_t_3_28848conv_t_3_28850*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Conv_T_3_layer_call_and_return_conditional_losses_280182"
 Conv_T_3/StatefulPartitionedCall�
Concat_2/PartitionedCallPartitionedCall)Conv_T_3/StatefulPartitionedCall:output:0#Max_Pool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������0(�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Concat_2_layer_call_and_return_conditional_losses_285532
Concat_2/PartitionedCall�
SPD_5/PartitionedCallPartitionedCall!Concat_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������0(�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_5_layer_call_and_return_conditional_losses_285892
SPD_5/PartitionedCall�
 Conv_T_4/StatefulPartitionedCallStatefulPartitionedCallSPD_5/PartitionedCall:output:0conv_t_4_28855conv_t_4_28857*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Conv_T_4_layer_call_and_return_conditional_losses_281432"
 Conv_T_4/StatefulPartitionedCall�
 Conv_T_5/StatefulPartitionedCallStatefulPartitionedCall)Conv_T_4/StatefulPartitionedCall:output:0conv_t_5_28860conv_t_5_28862*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Conv_T_5_layer_call_and_return_conditional_losses_281882"
 Conv_T_5/StatefulPartitionedCall�
IdentityIdentity)Conv_T_5/StatefulPartitionedCall:output:0^Conv_1/StatefulPartitionedCall^Conv_2/StatefulPartitionedCall^Conv_3/StatefulPartitionedCall^Conv_4/StatefulPartitionedCall^Conv_5/StatefulPartitionedCall!^Conv_T_1/StatefulPartitionedCall!^Conv_T_2/StatefulPartitionedCall!^Conv_T_3/StatefulPartitionedCall!^Conv_T_4/StatefulPartitionedCall!^Conv_T_5/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������:�����������::::::::::::::::::::2@
Conv_1/StatefulPartitionedCallConv_1/StatefulPartitionedCall2@
Conv_2/StatefulPartitionedCallConv_2/StatefulPartitionedCall2@
Conv_3/StatefulPartitionedCallConv_3/StatefulPartitionedCall2@
Conv_4/StatefulPartitionedCallConv_4/StatefulPartitionedCall2@
Conv_5/StatefulPartitionedCallConv_5/StatefulPartitionedCall2D
 Conv_T_1/StatefulPartitionedCall Conv_T_1/StatefulPartitionedCall2D
 Conv_T_2/StatefulPartitionedCall Conv_T_2/StatefulPartitionedCall2D
 Conv_T_3/StatefulPartitionedCall Conv_T_3/StatefulPartitionedCall2D
 Conv_T_4/StatefulPartitionedCall Conv_T_4/StatefulPartitionedCall2D
 Conv_T_5/StatefulPartitionedCall Conv_T_5/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:YU
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�e
�
U__inference_Facial_Landmark_Completion_layer_call_and_return_conditional_losses_28612
img_input_layer
mask_input_layer
conv_1_28241
conv_1_28243
conv_2_28269
conv_2_28271
conv_3_28336
conv_3_28338
conv_4_28364
conv_4_28366
conv_5_28431
conv_5_28433
conv_t_1_28476
conv_t_1_28478
conv_t_2_28536
conv_t_2_28538
conv_t_3_28541
conv_t_3_28543
conv_t_4_28601
conv_t_4_28603
conv_t_5_28606
conv_t_5_28608
identity��Conv_1/StatefulPartitionedCall�Conv_2/StatefulPartitionedCall�Conv_3/StatefulPartitionedCall�Conv_4/StatefulPartitionedCall�Conv_5/StatefulPartitionedCall� Conv_T_1/StatefulPartitionedCall� Conv_T_2/StatefulPartitionedCall� Conv_T_3/StatefulPartitionedCall� Conv_T_4/StatefulPartitionedCall� Conv_T_5/StatefulPartitionedCall�SPD_1/StatefulPartitionedCall�SPD_2/StatefulPartitionedCall�SPD_3/StatefulPartitionedCall�SPD_4/StatefulPartitionedCall�SPD_5/StatefulPartitionedCall�
#Concatenated_Inputs/PartitionedCallPartitionedCallimg_input_layermask_input_layer*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_Concatenated_Inputs_layer_call_and_return_conditional_losses_282102%
#Concatenated_Inputs/PartitionedCall�
Conv_1/StatefulPartitionedCallStatefulPartitionedCall,Concatenated_Inputs/PartitionedCall:output:0conv_1_28241conv_1_28243*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_Conv_1_layer_call_and_return_conditional_losses_282302 
Conv_1/StatefulPartitionedCall�
Max_Pool_1/PartitionedCallPartitionedCall'Conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������`P * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_Max_Pool_1_layer_call_and_return_conditional_losses_275312
Max_Pool_1/PartitionedCall�
Conv_2/StatefulPartitionedCallStatefulPartitionedCall#Max_Pool_1/PartitionedCall:output:0conv_2_28269conv_2_28271*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������`P@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_Conv_2_layer_call_and_return_conditional_losses_282582 
Conv_2/StatefulPartitionedCall�
Max_Pool_2/PartitionedCallPartitionedCall'Conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0(@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_Max_Pool_2_layer_call_and_return_conditional_losses_275432
Max_Pool_2/PartitionedCall�
SPD_1/StatefulPartitionedCallStatefulPartitionedCall#Max_Pool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0(@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_1_layer_call_and_return_conditional_losses_282972
SPD_1/StatefulPartitionedCall�
Conv_3/StatefulPartitionedCallStatefulPartitionedCall&SPD_1/StatefulPartitionedCall:output:0conv_3_28336conv_3_28338*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������0(�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_Conv_3_layer_call_and_return_conditional_losses_283252 
Conv_3/StatefulPartitionedCall�
Max_Pool_3/PartitionedCallPartitionedCall'Conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_Max_Pool_3_layer_call_and_return_conditional_losses_276232
Max_Pool_3/PartitionedCall�
Conv_4/StatefulPartitionedCallStatefulPartitionedCall#Max_Pool_3/PartitionedCall:output:0conv_4_28364conv_4_28366*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_Conv_4_layer_call_and_return_conditional_losses_283532 
Conv_4/StatefulPartitionedCall�
Max_Pool_4/PartitionedCallPartitionedCall'Conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_Max_Pool_4_layer_call_and_return_conditional_losses_276352
Max_Pool_4/PartitionedCall�
SPD_2/StatefulPartitionedCallStatefulPartitionedCall#Max_Pool_4/PartitionedCall:output:0^SPD_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_2_layer_call_and_return_conditional_losses_283922
SPD_2/StatefulPartitionedCall�
Conv_5/StatefulPartitionedCallStatefulPartitionedCall&SPD_2/StatefulPartitionedCall:output:0conv_5_28431conv_5_28433*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_Conv_5_layer_call_and_return_conditional_losses_284202 
Conv_5/StatefulPartitionedCall�
Max_Pool_5/PartitionedCallPartitionedCall'Conv_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_Max_Pool_5_layer_call_and_return_conditional_losses_277152
Max_Pool_5/PartitionedCall�
SPD_3/StatefulPartitionedCallStatefulPartitionedCall#Max_Pool_5/PartitionedCall:output:0^SPD_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_3_layer_call_and_return_conditional_losses_284592
SPD_3/StatefulPartitionedCall�
 Conv_T_1/StatefulPartitionedCallStatefulPartitionedCall&SPD_3/StatefulPartitionedCall:output:0conv_t_1_28476conv_t_1_28478*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Conv_T_1_layer_call_and_return_conditional_losses_278362"
 Conv_T_1/StatefulPartitionedCall�
Concat_1/PartitionedCallPartitionedCall)Conv_T_1/StatefulPartitionedCall:output:0#Max_Pool_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Concat_1_layer_call_and_return_conditional_losses_284882
Concat_1/PartitionedCall�
SPD_4/StatefulPartitionedCallStatefulPartitionedCall!Concat_1/PartitionedCall:output:0^SPD_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_4_layer_call_and_return_conditional_losses_285192
SPD_4/StatefulPartitionedCall�
 Conv_T_2/StatefulPartitionedCallStatefulPartitionedCall&SPD_4/StatefulPartitionedCall:output:0conv_t_2_28536conv_t_2_28538*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Conv_T_2_layer_call_and_return_conditional_losses_279612"
 Conv_T_2/StatefulPartitionedCall�
 Conv_T_3/StatefulPartitionedCallStatefulPartitionedCall)Conv_T_2/StatefulPartitionedCall:output:0conv_t_3_28541conv_t_3_28543*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Conv_T_3_layer_call_and_return_conditional_losses_280182"
 Conv_T_3/StatefulPartitionedCall�
Concat_2/PartitionedCallPartitionedCall)Conv_T_3/StatefulPartitionedCall:output:0#Max_Pool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������0(�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Concat_2_layer_call_and_return_conditional_losses_285532
Concat_2/PartitionedCall�
SPD_5/StatefulPartitionedCallStatefulPartitionedCall!Concat_2/PartitionedCall:output:0^SPD_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������0(�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_5_layer_call_and_return_conditional_losses_285842
SPD_5/StatefulPartitionedCall�
 Conv_T_4/StatefulPartitionedCallStatefulPartitionedCall&SPD_5/StatefulPartitionedCall:output:0conv_t_4_28601conv_t_4_28603*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Conv_T_4_layer_call_and_return_conditional_losses_281432"
 Conv_T_4/StatefulPartitionedCall�
 Conv_T_5/StatefulPartitionedCallStatefulPartitionedCall)Conv_T_4/StatefulPartitionedCall:output:0conv_t_5_28606conv_t_5_28608*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Conv_T_5_layer_call_and_return_conditional_losses_281882"
 Conv_T_5/StatefulPartitionedCall�
IdentityIdentity)Conv_T_5/StatefulPartitionedCall:output:0^Conv_1/StatefulPartitionedCall^Conv_2/StatefulPartitionedCall^Conv_3/StatefulPartitionedCall^Conv_4/StatefulPartitionedCall^Conv_5/StatefulPartitionedCall!^Conv_T_1/StatefulPartitionedCall!^Conv_T_2/StatefulPartitionedCall!^Conv_T_3/StatefulPartitionedCall!^Conv_T_4/StatefulPartitionedCall!^Conv_T_5/StatefulPartitionedCall^SPD_1/StatefulPartitionedCall^SPD_2/StatefulPartitionedCall^SPD_3/StatefulPartitionedCall^SPD_4/StatefulPartitionedCall^SPD_5/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������:�����������::::::::::::::::::::2@
Conv_1/StatefulPartitionedCallConv_1/StatefulPartitionedCall2@
Conv_2/StatefulPartitionedCallConv_2/StatefulPartitionedCall2@
Conv_3/StatefulPartitionedCallConv_3/StatefulPartitionedCall2@
Conv_4/StatefulPartitionedCallConv_4/StatefulPartitionedCall2@
Conv_5/StatefulPartitionedCallConv_5/StatefulPartitionedCall2D
 Conv_T_1/StatefulPartitionedCall Conv_T_1/StatefulPartitionedCall2D
 Conv_T_2/StatefulPartitionedCall Conv_T_2/StatefulPartitionedCall2D
 Conv_T_3/StatefulPartitionedCall Conv_T_3/StatefulPartitionedCall2D
 Conv_T_4/StatefulPartitionedCall Conv_T_4/StatefulPartitionedCall2D
 Conv_T_5/StatefulPartitionedCall Conv_T_5/StatefulPartitionedCall2>
SPD_1/StatefulPartitionedCallSPD_1/StatefulPartitionedCall2>
SPD_2/StatefulPartitionedCallSPD_2/StatefulPartitionedCall2>
SPD_3/StatefulPartitionedCallSPD_3/StatefulPartitionedCall2>
SPD_4/StatefulPartitionedCallSPD_4/StatefulPartitionedCall2>
SPD_5/StatefulPartitionedCallSPD_5/StatefulPartitionedCall:b ^
1
_output_shapes
:�����������
)
_user_specified_nameImg_Input_Layer:c_
1
_output_shapes
:�����������
*
_user_specified_nameMask_Input_Layer
�
a
E__inference_Max_Pool_2_layer_call_and_return_conditional_losses_27543

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
a
E__inference_Max_Pool_4_layer_call_and_return_conditional_losses_27635

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
d
H__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_28134

inputs
identity~
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+��������������������������� *
alpha%���>2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+��������������������������� :i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
A
%__inference_SPD_3_layer_call_fn_29800

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_3_layer_call_and_return_conditional_losses_277862
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
m
C__inference_Concat_2_layer_call_and_return_conditional_losses_28553

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:���������0(�2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:���������0(�2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+���������������������������@:���������0(@:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������0(@
 
_user_specified_nameinputs
�%
�
C__inference_Conv_T_2_layer_call_and_return_conditional_losses_27961

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :�2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������2	
BiasAdd�
leaky_re_lu_6/PartitionedCallPartitionedCallBiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_279522
leaky_re_lu_6/PartitionedCall�
IdentityIdentity&leaky_re_lu_6/PartitionedCall:output:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������:::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
^
@__inference_SPD_4_layer_call_and_return_conditional_losses_28524

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:���������
�2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:���������
�2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:���������
�:X T
0
_output_shapes
:���������
�
 
_user_specified_nameinputs
��
�
U__inference_Facial_Landmark_Completion_layer_call_and_return_conditional_losses_29211
inputs_0
inputs_1)
%conv_1_conv2d_readvariableop_resource*
&conv_1_biasadd_readvariableop_resource)
%conv_2_conv2d_readvariableop_resource*
&conv_2_biasadd_readvariableop_resource)
%conv_3_conv2d_readvariableop_resource*
&conv_3_biasadd_readvariableop_resource)
%conv_4_conv2d_readvariableop_resource*
&conv_4_biasadd_readvariableop_resource)
%conv_5_conv2d_readvariableop_resource*
&conv_5_biasadd_readvariableop_resource5
1conv_t_1_conv2d_transpose_readvariableop_resource,
(conv_t_1_biasadd_readvariableop_resource5
1conv_t_2_conv2d_transpose_readvariableop_resource,
(conv_t_2_biasadd_readvariableop_resource5
1conv_t_3_conv2d_transpose_readvariableop_resource,
(conv_t_3_biasadd_readvariableop_resource5
1conv_t_4_conv2d_transpose_readvariableop_resource,
(conv_t_4_biasadd_readvariableop_resource5
1conv_t_5_conv2d_transpose_readvariableop_resource,
(conv_t_5_biasadd_readvariableop_resource
identity��
Concatenated_Inputs/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
Concatenated_Inputs/concat/axis�
Concatenated_Inputs/concatConcatV2inputs_0inputs_1(Concatenated_Inputs/concat/axis:output:0*
N*
T0*1
_output_shapes
:�����������2
Concatenated_Inputs/concat�
Conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv_1/Conv2D/ReadVariableOp�
Conv_1/Conv2DConv2D#Concatenated_Inputs/concat:output:0$Conv_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
2
Conv_1/Conv2D�
Conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
Conv_1/BiasAdd/ReadVariableOp�
Conv_1/BiasAddBiasAddConv_1/Conv2D:output:0%Conv_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� 2
Conv_1/BiasAdd�
Conv_1/leaky_re_lu/LeakyRelu	LeakyReluConv_1/BiasAdd:output:0*1
_output_shapes
:����������� *
alpha%���>2
Conv_1/leaky_re_lu/LeakyRelu�
Max_Pool_1/MaxPoolMaxPool*Conv_1/leaky_re_lu/LeakyRelu:activations:0*/
_output_shapes
:���������`P *
ksize
*
paddingVALID*
strides
2
Max_Pool_1/MaxPool�
Conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv_2/Conv2D/ReadVariableOp�
Conv_2/Conv2DConv2DMax_Pool_1/MaxPool:output:0$Conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������`P@*
paddingSAME*
strides
2
Conv_2/Conv2D�
Conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
Conv_2/BiasAdd/ReadVariableOp�
Conv_2/BiasAddBiasAddConv_2/Conv2D:output:0%Conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������`P@2
Conv_2/BiasAdd�
Conv_2/leaky_re_lu_1/LeakyRelu	LeakyReluConv_2/BiasAdd:output:0*/
_output_shapes
:���������`P@*
alpha%���>2 
Conv_2/leaky_re_lu_1/LeakyRelu�
Max_Pool_2/MaxPoolMaxPool,Conv_2/leaky_re_lu_1/LeakyRelu:activations:0*/
_output_shapes
:���������0(@*
ksize
*
paddingVALID*
strides
2
Max_Pool_2/MaxPoole
SPD_1/ShapeShapeMax_Pool_2/MaxPool:output:0*
T0*
_output_shapes
:2
SPD_1/Shape�
SPD_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
SPD_1/strided_slice/stack�
SPD_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
SPD_1/strided_slice/stack_1�
SPD_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
SPD_1/strided_slice/stack_2�
SPD_1/strided_sliceStridedSliceSPD_1/Shape:output:0"SPD_1/strided_slice/stack:output:0$SPD_1/strided_slice/stack_1:output:0$SPD_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
SPD_1/strided_slice�
SPD_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
SPD_1/strided_slice_1/stack�
SPD_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
SPD_1/strided_slice_1/stack_1�
SPD_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
SPD_1/strided_slice_1/stack_2�
SPD_1/strided_slice_1StridedSliceSPD_1/Shape:output:0$SPD_1/strided_slice_1/stack:output:0&SPD_1/strided_slice_1/stack_1:output:0&SPD_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
SPD_1/strided_slice_1o
SPD_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
SPD_1/dropout/Const�
SPD_1/dropout/MulMulMax_Pool_2/MaxPool:output:0SPD_1/dropout/Const:output:0*
T0*/
_output_shapes
:���������0(@2
SPD_1/dropout/Mul�
$SPD_1/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$SPD_1/dropout/random_uniform/shape/1�
$SPD_1/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$SPD_1/dropout/random_uniform/shape/2�
"SPD_1/dropout/random_uniform/shapePackSPD_1/strided_slice:output:0-SPD_1/dropout/random_uniform/shape/1:output:0-SPD_1/dropout/random_uniform/shape/2:output:0SPD_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2$
"SPD_1/dropout/random_uniform/shape�
*SPD_1/dropout/random_uniform/RandomUniformRandomUniform+SPD_1/dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02,
*SPD_1/dropout/random_uniform/RandomUniform�
SPD_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
SPD_1/dropout/GreaterEqual/y�
SPD_1/dropout/GreaterEqualGreaterEqual3SPD_1/dropout/random_uniform/RandomUniform:output:0%SPD_1/dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
SPD_1/dropout/GreaterEqual�
SPD_1/dropout/CastCastSPD_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
SPD_1/dropout/Cast�
SPD_1/dropout/Mul_1MulSPD_1/dropout/Mul:z:0SPD_1/dropout/Cast:y:0*
T0*/
_output_shapes
:���������0(@2
SPD_1/dropout/Mul_1�
Conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
Conv_3/Conv2D/ReadVariableOp�
Conv_3/Conv2DConv2DSPD_1/dropout/Mul_1:z:0$Conv_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������0(�*
paddingSAME*
strides
2
Conv_3/Conv2D�
Conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
Conv_3/BiasAdd/ReadVariableOp�
Conv_3/BiasAddBiasAddConv_3/Conv2D:output:0%Conv_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������0(�2
Conv_3/BiasAdd�
Conv_3/leaky_re_lu_2/LeakyRelu	LeakyReluConv_3/BiasAdd:output:0*0
_output_shapes
:���������0(�*
alpha%���>2 
Conv_3/leaky_re_lu_2/LeakyRelu�
Max_Pool_3/MaxPoolMaxPool,Conv_3/leaky_re_lu_2/LeakyRelu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
Max_Pool_3/MaxPool�
Conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv_4/Conv2D/ReadVariableOp�
Conv_4/Conv2DConv2DMax_Pool_3/MaxPool:output:0$Conv_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv_4/Conv2D�
Conv_4/BiasAdd/ReadVariableOpReadVariableOp&conv_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
Conv_4/BiasAdd/ReadVariableOp�
Conv_4/BiasAddBiasAddConv_4/Conv2D:output:0%Conv_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
Conv_4/BiasAdd�
Conv_4/leaky_re_lu_3/LeakyRelu	LeakyReluConv_4/BiasAdd:output:0*0
_output_shapes
:����������*
alpha%���>2 
Conv_4/leaky_re_lu_3/LeakyRelu�
Max_Pool_4/MaxPoolMaxPool,Conv_4/leaky_re_lu_3/LeakyRelu:activations:0*0
_output_shapes
:���������
�*
ksize
*
paddingVALID*
strides
2
Max_Pool_4/MaxPoole
SPD_2/ShapeShapeMax_Pool_4/MaxPool:output:0*
T0*
_output_shapes
:2
SPD_2/Shape�
SPD_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
SPD_2/strided_slice/stack�
SPD_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
SPD_2/strided_slice/stack_1�
SPD_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
SPD_2/strided_slice/stack_2�
SPD_2/strided_sliceStridedSliceSPD_2/Shape:output:0"SPD_2/strided_slice/stack:output:0$SPD_2/strided_slice/stack_1:output:0$SPD_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
SPD_2/strided_slice�
SPD_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
SPD_2/strided_slice_1/stack�
SPD_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
SPD_2/strided_slice_1/stack_1�
SPD_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
SPD_2/strided_slice_1/stack_2�
SPD_2/strided_slice_1StridedSliceSPD_2/Shape:output:0$SPD_2/strided_slice_1/stack:output:0&SPD_2/strided_slice_1/stack_1:output:0&SPD_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
SPD_2/strided_slice_1o
SPD_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
SPD_2/dropout/Const�
SPD_2/dropout/MulMulMax_Pool_4/MaxPool:output:0SPD_2/dropout/Const:output:0*
T0*0
_output_shapes
:���������
�2
SPD_2/dropout/Mul�
$SPD_2/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$SPD_2/dropout/random_uniform/shape/1�
$SPD_2/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$SPD_2/dropout/random_uniform/shape/2�
"SPD_2/dropout/random_uniform/shapePackSPD_2/strided_slice:output:0-SPD_2/dropout/random_uniform/shape/1:output:0-SPD_2/dropout/random_uniform/shape/2:output:0SPD_2/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2$
"SPD_2/dropout/random_uniform/shape�
*SPD_2/dropout/random_uniform/RandomUniformRandomUniform+SPD_2/dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02,
*SPD_2/dropout/random_uniform/RandomUniform�
SPD_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
SPD_2/dropout/GreaterEqual/y�
SPD_2/dropout/GreaterEqualGreaterEqual3SPD_2/dropout/random_uniform/RandomUniform:output:0%SPD_2/dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
SPD_2/dropout/GreaterEqual�
SPD_2/dropout/CastCastSPD_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
SPD_2/dropout/Cast�
SPD_2/dropout/Mul_1MulSPD_2/dropout/Mul:z:0SPD_2/dropout/Cast:y:0*
T0*0
_output_shapes
:���������
�2
SPD_2/dropout/Mul_1�
Conv_5/Conv2D/ReadVariableOpReadVariableOp%conv_5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv_5/Conv2D/ReadVariableOp�
Conv_5/Conv2DConv2DSPD_2/dropout/Mul_1:z:0$Conv_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�*
paddingSAME*
strides
2
Conv_5/Conv2D�
Conv_5/BiasAdd/ReadVariableOpReadVariableOp&conv_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
Conv_5/BiasAdd/ReadVariableOp�
Conv_5/BiasAddBiasAddConv_5/Conv2D:output:0%Conv_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�2
Conv_5/BiasAdd�
Conv_5/leaky_re_lu_4/LeakyRelu	LeakyReluConv_5/BiasAdd:output:0*0
_output_shapes
:���������
�*
alpha%���>2 
Conv_5/leaky_re_lu_4/LeakyRelu�
Max_Pool_5/MaxPoolMaxPool,Conv_5/leaky_re_lu_4/LeakyRelu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
Max_Pool_5/MaxPoole
SPD_3/ShapeShapeMax_Pool_5/MaxPool:output:0*
T0*
_output_shapes
:2
SPD_3/Shape�
SPD_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
SPD_3/strided_slice/stack�
SPD_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
SPD_3/strided_slice/stack_1�
SPD_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
SPD_3/strided_slice/stack_2�
SPD_3/strided_sliceStridedSliceSPD_3/Shape:output:0"SPD_3/strided_slice/stack:output:0$SPD_3/strided_slice/stack_1:output:0$SPD_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
SPD_3/strided_slice�
SPD_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
SPD_3/strided_slice_1/stack�
SPD_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
SPD_3/strided_slice_1/stack_1�
SPD_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
SPD_3/strided_slice_1/stack_2�
SPD_3/strided_slice_1StridedSliceSPD_3/Shape:output:0$SPD_3/strided_slice_1/stack:output:0&SPD_3/strided_slice_1/stack_1:output:0&SPD_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
SPD_3/strided_slice_1o
SPD_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
SPD_3/dropout/Const�
SPD_3/dropout/MulMulMax_Pool_5/MaxPool:output:0SPD_3/dropout/Const:output:0*
T0*0
_output_shapes
:����������2
SPD_3/dropout/Mul�
$SPD_3/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$SPD_3/dropout/random_uniform/shape/1�
$SPD_3/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$SPD_3/dropout/random_uniform/shape/2�
"SPD_3/dropout/random_uniform/shapePackSPD_3/strided_slice:output:0-SPD_3/dropout/random_uniform/shape/1:output:0-SPD_3/dropout/random_uniform/shape/2:output:0SPD_3/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2$
"SPD_3/dropout/random_uniform/shape�
*SPD_3/dropout/random_uniform/RandomUniformRandomUniform+SPD_3/dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02,
*SPD_3/dropout/random_uniform/RandomUniform�
SPD_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
SPD_3/dropout/GreaterEqual/y�
SPD_3/dropout/GreaterEqualGreaterEqual3SPD_3/dropout/random_uniform/RandomUniform:output:0%SPD_3/dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
SPD_3/dropout/GreaterEqual�
SPD_3/dropout/CastCastSPD_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
SPD_3/dropout/Cast�
SPD_3/dropout/Mul_1MulSPD_3/dropout/Mul:z:0SPD_3/dropout/Cast:y:0*
T0*0
_output_shapes
:����������2
SPD_3/dropout/Mul_1g
Conv_T_1/ShapeShapeSPD_3/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
Conv_T_1/Shape�
Conv_T_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Conv_T_1/strided_slice/stack�
Conv_T_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
Conv_T_1/strided_slice/stack_1�
Conv_T_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
Conv_T_1/strided_slice/stack_2�
Conv_T_1/strided_sliceStridedSliceConv_T_1/Shape:output:0%Conv_T_1/strided_slice/stack:output:0'Conv_T_1/strided_slice/stack_1:output:0'Conv_T_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Conv_T_1/strided_slicef
Conv_T_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
Conv_T_1/stack/1f
Conv_T_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :
2
Conv_T_1/stack/2g
Conv_T_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :�2
Conv_T_1/stack/3�
Conv_T_1/stackPackConv_T_1/strided_slice:output:0Conv_T_1/stack/1:output:0Conv_T_1/stack/2:output:0Conv_T_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
Conv_T_1/stack�
Conv_T_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
Conv_T_1/strided_slice_1/stack�
 Conv_T_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 Conv_T_1/strided_slice_1/stack_1�
 Conv_T_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 Conv_T_1/strided_slice_1/stack_2�
Conv_T_1/strided_slice_1StridedSliceConv_T_1/stack:output:0'Conv_T_1/strided_slice_1/stack:output:0)Conv_T_1/strided_slice_1/stack_1:output:0)Conv_T_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Conv_T_1/strided_slice_1�
(Conv_T_1/conv2d_transpose/ReadVariableOpReadVariableOp1conv_t_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(Conv_T_1/conv2d_transpose/ReadVariableOp�
Conv_T_1/conv2d_transposeConv2DBackpropInputConv_T_1/stack:output:00Conv_T_1/conv2d_transpose/ReadVariableOp:value:0SPD_3/dropout/Mul_1:z:0*
T0*0
_output_shapes
:���������
�*
paddingSAME*
strides
2
Conv_T_1/conv2d_transpose�
Conv_T_1/BiasAdd/ReadVariableOpReadVariableOp(conv_t_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
Conv_T_1/BiasAdd/ReadVariableOp�
Conv_T_1/BiasAddBiasAdd"Conv_T_1/conv2d_transpose:output:0'Conv_T_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�2
Conv_T_1/BiasAdd�
 Conv_T_1/leaky_re_lu_5/LeakyRelu	LeakyReluConv_T_1/BiasAdd:output:0*0
_output_shapes
:���������
�*
alpha%���>2"
 Conv_T_1/leaky_re_lu_5/LeakyRelun
Concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Concat_1/concat/axis�
Concat_1/concatConcatV2.Conv_T_1/leaky_re_lu_5/LeakyRelu:activations:0Max_Pool_4/MaxPool:output:0Concat_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:���������
�2
Concat_1/concatb
SPD_4/ShapeShapeConcat_1/concat:output:0*
T0*
_output_shapes
:2
SPD_4/Shape�
SPD_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
SPD_4/strided_slice/stack�
SPD_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
SPD_4/strided_slice/stack_1�
SPD_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
SPD_4/strided_slice/stack_2�
SPD_4/strided_sliceStridedSliceSPD_4/Shape:output:0"SPD_4/strided_slice/stack:output:0$SPD_4/strided_slice/stack_1:output:0$SPD_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
SPD_4/strided_slice�
SPD_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
SPD_4/strided_slice_1/stack�
SPD_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
SPD_4/strided_slice_1/stack_1�
SPD_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
SPD_4/strided_slice_1/stack_2�
SPD_4/strided_slice_1StridedSliceSPD_4/Shape:output:0$SPD_4/strided_slice_1/stack:output:0&SPD_4/strided_slice_1/stack_1:output:0&SPD_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
SPD_4/strided_slice_1o
SPD_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
SPD_4/dropout/Const�
SPD_4/dropout/MulMulConcat_1/concat:output:0SPD_4/dropout/Const:output:0*
T0*0
_output_shapes
:���������
�2
SPD_4/dropout/Mul�
$SPD_4/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$SPD_4/dropout/random_uniform/shape/1�
$SPD_4/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$SPD_4/dropout/random_uniform/shape/2�
"SPD_4/dropout/random_uniform/shapePackSPD_4/strided_slice:output:0-SPD_4/dropout/random_uniform/shape/1:output:0-SPD_4/dropout/random_uniform/shape/2:output:0SPD_4/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2$
"SPD_4/dropout/random_uniform/shape�
*SPD_4/dropout/random_uniform/RandomUniformRandomUniform+SPD_4/dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02,
*SPD_4/dropout/random_uniform/RandomUniform�
SPD_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
SPD_4/dropout/GreaterEqual/y�
SPD_4/dropout/GreaterEqualGreaterEqual3SPD_4/dropout/random_uniform/RandomUniform:output:0%SPD_4/dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
SPD_4/dropout/GreaterEqual�
SPD_4/dropout/CastCastSPD_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
SPD_4/dropout/Cast�
SPD_4/dropout/Mul_1MulSPD_4/dropout/Mul:z:0SPD_4/dropout/Cast:y:0*
T0*0
_output_shapes
:���������
�2
SPD_4/dropout/Mul_1g
Conv_T_2/ShapeShapeSPD_4/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
Conv_T_2/Shape�
Conv_T_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Conv_T_2/strided_slice/stack�
Conv_T_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
Conv_T_2/strided_slice/stack_1�
Conv_T_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
Conv_T_2/strided_slice/stack_2�
Conv_T_2/strided_sliceStridedSliceConv_T_2/Shape:output:0%Conv_T_2/strided_slice/stack:output:0'Conv_T_2/strided_slice/stack_1:output:0'Conv_T_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Conv_T_2/strided_slicef
Conv_T_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
Conv_T_2/stack/1f
Conv_T_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
Conv_T_2/stack/2g
Conv_T_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :�2
Conv_T_2/stack/3�
Conv_T_2/stackPackConv_T_2/strided_slice:output:0Conv_T_2/stack/1:output:0Conv_T_2/stack/2:output:0Conv_T_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
Conv_T_2/stack�
Conv_T_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
Conv_T_2/strided_slice_1/stack�
 Conv_T_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 Conv_T_2/strided_slice_1/stack_1�
 Conv_T_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 Conv_T_2/strided_slice_1/stack_2�
Conv_T_2/strided_slice_1StridedSliceConv_T_2/stack:output:0'Conv_T_2/strided_slice_1/stack:output:0)Conv_T_2/strided_slice_1/stack_1:output:0)Conv_T_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Conv_T_2/strided_slice_1�
(Conv_T_2/conv2d_transpose/ReadVariableOpReadVariableOp1conv_t_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(Conv_T_2/conv2d_transpose/ReadVariableOp�
Conv_T_2/conv2d_transposeConv2DBackpropInputConv_T_2/stack:output:00Conv_T_2/conv2d_transpose/ReadVariableOp:value:0SPD_4/dropout/Mul_1:z:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv_T_2/conv2d_transpose�
Conv_T_2/BiasAdd/ReadVariableOpReadVariableOp(conv_t_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
Conv_T_2/BiasAdd/ReadVariableOp�
Conv_T_2/BiasAddBiasAdd"Conv_T_2/conv2d_transpose:output:0'Conv_T_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
Conv_T_2/BiasAdd�
 Conv_T_2/leaky_re_lu_6/LeakyRelu	LeakyReluConv_T_2/BiasAdd:output:0*0
_output_shapes
:����������*
alpha%���>2"
 Conv_T_2/leaky_re_lu_6/LeakyRelu~
Conv_T_3/ShapeShape.Conv_T_2/leaky_re_lu_6/LeakyRelu:activations:0*
T0*
_output_shapes
:2
Conv_T_3/Shape�
Conv_T_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Conv_T_3/strided_slice/stack�
Conv_T_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
Conv_T_3/strided_slice/stack_1�
Conv_T_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
Conv_T_3/strided_slice/stack_2�
Conv_T_3/strided_sliceStridedSliceConv_T_3/Shape:output:0%Conv_T_3/strided_slice/stack:output:0'Conv_T_3/strided_slice/stack_1:output:0'Conv_T_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Conv_T_3/strided_slicef
Conv_T_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :02
Conv_T_3/stack/1f
Conv_T_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :(2
Conv_T_3/stack/2f
Conv_T_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
Conv_T_3/stack/3�
Conv_T_3/stackPackConv_T_3/strided_slice:output:0Conv_T_3/stack/1:output:0Conv_T_3/stack/2:output:0Conv_T_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
Conv_T_3/stack�
Conv_T_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
Conv_T_3/strided_slice_1/stack�
 Conv_T_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 Conv_T_3/strided_slice_1/stack_1�
 Conv_T_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 Conv_T_3/strided_slice_1/stack_2�
Conv_T_3/strided_slice_1StridedSliceConv_T_3/stack:output:0'Conv_T_3/strided_slice_1/stack:output:0)Conv_T_3/strided_slice_1/stack_1:output:0)Conv_T_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Conv_T_3/strided_slice_1�
(Conv_T_3/conv2d_transpose/ReadVariableOpReadVariableOp1conv_t_3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype02*
(Conv_T_3/conv2d_transpose/ReadVariableOp�
Conv_T_3/conv2d_transposeConv2DBackpropInputConv_T_3/stack:output:00Conv_T_3/conv2d_transpose/ReadVariableOp:value:0.Conv_T_2/leaky_re_lu_6/LeakyRelu:activations:0*
T0*/
_output_shapes
:���������0(@*
paddingSAME*
strides
2
Conv_T_3/conv2d_transpose�
Conv_T_3/BiasAdd/ReadVariableOpReadVariableOp(conv_t_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
Conv_T_3/BiasAdd/ReadVariableOp�
Conv_T_3/BiasAddBiasAdd"Conv_T_3/conv2d_transpose:output:0'Conv_T_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0(@2
Conv_T_3/BiasAdd�
 Conv_T_3/leaky_re_lu_7/LeakyRelu	LeakyReluConv_T_3/BiasAdd:output:0*/
_output_shapes
:���������0(@*
alpha%���>2"
 Conv_T_3/leaky_re_lu_7/LeakyRelun
Concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Concat_2/concat/axis�
Concat_2/concatConcatV2.Conv_T_3/leaky_re_lu_7/LeakyRelu:activations:0Max_Pool_2/MaxPool:output:0Concat_2/concat/axis:output:0*
N*
T0*0
_output_shapes
:���������0(�2
Concat_2/concatb
SPD_5/ShapeShapeConcat_2/concat:output:0*
T0*
_output_shapes
:2
SPD_5/Shape�
SPD_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
SPD_5/strided_slice/stack�
SPD_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
SPD_5/strided_slice/stack_1�
SPD_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
SPD_5/strided_slice/stack_2�
SPD_5/strided_sliceStridedSliceSPD_5/Shape:output:0"SPD_5/strided_slice/stack:output:0$SPD_5/strided_slice/stack_1:output:0$SPD_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
SPD_5/strided_slice�
SPD_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
SPD_5/strided_slice_1/stack�
SPD_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
SPD_5/strided_slice_1/stack_1�
SPD_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
SPD_5/strided_slice_1/stack_2�
SPD_5/strided_slice_1StridedSliceSPD_5/Shape:output:0$SPD_5/strided_slice_1/stack:output:0&SPD_5/strided_slice_1/stack_1:output:0&SPD_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
SPD_5/strided_slice_1o
SPD_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
SPD_5/dropout/Const�
SPD_5/dropout/MulMulConcat_2/concat:output:0SPD_5/dropout/Const:output:0*
T0*0
_output_shapes
:���������0(�2
SPD_5/dropout/Mul�
$SPD_5/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$SPD_5/dropout/random_uniform/shape/1�
$SPD_5/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$SPD_5/dropout/random_uniform/shape/2�
"SPD_5/dropout/random_uniform/shapePackSPD_5/strided_slice:output:0-SPD_5/dropout/random_uniform/shape/1:output:0-SPD_5/dropout/random_uniform/shape/2:output:0SPD_5/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2$
"SPD_5/dropout/random_uniform/shape�
*SPD_5/dropout/random_uniform/RandomUniformRandomUniform+SPD_5/dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02,
*SPD_5/dropout/random_uniform/RandomUniform�
SPD_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
SPD_5/dropout/GreaterEqual/y�
SPD_5/dropout/GreaterEqualGreaterEqual3SPD_5/dropout/random_uniform/RandomUniform:output:0%SPD_5/dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
SPD_5/dropout/GreaterEqual�
SPD_5/dropout/CastCastSPD_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
SPD_5/dropout/Cast�
SPD_5/dropout/Mul_1MulSPD_5/dropout/Mul:z:0SPD_5/dropout/Cast:y:0*
T0*0
_output_shapes
:���������0(�2
SPD_5/dropout/Mul_1g
Conv_T_4/ShapeShapeSPD_5/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
Conv_T_4/Shape�
Conv_T_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Conv_T_4/strided_slice/stack�
Conv_T_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
Conv_T_4/strided_slice/stack_1�
Conv_T_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
Conv_T_4/strided_slice/stack_2�
Conv_T_4/strided_sliceStridedSliceConv_T_4/Shape:output:0%Conv_T_4/strided_slice/stack:output:0'Conv_T_4/strided_slice/stack_1:output:0'Conv_T_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Conv_T_4/strided_slicef
Conv_T_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :`2
Conv_T_4/stack/1f
Conv_T_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :P2
Conv_T_4/stack/2f
Conv_T_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Conv_T_4/stack/3�
Conv_T_4/stackPackConv_T_4/strided_slice:output:0Conv_T_4/stack/1:output:0Conv_T_4/stack/2:output:0Conv_T_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
Conv_T_4/stack�
Conv_T_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
Conv_T_4/strided_slice_1/stack�
 Conv_T_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 Conv_T_4/strided_slice_1/stack_1�
 Conv_T_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 Conv_T_4/strided_slice_1/stack_2�
Conv_T_4/strided_slice_1StridedSliceConv_T_4/stack:output:0'Conv_T_4/strided_slice_1/stack:output:0)Conv_T_4/strided_slice_1/stack_1:output:0)Conv_T_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Conv_T_4/strided_slice_1�
(Conv_T_4/conv2d_transpose/ReadVariableOpReadVariableOp1conv_t_4_conv2d_transpose_readvariableop_resource*'
_output_shapes
: �*
dtype02*
(Conv_T_4/conv2d_transpose/ReadVariableOp�
Conv_T_4/conv2d_transposeConv2DBackpropInputConv_T_4/stack:output:00Conv_T_4/conv2d_transpose/ReadVariableOp:value:0SPD_5/dropout/Mul_1:z:0*
T0*/
_output_shapes
:���������`P *
paddingSAME*
strides
2
Conv_T_4/conv2d_transpose�
Conv_T_4/BiasAdd/ReadVariableOpReadVariableOp(conv_t_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
Conv_T_4/BiasAdd/ReadVariableOp�
Conv_T_4/BiasAddBiasAdd"Conv_T_4/conv2d_transpose:output:0'Conv_T_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������`P 2
Conv_T_4/BiasAdd�
 Conv_T_4/leaky_re_lu_8/LeakyRelu	LeakyReluConv_T_4/BiasAdd:output:0*/
_output_shapes
:���������`P *
alpha%���>2"
 Conv_T_4/leaky_re_lu_8/LeakyRelu~
Conv_T_5/ShapeShape.Conv_T_4/leaky_re_lu_8/LeakyRelu:activations:0*
T0*
_output_shapes
:2
Conv_T_5/Shape�
Conv_T_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Conv_T_5/strided_slice/stack�
Conv_T_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
Conv_T_5/strided_slice/stack_1�
Conv_T_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
Conv_T_5/strided_slice/stack_2�
Conv_T_5/strided_sliceStridedSliceConv_T_5/Shape:output:0%Conv_T_5/strided_slice/stack:output:0'Conv_T_5/strided_slice/stack_1:output:0'Conv_T_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Conv_T_5/strided_sliceg
Conv_T_5/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�2
Conv_T_5/stack/1g
Conv_T_5/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�2
Conv_T_5/stack/2f
Conv_T_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
Conv_T_5/stack/3�
Conv_T_5/stackPackConv_T_5/strided_slice:output:0Conv_T_5/stack/1:output:0Conv_T_5/stack/2:output:0Conv_T_5/stack/3:output:0*
N*
T0*
_output_shapes
:2
Conv_T_5/stack�
Conv_T_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
Conv_T_5/strided_slice_1/stack�
 Conv_T_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 Conv_T_5/strided_slice_1/stack_1�
 Conv_T_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 Conv_T_5/strided_slice_1/stack_2�
Conv_T_5/strided_slice_1StridedSliceConv_T_5/stack:output:0'Conv_T_5/strided_slice_1/stack:output:0)Conv_T_5/strided_slice_1/stack_1:output:0)Conv_T_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Conv_T_5/strided_slice_1�
(Conv_T_5/conv2d_transpose/ReadVariableOpReadVariableOp1conv_t_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02*
(Conv_T_5/conv2d_transpose/ReadVariableOp�
Conv_T_5/conv2d_transposeConv2DBackpropInputConv_T_5/stack:output:00Conv_T_5/conv2d_transpose/ReadVariableOp:value:0.Conv_T_4/leaky_re_lu_8/LeakyRelu:activations:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
Conv_T_5/conv2d_transpose�
Conv_T_5/BiasAdd/ReadVariableOpReadVariableOp(conv_t_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
Conv_T_5/BiasAdd/ReadVariableOp�
Conv_T_5/BiasAddBiasAdd"Conv_T_5/conv2d_transpose:output:0'Conv_T_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
Conv_T_5/BiasAdd�
Conv_T_5/SigmoidSigmoidConv_T_5/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2
Conv_T_5/Sigmoidr
IdentityIdentityConv_T_5/Sigmoid:y:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������:�����������:::::::::::::::::::::[ W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/1
�
^
@__inference_SPD_5_layer_call_and_return_conditional_losses_29930

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:���������0(�2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:���������0(�2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:���������0(�:X T
0
_output_shapes
:���������0(�
 
_user_specified_nameinputs
�d
�
U__inference_Facial_Landmark_Completion_layer_call_and_return_conditional_losses_28752

inputs
inputs_1
conv_1_28689
conv_1_28691
conv_2_28695
conv_2_28697
conv_3_28702
conv_3_28704
conv_4_28708
conv_4_28710
conv_5_28715
conv_5_28717
conv_t_1_28722
conv_t_1_28724
conv_t_2_28729
conv_t_2_28731
conv_t_3_28734
conv_t_3_28736
conv_t_4_28741
conv_t_4_28743
conv_t_5_28746
conv_t_5_28748
identity��Conv_1/StatefulPartitionedCall�Conv_2/StatefulPartitionedCall�Conv_3/StatefulPartitionedCall�Conv_4/StatefulPartitionedCall�Conv_5/StatefulPartitionedCall� Conv_T_1/StatefulPartitionedCall� Conv_T_2/StatefulPartitionedCall� Conv_T_3/StatefulPartitionedCall� Conv_T_4/StatefulPartitionedCall� Conv_T_5/StatefulPartitionedCall�SPD_1/StatefulPartitionedCall�SPD_2/StatefulPartitionedCall�SPD_3/StatefulPartitionedCall�SPD_4/StatefulPartitionedCall�SPD_5/StatefulPartitionedCall�
#Concatenated_Inputs/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_Concatenated_Inputs_layer_call_and_return_conditional_losses_282102%
#Concatenated_Inputs/PartitionedCall�
Conv_1/StatefulPartitionedCallStatefulPartitionedCall,Concatenated_Inputs/PartitionedCall:output:0conv_1_28689conv_1_28691*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_Conv_1_layer_call_and_return_conditional_losses_282302 
Conv_1/StatefulPartitionedCall�
Max_Pool_1/PartitionedCallPartitionedCall'Conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������`P * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_Max_Pool_1_layer_call_and_return_conditional_losses_275312
Max_Pool_1/PartitionedCall�
Conv_2/StatefulPartitionedCallStatefulPartitionedCall#Max_Pool_1/PartitionedCall:output:0conv_2_28695conv_2_28697*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������`P@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_Conv_2_layer_call_and_return_conditional_losses_282582 
Conv_2/StatefulPartitionedCall�
Max_Pool_2/PartitionedCallPartitionedCall'Conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0(@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_Max_Pool_2_layer_call_and_return_conditional_losses_275432
Max_Pool_2/PartitionedCall�
SPD_1/StatefulPartitionedCallStatefulPartitionedCall#Max_Pool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0(@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_1_layer_call_and_return_conditional_losses_282972
SPD_1/StatefulPartitionedCall�
Conv_3/StatefulPartitionedCallStatefulPartitionedCall&SPD_1/StatefulPartitionedCall:output:0conv_3_28702conv_3_28704*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������0(�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_Conv_3_layer_call_and_return_conditional_losses_283252 
Conv_3/StatefulPartitionedCall�
Max_Pool_3/PartitionedCallPartitionedCall'Conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_Max_Pool_3_layer_call_and_return_conditional_losses_276232
Max_Pool_3/PartitionedCall�
Conv_4/StatefulPartitionedCallStatefulPartitionedCall#Max_Pool_3/PartitionedCall:output:0conv_4_28708conv_4_28710*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_Conv_4_layer_call_and_return_conditional_losses_283532 
Conv_4/StatefulPartitionedCall�
Max_Pool_4/PartitionedCallPartitionedCall'Conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_Max_Pool_4_layer_call_and_return_conditional_losses_276352
Max_Pool_4/PartitionedCall�
SPD_2/StatefulPartitionedCallStatefulPartitionedCall#Max_Pool_4/PartitionedCall:output:0^SPD_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_2_layer_call_and_return_conditional_losses_283922
SPD_2/StatefulPartitionedCall�
Conv_5/StatefulPartitionedCallStatefulPartitionedCall&SPD_2/StatefulPartitionedCall:output:0conv_5_28715conv_5_28717*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_Conv_5_layer_call_and_return_conditional_losses_284202 
Conv_5/StatefulPartitionedCall�
Max_Pool_5/PartitionedCallPartitionedCall'Conv_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_Max_Pool_5_layer_call_and_return_conditional_losses_277152
Max_Pool_5/PartitionedCall�
SPD_3/StatefulPartitionedCallStatefulPartitionedCall#Max_Pool_5/PartitionedCall:output:0^SPD_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_3_layer_call_and_return_conditional_losses_284592
SPD_3/StatefulPartitionedCall�
 Conv_T_1/StatefulPartitionedCallStatefulPartitionedCall&SPD_3/StatefulPartitionedCall:output:0conv_t_1_28722conv_t_1_28724*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Conv_T_1_layer_call_and_return_conditional_losses_278362"
 Conv_T_1/StatefulPartitionedCall�
Concat_1/PartitionedCallPartitionedCall)Conv_T_1/StatefulPartitionedCall:output:0#Max_Pool_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Concat_1_layer_call_and_return_conditional_losses_284882
Concat_1/PartitionedCall�
SPD_4/StatefulPartitionedCallStatefulPartitionedCall!Concat_1/PartitionedCall:output:0^SPD_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_4_layer_call_and_return_conditional_losses_285192
SPD_4/StatefulPartitionedCall�
 Conv_T_2/StatefulPartitionedCallStatefulPartitionedCall&SPD_4/StatefulPartitionedCall:output:0conv_t_2_28729conv_t_2_28731*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Conv_T_2_layer_call_and_return_conditional_losses_279612"
 Conv_T_2/StatefulPartitionedCall�
 Conv_T_3/StatefulPartitionedCallStatefulPartitionedCall)Conv_T_2/StatefulPartitionedCall:output:0conv_t_3_28734conv_t_3_28736*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Conv_T_3_layer_call_and_return_conditional_losses_280182"
 Conv_T_3/StatefulPartitionedCall�
Concat_2/PartitionedCallPartitionedCall)Conv_T_3/StatefulPartitionedCall:output:0#Max_Pool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������0(�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Concat_2_layer_call_and_return_conditional_losses_285532
Concat_2/PartitionedCall�
SPD_5/StatefulPartitionedCallStatefulPartitionedCall!Concat_2/PartitionedCall:output:0^SPD_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������0(�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_5_layer_call_and_return_conditional_losses_285842
SPD_5/StatefulPartitionedCall�
 Conv_T_4/StatefulPartitionedCallStatefulPartitionedCall&SPD_5/StatefulPartitionedCall:output:0conv_t_4_28741conv_t_4_28743*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Conv_T_4_layer_call_and_return_conditional_losses_281432"
 Conv_T_4/StatefulPartitionedCall�
 Conv_T_5/StatefulPartitionedCallStatefulPartitionedCall)Conv_T_4/StatefulPartitionedCall:output:0conv_t_5_28746conv_t_5_28748*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Conv_T_5_layer_call_and_return_conditional_losses_281882"
 Conv_T_5/StatefulPartitionedCall�
IdentityIdentity)Conv_T_5/StatefulPartitionedCall:output:0^Conv_1/StatefulPartitionedCall^Conv_2/StatefulPartitionedCall^Conv_3/StatefulPartitionedCall^Conv_4/StatefulPartitionedCall^Conv_5/StatefulPartitionedCall!^Conv_T_1/StatefulPartitionedCall!^Conv_T_2/StatefulPartitionedCall!^Conv_T_3/StatefulPartitionedCall!^Conv_T_4/StatefulPartitionedCall!^Conv_T_5/StatefulPartitionedCall^SPD_1/StatefulPartitionedCall^SPD_2/StatefulPartitionedCall^SPD_3/StatefulPartitionedCall^SPD_4/StatefulPartitionedCall^SPD_5/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������:�����������::::::::::::::::::::2@
Conv_1/StatefulPartitionedCallConv_1/StatefulPartitionedCall2@
Conv_2/StatefulPartitionedCallConv_2/StatefulPartitionedCall2@
Conv_3/StatefulPartitionedCallConv_3/StatefulPartitionedCall2@
Conv_4/StatefulPartitionedCallConv_4/StatefulPartitionedCall2@
Conv_5/StatefulPartitionedCallConv_5/StatefulPartitionedCall2D
 Conv_T_1/StatefulPartitionedCall Conv_T_1/StatefulPartitionedCall2D
 Conv_T_2/StatefulPartitionedCall Conv_T_2/StatefulPartitionedCall2D
 Conv_T_3/StatefulPartitionedCall Conv_T_3/StatefulPartitionedCall2D
 Conv_T_4/StatefulPartitionedCall Conv_T_4/StatefulPartitionedCall2D
 Conv_T_5/StatefulPartitionedCall Conv_T_5/StatefulPartitionedCall2>
SPD_1/StatefulPartitionedCallSPD_1/StatefulPartitionedCall2>
SPD_2/StatefulPartitionedCallSPD_2/StatefulPartitionedCall2>
SPD_3/StatefulPartitionedCallSPD_3/StatefulPartitionedCall2>
SPD_4/StatefulPartitionedCallSPD_4/StatefulPartitionedCall2>
SPD_5/StatefulPartitionedCallSPD_5/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:YU
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
_
@__inference_SPD_4_layer_call_and_return_conditional_losses_28519

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:���������
�2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:���������
�2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:���������
�2

Identity"
identityIdentity:output:0*/
_input_shapes
:���������
�:X T
0
_output_shapes
:���������
�
 
_user_specified_nameinputs
�
^
%__inference_SPD_3_layer_call_fn_29757

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_3_layer_call_and_return_conditional_losses_284592
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
A__inference_Conv_5_layer_call_and_return_conditional_losses_29715

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�2	
BiasAdd�
leaky_re_lu_4/LeakyRelu	LeakyReluBiasAdd:output:0*0
_output_shapes
:���������
�*
alpha%���>2
leaky_re_lu_4/LeakyRelu�
IdentityIdentity%leaky_re_lu_4/LeakyRelu:activations:0*
T0*0
_output_shapes
:���������
�2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:���������
�:::X T
0
_output_shapes
:���������
�
 
_user_specified_nameinputs
�
�
:__inference_Facial_Landmark_Completion_layer_call_fn_29413
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_Facial_Landmark_Completion_layer_call_and_return_conditional_losses_287522
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������:�����������::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/1
��
�
U__inference_Facial_Landmark_Completion_layer_call_and_return_conditional_losses_29367
inputs_0
inputs_1)
%conv_1_conv2d_readvariableop_resource*
&conv_1_biasadd_readvariableop_resource)
%conv_2_conv2d_readvariableop_resource*
&conv_2_biasadd_readvariableop_resource)
%conv_3_conv2d_readvariableop_resource*
&conv_3_biasadd_readvariableop_resource)
%conv_4_conv2d_readvariableop_resource*
&conv_4_biasadd_readvariableop_resource)
%conv_5_conv2d_readvariableop_resource*
&conv_5_biasadd_readvariableop_resource5
1conv_t_1_conv2d_transpose_readvariableop_resource,
(conv_t_1_biasadd_readvariableop_resource5
1conv_t_2_conv2d_transpose_readvariableop_resource,
(conv_t_2_biasadd_readvariableop_resource5
1conv_t_3_conv2d_transpose_readvariableop_resource,
(conv_t_3_biasadd_readvariableop_resource5
1conv_t_4_conv2d_transpose_readvariableop_resource,
(conv_t_4_biasadd_readvariableop_resource5
1conv_t_5_conv2d_transpose_readvariableop_resource,
(conv_t_5_biasadd_readvariableop_resource
identity��
Concatenated_Inputs/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
Concatenated_Inputs/concat/axis�
Concatenated_Inputs/concatConcatV2inputs_0inputs_1(Concatenated_Inputs/concat/axis:output:0*
N*
T0*1
_output_shapes
:�����������2
Concatenated_Inputs/concat�
Conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv_1/Conv2D/ReadVariableOp�
Conv_1/Conv2DConv2D#Concatenated_Inputs/concat:output:0$Conv_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
2
Conv_1/Conv2D�
Conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
Conv_1/BiasAdd/ReadVariableOp�
Conv_1/BiasAddBiasAddConv_1/Conv2D:output:0%Conv_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� 2
Conv_1/BiasAdd�
Conv_1/leaky_re_lu/LeakyRelu	LeakyReluConv_1/BiasAdd:output:0*1
_output_shapes
:����������� *
alpha%���>2
Conv_1/leaky_re_lu/LeakyRelu�
Max_Pool_1/MaxPoolMaxPool*Conv_1/leaky_re_lu/LeakyRelu:activations:0*/
_output_shapes
:���������`P *
ksize
*
paddingVALID*
strides
2
Max_Pool_1/MaxPool�
Conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv_2/Conv2D/ReadVariableOp�
Conv_2/Conv2DConv2DMax_Pool_1/MaxPool:output:0$Conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������`P@*
paddingSAME*
strides
2
Conv_2/Conv2D�
Conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
Conv_2/BiasAdd/ReadVariableOp�
Conv_2/BiasAddBiasAddConv_2/Conv2D:output:0%Conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������`P@2
Conv_2/BiasAdd�
Conv_2/leaky_re_lu_1/LeakyRelu	LeakyReluConv_2/BiasAdd:output:0*/
_output_shapes
:���������`P@*
alpha%���>2 
Conv_2/leaky_re_lu_1/LeakyRelu�
Max_Pool_2/MaxPoolMaxPool,Conv_2/leaky_re_lu_1/LeakyRelu:activations:0*/
_output_shapes
:���������0(@*
ksize
*
paddingVALID*
strides
2
Max_Pool_2/MaxPool�
SPD_1/IdentityIdentityMax_Pool_2/MaxPool:output:0*
T0*/
_output_shapes
:���������0(@2
SPD_1/Identity�
Conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
Conv_3/Conv2D/ReadVariableOp�
Conv_3/Conv2DConv2DSPD_1/Identity:output:0$Conv_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������0(�*
paddingSAME*
strides
2
Conv_3/Conv2D�
Conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
Conv_3/BiasAdd/ReadVariableOp�
Conv_3/BiasAddBiasAddConv_3/Conv2D:output:0%Conv_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������0(�2
Conv_3/BiasAdd�
Conv_3/leaky_re_lu_2/LeakyRelu	LeakyReluConv_3/BiasAdd:output:0*0
_output_shapes
:���������0(�*
alpha%���>2 
Conv_3/leaky_re_lu_2/LeakyRelu�
Max_Pool_3/MaxPoolMaxPool,Conv_3/leaky_re_lu_2/LeakyRelu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
Max_Pool_3/MaxPool�
Conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv_4/Conv2D/ReadVariableOp�
Conv_4/Conv2DConv2DMax_Pool_3/MaxPool:output:0$Conv_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv_4/Conv2D�
Conv_4/BiasAdd/ReadVariableOpReadVariableOp&conv_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
Conv_4/BiasAdd/ReadVariableOp�
Conv_4/BiasAddBiasAddConv_4/Conv2D:output:0%Conv_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
Conv_4/BiasAdd�
Conv_4/leaky_re_lu_3/LeakyRelu	LeakyReluConv_4/BiasAdd:output:0*0
_output_shapes
:����������*
alpha%���>2 
Conv_4/leaky_re_lu_3/LeakyRelu�
Max_Pool_4/MaxPoolMaxPool,Conv_4/leaky_re_lu_3/LeakyRelu:activations:0*0
_output_shapes
:���������
�*
ksize
*
paddingVALID*
strides
2
Max_Pool_4/MaxPool�
SPD_2/IdentityIdentityMax_Pool_4/MaxPool:output:0*
T0*0
_output_shapes
:���������
�2
SPD_2/Identity�
Conv_5/Conv2D/ReadVariableOpReadVariableOp%conv_5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv_5/Conv2D/ReadVariableOp�
Conv_5/Conv2DConv2DSPD_2/Identity:output:0$Conv_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�*
paddingSAME*
strides
2
Conv_5/Conv2D�
Conv_5/BiasAdd/ReadVariableOpReadVariableOp&conv_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
Conv_5/BiasAdd/ReadVariableOp�
Conv_5/BiasAddBiasAddConv_5/Conv2D:output:0%Conv_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�2
Conv_5/BiasAdd�
Conv_5/leaky_re_lu_4/LeakyRelu	LeakyReluConv_5/BiasAdd:output:0*0
_output_shapes
:���������
�*
alpha%���>2 
Conv_5/leaky_re_lu_4/LeakyRelu�
Max_Pool_5/MaxPoolMaxPool,Conv_5/leaky_re_lu_4/LeakyRelu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
Max_Pool_5/MaxPool�
SPD_3/IdentityIdentityMax_Pool_5/MaxPool:output:0*
T0*0
_output_shapes
:����������2
SPD_3/Identityg
Conv_T_1/ShapeShapeSPD_3/Identity:output:0*
T0*
_output_shapes
:2
Conv_T_1/Shape�
Conv_T_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Conv_T_1/strided_slice/stack�
Conv_T_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
Conv_T_1/strided_slice/stack_1�
Conv_T_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
Conv_T_1/strided_slice/stack_2�
Conv_T_1/strided_sliceStridedSliceConv_T_1/Shape:output:0%Conv_T_1/strided_slice/stack:output:0'Conv_T_1/strided_slice/stack_1:output:0'Conv_T_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Conv_T_1/strided_slicef
Conv_T_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
Conv_T_1/stack/1f
Conv_T_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :
2
Conv_T_1/stack/2g
Conv_T_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :�2
Conv_T_1/stack/3�
Conv_T_1/stackPackConv_T_1/strided_slice:output:0Conv_T_1/stack/1:output:0Conv_T_1/stack/2:output:0Conv_T_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
Conv_T_1/stack�
Conv_T_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
Conv_T_1/strided_slice_1/stack�
 Conv_T_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 Conv_T_1/strided_slice_1/stack_1�
 Conv_T_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 Conv_T_1/strided_slice_1/stack_2�
Conv_T_1/strided_slice_1StridedSliceConv_T_1/stack:output:0'Conv_T_1/strided_slice_1/stack:output:0)Conv_T_1/strided_slice_1/stack_1:output:0)Conv_T_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Conv_T_1/strided_slice_1�
(Conv_T_1/conv2d_transpose/ReadVariableOpReadVariableOp1conv_t_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(Conv_T_1/conv2d_transpose/ReadVariableOp�
Conv_T_1/conv2d_transposeConv2DBackpropInputConv_T_1/stack:output:00Conv_T_1/conv2d_transpose/ReadVariableOp:value:0SPD_3/Identity:output:0*
T0*0
_output_shapes
:���������
�*
paddingSAME*
strides
2
Conv_T_1/conv2d_transpose�
Conv_T_1/BiasAdd/ReadVariableOpReadVariableOp(conv_t_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
Conv_T_1/BiasAdd/ReadVariableOp�
Conv_T_1/BiasAddBiasAdd"Conv_T_1/conv2d_transpose:output:0'Conv_T_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�2
Conv_T_1/BiasAdd�
 Conv_T_1/leaky_re_lu_5/LeakyRelu	LeakyReluConv_T_1/BiasAdd:output:0*0
_output_shapes
:���������
�*
alpha%���>2"
 Conv_T_1/leaky_re_lu_5/LeakyRelun
Concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Concat_1/concat/axis�
Concat_1/concatConcatV2.Conv_T_1/leaky_re_lu_5/LeakyRelu:activations:0Max_Pool_4/MaxPool:output:0Concat_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:���������
�2
Concat_1/concat�
SPD_4/IdentityIdentityConcat_1/concat:output:0*
T0*0
_output_shapes
:���������
�2
SPD_4/Identityg
Conv_T_2/ShapeShapeSPD_4/Identity:output:0*
T0*
_output_shapes
:2
Conv_T_2/Shape�
Conv_T_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Conv_T_2/strided_slice/stack�
Conv_T_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
Conv_T_2/strided_slice/stack_1�
Conv_T_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
Conv_T_2/strided_slice/stack_2�
Conv_T_2/strided_sliceStridedSliceConv_T_2/Shape:output:0%Conv_T_2/strided_slice/stack:output:0'Conv_T_2/strided_slice/stack_1:output:0'Conv_T_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Conv_T_2/strided_slicef
Conv_T_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
Conv_T_2/stack/1f
Conv_T_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
Conv_T_2/stack/2g
Conv_T_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :�2
Conv_T_2/stack/3�
Conv_T_2/stackPackConv_T_2/strided_slice:output:0Conv_T_2/stack/1:output:0Conv_T_2/stack/2:output:0Conv_T_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
Conv_T_2/stack�
Conv_T_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
Conv_T_2/strided_slice_1/stack�
 Conv_T_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 Conv_T_2/strided_slice_1/stack_1�
 Conv_T_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 Conv_T_2/strided_slice_1/stack_2�
Conv_T_2/strided_slice_1StridedSliceConv_T_2/stack:output:0'Conv_T_2/strided_slice_1/stack:output:0)Conv_T_2/strided_slice_1/stack_1:output:0)Conv_T_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Conv_T_2/strided_slice_1�
(Conv_T_2/conv2d_transpose/ReadVariableOpReadVariableOp1conv_t_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype02*
(Conv_T_2/conv2d_transpose/ReadVariableOp�
Conv_T_2/conv2d_transposeConv2DBackpropInputConv_T_2/stack:output:00Conv_T_2/conv2d_transpose/ReadVariableOp:value:0SPD_4/Identity:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv_T_2/conv2d_transpose�
Conv_T_2/BiasAdd/ReadVariableOpReadVariableOp(conv_t_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
Conv_T_2/BiasAdd/ReadVariableOp�
Conv_T_2/BiasAddBiasAdd"Conv_T_2/conv2d_transpose:output:0'Conv_T_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
Conv_T_2/BiasAdd�
 Conv_T_2/leaky_re_lu_6/LeakyRelu	LeakyReluConv_T_2/BiasAdd:output:0*0
_output_shapes
:����������*
alpha%���>2"
 Conv_T_2/leaky_re_lu_6/LeakyRelu~
Conv_T_3/ShapeShape.Conv_T_2/leaky_re_lu_6/LeakyRelu:activations:0*
T0*
_output_shapes
:2
Conv_T_3/Shape�
Conv_T_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Conv_T_3/strided_slice/stack�
Conv_T_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
Conv_T_3/strided_slice/stack_1�
Conv_T_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
Conv_T_3/strided_slice/stack_2�
Conv_T_3/strided_sliceStridedSliceConv_T_3/Shape:output:0%Conv_T_3/strided_slice/stack:output:0'Conv_T_3/strided_slice/stack_1:output:0'Conv_T_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Conv_T_3/strided_slicef
Conv_T_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :02
Conv_T_3/stack/1f
Conv_T_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :(2
Conv_T_3/stack/2f
Conv_T_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
Conv_T_3/stack/3�
Conv_T_3/stackPackConv_T_3/strided_slice:output:0Conv_T_3/stack/1:output:0Conv_T_3/stack/2:output:0Conv_T_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
Conv_T_3/stack�
Conv_T_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
Conv_T_3/strided_slice_1/stack�
 Conv_T_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 Conv_T_3/strided_slice_1/stack_1�
 Conv_T_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 Conv_T_3/strided_slice_1/stack_2�
Conv_T_3/strided_slice_1StridedSliceConv_T_3/stack:output:0'Conv_T_3/strided_slice_1/stack:output:0)Conv_T_3/strided_slice_1/stack_1:output:0)Conv_T_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Conv_T_3/strided_slice_1�
(Conv_T_3/conv2d_transpose/ReadVariableOpReadVariableOp1conv_t_3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype02*
(Conv_T_3/conv2d_transpose/ReadVariableOp�
Conv_T_3/conv2d_transposeConv2DBackpropInputConv_T_3/stack:output:00Conv_T_3/conv2d_transpose/ReadVariableOp:value:0.Conv_T_2/leaky_re_lu_6/LeakyRelu:activations:0*
T0*/
_output_shapes
:���������0(@*
paddingSAME*
strides
2
Conv_T_3/conv2d_transpose�
Conv_T_3/BiasAdd/ReadVariableOpReadVariableOp(conv_t_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
Conv_T_3/BiasAdd/ReadVariableOp�
Conv_T_3/BiasAddBiasAdd"Conv_T_3/conv2d_transpose:output:0'Conv_T_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0(@2
Conv_T_3/BiasAdd�
 Conv_T_3/leaky_re_lu_7/LeakyRelu	LeakyReluConv_T_3/BiasAdd:output:0*/
_output_shapes
:���������0(@*
alpha%���>2"
 Conv_T_3/leaky_re_lu_7/LeakyRelun
Concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Concat_2/concat/axis�
Concat_2/concatConcatV2.Conv_T_3/leaky_re_lu_7/LeakyRelu:activations:0Max_Pool_2/MaxPool:output:0Concat_2/concat/axis:output:0*
N*
T0*0
_output_shapes
:���������0(�2
Concat_2/concat�
SPD_5/IdentityIdentityConcat_2/concat:output:0*
T0*0
_output_shapes
:���������0(�2
SPD_5/Identityg
Conv_T_4/ShapeShapeSPD_5/Identity:output:0*
T0*
_output_shapes
:2
Conv_T_4/Shape�
Conv_T_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Conv_T_4/strided_slice/stack�
Conv_T_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
Conv_T_4/strided_slice/stack_1�
Conv_T_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
Conv_T_4/strided_slice/stack_2�
Conv_T_4/strided_sliceStridedSliceConv_T_4/Shape:output:0%Conv_T_4/strided_slice/stack:output:0'Conv_T_4/strided_slice/stack_1:output:0'Conv_T_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Conv_T_4/strided_slicef
Conv_T_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :`2
Conv_T_4/stack/1f
Conv_T_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :P2
Conv_T_4/stack/2f
Conv_T_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Conv_T_4/stack/3�
Conv_T_4/stackPackConv_T_4/strided_slice:output:0Conv_T_4/stack/1:output:0Conv_T_4/stack/2:output:0Conv_T_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
Conv_T_4/stack�
Conv_T_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
Conv_T_4/strided_slice_1/stack�
 Conv_T_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 Conv_T_4/strided_slice_1/stack_1�
 Conv_T_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 Conv_T_4/strided_slice_1/stack_2�
Conv_T_4/strided_slice_1StridedSliceConv_T_4/stack:output:0'Conv_T_4/strided_slice_1/stack:output:0)Conv_T_4/strided_slice_1/stack_1:output:0)Conv_T_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Conv_T_4/strided_slice_1�
(Conv_T_4/conv2d_transpose/ReadVariableOpReadVariableOp1conv_t_4_conv2d_transpose_readvariableop_resource*'
_output_shapes
: �*
dtype02*
(Conv_T_4/conv2d_transpose/ReadVariableOp�
Conv_T_4/conv2d_transposeConv2DBackpropInputConv_T_4/stack:output:00Conv_T_4/conv2d_transpose/ReadVariableOp:value:0SPD_5/Identity:output:0*
T0*/
_output_shapes
:���������`P *
paddingSAME*
strides
2
Conv_T_4/conv2d_transpose�
Conv_T_4/BiasAdd/ReadVariableOpReadVariableOp(conv_t_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
Conv_T_4/BiasAdd/ReadVariableOp�
Conv_T_4/BiasAddBiasAdd"Conv_T_4/conv2d_transpose:output:0'Conv_T_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������`P 2
Conv_T_4/BiasAdd�
 Conv_T_4/leaky_re_lu_8/LeakyRelu	LeakyReluConv_T_4/BiasAdd:output:0*/
_output_shapes
:���������`P *
alpha%���>2"
 Conv_T_4/leaky_re_lu_8/LeakyRelu~
Conv_T_5/ShapeShape.Conv_T_4/leaky_re_lu_8/LeakyRelu:activations:0*
T0*
_output_shapes
:2
Conv_T_5/Shape�
Conv_T_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Conv_T_5/strided_slice/stack�
Conv_T_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
Conv_T_5/strided_slice/stack_1�
Conv_T_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
Conv_T_5/strided_slice/stack_2�
Conv_T_5/strided_sliceStridedSliceConv_T_5/Shape:output:0%Conv_T_5/strided_slice/stack:output:0'Conv_T_5/strided_slice/stack_1:output:0'Conv_T_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Conv_T_5/strided_sliceg
Conv_T_5/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�2
Conv_T_5/stack/1g
Conv_T_5/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�2
Conv_T_5/stack/2f
Conv_T_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
Conv_T_5/stack/3�
Conv_T_5/stackPackConv_T_5/strided_slice:output:0Conv_T_5/stack/1:output:0Conv_T_5/stack/2:output:0Conv_T_5/stack/3:output:0*
N*
T0*
_output_shapes
:2
Conv_T_5/stack�
Conv_T_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
Conv_T_5/strided_slice_1/stack�
 Conv_T_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 Conv_T_5/strided_slice_1/stack_1�
 Conv_T_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 Conv_T_5/strided_slice_1/stack_2�
Conv_T_5/strided_slice_1StridedSliceConv_T_5/stack:output:0'Conv_T_5/strided_slice_1/stack:output:0)Conv_T_5/strided_slice_1/stack_1:output:0)Conv_T_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Conv_T_5/strided_slice_1�
(Conv_T_5/conv2d_transpose/ReadVariableOpReadVariableOp1conv_t_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02*
(Conv_T_5/conv2d_transpose/ReadVariableOp�
Conv_T_5/conv2d_transposeConv2DBackpropInputConv_T_5/stack:output:00Conv_T_5/conv2d_transpose/ReadVariableOp:value:0.Conv_T_4/leaky_re_lu_8/LeakyRelu:activations:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
Conv_T_5/conv2d_transpose�
Conv_T_5/BiasAdd/ReadVariableOpReadVariableOp(conv_t_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
Conv_T_5/BiasAdd/ReadVariableOp�
Conv_T_5/BiasAddBiasAdd"Conv_T_5/conv2d_transpose:output:0'Conv_T_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
Conv_T_5/BiasAdd�
Conv_T_5/SigmoidSigmoidConv_T_5/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2
Conv_T_5/Sigmoidr
IdentityIdentityConv_T_5/Sigmoid:y:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������:�����������:::::::::::::::::::::[ W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/1
�
{
&__inference_Conv_1_layer_call_fn_29492

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_Conv_1_layer_call_and_return_conditional_losses_282302
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:����������� 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
^
%__inference_SPD_5_layer_call_fn_29935

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������0(�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_5_layer_call_and_return_conditional_losses_285842
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:���������0(�2

Identity"
identityIdentity:output:0*/
_input_shapes
:���������0(�22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������0(�
 
_user_specified_nameinputs
�
^
@__inference_SPD_2_layer_call_and_return_conditional_losses_29656

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4������������������������������������2

Identity�

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
^
@__inference_SPD_4_layer_call_and_return_conditional_losses_27911

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4������������������������������������2

Identity�

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
A
%__inference_SPD_5_layer_call_fn_29978

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_5_layer_call_and_return_conditional_losses_280932
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
_
@__inference_SPD_2_layer_call_and_return_conditional_losses_29651

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const�
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul_1�
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
T
(__inference_Concat_2_layer_call_fn_29902
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������0(�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Concat_2_layer_call_and_return_conditional_losses_285532
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:���������0(�2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+���������������������������@:���������0(@:k g
A
_output_shapes/
-:+���������������������������@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������0(@
"
_user_specified_name
inputs/1
�%
�
C__inference_Conv_T_3_layer_call_and_return_conditional_losses_28018

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAdd�
leaky_re_lu_7/PartitionedCallPartitionedCallBiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_280092
leaky_re_lu_7/PartitionedCall�
IdentityIdentity&leaky_re_lu_7/PartitionedCall:output:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������:::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
^
@__inference_SPD_4_layer_call_and_return_conditional_losses_29841

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4������������������������������������2

Identity�

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
F
*__inference_Max_Pool_4_layer_call_fn_27641

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_Max_Pool_4_layer_call_and_return_conditional_losses_276352
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
{
&__inference_Conv_5_layer_call_fn_29724

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_Conv_5_layer_call_and_return_conditional_losses_284202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:���������
�2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:���������
�::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������
�
 
_user_specified_nameinputs
�	
�
A__inference_Conv_4_layer_call_and_return_conditional_losses_28353

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdd�
leaky_re_lu_3/LeakyRelu	LeakyReluBiasAdd:output:0*0
_output_shapes
:����������*
alpha%���>2
leaky_re_lu_3/LeakyRelu�
IdentityIdentity%leaky_re_lu_3/LeakyRelu:activations:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������:::X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
A__inference_Conv_1_layer_call_and_return_conditional_losses_28230

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� 2	
BiasAdd�
leaky_re_lu/LeakyRelu	LeakyReluBiasAdd:output:0*1
_output_shapes
:����������� *
alpha%���>2
leaky_re_lu/LeakyRelu�
IdentityIdentity#leaky_re_lu/LeakyRelu:activations:0*
T0*1
_output_shapes
:����������� 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������:::Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
I
-__inference_leaky_re_lu_8_layer_call_fn_30018

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_281342
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+��������������������������� :i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
^
@__inference_SPD_1_layer_call_and_return_conditional_losses_29578

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4������������������������������������2

Identity�

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
A
%__inference_SPD_2_layer_call_fn_29704

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_2_layer_call_and_return_conditional_losses_283972
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:���������
�2

Identity"
identityIdentity:output:0*/
_input_shapes
:���������
�:X T
0
_output_shapes
:���������
�
 
_user_specified_nameinputs
�
A
%__inference_SPD_3_layer_call_fn_29762

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_3_layer_call_and_return_conditional_losses_284642
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
A
%__inference_SPD_5_layer_call_fn_29940

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������0(�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_5_layer_call_and_return_conditional_losses_285892
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:���������0(�2

Identity"
identityIdentity:output:0*/
_input_shapes
:���������0(�:X T
0
_output_shapes
:���������0(�
 
_user_specified_nameinputs
�
d
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_27952

inputs
identity
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,����������������������������*
alpha%���>2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
^
%__inference_SPD_1_layer_call_fn_29545

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0(@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_1_layer_call_and_return_conditional_losses_282972
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������0(@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������0(@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������0(@
 
_user_specified_nameinputs
�
_
@__inference_SPD_4_layer_call_and_return_conditional_losses_27901

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const�
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul_1�
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
I
-__inference_leaky_re_lu_7_layer_call_fn_30008

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_280092
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������@:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
��
�
 __inference__wrapped_model_27525
img_input_layer
mask_input_layerD
@facial_landmark_completion_conv_1_conv2d_readvariableop_resourceE
Afacial_landmark_completion_conv_1_biasadd_readvariableop_resourceD
@facial_landmark_completion_conv_2_conv2d_readvariableop_resourceE
Afacial_landmark_completion_conv_2_biasadd_readvariableop_resourceD
@facial_landmark_completion_conv_3_conv2d_readvariableop_resourceE
Afacial_landmark_completion_conv_3_biasadd_readvariableop_resourceD
@facial_landmark_completion_conv_4_conv2d_readvariableop_resourceE
Afacial_landmark_completion_conv_4_biasadd_readvariableop_resourceD
@facial_landmark_completion_conv_5_conv2d_readvariableop_resourceE
Afacial_landmark_completion_conv_5_biasadd_readvariableop_resourceP
Lfacial_landmark_completion_conv_t_1_conv2d_transpose_readvariableop_resourceG
Cfacial_landmark_completion_conv_t_1_biasadd_readvariableop_resourceP
Lfacial_landmark_completion_conv_t_2_conv2d_transpose_readvariableop_resourceG
Cfacial_landmark_completion_conv_t_2_biasadd_readvariableop_resourceP
Lfacial_landmark_completion_conv_t_3_conv2d_transpose_readvariableop_resourceG
Cfacial_landmark_completion_conv_t_3_biasadd_readvariableop_resourceP
Lfacial_landmark_completion_conv_t_4_conv2d_transpose_readvariableop_resourceG
Cfacial_landmark_completion_conv_t_4_biasadd_readvariableop_resourceP
Lfacial_landmark_completion_conv_t_5_conv2d_transpose_readvariableop_resourceG
Cfacial_landmark_completion_conv_t_5_biasadd_readvariableop_resource
identity��
:Facial_Landmark_Completion/Concatenated_Inputs/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2<
:Facial_Landmark_Completion/Concatenated_Inputs/concat/axis�
5Facial_Landmark_Completion/Concatenated_Inputs/concatConcatV2img_input_layermask_input_layerCFacial_Landmark_Completion/Concatenated_Inputs/concat/axis:output:0*
N*
T0*1
_output_shapes
:�����������27
5Facial_Landmark_Completion/Concatenated_Inputs/concat�
7Facial_Landmark_Completion/Conv_1/Conv2D/ReadVariableOpReadVariableOp@facial_landmark_completion_conv_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype029
7Facial_Landmark_Completion/Conv_1/Conv2D/ReadVariableOp�
(Facial_Landmark_Completion/Conv_1/Conv2DConv2D>Facial_Landmark_Completion/Concatenated_Inputs/concat:output:0?Facial_Landmark_Completion/Conv_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
2*
(Facial_Landmark_Completion/Conv_1/Conv2D�
8Facial_Landmark_Completion/Conv_1/BiasAdd/ReadVariableOpReadVariableOpAfacial_landmark_completion_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02:
8Facial_Landmark_Completion/Conv_1/BiasAdd/ReadVariableOp�
)Facial_Landmark_Completion/Conv_1/BiasAddBiasAdd1Facial_Landmark_Completion/Conv_1/Conv2D:output:0@Facial_Landmark_Completion/Conv_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� 2+
)Facial_Landmark_Completion/Conv_1/BiasAdd�
7Facial_Landmark_Completion/Conv_1/leaky_re_lu/LeakyRelu	LeakyRelu2Facial_Landmark_Completion/Conv_1/BiasAdd:output:0*1
_output_shapes
:����������� *
alpha%���>29
7Facial_Landmark_Completion/Conv_1/leaky_re_lu/LeakyRelu�
-Facial_Landmark_Completion/Max_Pool_1/MaxPoolMaxPoolEFacial_Landmark_Completion/Conv_1/leaky_re_lu/LeakyRelu:activations:0*/
_output_shapes
:���������`P *
ksize
*
paddingVALID*
strides
2/
-Facial_Landmark_Completion/Max_Pool_1/MaxPool�
7Facial_Landmark_Completion/Conv_2/Conv2D/ReadVariableOpReadVariableOp@facial_landmark_completion_conv_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype029
7Facial_Landmark_Completion/Conv_2/Conv2D/ReadVariableOp�
(Facial_Landmark_Completion/Conv_2/Conv2DConv2D6Facial_Landmark_Completion/Max_Pool_1/MaxPool:output:0?Facial_Landmark_Completion/Conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������`P@*
paddingSAME*
strides
2*
(Facial_Landmark_Completion/Conv_2/Conv2D�
8Facial_Landmark_Completion/Conv_2/BiasAdd/ReadVariableOpReadVariableOpAfacial_landmark_completion_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8Facial_Landmark_Completion/Conv_2/BiasAdd/ReadVariableOp�
)Facial_Landmark_Completion/Conv_2/BiasAddBiasAdd1Facial_Landmark_Completion/Conv_2/Conv2D:output:0@Facial_Landmark_Completion/Conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������`P@2+
)Facial_Landmark_Completion/Conv_2/BiasAdd�
9Facial_Landmark_Completion/Conv_2/leaky_re_lu_1/LeakyRelu	LeakyRelu2Facial_Landmark_Completion/Conv_2/BiasAdd:output:0*/
_output_shapes
:���������`P@*
alpha%���>2;
9Facial_Landmark_Completion/Conv_2/leaky_re_lu_1/LeakyRelu�
-Facial_Landmark_Completion/Max_Pool_2/MaxPoolMaxPoolGFacial_Landmark_Completion/Conv_2/leaky_re_lu_1/LeakyRelu:activations:0*/
_output_shapes
:���������0(@*
ksize
*
paddingVALID*
strides
2/
-Facial_Landmark_Completion/Max_Pool_2/MaxPool�
)Facial_Landmark_Completion/SPD_1/IdentityIdentity6Facial_Landmark_Completion/Max_Pool_2/MaxPool:output:0*
T0*/
_output_shapes
:���������0(@2+
)Facial_Landmark_Completion/SPD_1/Identity�
7Facial_Landmark_Completion/Conv_3/Conv2D/ReadVariableOpReadVariableOp@facial_landmark_completion_conv_3_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype029
7Facial_Landmark_Completion/Conv_3/Conv2D/ReadVariableOp�
(Facial_Landmark_Completion/Conv_3/Conv2DConv2D2Facial_Landmark_Completion/SPD_1/Identity:output:0?Facial_Landmark_Completion/Conv_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������0(�*
paddingSAME*
strides
2*
(Facial_Landmark_Completion/Conv_3/Conv2D�
8Facial_Landmark_Completion/Conv_3/BiasAdd/ReadVariableOpReadVariableOpAfacial_landmark_completion_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8Facial_Landmark_Completion/Conv_3/BiasAdd/ReadVariableOp�
)Facial_Landmark_Completion/Conv_3/BiasAddBiasAdd1Facial_Landmark_Completion/Conv_3/Conv2D:output:0@Facial_Landmark_Completion/Conv_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������0(�2+
)Facial_Landmark_Completion/Conv_3/BiasAdd�
9Facial_Landmark_Completion/Conv_3/leaky_re_lu_2/LeakyRelu	LeakyRelu2Facial_Landmark_Completion/Conv_3/BiasAdd:output:0*0
_output_shapes
:���������0(�*
alpha%���>2;
9Facial_Landmark_Completion/Conv_3/leaky_re_lu_2/LeakyRelu�
-Facial_Landmark_Completion/Max_Pool_3/MaxPoolMaxPoolGFacial_Landmark_Completion/Conv_3/leaky_re_lu_2/LeakyRelu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2/
-Facial_Landmark_Completion/Max_Pool_3/MaxPool�
7Facial_Landmark_Completion/Conv_4/Conv2D/ReadVariableOpReadVariableOp@facial_landmark_completion_conv_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype029
7Facial_Landmark_Completion/Conv_4/Conv2D/ReadVariableOp�
(Facial_Landmark_Completion/Conv_4/Conv2DConv2D6Facial_Landmark_Completion/Max_Pool_3/MaxPool:output:0?Facial_Landmark_Completion/Conv_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2*
(Facial_Landmark_Completion/Conv_4/Conv2D�
8Facial_Landmark_Completion/Conv_4/BiasAdd/ReadVariableOpReadVariableOpAfacial_landmark_completion_conv_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8Facial_Landmark_Completion/Conv_4/BiasAdd/ReadVariableOp�
)Facial_Landmark_Completion/Conv_4/BiasAddBiasAdd1Facial_Landmark_Completion/Conv_4/Conv2D:output:0@Facial_Landmark_Completion/Conv_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2+
)Facial_Landmark_Completion/Conv_4/BiasAdd�
9Facial_Landmark_Completion/Conv_4/leaky_re_lu_3/LeakyRelu	LeakyRelu2Facial_Landmark_Completion/Conv_4/BiasAdd:output:0*0
_output_shapes
:����������*
alpha%���>2;
9Facial_Landmark_Completion/Conv_4/leaky_re_lu_3/LeakyRelu�
-Facial_Landmark_Completion/Max_Pool_4/MaxPoolMaxPoolGFacial_Landmark_Completion/Conv_4/leaky_re_lu_3/LeakyRelu:activations:0*0
_output_shapes
:���������
�*
ksize
*
paddingVALID*
strides
2/
-Facial_Landmark_Completion/Max_Pool_4/MaxPool�
)Facial_Landmark_Completion/SPD_2/IdentityIdentity6Facial_Landmark_Completion/Max_Pool_4/MaxPool:output:0*
T0*0
_output_shapes
:���������
�2+
)Facial_Landmark_Completion/SPD_2/Identity�
7Facial_Landmark_Completion/Conv_5/Conv2D/ReadVariableOpReadVariableOp@facial_landmark_completion_conv_5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype029
7Facial_Landmark_Completion/Conv_5/Conv2D/ReadVariableOp�
(Facial_Landmark_Completion/Conv_5/Conv2DConv2D2Facial_Landmark_Completion/SPD_2/Identity:output:0?Facial_Landmark_Completion/Conv_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�*
paddingSAME*
strides
2*
(Facial_Landmark_Completion/Conv_5/Conv2D�
8Facial_Landmark_Completion/Conv_5/BiasAdd/ReadVariableOpReadVariableOpAfacial_landmark_completion_conv_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8Facial_Landmark_Completion/Conv_5/BiasAdd/ReadVariableOp�
)Facial_Landmark_Completion/Conv_5/BiasAddBiasAdd1Facial_Landmark_Completion/Conv_5/Conv2D:output:0@Facial_Landmark_Completion/Conv_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�2+
)Facial_Landmark_Completion/Conv_5/BiasAdd�
9Facial_Landmark_Completion/Conv_5/leaky_re_lu_4/LeakyRelu	LeakyRelu2Facial_Landmark_Completion/Conv_5/BiasAdd:output:0*0
_output_shapes
:���������
�*
alpha%���>2;
9Facial_Landmark_Completion/Conv_5/leaky_re_lu_4/LeakyRelu�
-Facial_Landmark_Completion/Max_Pool_5/MaxPoolMaxPoolGFacial_Landmark_Completion/Conv_5/leaky_re_lu_4/LeakyRelu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2/
-Facial_Landmark_Completion/Max_Pool_5/MaxPool�
)Facial_Landmark_Completion/SPD_3/IdentityIdentity6Facial_Landmark_Completion/Max_Pool_5/MaxPool:output:0*
T0*0
_output_shapes
:����������2+
)Facial_Landmark_Completion/SPD_3/Identity�
)Facial_Landmark_Completion/Conv_T_1/ShapeShape2Facial_Landmark_Completion/SPD_3/Identity:output:0*
T0*
_output_shapes
:2+
)Facial_Landmark_Completion/Conv_T_1/Shape�
7Facial_Landmark_Completion/Conv_T_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7Facial_Landmark_Completion/Conv_T_1/strided_slice/stack�
9Facial_Landmark_Completion/Conv_T_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9Facial_Landmark_Completion/Conv_T_1/strided_slice/stack_1�
9Facial_Landmark_Completion/Conv_T_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9Facial_Landmark_Completion/Conv_T_1/strided_slice/stack_2�
1Facial_Landmark_Completion/Conv_T_1/strided_sliceStridedSlice2Facial_Landmark_Completion/Conv_T_1/Shape:output:0@Facial_Landmark_Completion/Conv_T_1/strided_slice/stack:output:0BFacial_Landmark_Completion/Conv_T_1/strided_slice/stack_1:output:0BFacial_Landmark_Completion/Conv_T_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1Facial_Landmark_Completion/Conv_T_1/strided_slice�
+Facial_Landmark_Completion/Conv_T_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+Facial_Landmark_Completion/Conv_T_1/stack/1�
+Facial_Landmark_Completion/Conv_T_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :
2-
+Facial_Landmark_Completion/Conv_T_1/stack/2�
+Facial_Landmark_Completion/Conv_T_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :�2-
+Facial_Landmark_Completion/Conv_T_1/stack/3�
)Facial_Landmark_Completion/Conv_T_1/stackPack:Facial_Landmark_Completion/Conv_T_1/strided_slice:output:04Facial_Landmark_Completion/Conv_T_1/stack/1:output:04Facial_Landmark_Completion/Conv_T_1/stack/2:output:04Facial_Landmark_Completion/Conv_T_1/stack/3:output:0*
N*
T0*
_output_shapes
:2+
)Facial_Landmark_Completion/Conv_T_1/stack�
9Facial_Landmark_Completion/Conv_T_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9Facial_Landmark_Completion/Conv_T_1/strided_slice_1/stack�
;Facial_Landmark_Completion/Conv_T_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;Facial_Landmark_Completion/Conv_T_1/strided_slice_1/stack_1�
;Facial_Landmark_Completion/Conv_T_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;Facial_Landmark_Completion/Conv_T_1/strided_slice_1/stack_2�
3Facial_Landmark_Completion/Conv_T_1/strided_slice_1StridedSlice2Facial_Landmark_Completion/Conv_T_1/stack:output:0BFacial_Landmark_Completion/Conv_T_1/strided_slice_1/stack:output:0DFacial_Landmark_Completion/Conv_T_1/strided_slice_1/stack_1:output:0DFacial_Landmark_Completion/Conv_T_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3Facial_Landmark_Completion/Conv_T_1/strided_slice_1�
CFacial_Landmark_Completion/Conv_T_1/conv2d_transpose/ReadVariableOpReadVariableOpLfacial_landmark_completion_conv_t_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype02E
CFacial_Landmark_Completion/Conv_T_1/conv2d_transpose/ReadVariableOp�
4Facial_Landmark_Completion/Conv_T_1/conv2d_transposeConv2DBackpropInput2Facial_Landmark_Completion/Conv_T_1/stack:output:0KFacial_Landmark_Completion/Conv_T_1/conv2d_transpose/ReadVariableOp:value:02Facial_Landmark_Completion/SPD_3/Identity:output:0*
T0*0
_output_shapes
:���������
�*
paddingSAME*
strides
26
4Facial_Landmark_Completion/Conv_T_1/conv2d_transpose�
:Facial_Landmark_Completion/Conv_T_1/BiasAdd/ReadVariableOpReadVariableOpCfacial_landmark_completion_conv_t_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02<
:Facial_Landmark_Completion/Conv_T_1/BiasAdd/ReadVariableOp�
+Facial_Landmark_Completion/Conv_T_1/BiasAddBiasAdd=Facial_Landmark_Completion/Conv_T_1/conv2d_transpose:output:0BFacial_Landmark_Completion/Conv_T_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�2-
+Facial_Landmark_Completion/Conv_T_1/BiasAdd�
;Facial_Landmark_Completion/Conv_T_1/leaky_re_lu_5/LeakyRelu	LeakyRelu4Facial_Landmark_Completion/Conv_T_1/BiasAdd:output:0*0
_output_shapes
:���������
�*
alpha%���>2=
;Facial_Landmark_Completion/Conv_T_1/leaky_re_lu_5/LeakyRelu�
/Facial_Landmark_Completion/Concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :21
/Facial_Landmark_Completion/Concat_1/concat/axis�
*Facial_Landmark_Completion/Concat_1/concatConcatV2IFacial_Landmark_Completion/Conv_T_1/leaky_re_lu_5/LeakyRelu:activations:06Facial_Landmark_Completion/Max_Pool_4/MaxPool:output:08Facial_Landmark_Completion/Concat_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:���������
�2,
*Facial_Landmark_Completion/Concat_1/concat�
)Facial_Landmark_Completion/SPD_4/IdentityIdentity3Facial_Landmark_Completion/Concat_1/concat:output:0*
T0*0
_output_shapes
:���������
�2+
)Facial_Landmark_Completion/SPD_4/Identity�
)Facial_Landmark_Completion/Conv_T_2/ShapeShape2Facial_Landmark_Completion/SPD_4/Identity:output:0*
T0*
_output_shapes
:2+
)Facial_Landmark_Completion/Conv_T_2/Shape�
7Facial_Landmark_Completion/Conv_T_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7Facial_Landmark_Completion/Conv_T_2/strided_slice/stack�
9Facial_Landmark_Completion/Conv_T_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9Facial_Landmark_Completion/Conv_T_2/strided_slice/stack_1�
9Facial_Landmark_Completion/Conv_T_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9Facial_Landmark_Completion/Conv_T_2/strided_slice/stack_2�
1Facial_Landmark_Completion/Conv_T_2/strided_sliceStridedSlice2Facial_Landmark_Completion/Conv_T_2/Shape:output:0@Facial_Landmark_Completion/Conv_T_2/strided_slice/stack:output:0BFacial_Landmark_Completion/Conv_T_2/strided_slice/stack_1:output:0BFacial_Landmark_Completion/Conv_T_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1Facial_Landmark_Completion/Conv_T_2/strided_slice�
+Facial_Landmark_Completion/Conv_T_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+Facial_Landmark_Completion/Conv_T_2/stack/1�
+Facial_Landmark_Completion/Conv_T_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+Facial_Landmark_Completion/Conv_T_2/stack/2�
+Facial_Landmark_Completion/Conv_T_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :�2-
+Facial_Landmark_Completion/Conv_T_2/stack/3�
)Facial_Landmark_Completion/Conv_T_2/stackPack:Facial_Landmark_Completion/Conv_T_2/strided_slice:output:04Facial_Landmark_Completion/Conv_T_2/stack/1:output:04Facial_Landmark_Completion/Conv_T_2/stack/2:output:04Facial_Landmark_Completion/Conv_T_2/stack/3:output:0*
N*
T0*
_output_shapes
:2+
)Facial_Landmark_Completion/Conv_T_2/stack�
9Facial_Landmark_Completion/Conv_T_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9Facial_Landmark_Completion/Conv_T_2/strided_slice_1/stack�
;Facial_Landmark_Completion/Conv_T_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;Facial_Landmark_Completion/Conv_T_2/strided_slice_1/stack_1�
;Facial_Landmark_Completion/Conv_T_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;Facial_Landmark_Completion/Conv_T_2/strided_slice_1/stack_2�
3Facial_Landmark_Completion/Conv_T_2/strided_slice_1StridedSlice2Facial_Landmark_Completion/Conv_T_2/stack:output:0BFacial_Landmark_Completion/Conv_T_2/strided_slice_1/stack:output:0DFacial_Landmark_Completion/Conv_T_2/strided_slice_1/stack_1:output:0DFacial_Landmark_Completion/Conv_T_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3Facial_Landmark_Completion/Conv_T_2/strided_slice_1�
CFacial_Landmark_Completion/Conv_T_2/conv2d_transpose/ReadVariableOpReadVariableOpLfacial_landmark_completion_conv_t_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype02E
CFacial_Landmark_Completion/Conv_T_2/conv2d_transpose/ReadVariableOp�
4Facial_Landmark_Completion/Conv_T_2/conv2d_transposeConv2DBackpropInput2Facial_Landmark_Completion/Conv_T_2/stack:output:0KFacial_Landmark_Completion/Conv_T_2/conv2d_transpose/ReadVariableOp:value:02Facial_Landmark_Completion/SPD_4/Identity:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
26
4Facial_Landmark_Completion/Conv_T_2/conv2d_transpose�
:Facial_Landmark_Completion/Conv_T_2/BiasAdd/ReadVariableOpReadVariableOpCfacial_landmark_completion_conv_t_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02<
:Facial_Landmark_Completion/Conv_T_2/BiasAdd/ReadVariableOp�
+Facial_Landmark_Completion/Conv_T_2/BiasAddBiasAdd=Facial_Landmark_Completion/Conv_T_2/conv2d_transpose:output:0BFacial_Landmark_Completion/Conv_T_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2-
+Facial_Landmark_Completion/Conv_T_2/BiasAdd�
;Facial_Landmark_Completion/Conv_T_2/leaky_re_lu_6/LeakyRelu	LeakyRelu4Facial_Landmark_Completion/Conv_T_2/BiasAdd:output:0*0
_output_shapes
:����������*
alpha%���>2=
;Facial_Landmark_Completion/Conv_T_2/leaky_re_lu_6/LeakyRelu�
)Facial_Landmark_Completion/Conv_T_3/ShapeShapeIFacial_Landmark_Completion/Conv_T_2/leaky_re_lu_6/LeakyRelu:activations:0*
T0*
_output_shapes
:2+
)Facial_Landmark_Completion/Conv_T_3/Shape�
7Facial_Landmark_Completion/Conv_T_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7Facial_Landmark_Completion/Conv_T_3/strided_slice/stack�
9Facial_Landmark_Completion/Conv_T_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9Facial_Landmark_Completion/Conv_T_3/strided_slice/stack_1�
9Facial_Landmark_Completion/Conv_T_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9Facial_Landmark_Completion/Conv_T_3/strided_slice/stack_2�
1Facial_Landmark_Completion/Conv_T_3/strided_sliceStridedSlice2Facial_Landmark_Completion/Conv_T_3/Shape:output:0@Facial_Landmark_Completion/Conv_T_3/strided_slice/stack:output:0BFacial_Landmark_Completion/Conv_T_3/strided_slice/stack_1:output:0BFacial_Landmark_Completion/Conv_T_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1Facial_Landmark_Completion/Conv_T_3/strided_slice�
+Facial_Landmark_Completion/Conv_T_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :02-
+Facial_Landmark_Completion/Conv_T_3/stack/1�
+Facial_Landmark_Completion/Conv_T_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :(2-
+Facial_Landmark_Completion/Conv_T_3/stack/2�
+Facial_Landmark_Completion/Conv_T_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2-
+Facial_Landmark_Completion/Conv_T_3/stack/3�
)Facial_Landmark_Completion/Conv_T_3/stackPack:Facial_Landmark_Completion/Conv_T_3/strided_slice:output:04Facial_Landmark_Completion/Conv_T_3/stack/1:output:04Facial_Landmark_Completion/Conv_T_3/stack/2:output:04Facial_Landmark_Completion/Conv_T_3/stack/3:output:0*
N*
T0*
_output_shapes
:2+
)Facial_Landmark_Completion/Conv_T_3/stack�
9Facial_Landmark_Completion/Conv_T_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9Facial_Landmark_Completion/Conv_T_3/strided_slice_1/stack�
;Facial_Landmark_Completion/Conv_T_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;Facial_Landmark_Completion/Conv_T_3/strided_slice_1/stack_1�
;Facial_Landmark_Completion/Conv_T_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;Facial_Landmark_Completion/Conv_T_3/strided_slice_1/stack_2�
3Facial_Landmark_Completion/Conv_T_3/strided_slice_1StridedSlice2Facial_Landmark_Completion/Conv_T_3/stack:output:0BFacial_Landmark_Completion/Conv_T_3/strided_slice_1/stack:output:0DFacial_Landmark_Completion/Conv_T_3/strided_slice_1/stack_1:output:0DFacial_Landmark_Completion/Conv_T_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3Facial_Landmark_Completion/Conv_T_3/strided_slice_1�
CFacial_Landmark_Completion/Conv_T_3/conv2d_transpose/ReadVariableOpReadVariableOpLfacial_landmark_completion_conv_t_3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype02E
CFacial_Landmark_Completion/Conv_T_3/conv2d_transpose/ReadVariableOp�
4Facial_Landmark_Completion/Conv_T_3/conv2d_transposeConv2DBackpropInput2Facial_Landmark_Completion/Conv_T_3/stack:output:0KFacial_Landmark_Completion/Conv_T_3/conv2d_transpose/ReadVariableOp:value:0IFacial_Landmark_Completion/Conv_T_2/leaky_re_lu_6/LeakyRelu:activations:0*
T0*/
_output_shapes
:���������0(@*
paddingSAME*
strides
26
4Facial_Landmark_Completion/Conv_T_3/conv2d_transpose�
:Facial_Landmark_Completion/Conv_T_3/BiasAdd/ReadVariableOpReadVariableOpCfacial_landmark_completion_conv_t_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:Facial_Landmark_Completion/Conv_T_3/BiasAdd/ReadVariableOp�
+Facial_Landmark_Completion/Conv_T_3/BiasAddBiasAdd=Facial_Landmark_Completion/Conv_T_3/conv2d_transpose:output:0BFacial_Landmark_Completion/Conv_T_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0(@2-
+Facial_Landmark_Completion/Conv_T_3/BiasAdd�
;Facial_Landmark_Completion/Conv_T_3/leaky_re_lu_7/LeakyRelu	LeakyRelu4Facial_Landmark_Completion/Conv_T_3/BiasAdd:output:0*/
_output_shapes
:���������0(@*
alpha%���>2=
;Facial_Landmark_Completion/Conv_T_3/leaky_re_lu_7/LeakyRelu�
/Facial_Landmark_Completion/Concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :21
/Facial_Landmark_Completion/Concat_2/concat/axis�
*Facial_Landmark_Completion/Concat_2/concatConcatV2IFacial_Landmark_Completion/Conv_T_3/leaky_re_lu_7/LeakyRelu:activations:06Facial_Landmark_Completion/Max_Pool_2/MaxPool:output:08Facial_Landmark_Completion/Concat_2/concat/axis:output:0*
N*
T0*0
_output_shapes
:���������0(�2,
*Facial_Landmark_Completion/Concat_2/concat�
)Facial_Landmark_Completion/SPD_5/IdentityIdentity3Facial_Landmark_Completion/Concat_2/concat:output:0*
T0*0
_output_shapes
:���������0(�2+
)Facial_Landmark_Completion/SPD_5/Identity�
)Facial_Landmark_Completion/Conv_T_4/ShapeShape2Facial_Landmark_Completion/SPD_5/Identity:output:0*
T0*
_output_shapes
:2+
)Facial_Landmark_Completion/Conv_T_4/Shape�
7Facial_Landmark_Completion/Conv_T_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7Facial_Landmark_Completion/Conv_T_4/strided_slice/stack�
9Facial_Landmark_Completion/Conv_T_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9Facial_Landmark_Completion/Conv_T_4/strided_slice/stack_1�
9Facial_Landmark_Completion/Conv_T_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9Facial_Landmark_Completion/Conv_T_4/strided_slice/stack_2�
1Facial_Landmark_Completion/Conv_T_4/strided_sliceStridedSlice2Facial_Landmark_Completion/Conv_T_4/Shape:output:0@Facial_Landmark_Completion/Conv_T_4/strided_slice/stack:output:0BFacial_Landmark_Completion/Conv_T_4/strided_slice/stack_1:output:0BFacial_Landmark_Completion/Conv_T_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1Facial_Landmark_Completion/Conv_T_4/strided_slice�
+Facial_Landmark_Completion/Conv_T_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :`2-
+Facial_Landmark_Completion/Conv_T_4/stack/1�
+Facial_Landmark_Completion/Conv_T_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :P2-
+Facial_Landmark_Completion/Conv_T_4/stack/2�
+Facial_Landmark_Completion/Conv_T_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2-
+Facial_Landmark_Completion/Conv_T_4/stack/3�
)Facial_Landmark_Completion/Conv_T_4/stackPack:Facial_Landmark_Completion/Conv_T_4/strided_slice:output:04Facial_Landmark_Completion/Conv_T_4/stack/1:output:04Facial_Landmark_Completion/Conv_T_4/stack/2:output:04Facial_Landmark_Completion/Conv_T_4/stack/3:output:0*
N*
T0*
_output_shapes
:2+
)Facial_Landmark_Completion/Conv_T_4/stack�
9Facial_Landmark_Completion/Conv_T_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9Facial_Landmark_Completion/Conv_T_4/strided_slice_1/stack�
;Facial_Landmark_Completion/Conv_T_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;Facial_Landmark_Completion/Conv_T_4/strided_slice_1/stack_1�
;Facial_Landmark_Completion/Conv_T_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;Facial_Landmark_Completion/Conv_T_4/strided_slice_1/stack_2�
3Facial_Landmark_Completion/Conv_T_4/strided_slice_1StridedSlice2Facial_Landmark_Completion/Conv_T_4/stack:output:0BFacial_Landmark_Completion/Conv_T_4/strided_slice_1/stack:output:0DFacial_Landmark_Completion/Conv_T_4/strided_slice_1/stack_1:output:0DFacial_Landmark_Completion/Conv_T_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3Facial_Landmark_Completion/Conv_T_4/strided_slice_1�
CFacial_Landmark_Completion/Conv_T_4/conv2d_transpose/ReadVariableOpReadVariableOpLfacial_landmark_completion_conv_t_4_conv2d_transpose_readvariableop_resource*'
_output_shapes
: �*
dtype02E
CFacial_Landmark_Completion/Conv_T_4/conv2d_transpose/ReadVariableOp�
4Facial_Landmark_Completion/Conv_T_4/conv2d_transposeConv2DBackpropInput2Facial_Landmark_Completion/Conv_T_4/stack:output:0KFacial_Landmark_Completion/Conv_T_4/conv2d_transpose/ReadVariableOp:value:02Facial_Landmark_Completion/SPD_5/Identity:output:0*
T0*/
_output_shapes
:���������`P *
paddingSAME*
strides
26
4Facial_Landmark_Completion/Conv_T_4/conv2d_transpose�
:Facial_Landmark_Completion/Conv_T_4/BiasAdd/ReadVariableOpReadVariableOpCfacial_landmark_completion_conv_t_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02<
:Facial_Landmark_Completion/Conv_T_4/BiasAdd/ReadVariableOp�
+Facial_Landmark_Completion/Conv_T_4/BiasAddBiasAdd=Facial_Landmark_Completion/Conv_T_4/conv2d_transpose:output:0BFacial_Landmark_Completion/Conv_T_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������`P 2-
+Facial_Landmark_Completion/Conv_T_4/BiasAdd�
;Facial_Landmark_Completion/Conv_T_4/leaky_re_lu_8/LeakyRelu	LeakyRelu4Facial_Landmark_Completion/Conv_T_4/BiasAdd:output:0*/
_output_shapes
:���������`P *
alpha%���>2=
;Facial_Landmark_Completion/Conv_T_4/leaky_re_lu_8/LeakyRelu�
)Facial_Landmark_Completion/Conv_T_5/ShapeShapeIFacial_Landmark_Completion/Conv_T_4/leaky_re_lu_8/LeakyRelu:activations:0*
T0*
_output_shapes
:2+
)Facial_Landmark_Completion/Conv_T_5/Shape�
7Facial_Landmark_Completion/Conv_T_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7Facial_Landmark_Completion/Conv_T_5/strided_slice/stack�
9Facial_Landmark_Completion/Conv_T_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9Facial_Landmark_Completion/Conv_T_5/strided_slice/stack_1�
9Facial_Landmark_Completion/Conv_T_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9Facial_Landmark_Completion/Conv_T_5/strided_slice/stack_2�
1Facial_Landmark_Completion/Conv_T_5/strided_sliceStridedSlice2Facial_Landmark_Completion/Conv_T_5/Shape:output:0@Facial_Landmark_Completion/Conv_T_5/strided_slice/stack:output:0BFacial_Landmark_Completion/Conv_T_5/strided_slice/stack_1:output:0BFacial_Landmark_Completion/Conv_T_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1Facial_Landmark_Completion/Conv_T_5/strided_slice�
+Facial_Landmark_Completion/Conv_T_5/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�2-
+Facial_Landmark_Completion/Conv_T_5/stack/1�
+Facial_Landmark_Completion/Conv_T_5/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�2-
+Facial_Landmark_Completion/Conv_T_5/stack/2�
+Facial_Landmark_Completion/Conv_T_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2-
+Facial_Landmark_Completion/Conv_T_5/stack/3�
)Facial_Landmark_Completion/Conv_T_5/stackPack:Facial_Landmark_Completion/Conv_T_5/strided_slice:output:04Facial_Landmark_Completion/Conv_T_5/stack/1:output:04Facial_Landmark_Completion/Conv_T_5/stack/2:output:04Facial_Landmark_Completion/Conv_T_5/stack/3:output:0*
N*
T0*
_output_shapes
:2+
)Facial_Landmark_Completion/Conv_T_5/stack�
9Facial_Landmark_Completion/Conv_T_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9Facial_Landmark_Completion/Conv_T_5/strided_slice_1/stack�
;Facial_Landmark_Completion/Conv_T_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;Facial_Landmark_Completion/Conv_T_5/strided_slice_1/stack_1�
;Facial_Landmark_Completion/Conv_T_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;Facial_Landmark_Completion/Conv_T_5/strided_slice_1/stack_2�
3Facial_Landmark_Completion/Conv_T_5/strided_slice_1StridedSlice2Facial_Landmark_Completion/Conv_T_5/stack:output:0BFacial_Landmark_Completion/Conv_T_5/strided_slice_1/stack:output:0DFacial_Landmark_Completion/Conv_T_5/strided_slice_1/stack_1:output:0DFacial_Landmark_Completion/Conv_T_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3Facial_Landmark_Completion/Conv_T_5/strided_slice_1�
CFacial_Landmark_Completion/Conv_T_5/conv2d_transpose/ReadVariableOpReadVariableOpLfacial_landmark_completion_conv_t_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02E
CFacial_Landmark_Completion/Conv_T_5/conv2d_transpose/ReadVariableOp�
4Facial_Landmark_Completion/Conv_T_5/conv2d_transposeConv2DBackpropInput2Facial_Landmark_Completion/Conv_T_5/stack:output:0KFacial_Landmark_Completion/Conv_T_5/conv2d_transpose/ReadVariableOp:value:0IFacial_Landmark_Completion/Conv_T_4/leaky_re_lu_8/LeakyRelu:activations:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
26
4Facial_Landmark_Completion/Conv_T_5/conv2d_transpose�
:Facial_Landmark_Completion/Conv_T_5/BiasAdd/ReadVariableOpReadVariableOpCfacial_landmark_completion_conv_t_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02<
:Facial_Landmark_Completion/Conv_T_5/BiasAdd/ReadVariableOp�
+Facial_Landmark_Completion/Conv_T_5/BiasAddBiasAdd=Facial_Landmark_Completion/Conv_T_5/conv2d_transpose:output:0BFacial_Landmark_Completion/Conv_T_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2-
+Facial_Landmark_Completion/Conv_T_5/BiasAdd�
+Facial_Landmark_Completion/Conv_T_5/SigmoidSigmoid4Facial_Landmark_Completion/Conv_T_5/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2-
+Facial_Landmark_Completion/Conv_T_5/Sigmoid�
IdentityIdentity/Facial_Landmark_Completion/Conv_T_5/Sigmoid:y:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������:�����������:::::::::::::::::::::b ^
1
_output_shapes
:�����������
)
_user_specified_nameImg_Input_Layer:c_
1
_output_shapes
:�����������
*
_user_specified_nameMask_Input_Layer
�	
�
A__inference_Conv_3_layer_call_and_return_conditional_losses_29599

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������0(�*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������0(�2	
BiasAdd�
leaky_re_lu_2/LeakyRelu	LeakyReluBiasAdd:output:0*0
_output_shapes
:���������0(�*
alpha%���>2
leaky_re_lu_2/LeakyRelu�
IdentityIdentity%leaky_re_lu_2/LeakyRelu:activations:0*
T0*0
_output_shapes
:���������0(�2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������0(@:::W S
/
_output_shapes
:���������0(@
 
_user_specified_nameinputs
�
^
@__inference_SPD_3_layer_call_and_return_conditional_losses_27786

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4������������������������������������2

Identity�

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
_
@__inference_SPD_5_layer_call_and_return_conditional_losses_28083

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const�
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul_1�
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
}
(__inference_Conv_T_1_layer_call_fn_27846

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Conv_T_1_layer_call_and_return_conditional_losses_278362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
_
@__inference_SPD_5_layer_call_and_return_conditional_losses_29925

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:���������0(�2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:���������0(�2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:���������0(�2

Identity"
identityIdentity:output:0*/
_input_shapes
:���������0(�:X T
0
_output_shapes
:���������0(�
 
_user_specified_nameinputs
�
}
(__inference_Conv_T_5_layer_call_fn_28198

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Conv_T_5_layer_call_and_return_conditional_losses_281882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�]
�
U__inference_Facial_Landmark_Completion_layer_call_and_return_conditional_losses_28680
img_input_layer
mask_input_layer
conv_1_28617
conv_1_28619
conv_2_28623
conv_2_28625
conv_3_28630
conv_3_28632
conv_4_28636
conv_4_28638
conv_5_28643
conv_5_28645
conv_t_1_28650
conv_t_1_28652
conv_t_2_28657
conv_t_2_28659
conv_t_3_28662
conv_t_3_28664
conv_t_4_28669
conv_t_4_28671
conv_t_5_28674
conv_t_5_28676
identity��Conv_1/StatefulPartitionedCall�Conv_2/StatefulPartitionedCall�Conv_3/StatefulPartitionedCall�Conv_4/StatefulPartitionedCall�Conv_5/StatefulPartitionedCall� Conv_T_1/StatefulPartitionedCall� Conv_T_2/StatefulPartitionedCall� Conv_T_3/StatefulPartitionedCall� Conv_T_4/StatefulPartitionedCall� Conv_T_5/StatefulPartitionedCall�
#Concatenated_Inputs/PartitionedCallPartitionedCallimg_input_layermask_input_layer*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_Concatenated_Inputs_layer_call_and_return_conditional_losses_282102%
#Concatenated_Inputs/PartitionedCall�
Conv_1/StatefulPartitionedCallStatefulPartitionedCall,Concatenated_Inputs/PartitionedCall:output:0conv_1_28617conv_1_28619*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_Conv_1_layer_call_and_return_conditional_losses_282302 
Conv_1/StatefulPartitionedCall�
Max_Pool_1/PartitionedCallPartitionedCall'Conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������`P * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_Max_Pool_1_layer_call_and_return_conditional_losses_275312
Max_Pool_1/PartitionedCall�
Conv_2/StatefulPartitionedCallStatefulPartitionedCall#Max_Pool_1/PartitionedCall:output:0conv_2_28623conv_2_28625*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������`P@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_Conv_2_layer_call_and_return_conditional_losses_282582 
Conv_2/StatefulPartitionedCall�
Max_Pool_2/PartitionedCallPartitionedCall'Conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0(@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_Max_Pool_2_layer_call_and_return_conditional_losses_275432
Max_Pool_2/PartitionedCall�
SPD_1/PartitionedCallPartitionedCall#Max_Pool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0(@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_1_layer_call_and_return_conditional_losses_283022
SPD_1/PartitionedCall�
Conv_3/StatefulPartitionedCallStatefulPartitionedCallSPD_1/PartitionedCall:output:0conv_3_28630conv_3_28632*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������0(�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_Conv_3_layer_call_and_return_conditional_losses_283252 
Conv_3/StatefulPartitionedCall�
Max_Pool_3/PartitionedCallPartitionedCall'Conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_Max_Pool_3_layer_call_and_return_conditional_losses_276232
Max_Pool_3/PartitionedCall�
Conv_4/StatefulPartitionedCallStatefulPartitionedCall#Max_Pool_3/PartitionedCall:output:0conv_4_28636conv_4_28638*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_Conv_4_layer_call_and_return_conditional_losses_283532 
Conv_4/StatefulPartitionedCall�
Max_Pool_4/PartitionedCallPartitionedCall'Conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_Max_Pool_4_layer_call_and_return_conditional_losses_276352
Max_Pool_4/PartitionedCall�
SPD_2/PartitionedCallPartitionedCall#Max_Pool_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_2_layer_call_and_return_conditional_losses_283972
SPD_2/PartitionedCall�
Conv_5/StatefulPartitionedCallStatefulPartitionedCallSPD_2/PartitionedCall:output:0conv_5_28643conv_5_28645*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_Conv_5_layer_call_and_return_conditional_losses_284202 
Conv_5/StatefulPartitionedCall�
Max_Pool_5/PartitionedCallPartitionedCall'Conv_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_Max_Pool_5_layer_call_and_return_conditional_losses_277152
Max_Pool_5/PartitionedCall�
SPD_3/PartitionedCallPartitionedCall#Max_Pool_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_3_layer_call_and_return_conditional_losses_284642
SPD_3/PartitionedCall�
 Conv_T_1/StatefulPartitionedCallStatefulPartitionedCallSPD_3/PartitionedCall:output:0conv_t_1_28650conv_t_1_28652*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Conv_T_1_layer_call_and_return_conditional_losses_278362"
 Conv_T_1/StatefulPartitionedCall�
Concat_1/PartitionedCallPartitionedCall)Conv_T_1/StatefulPartitionedCall:output:0#Max_Pool_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Concat_1_layer_call_and_return_conditional_losses_284882
Concat_1/PartitionedCall�
SPD_4/PartitionedCallPartitionedCall!Concat_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_4_layer_call_and_return_conditional_losses_285242
SPD_4/PartitionedCall�
 Conv_T_2/StatefulPartitionedCallStatefulPartitionedCallSPD_4/PartitionedCall:output:0conv_t_2_28657conv_t_2_28659*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Conv_T_2_layer_call_and_return_conditional_losses_279612"
 Conv_T_2/StatefulPartitionedCall�
 Conv_T_3/StatefulPartitionedCallStatefulPartitionedCall)Conv_T_2/StatefulPartitionedCall:output:0conv_t_3_28662conv_t_3_28664*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Conv_T_3_layer_call_and_return_conditional_losses_280182"
 Conv_T_3/StatefulPartitionedCall�
Concat_2/PartitionedCallPartitionedCall)Conv_T_3/StatefulPartitionedCall:output:0#Max_Pool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������0(�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Concat_2_layer_call_and_return_conditional_losses_285532
Concat_2/PartitionedCall�
SPD_5/PartitionedCallPartitionedCall!Concat_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������0(�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_5_layer_call_and_return_conditional_losses_285892
SPD_5/PartitionedCall�
 Conv_T_4/StatefulPartitionedCallStatefulPartitionedCallSPD_5/PartitionedCall:output:0conv_t_4_28669conv_t_4_28671*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Conv_T_4_layer_call_and_return_conditional_losses_281432"
 Conv_T_4/StatefulPartitionedCall�
 Conv_T_5/StatefulPartitionedCallStatefulPartitionedCall)Conv_T_4/StatefulPartitionedCall:output:0conv_t_5_28674conv_t_5_28676*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Conv_T_5_layer_call_and_return_conditional_losses_281882"
 Conv_T_5/StatefulPartitionedCall�
IdentityIdentity)Conv_T_5/StatefulPartitionedCall:output:0^Conv_1/StatefulPartitionedCall^Conv_2/StatefulPartitionedCall^Conv_3/StatefulPartitionedCall^Conv_4/StatefulPartitionedCall^Conv_5/StatefulPartitionedCall!^Conv_T_1/StatefulPartitionedCall!^Conv_T_2/StatefulPartitionedCall!^Conv_T_3/StatefulPartitionedCall!^Conv_T_4/StatefulPartitionedCall!^Conv_T_5/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������:�����������::::::::::::::::::::2@
Conv_1/StatefulPartitionedCallConv_1/StatefulPartitionedCall2@
Conv_2/StatefulPartitionedCallConv_2/StatefulPartitionedCall2@
Conv_3/StatefulPartitionedCallConv_3/StatefulPartitionedCall2@
Conv_4/StatefulPartitionedCallConv_4/StatefulPartitionedCall2@
Conv_5/StatefulPartitionedCallConv_5/StatefulPartitionedCall2D
 Conv_T_1/StatefulPartitionedCall Conv_T_1/StatefulPartitionedCall2D
 Conv_T_2/StatefulPartitionedCall Conv_T_2/StatefulPartitionedCall2D
 Conv_T_3/StatefulPartitionedCall Conv_T_3/StatefulPartitionedCall2D
 Conv_T_4/StatefulPartitionedCall Conv_T_4/StatefulPartitionedCall2D
 Conv_T_5/StatefulPartitionedCall Conv_T_5/StatefulPartitionedCall:b ^
1
_output_shapes
:�����������
)
_user_specified_nameImg_Input_Layer:c_
1
_output_shapes
:�����������
*
_user_specified_nameMask_Input_Layer
�
A
%__inference_SPD_1_layer_call_fn_29588

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_1_layer_call_and_return_conditional_losses_276142
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
^
@__inference_SPD_2_layer_call_and_return_conditional_losses_28397

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:���������
�2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:���������
�2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:���������
�:X T
0
_output_shapes
:���������
�
 
_user_specified_nameinputs
�
m
C__inference_Concat_1_layer_call_and_return_conditional_losses_28488

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:���������
�2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:���������
�2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:,����������������������������:���������
�:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:XT
0
_output_shapes
:���������
�
 
_user_specified_nameinputs
�	
�
A__inference_Conv_4_layer_call_and_return_conditional_losses_29619

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdd�
leaky_re_lu_3/LeakyRelu	LeakyReluBiasAdd:output:0*0
_output_shapes
:����������*
alpha%���>2
leaky_re_lu_3/LeakyRelu�
IdentityIdentity%leaky_re_lu_3/LeakyRelu:activations:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������:::X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
^
@__inference_SPD_5_layer_call_and_return_conditional_losses_29968

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4������������������������������������2

Identity�

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�%
�
C__inference_Conv_T_1_layer_call_and_return_conditional_losses_27836

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :�2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������2	
BiasAdd�
leaky_re_lu_5/PartitionedCallPartitionedCallBiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_278272
leaky_re_lu_5/PartitionedCall�
IdentityIdentity&leaky_re_lu_5/PartitionedCall:output:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������:::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
_
@__inference_SPD_3_layer_call_and_return_conditional_losses_29747

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:����������2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
_
@__inference_SPD_4_layer_call_and_return_conditional_losses_29836

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const�
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul_1�
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
x
N__inference_Concatenated_Inputs_layer_call_and_return_conditional_losses_28210

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:�����������2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*M
_input_shapes<
::�����������:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:YU
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�	
�
A__inference_Conv_3_layer_call_and_return_conditional_losses_28325

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������0(�*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������0(�2	
BiasAdd�
leaky_re_lu_2/LeakyRelu	LeakyReluBiasAdd:output:0*0
_output_shapes
:���������0(�*
alpha%���>2
leaky_re_lu_2/LeakyRelu�
IdentityIdentity%leaky_re_lu_2/LeakyRelu:activations:0*
T0*0
_output_shapes
:���������0(�2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������0(@:::W S
/
_output_shapes
:���������0(@
 
_user_specified_nameinputs
�
I
-__inference_leaky_re_lu_6_layer_call_fn_29998

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_279522
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
��
�
__inference__traced_save_30249
file_prefix,
(savev2_conv_1_kernel_read_readvariableop*
&savev2_conv_1_bias_read_readvariableop,
(savev2_conv_2_kernel_read_readvariableop*
&savev2_conv_2_bias_read_readvariableop,
(savev2_conv_3_kernel_read_readvariableop*
&savev2_conv_3_bias_read_readvariableop,
(savev2_conv_4_kernel_read_readvariableop*
&savev2_conv_4_bias_read_readvariableop,
(savev2_conv_5_kernel_read_readvariableop*
&savev2_conv_5_bias_read_readvariableop.
*savev2_conv_t_1_kernel_read_readvariableop,
(savev2_conv_t_1_bias_read_readvariableop.
*savev2_conv_t_2_kernel_read_readvariableop,
(savev2_conv_t_2_bias_read_readvariableop.
*savev2_conv_t_3_kernel_read_readvariableop,
(savev2_conv_t_3_bias_read_readvariableop.
*savev2_conv_t_4_kernel_read_readvariableop,
(savev2_conv_t_4_bias_read_readvariableop.
*savev2_conv_t_5_kernel_read_readvariableop,
(savev2_conv_t_5_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_conv_1_kernel_m_read_readvariableop1
-savev2_adam_conv_1_bias_m_read_readvariableop3
/savev2_adam_conv_2_kernel_m_read_readvariableop1
-savev2_adam_conv_2_bias_m_read_readvariableop3
/savev2_adam_conv_3_kernel_m_read_readvariableop1
-savev2_adam_conv_3_bias_m_read_readvariableop3
/savev2_adam_conv_4_kernel_m_read_readvariableop1
-savev2_adam_conv_4_bias_m_read_readvariableop3
/savev2_adam_conv_5_kernel_m_read_readvariableop1
-savev2_adam_conv_5_bias_m_read_readvariableop5
1savev2_adam_conv_t_1_kernel_m_read_readvariableop3
/savev2_adam_conv_t_1_bias_m_read_readvariableop5
1savev2_adam_conv_t_2_kernel_m_read_readvariableop3
/savev2_adam_conv_t_2_bias_m_read_readvariableop5
1savev2_adam_conv_t_3_kernel_m_read_readvariableop3
/savev2_adam_conv_t_3_bias_m_read_readvariableop5
1savev2_adam_conv_t_4_kernel_m_read_readvariableop3
/savev2_adam_conv_t_4_bias_m_read_readvariableop5
1savev2_adam_conv_t_5_kernel_m_read_readvariableop3
/savev2_adam_conv_t_5_bias_m_read_readvariableop3
/savev2_adam_conv_1_kernel_v_read_readvariableop1
-savev2_adam_conv_1_bias_v_read_readvariableop3
/savev2_adam_conv_2_kernel_v_read_readvariableop1
-savev2_adam_conv_2_bias_v_read_readvariableop3
/savev2_adam_conv_3_kernel_v_read_readvariableop1
-savev2_adam_conv_3_bias_v_read_readvariableop3
/savev2_adam_conv_4_kernel_v_read_readvariableop1
-savev2_adam_conv_4_bias_v_read_readvariableop3
/savev2_adam_conv_5_kernel_v_read_readvariableop1
-savev2_adam_conv_5_bias_v_read_readvariableop5
1savev2_adam_conv_t_1_kernel_v_read_readvariableop3
/savev2_adam_conv_t_1_bias_v_read_readvariableop5
1savev2_adam_conv_t_2_kernel_v_read_readvariableop3
/savev2_adam_conv_t_2_bias_v_read_readvariableop5
1savev2_adam_conv_t_3_kernel_v_read_readvariableop3
/savev2_adam_conv_t_3_bias_v_read_readvariableop5
1savev2_adam_conv_t_4_kernel_v_read_readvariableop3
/savev2_adam_conv_t_4_bias_v_read_readvariableop5
1savev2_adam_conv_t_5_kernel_v_read_readvariableop3
/savev2_adam_conv_t_5_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_2e5383f653bf4e39ad4e87b2dec75cd4/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�'
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*�&
value�&B�&FB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*�
value�B�FB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv_1_kernel_read_readvariableop&savev2_conv_1_bias_read_readvariableop(savev2_conv_2_kernel_read_readvariableop&savev2_conv_2_bias_read_readvariableop(savev2_conv_3_kernel_read_readvariableop&savev2_conv_3_bias_read_readvariableop(savev2_conv_4_kernel_read_readvariableop&savev2_conv_4_bias_read_readvariableop(savev2_conv_5_kernel_read_readvariableop&savev2_conv_5_bias_read_readvariableop*savev2_conv_t_1_kernel_read_readvariableop(savev2_conv_t_1_bias_read_readvariableop*savev2_conv_t_2_kernel_read_readvariableop(savev2_conv_t_2_bias_read_readvariableop*savev2_conv_t_3_kernel_read_readvariableop(savev2_conv_t_3_bias_read_readvariableop*savev2_conv_t_4_kernel_read_readvariableop(savev2_conv_t_4_bias_read_readvariableop*savev2_conv_t_5_kernel_read_readvariableop(savev2_conv_t_5_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_conv_1_kernel_m_read_readvariableop-savev2_adam_conv_1_bias_m_read_readvariableop/savev2_adam_conv_2_kernel_m_read_readvariableop-savev2_adam_conv_2_bias_m_read_readvariableop/savev2_adam_conv_3_kernel_m_read_readvariableop-savev2_adam_conv_3_bias_m_read_readvariableop/savev2_adam_conv_4_kernel_m_read_readvariableop-savev2_adam_conv_4_bias_m_read_readvariableop/savev2_adam_conv_5_kernel_m_read_readvariableop-savev2_adam_conv_5_bias_m_read_readvariableop1savev2_adam_conv_t_1_kernel_m_read_readvariableop/savev2_adam_conv_t_1_bias_m_read_readvariableop1savev2_adam_conv_t_2_kernel_m_read_readvariableop/savev2_adam_conv_t_2_bias_m_read_readvariableop1savev2_adam_conv_t_3_kernel_m_read_readvariableop/savev2_adam_conv_t_3_bias_m_read_readvariableop1savev2_adam_conv_t_4_kernel_m_read_readvariableop/savev2_adam_conv_t_4_bias_m_read_readvariableop1savev2_adam_conv_t_5_kernel_m_read_readvariableop/savev2_adam_conv_t_5_bias_m_read_readvariableop/savev2_adam_conv_1_kernel_v_read_readvariableop-savev2_adam_conv_1_bias_v_read_readvariableop/savev2_adam_conv_2_kernel_v_read_readvariableop-savev2_adam_conv_2_bias_v_read_readvariableop/savev2_adam_conv_3_kernel_v_read_readvariableop-savev2_adam_conv_3_bias_v_read_readvariableop/savev2_adam_conv_4_kernel_v_read_readvariableop-savev2_adam_conv_4_bias_v_read_readvariableop/savev2_adam_conv_5_kernel_v_read_readvariableop-savev2_adam_conv_5_bias_v_read_readvariableop1savev2_adam_conv_t_1_kernel_v_read_readvariableop/savev2_adam_conv_t_1_bias_v_read_readvariableop1savev2_adam_conv_t_2_kernel_v_read_readvariableop/savev2_adam_conv_t_2_bias_v_read_readvariableop1savev2_adam_conv_t_3_kernel_v_read_readvariableop/savev2_adam_conv_t_3_bias_v_read_readvariableop1savev2_adam_conv_t_4_kernel_v_read_readvariableop/savev2_adam_conv_t_4_bias_v_read_readvariableop1savev2_adam_conv_t_5_kernel_v_read_readvariableop/savev2_adam_conv_t_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *T
dtypesJ
H2F	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : : @:@:@�:�:��:�:��:�:��:�:��:�:@�:@: �: : :: : : : : : : : : : : : @:@:@�:�:��:�:��:�:��:�:��:�:@�:@: �: : :: : : @:@:@�:�:��:�:��:�:��:�:��:�:@�:@: �: : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.	*
(
_output_shapes
:��:!


_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:-)
'
_output_shapes
:@�: 

_output_shapes
:@:-)
'
_output_shapes
: �: 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :, (
&
_output_shapes
: @: !

_output_shapes
:@:-")
'
_output_shapes
:@�:!#

_output_shapes	
:�:.$*
(
_output_shapes
:��:!%

_output_shapes	
:�:.&*
(
_output_shapes
:��:!'

_output_shapes	
:�:.(*
(
_output_shapes
:��:!)

_output_shapes	
:�:.**
(
_output_shapes
:��:!+

_output_shapes	
:�:-,)
'
_output_shapes
:@�: -

_output_shapes
:@:-.)
'
_output_shapes
: �: /

_output_shapes
: :,0(
&
_output_shapes
: : 1

_output_shapes
::,2(
&
_output_shapes
: : 3

_output_shapes
: :,4(
&
_output_shapes
: @: 5

_output_shapes
:@:-6)
'
_output_shapes
:@�:!7

_output_shapes	
:�:.8*
(
_output_shapes
:��:!9

_output_shapes	
:�:.:*
(
_output_shapes
:��:!;

_output_shapes	
:�:.<*
(
_output_shapes
:��:!=

_output_shapes	
:�:.>*
(
_output_shapes
:��:!?

_output_shapes	
:�:-@)
'
_output_shapes
:@�: A

_output_shapes
:@:-B)
'
_output_shapes
: �: C

_output_shapes
: :,D(
&
_output_shapes
: : E

_output_shapes
::F

_output_shapes
: 
�
A
%__inference_SPD_4_layer_call_fn_29889

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_4_layer_call_and_return_conditional_losses_285242
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:���������
�2

Identity"
identityIdentity:output:0*/
_input_shapes
:���������
�:X T
0
_output_shapes
:���������
�
 
_user_specified_nameinputs
�
^
%__inference_SPD_2_layer_call_fn_29661

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_2_layer_call_and_return_conditional_losses_276962
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
F
*__inference_Max_Pool_2_layer_call_fn_27549

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_Max_Pool_2_layer_call_and_return_conditional_losses_275432
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
_
@__inference_SPD_2_layer_call_and_return_conditional_losses_27696

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const�
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul_1�
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
{
&__inference_Conv_2_layer_call_fn_29512

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������`P@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_Conv_2_layer_call_and_return_conditional_losses_282582
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������`P@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������`P ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������`P 
 
_user_specified_nameinputs
�
_
@__inference_SPD_5_layer_call_and_return_conditional_losses_29963

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const�
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul_1�
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
d
H__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_30013

inputs
identity~
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+��������������������������� *
alpha%���>2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+��������������������������� :i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
z
N__inference_Concatenated_Inputs_layer_call_and_return_conditional_losses_29466
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:�����������2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*M
_input_shapes<
::�����������:�����������:[ W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/1
�	
�
A__inference_Conv_1_layer_call_and_return_conditional_losses_29483

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� 2	
BiasAdd�
leaky_re_lu/LeakyRelu	LeakyReluBiasAdd:output:0*1
_output_shapes
:����������� *
alpha%���>2
leaky_re_lu/LeakyRelu�
IdentityIdentity#leaky_re_lu/LeakyRelu:activations:0*
T0*1
_output_shapes
:����������� 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������:::Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
}
(__inference_Conv_T_3_layer_call_fn_28028

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Conv_T_3_layer_call_and_return_conditional_losses_280182
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
_
@__inference_SPD_3_layer_call_and_return_conditional_losses_28459

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:����������2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
E__inference_Max_Pool_1_layer_call_and_return_conditional_losses_27531

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
^
@__inference_SPD_3_layer_call_and_return_conditional_losses_28464

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:����������2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
{
&__inference_Conv_4_layer_call_fn_29628

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_Conv_4_layer_call_and_return_conditional_losses_283532
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
^
@__inference_SPD_5_layer_call_and_return_conditional_losses_28093

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4������������������������������������2

Identity�

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
d
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_29993

inputs
identity
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,����������������������������*
alpha%���>2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
̟
�#
!__inference__traced_restore_30466
file_prefix"
assignvariableop_conv_1_kernel"
assignvariableop_1_conv_1_bias$
 assignvariableop_2_conv_2_kernel"
assignvariableop_3_conv_2_bias$
 assignvariableop_4_conv_3_kernel"
assignvariableop_5_conv_3_bias$
 assignvariableop_6_conv_4_kernel"
assignvariableop_7_conv_4_bias$
 assignvariableop_8_conv_5_kernel"
assignvariableop_9_conv_5_bias'
#assignvariableop_10_conv_t_1_kernel%
!assignvariableop_11_conv_t_1_bias'
#assignvariableop_12_conv_t_2_kernel%
!assignvariableop_13_conv_t_2_bias'
#assignvariableop_14_conv_t_3_kernel%
!assignvariableop_15_conv_t_3_bias'
#assignvariableop_16_conv_t_4_kernel%
!assignvariableop_17_conv_t_4_bias'
#assignvariableop_18_conv_t_5_kernel%
!assignvariableop_19_conv_t_5_bias!
assignvariableop_20_adam_iter#
assignvariableop_21_adam_beta_1#
assignvariableop_22_adam_beta_2"
assignvariableop_23_adam_decay*
&assignvariableop_24_adam_learning_rate
assignvariableop_25_total
assignvariableop_26_count
assignvariableop_27_total_1
assignvariableop_28_count_1,
(assignvariableop_29_adam_conv_1_kernel_m*
&assignvariableop_30_adam_conv_1_bias_m,
(assignvariableop_31_adam_conv_2_kernel_m*
&assignvariableop_32_adam_conv_2_bias_m,
(assignvariableop_33_adam_conv_3_kernel_m*
&assignvariableop_34_adam_conv_3_bias_m,
(assignvariableop_35_adam_conv_4_kernel_m*
&assignvariableop_36_adam_conv_4_bias_m,
(assignvariableop_37_adam_conv_5_kernel_m*
&assignvariableop_38_adam_conv_5_bias_m.
*assignvariableop_39_adam_conv_t_1_kernel_m,
(assignvariableop_40_adam_conv_t_1_bias_m.
*assignvariableop_41_adam_conv_t_2_kernel_m,
(assignvariableop_42_adam_conv_t_2_bias_m.
*assignvariableop_43_adam_conv_t_3_kernel_m,
(assignvariableop_44_adam_conv_t_3_bias_m.
*assignvariableop_45_adam_conv_t_4_kernel_m,
(assignvariableop_46_adam_conv_t_4_bias_m.
*assignvariableop_47_adam_conv_t_5_kernel_m,
(assignvariableop_48_adam_conv_t_5_bias_m,
(assignvariableop_49_adam_conv_1_kernel_v*
&assignvariableop_50_adam_conv_1_bias_v,
(assignvariableop_51_adam_conv_2_kernel_v*
&assignvariableop_52_adam_conv_2_bias_v,
(assignvariableop_53_adam_conv_3_kernel_v*
&assignvariableop_54_adam_conv_3_bias_v,
(assignvariableop_55_adam_conv_4_kernel_v*
&assignvariableop_56_adam_conv_4_bias_v,
(assignvariableop_57_adam_conv_5_kernel_v*
&assignvariableop_58_adam_conv_5_bias_v.
*assignvariableop_59_adam_conv_t_1_kernel_v,
(assignvariableop_60_adam_conv_t_1_bias_v.
*assignvariableop_61_adam_conv_t_2_kernel_v,
(assignvariableop_62_adam_conv_t_2_bias_v.
*assignvariableop_63_adam_conv_t_3_kernel_v,
(assignvariableop_64_adam_conv_t_3_bias_v.
*assignvariableop_65_adam_conv_t_4_kernel_v,
(assignvariableop_66_adam_conv_t_4_bias_v.
*assignvariableop_67_adam_conv_t_5_kernel_v,
(assignvariableop_68_adam_conv_t_5_bias_v
identity_70��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�'
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*�&
value�&B�&FB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*�
value�B�FB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*T
dtypesJ
H2F	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_conv_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp assignvariableop_8_conv_5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_conv_5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv_t_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv_t_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv_t_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv_t_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv_t_3_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv_t_3_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp#assignvariableop_16_conv_t_4_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp!assignvariableop_17_conv_t_4_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv_t_5_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv_t_5_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_iterIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_beta_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_beta_2Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_decayIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_learning_rateIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_conv_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_conv_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_conv_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_conv_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_conv_3_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_conv_3_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_conv_4_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_conv_4_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_conv_5_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_conv_5_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv_t_1_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv_t_1_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv_t_2_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv_t_2_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_conv_t_3_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv_t_3_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_conv_t_4_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_conv_t_4_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_conv_t_5_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_conv_t_5_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_conv_1_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp&assignvariableop_50_adam_conv_1_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_conv_2_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp&assignvariableop_52_adam_conv_2_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOp(assignvariableop_53_adam_conv_3_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54�
AssignVariableOp_54AssignVariableOp&assignvariableop_54_adam_conv_3_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55�
AssignVariableOp_55AssignVariableOp(assignvariableop_55_adam_conv_4_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56�
AssignVariableOp_56AssignVariableOp&assignvariableop_56_adam_conv_4_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57�
AssignVariableOp_57AssignVariableOp(assignvariableop_57_adam_conv_5_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58�
AssignVariableOp_58AssignVariableOp&assignvariableop_58_adam_conv_5_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59�
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_conv_t_1_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60�
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_conv_t_1_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61�
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_conv_t_2_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62�
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_conv_t_2_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63�
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_conv_t_3_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64�
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_conv_t_3_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65�
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_conv_t_4_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66�
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_conv_t_4_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67�
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_conv_t_5_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68�
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_conv_t_5_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_689
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_69Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_69�
Identity_70IdentityIdentity_69:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_70"#
identity_70Identity_70:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
^
%__inference_SPD_3_layer_call_fn_29795

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_3_layer_call_and_return_conditional_losses_277762
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
d
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_30003

inputs
identity~
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+���������������������������@*
alpha%���>2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������@:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
^
@__inference_SPD_1_layer_call_and_return_conditional_losses_28302

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:���������0(@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������0(@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������0(@:W S
/
_output_shapes
:���������0(@
 
_user_specified_nameinputs
�
_
@__inference_SPD_1_layer_call_and_return_conditional_losses_29573

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const�
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul_1�
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_28965
img_input_layer
mask_input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallimg_input_layermask_input_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__wrapped_model_275252
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������:�����������::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:�����������
)
_user_specified_nameImg_Input_Layer:c_
1
_output_shapes
:�����������
*
_user_specified_nameMask_Input_Layer
�
_
@__inference_SPD_4_layer_call_and_return_conditional_losses_29874

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:���������
�2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:���������
�2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:���������
�2

Identity"
identityIdentity:output:0*/
_input_shapes
:���������
�:X T
0
_output_shapes
:���������
�
 
_user_specified_nameinputs
�
_
@__inference_SPD_5_layer_call_and_return_conditional_losses_28584

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:���������0(�2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:���������0(�2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:���������0(�2

Identity"
identityIdentity:output:0*/
_input_shapes
:���������0(�:X T
0
_output_shapes
:���������0(�
 
_user_specified_nameinputs
�
^
@__inference_SPD_4_layer_call_and_return_conditional_losses_29879

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:���������
�2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:���������
�2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:���������
�:X T
0
_output_shapes
:���������
�
 
_user_specified_nameinputs
�
F
*__inference_Max_Pool_1_layer_call_fn_27537

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_Max_Pool_1_layer_call_and_return_conditional_losses_275312
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
d
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_27827

inputs
identity
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,����������������������������*
alpha%���>2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
^
@__inference_SPD_2_layer_call_and_return_conditional_losses_27706

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4������������������������������������2

Identity�

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
d
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_29983

inputs
identity
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,����������������������������*
alpha%���>2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
A
%__inference_SPD_1_layer_call_fn_29550

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0(@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_1_layer_call_and_return_conditional_losses_283022
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������0(@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������0(@:W S
/
_output_shapes
:���������0(@
 
_user_specified_nameinputs
�
_
@__inference_SPD_3_layer_call_and_return_conditional_losses_27776

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const�
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul_1�
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
_
@__inference_SPD_2_layer_call_and_return_conditional_losses_29689

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:���������
�2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:���������
�2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:���������
�2

Identity"
identityIdentity:output:0*/
_input_shapes
:���������
�:X T
0
_output_shapes
:���������
�
 
_user_specified_nameinputs
�
a
E__inference_Max_Pool_5_layer_call_and_return_conditional_losses_27715

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
A
%__inference_SPD_2_layer_call_fn_29666

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_2_layer_call_and_return_conditional_losses_277062
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
_
@__inference_SPD_2_layer_call_and_return_conditional_losses_28392

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:���������
�2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:���������
�2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:���������
�2

Identity"
identityIdentity:output:0*/
_input_shapes
:���������
�:X T
0
_output_shapes
:���������
�
 
_user_specified_nameinputs
�	
�
A__inference_Conv_2_layer_call_and_return_conditional_losses_29503

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������`P@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������`P@2	
BiasAdd�
leaky_re_lu_1/LeakyRelu	LeakyReluBiasAdd:output:0*/
_output_shapes
:���������`P@*
alpha%���>2
leaky_re_lu_1/LeakyRelu�
IdentityIdentity%leaky_re_lu_1/LeakyRelu:activations:0*
T0*/
_output_shapes
:���������`P@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������`P :::W S
/
_output_shapes
:���������`P 
 
_user_specified_nameinputs
�
^
%__inference_SPD_4_layer_call_fn_29846

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_4_layer_call_and_return_conditional_losses_279012
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
^
@__inference_SPD_3_layer_call_and_return_conditional_losses_29790

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4������������������������������������2

Identity�

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
:__inference_Facial_Landmark_Completion_layer_call_fn_29459
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_Facial_Landmark_Completion_layer_call_and_return_conditional_losses_288662
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������:�����������::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/1
�
d
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_28009

inputs
identity~
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+���������������������������@*
alpha%���>2
	LeakyRelu�
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+���������������������������@:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
A
%__inference_SPD_4_layer_call_fn_29851

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_4_layer_call_and_return_conditional_losses_279112
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
^
@__inference_SPD_1_layer_call_and_return_conditional_losses_27614

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4������������������������������������2

Identity�

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�	
�
A__inference_Conv_5_layer_call_and_return_conditional_losses_28420

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�2	
BiasAdd�
leaky_re_lu_4/LeakyRelu	LeakyReluBiasAdd:output:0*0
_output_shapes
:���������
�*
alpha%���>2
leaky_re_lu_4/LeakyRelu�
IdentityIdentity%leaky_re_lu_4/LeakyRelu:activations:0*
T0*0
_output_shapes
:���������
�2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:���������
�:::X T
0
_output_shapes
:���������
�
 
_user_specified_nameinputs
�
^
@__inference_SPD_2_layer_call_and_return_conditional_losses_29694

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:���������
�2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:���������
�2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:���������
�:X T
0
_output_shapes
:���������
�
 
_user_specified_nameinputs
�
o
C__inference_Concat_2_layer_call_and_return_conditional_losses_29896
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:���������0(�2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:���������0(�2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+���������������������������@:���������0(@:k g
A
_output_shapes/
-:+���������������������������@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:���������0(@
"
_user_specified_name
inputs/1
�
_
@__inference_SPD_1_layer_call_and_return_conditional_losses_27604

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const�
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul_1�
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
I
-__inference_leaky_re_lu_5_layer_call_fn_29988

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_278272
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
}
(__inference_Conv_T_2_layer_call_fn_27971

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Conv_T_2_layer_call_and_return_conditional_losses_279612
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
{
&__inference_Conv_3_layer_call_fn_29608

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������0(�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_Conv_3_layer_call_and_return_conditional_losses_283252
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:���������0(�2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������0(@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������0(@
 
_user_specified_nameinputs
�
a
E__inference_Max_Pool_3_layer_call_and_return_conditional_losses_27623

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
^
@__inference_SPD_5_layer_call_and_return_conditional_losses_28589

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:���������0(�2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:���������0(�2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:���������0(�:X T
0
_output_shapes
:���������0(�
 
_user_specified_nameinputs
�
^
%__inference_SPD_2_layer_call_fn_29699

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_2_layer_call_and_return_conditional_losses_283922
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:���������
�2

Identity"
identityIdentity:output:0*/
_input_shapes
:���������
�22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������
�
 
_user_specified_nameinputs
�
F
*__inference_Max_Pool_5_layer_call_fn_27721

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_Max_Pool_5_layer_call_and_return_conditional_losses_277152
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
F
*__inference_Max_Pool_3_layer_call_fn_27629

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_Max_Pool_3_layer_call_and_return_conditional_losses_276232
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
^
%__inference_SPD_4_layer_call_fn_29884

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_4_layer_call_and_return_conditional_losses_285192
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:���������
�2

Identity"
identityIdentity:output:0*/
_input_shapes
:���������
�22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������
�
 
_user_specified_nameinputs
�
^
@__inference_SPD_3_layer_call_and_return_conditional_losses_29752

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:����������2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
^
%__inference_SPD_1_layer_call_fn_29583

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_1_layer_call_and_return_conditional_losses_276042
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
T
(__inference_Concat_1_layer_call_fn_29813
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Concat_1_layer_call_and_return_conditional_losses_284882
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:���������
�2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:,����������������������������:���������
�:l h
B
_output_shapes0
.:,����������������������������
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:���������
�
"
_user_specified_name
inputs/1
�
o
C__inference_Concat_1_layer_call_and_return_conditional_losses_29807
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:���������
�2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:���������
�2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:,����������������������������:���������
�:l h
B
_output_shapes0
.:,����������������������������
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:���������
�
"
_user_specified_name
inputs/1
�
�
:__inference_Facial_Landmark_Completion_layer_call_fn_28795
img_input_layer
mask_input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallimg_input_layermask_input_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_Facial_Landmark_Completion_layer_call_and_return_conditional_losses_287522
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������:�����������::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:�����������
)
_user_specified_nameImg_Input_Layer:c_
1
_output_shapes
:�����������
*
_user_specified_nameMask_Input_Layer
�
�
:__inference_Facial_Landmark_Completion_layer_call_fn_28909
img_input_layer
mask_input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallimg_input_layermask_input_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_Facial_Landmark_Completion_layer_call_and_return_conditional_losses_288662
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������:�����������::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:�����������
)
_user_specified_nameImg_Input_Layer:c_
1
_output_shapes
:�����������
*
_user_specified_nameMask_Input_Layer
�
^
@__inference_SPD_1_layer_call_and_return_conditional_losses_29540

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:���������0(@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������0(@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������0(@:W S
/
_output_shapes
:���������0(@
 
_user_specified_nameinputs
�
}
(__inference_Conv_T_4_layer_call_fn_28153

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Conv_T_4_layer_call_and_return_conditional_losses_281432
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�"
�
C__inference_Conv_T_5_layer_call_and_return_conditional_losses_28188

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2	
Sigmoidy
IdentityIdentitySigmoid:y:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� :::i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
_
@__inference_SPD_1_layer_call_and_return_conditional_losses_28297

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������0(@2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������0(@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������0(@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������0(@:W S
/
_output_shapes
:���������0(@
 
_user_specified_nameinputs
�
_
3__inference_Concatenated_Inputs_layer_call_fn_29472
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_Concatenated_Inputs_layer_call_and_return_conditional_losses_282102
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*M
_input_shapes<
::�����������:�����������:[ W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:�����������
"
_user_specified_name
inputs/1
�
_
@__inference_SPD_1_layer_call_and_return_conditional_losses_29535

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������0(@2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������0(@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������0(@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������0(@:W S
/
_output_shapes
:���������0(@
 
_user_specified_nameinputs
�	
�
A__inference_Conv_2_layer_call_and_return_conditional_losses_28258

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������`P@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������`P@2	
BiasAdd�
leaky_re_lu_1/LeakyRelu	LeakyReluBiasAdd:output:0*/
_output_shapes
:���������`P@*
alpha%���>2
leaky_re_lu_1/LeakyRelu�
IdentityIdentity%leaky_re_lu_1/LeakyRelu:activations:0*
T0*/
_output_shapes
:���������`P@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������`P :::W S
/
_output_shapes
:���������`P 
 
_user_specified_nameinputs
�
_
@__inference_SPD_3_layer_call_and_return_conditional_losses_29785

inputs
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const�
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul�
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1�
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"������������������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4������������������������������������2
dropout/Mul_1�
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
^
%__inference_SPD_5_layer_call_fn_29973

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_SPD_5_layer_call_and_return_conditional_losses_280832
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�%
�
C__inference_Conv_T_4_layer_call_and_return_conditional_losses_28143

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity�D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
: �*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� 2	
BiasAdd�
leaky_re_lu_8/PartitionedCallPartitionedCallBiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_281342
leaky_re_lu_8/PartitionedCall�
IdentityIdentity&leaky_re_lu_8/PartitionedCall:output:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������:::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
U
Img_Input_LayerB
!serving_default_Img_Input_Layer:0�����������
W
Mask_Input_LayerC
"serving_default_Mask_Input_Layer:0�����������F
Conv_T_5:
StatefulPartitionedCall:0�����������tensorflow/serving/predict:ُ
��
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
layer-15
layer_with_weights-5
layer-16
layer-17
layer-18
layer_with_weights-6
layer-19
layer_with_weights-7
layer-20
layer-21
layer-22
layer_with_weights-8
layer-23
layer_with_weights-9
layer-24
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
�_default_save_signature
�__call__
+�&call_and_return_all_conditional_losses"��
_tf_keras_network��{"class_name": "Functional", "name": "Facial_Landmark_Completion", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Facial_Landmark_Completion", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 160, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Img_Input_Layer"}, "name": "Img_Input_Layer", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 160, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Mask_Input_Layer"}, "name": "Mask_Input_Layer", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "Concatenated_Inputs", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Concatenated_Inputs", "inbound_nodes": [[["Img_Input_Layer", 0, 0, {}], ["Mask_Input_Layer", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Conv_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv_1", "inbound_nodes": [[["Concatenated_Inputs", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Max_Pool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Max_Pool_1", "inbound_nodes": [[["Conv_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Conv_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv_2", "inbound_nodes": [[["Max_Pool_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Max_Pool_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Max_Pool_2", "inbound_nodes": [[["Conv_2", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "SPD_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "SPD_1", "inbound_nodes": [[["Max_Pool_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Conv_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv_3", "inbound_nodes": [[["SPD_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Max_Pool_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Max_Pool_3", "inbound_nodes": [[["Conv_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Conv_4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv_4", "inbound_nodes": [[["Max_Pool_3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Max_Pool_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Max_Pool_4", "inbound_nodes": [[["Conv_4", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "SPD_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "SPD_2", "inbound_nodes": [[["Max_Pool_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Conv_5", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv_5", "inbound_nodes": [[["SPD_2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Max_Pool_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Max_Pool_5", "inbound_nodes": [[["Conv_5", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "SPD_3", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "SPD_3", "inbound_nodes": [[["Max_Pool_5", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Conv_T_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Conv_T_1", "inbound_nodes": [[["SPD_3", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Concat_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Concat_1", "inbound_nodes": [[["Conv_T_1", 0, 0, {}], ["Max_Pool_4", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "SPD_4", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "SPD_4", "inbound_nodes": [[["Concat_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Conv_T_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Conv_T_2", "inbound_nodes": [[["SPD_4", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Conv_T_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Conv_T_3", "inbound_nodes": [[["Conv_T_2", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Concat_2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Concat_2", "inbound_nodes": [[["Conv_T_3", 0, 0, {}], ["Max_Pool_2", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "SPD_5", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "SPD_5", "inbound_nodes": [[["Concat_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Conv_T_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Conv_T_4", "inbound_nodes": [[["SPD_5", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Conv_T_5", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Conv_T_5", "inbound_nodes": [[["Conv_T_4", 0, 0, {}]]]}], "input_layers": [["Img_Input_Layer", 0, 0], ["Mask_Input_Layer", 0, 0]], "output_layers": [["Conv_T_5", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 192, 160, 3]}, {"class_name": "TensorShape", "items": [null, 192, 160, 1]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Facial_Landmark_Completion", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 160, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Img_Input_Layer"}, "name": "Img_Input_Layer", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 160, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Mask_Input_Layer"}, "name": "Mask_Input_Layer", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "Concatenated_Inputs", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Concatenated_Inputs", "inbound_nodes": [[["Img_Input_Layer", 0, 0, {}], ["Mask_Input_Layer", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Conv_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv_1", "inbound_nodes": [[["Concatenated_Inputs", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Max_Pool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Max_Pool_1", "inbound_nodes": [[["Conv_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Conv_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv_2", "inbound_nodes": [[["Max_Pool_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Max_Pool_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Max_Pool_2", "inbound_nodes": [[["Conv_2", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "SPD_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "SPD_1", "inbound_nodes": [[["Max_Pool_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Conv_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv_3", "inbound_nodes": [[["SPD_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Max_Pool_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Max_Pool_3", "inbound_nodes": [[["Conv_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Conv_4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv_4", "inbound_nodes": [[["Max_Pool_3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Max_Pool_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Max_Pool_4", "inbound_nodes": [[["Conv_4", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "SPD_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "SPD_2", "inbound_nodes": [[["Max_Pool_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Conv_5", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv_5", "inbound_nodes": [[["SPD_2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Max_Pool_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Max_Pool_5", "inbound_nodes": [[["Conv_5", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "SPD_3", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "SPD_3", "inbound_nodes": [[["Max_Pool_5", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Conv_T_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Conv_T_1", "inbound_nodes": [[["SPD_3", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Concat_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Concat_1", "inbound_nodes": [[["Conv_T_1", 0, 0, {}], ["Max_Pool_4", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "SPD_4", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "SPD_4", "inbound_nodes": [[["Concat_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Conv_T_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Conv_T_2", "inbound_nodes": [[["SPD_4", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Conv_T_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Conv_T_3", "inbound_nodes": [[["Conv_T_2", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Concat_2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Concat_2", "inbound_nodes": [[["Conv_T_3", 0, 0, {}], ["Max_Pool_2", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "SPD_5", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "SPD_5", "inbound_nodes": [[["Concat_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Conv_T_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Conv_T_4", "inbound_nodes": [[["SPD_5", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Conv_T_5", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Conv_T_5", "inbound_nodes": [[["Conv_T_4", 0, 0, {}]]]}], "input_layers": [["Img_Input_Layer", 0, 0], ["Mask_Input_Layer", 0, 0]], "output_layers": [["Conv_T_5", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["mse"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-06, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "Img_Input_Layer", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 160, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 160, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Img_Input_Layer"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "Mask_Input_Layer", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 160, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 160, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Mask_Input_Layer"}}
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Concatenate", "name": "Concatenated_Inputs", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Concatenated_Inputs", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 192, 160, 3]}, {"class_name": "TensorShape", "items": [null, 192, 160, 1]}]}
�

$
activation

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�	
_tf_keras_layer�	{"class_name": "Conv2D", "name": "Conv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Conv_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 192, 160, 4]}}
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "Max_Pool_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Max_Pool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�

/
activation

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�	
_tf_keras_layer�	{"class_name": "Conv2D", "name": "Conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Conv_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 80, 32]}}
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "Max_Pool_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Max_Pool_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "SpatialDropout2D", "name": "SPD_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "SPD_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
>
activation

?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�	
_tf_keras_layer�	{"class_name": "Conv2D", "name": "Conv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Conv_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 40, 64]}}
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "Max_Pool_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Max_Pool_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
I
activation

Jkernel
Kbias
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�	
_tf_keras_layer�	{"class_name": "Conv2D", "name": "Conv_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Conv_4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 20, 128]}}
�
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "Max_Pool_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Max_Pool_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "SpatialDropout2D", "name": "SPD_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "SPD_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
X
activation

Ykernel
Zbias
[	variables
\trainable_variables
]regularization_losses
^	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�	
_tf_keras_layer�	{"class_name": "Conv2D", "name": "Conv_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Conv_5", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 10, 256]}}
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "Max_Pool_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Max_Pool_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "SpatialDropout2D", "name": "SPD_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "SPD_3", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
g
activation

hkernel
ibias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�	
_tf_keras_layer�	{"class_name": "Conv2DTranspose", "name": "Conv_T_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Conv_T_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 5, 512]}}
�
n	variables
otrainable_variables
pregularization_losses
q	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Concatenate", "name": "Concat_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Concat_1", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 12, 10, 256]}, {"class_name": "TensorShape", "items": [null, 12, 10, 256]}]}
�
r	variables
strainable_variables
tregularization_losses
u	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "SpatialDropout2D", "name": "SPD_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "SPD_4", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
v
activation

wkernel
xbias
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�	
_tf_keras_layer�	{"class_name": "Conv2DTranspose", "name": "Conv_T_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Conv_T_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 10, 512]}}
�
}
activation

~kernel
bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�	
_tf_keras_layer�	{"class_name": "Conv2DTranspose", "name": "Conv_T_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Conv_T_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 20, 128]}}
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Concatenate", "name": "Concat_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Concat_2", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 48, 40, 64]}, {"class_name": "TensorShape", "items": [null, 48, 40, 64]}]}
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "SpatialDropout2D", "name": "SPD_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "SPD_5", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
�
activation
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�	
_tf_keras_layer�	{"class_name": "Conv2DTranspose", "name": "Conv_T_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Conv_T_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 40, 128]}}
�

�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "Conv_T_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Conv_T_5", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 80, 32]}}
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate%m�&m�0m�1m�?m�@m�Jm�Km�Ym�Zm�hm�im�wm�xm�~m�m�	�m�	�m�	�m�	�m�%v�&v�0v�1v�?v�@v�Jv�Kv�Yv�Zv�hv�iv�wv�xv�~v�v�	�v�	�v�	�v�	�v�"
	optimizer
�
%0
&1
02
13
?4
@5
J6
K7
Y8
Z9
h10
i11
w12
x13
~14
15
�16
�17
�18
�19"
trackable_list_wrapper
�
%0
&1
02
13
?4
@5
J6
K7
Y8
Z9
h10
i11
w12
x13
~14
15
�16
�17
�18
�19"
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
	variables
trainable_variables
�layers
�non_trainable_variables
regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
 	variables
!trainable_variables
�layers
�non_trainable_variables
"regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
':% 2Conv_1/kernel
: 2Conv_1/bias
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
'	variables
(trainable_variables
�layers
�non_trainable_variables
)regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
+	variables
,trainable_variables
�layers
�non_trainable_variables
-regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
':% @2Conv_2/kernel
:@2Conv_2/bias
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
2	variables
3trainable_variables
�layers
�non_trainable_variables
4regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
6	variables
7trainable_variables
�layers
�non_trainable_variables
8regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
:	variables
;trainable_variables
�layers
�non_trainable_variables
<regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
(:&@�2Conv_3/kernel
:�2Conv_3/bias
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
A	variables
Btrainable_variables
�layers
�non_trainable_variables
Cregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
E	variables
Ftrainable_variables
�layers
�non_trainable_variables
Gregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
):'��2Conv_4/kernel
:�2Conv_4/bias
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
L	variables
Mtrainable_variables
�layers
�non_trainable_variables
Nregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
P	variables
Qtrainable_variables
�layers
�non_trainable_variables
Rregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
T	variables
Utrainable_variables
�layers
�non_trainable_variables
Vregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
):'��2Conv_5/kernel
:�2Conv_5/bias
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
[	variables
\trainable_variables
�layers
�non_trainable_variables
]regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
_	variables
`trainable_variables
�layers
�non_trainable_variables
aregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
c	variables
dtrainable_variables
�layers
�non_trainable_variables
eregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
+:)��2Conv_T_1/kernel
:�2Conv_T_1/bias
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
j	variables
ktrainable_variables
�layers
�non_trainable_variables
lregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
n	variables
otrainable_variables
�layers
�non_trainable_variables
pregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
r	variables
strainable_variables
�layers
�non_trainable_variables
tregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
+:)��2Conv_T_2/kernel
:�2Conv_T_2/bias
.
w0
x1"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
y	variables
ztrainable_variables
�layers
�non_trainable_variables
{regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
*:(@�2Conv_T_3/kernel
:@2Conv_T_3/bias
.
~0
1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
*:( �2Conv_T_4/kernel
: 2Conv_T_4/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):' 2Conv_T_5/kernel
:2Conv_T_5/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
$0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
/0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
>0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
I0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
X0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
g0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
v0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
}0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 �layer_regularization_losses
�layer_metrics
�metrics
�	variables
�trainable_variables
�layers
�non_trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
,:* 2Adam/Conv_1/kernel/m
: 2Adam/Conv_1/bias/m
,:* @2Adam/Conv_2/kernel/m
:@2Adam/Conv_2/bias/m
-:+@�2Adam/Conv_3/kernel/m
:�2Adam/Conv_3/bias/m
.:,��2Adam/Conv_4/kernel/m
:�2Adam/Conv_4/bias/m
.:,��2Adam/Conv_5/kernel/m
:�2Adam/Conv_5/bias/m
0:.��2Adam/Conv_T_1/kernel/m
!:�2Adam/Conv_T_1/bias/m
0:.��2Adam/Conv_T_2/kernel/m
!:�2Adam/Conv_T_2/bias/m
/:-@�2Adam/Conv_T_3/kernel/m
 :@2Adam/Conv_T_3/bias/m
/:- �2Adam/Conv_T_4/kernel/m
 : 2Adam/Conv_T_4/bias/m
.:, 2Adam/Conv_T_5/kernel/m
 :2Adam/Conv_T_5/bias/m
,:* 2Adam/Conv_1/kernel/v
: 2Adam/Conv_1/bias/v
,:* @2Adam/Conv_2/kernel/v
:@2Adam/Conv_2/bias/v
-:+@�2Adam/Conv_3/kernel/v
:�2Adam/Conv_3/bias/v
.:,��2Adam/Conv_4/kernel/v
:�2Adam/Conv_4/bias/v
.:,��2Adam/Conv_5/kernel/v
:�2Adam/Conv_5/bias/v
0:.��2Adam/Conv_T_1/kernel/v
!:�2Adam/Conv_T_1/bias/v
0:.��2Adam/Conv_T_2/kernel/v
!:�2Adam/Conv_T_2/bias/v
/:-@�2Adam/Conv_T_3/kernel/v
 :@2Adam/Conv_T_3/bias/v
/:- �2Adam/Conv_T_4/kernel/v
 : 2Adam/Conv_T_4/bias/v
.:, 2Adam/Conv_T_5/kernel/v
 :2Adam/Conv_T_5/bias/v
�2�
 __inference__wrapped_model_27525�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *s�p
n�k
3�0
Img_Input_Layer�����������
4�1
Mask_Input_Layer�����������
�2�
:__inference_Facial_Landmark_Completion_layer_call_fn_29459
:__inference_Facial_Landmark_Completion_layer_call_fn_28795
:__inference_Facial_Landmark_Completion_layer_call_fn_28909
:__inference_Facial_Landmark_Completion_layer_call_fn_29413�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
U__inference_Facial_Landmark_Completion_layer_call_and_return_conditional_losses_28680
U__inference_Facial_Landmark_Completion_layer_call_and_return_conditional_losses_28612
U__inference_Facial_Landmark_Completion_layer_call_and_return_conditional_losses_29367
U__inference_Facial_Landmark_Completion_layer_call_and_return_conditional_losses_29211�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
3__inference_Concatenated_Inputs_layer_call_fn_29472�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
N__inference_Concatenated_Inputs_layer_call_and_return_conditional_losses_29466�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_Conv_1_layer_call_fn_29492�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_Conv_1_layer_call_and_return_conditional_losses_29483�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_Max_Pool_1_layer_call_fn_27537�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
E__inference_Max_Pool_1_layer_call_and_return_conditional_losses_27531�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
&__inference_Conv_2_layer_call_fn_29512�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_Conv_2_layer_call_and_return_conditional_losses_29503�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_Max_Pool_2_layer_call_fn_27549�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
E__inference_Max_Pool_2_layer_call_and_return_conditional_losses_27543�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
%__inference_SPD_1_layer_call_fn_29588
%__inference_SPD_1_layer_call_fn_29583
%__inference_SPD_1_layer_call_fn_29545
%__inference_SPD_1_layer_call_fn_29550�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
@__inference_SPD_1_layer_call_and_return_conditional_losses_29535
@__inference_SPD_1_layer_call_and_return_conditional_losses_29573
@__inference_SPD_1_layer_call_and_return_conditional_losses_29540
@__inference_SPD_1_layer_call_and_return_conditional_losses_29578�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
&__inference_Conv_3_layer_call_fn_29608�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_Conv_3_layer_call_and_return_conditional_losses_29599�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_Max_Pool_3_layer_call_fn_27629�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
E__inference_Max_Pool_3_layer_call_and_return_conditional_losses_27623�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
&__inference_Conv_4_layer_call_fn_29628�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_Conv_4_layer_call_and_return_conditional_losses_29619�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_Max_Pool_4_layer_call_fn_27641�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
E__inference_Max_Pool_4_layer_call_and_return_conditional_losses_27635�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
%__inference_SPD_2_layer_call_fn_29699
%__inference_SPD_2_layer_call_fn_29661
%__inference_SPD_2_layer_call_fn_29704
%__inference_SPD_2_layer_call_fn_29666�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
@__inference_SPD_2_layer_call_and_return_conditional_losses_29694
@__inference_SPD_2_layer_call_and_return_conditional_losses_29651
@__inference_SPD_2_layer_call_and_return_conditional_losses_29656
@__inference_SPD_2_layer_call_and_return_conditional_losses_29689�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
&__inference_Conv_5_layer_call_fn_29724�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_Conv_5_layer_call_and_return_conditional_losses_29715�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_Max_Pool_5_layer_call_fn_27721�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
E__inference_Max_Pool_5_layer_call_and_return_conditional_losses_27715�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
%__inference_SPD_3_layer_call_fn_29757
%__inference_SPD_3_layer_call_fn_29800
%__inference_SPD_3_layer_call_fn_29795
%__inference_SPD_3_layer_call_fn_29762�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
@__inference_SPD_3_layer_call_and_return_conditional_losses_29785
@__inference_SPD_3_layer_call_and_return_conditional_losses_29747
@__inference_SPD_3_layer_call_and_return_conditional_losses_29790
@__inference_SPD_3_layer_call_and_return_conditional_losses_29752�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_Conv_T_1_layer_call_fn_27846�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
C__inference_Conv_T_1_layer_call_and_return_conditional_losses_27836�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
(__inference_Concat_1_layer_call_fn_29813�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_Concat_1_layer_call_and_return_conditional_losses_29807�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_SPD_4_layer_call_fn_29884
%__inference_SPD_4_layer_call_fn_29846
%__inference_SPD_4_layer_call_fn_29889
%__inference_SPD_4_layer_call_fn_29851�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
@__inference_SPD_4_layer_call_and_return_conditional_losses_29836
@__inference_SPD_4_layer_call_and_return_conditional_losses_29841
@__inference_SPD_4_layer_call_and_return_conditional_losses_29874
@__inference_SPD_4_layer_call_and_return_conditional_losses_29879�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_Conv_T_2_layer_call_fn_27971�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
C__inference_Conv_T_2_layer_call_and_return_conditional_losses_27961�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
(__inference_Conv_T_3_layer_call_fn_28028�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
C__inference_Conv_T_3_layer_call_and_return_conditional_losses_28018�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
(__inference_Concat_2_layer_call_fn_29902�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_Concat_2_layer_call_and_return_conditional_losses_29896�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_SPD_5_layer_call_fn_29935
%__inference_SPD_5_layer_call_fn_29940
%__inference_SPD_5_layer_call_fn_29973
%__inference_SPD_5_layer_call_fn_29978�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
@__inference_SPD_5_layer_call_and_return_conditional_losses_29930
@__inference_SPD_5_layer_call_and_return_conditional_losses_29963
@__inference_SPD_5_layer_call_and_return_conditional_losses_29968
@__inference_SPD_5_layer_call_and_return_conditional_losses_29925�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_Conv_T_4_layer_call_fn_28153�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
C__inference_Conv_T_4_layer_call_and_return_conditional_losses_28143�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
(__inference_Conv_T_5_layer_call_fn_28198�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
C__inference_Conv_T_5_layer_call_and_return_conditional_losses_28188�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
JBH
#__inference_signature_wrapper_28965Img_Input_LayerMask_Input_Layer
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_leaky_re_lu_5_layer_call_fn_29988�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_29983�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_leaky_re_lu_6_layer_call_fn_29998�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_29993�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_leaky_re_lu_7_layer_call_fn_30008�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_30003�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_leaky_re_lu_8_layer_call_fn_30018�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_30013�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
C__inference_Concat_1_layer_call_and_return_conditional_losses_29807�~�{
t�q
o�l
=�:
inputs/0,����������������������������
+�(
inputs/1���������
�
� ".�+
$�!
0���������
�
� �
(__inference_Concat_1_layer_call_fn_29813�~�{
t�q
o�l
=�:
inputs/0,����������������������������
+�(
inputs/1���������
�
� "!����������
��
C__inference_Concat_2_layer_call_and_return_conditional_losses_29896�|�y
r�o
m�j
<�9
inputs/0+���������������������������@
*�'
inputs/1���������0(@
� ".�+
$�!
0���������0(�
� �
(__inference_Concat_2_layer_call_fn_29902�|�y
r�o
m�j
<�9
inputs/0+���������������������������@
*�'
inputs/1���������0(@
� "!����������0(��
N__inference_Concatenated_Inputs_layer_call_and_return_conditional_losses_29466�n�k
d�a
_�\
,�)
inputs/0�����������
,�)
inputs/1�����������
� "/�,
%�"
0�����������
� �
3__inference_Concatenated_Inputs_layer_call_fn_29472�n�k
d�a
_�\
,�)
inputs/0�����������
,�)
inputs/1�����������
� ""�������������
A__inference_Conv_1_layer_call_and_return_conditional_losses_29483p%&9�6
/�,
*�'
inputs�����������
� "/�,
%�"
0����������� 
� �
&__inference_Conv_1_layer_call_fn_29492c%&9�6
/�,
*�'
inputs�����������
� ""������������ �
A__inference_Conv_2_layer_call_and_return_conditional_losses_29503l017�4
-�*
(�%
inputs���������`P 
� "-�*
#� 
0���������`P@
� �
&__inference_Conv_2_layer_call_fn_29512_017�4
-�*
(�%
inputs���������`P 
� " ����������`P@�
A__inference_Conv_3_layer_call_and_return_conditional_losses_29599m?@7�4
-�*
(�%
inputs���������0(@
� ".�+
$�!
0���������0(�
� �
&__inference_Conv_3_layer_call_fn_29608`?@7�4
-�*
(�%
inputs���������0(@
� "!����������0(��
A__inference_Conv_4_layer_call_and_return_conditional_losses_29619nJK8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
&__inference_Conv_4_layer_call_fn_29628aJK8�5
.�+
)�&
inputs����������
� "!������������
A__inference_Conv_5_layer_call_and_return_conditional_losses_29715nYZ8�5
.�+
)�&
inputs���������
�
� ".�+
$�!
0���������
�
� �
&__inference_Conv_5_layer_call_fn_29724aYZ8�5
.�+
)�&
inputs���������
�
� "!����������
��
C__inference_Conv_T_1_layer_call_and_return_conditional_losses_27836�hiJ�G
@�=
;�8
inputs,����������������������������
� "@�=
6�3
0,����������������������������
� �
(__inference_Conv_T_1_layer_call_fn_27846�hiJ�G
@�=
;�8
inputs,����������������������������
� "3�0,�����������������������������
C__inference_Conv_T_2_layer_call_and_return_conditional_losses_27961�wxJ�G
@�=
;�8
inputs,����������������������������
� "@�=
6�3
0,����������������������������
� �
(__inference_Conv_T_2_layer_call_fn_27971�wxJ�G
@�=
;�8
inputs,����������������������������
� "3�0,�����������������������������
C__inference_Conv_T_3_layer_call_and_return_conditional_losses_28018�~J�G
@�=
;�8
inputs,����������������������������
� "?�<
5�2
0+���������������������������@
� �
(__inference_Conv_T_3_layer_call_fn_28028�~J�G
@�=
;�8
inputs,����������������������������
� "2�/+���������������������������@�
C__inference_Conv_T_4_layer_call_and_return_conditional_losses_28143���J�G
@�=
;�8
inputs,����������������������������
� "?�<
5�2
0+��������������������������� 
� �
(__inference_Conv_T_4_layer_call_fn_28153���J�G
@�=
;�8
inputs,����������������������������
� "2�/+��������������������������� �
C__inference_Conv_T_5_layer_call_and_return_conditional_losses_28188���I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+���������������������������
� �
(__inference_Conv_T_5_layer_call_fn_28198���I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+����������������������������
U__inference_Facial_Landmark_Completion_layer_call_and_return_conditional_losses_28612�%&01?@JKYZhiwx~�������
{�x
n�k
3�0
Img_Input_Layer�����������
4�1
Mask_Input_Layer�����������
p

 
� "?�<
5�2
0+���������������������������
� �
U__inference_Facial_Landmark_Completion_layer_call_and_return_conditional_losses_28680�%&01?@JKYZhiwx~�������
{�x
n�k
3�0
Img_Input_Layer�����������
4�1
Mask_Input_Layer�����������
p 

 
� "?�<
5�2
0+���������������������������
� �
U__inference_Facial_Landmark_Completion_layer_call_and_return_conditional_losses_29211�%&01?@JKYZhiwx~����v�s
l�i
_�\
,�)
inputs/0�����������
,�)
inputs/1�����������
p

 
� "/�,
%�"
0�����������
� �
U__inference_Facial_Landmark_Completion_layer_call_and_return_conditional_losses_29367�%&01?@JKYZhiwx~����v�s
l�i
_�\
,�)
inputs/0�����������
,�)
inputs/1�����������
p 

 
� "/�,
%�"
0�����������
� �
:__inference_Facial_Landmark_Completion_layer_call_fn_28795�%&01?@JKYZhiwx~�������
{�x
n�k
3�0
Img_Input_Layer�����������
4�1
Mask_Input_Layer�����������
p

 
� "2�/+����������������������������
:__inference_Facial_Landmark_Completion_layer_call_fn_28909�%&01?@JKYZhiwx~�������
{�x
n�k
3�0
Img_Input_Layer�����������
4�1
Mask_Input_Layer�����������
p 

 
� "2�/+����������������������������
:__inference_Facial_Landmark_Completion_layer_call_fn_29413�%&01?@JKYZhiwx~����v�s
l�i
_�\
,�)
inputs/0�����������
,�)
inputs/1�����������
p

 
� "2�/+����������������������������
:__inference_Facial_Landmark_Completion_layer_call_fn_29459�%&01?@JKYZhiwx~����v�s
l�i
_�\
,�)
inputs/0�����������
,�)
inputs/1�����������
p 

 
� "2�/+����������������������������
E__inference_Max_Pool_1_layer_call_and_return_conditional_losses_27531�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
*__inference_Max_Pool_1_layer_call_fn_27537�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
E__inference_Max_Pool_2_layer_call_and_return_conditional_losses_27543�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
*__inference_Max_Pool_2_layer_call_fn_27549�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
E__inference_Max_Pool_3_layer_call_and_return_conditional_losses_27623�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
*__inference_Max_Pool_3_layer_call_fn_27629�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
E__inference_Max_Pool_4_layer_call_and_return_conditional_losses_27635�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
*__inference_Max_Pool_4_layer_call_fn_27641�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
E__inference_Max_Pool_5_layer_call_and_return_conditional_losses_27715�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
*__inference_Max_Pool_5_layer_call_fn_27721�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
@__inference_SPD_1_layer_call_and_return_conditional_losses_29535l;�8
1�.
(�%
inputs���������0(@
p
� "-�*
#� 
0���������0(@
� �
@__inference_SPD_1_layer_call_and_return_conditional_losses_29540l;�8
1�.
(�%
inputs���������0(@
p 
� "-�*
#� 
0���������0(@
� �
@__inference_SPD_1_layer_call_and_return_conditional_losses_29573�V�S
L�I
C�@
inputs4������������������������������������
p
� "H�E
>�;
04������������������������������������
� �
@__inference_SPD_1_layer_call_and_return_conditional_losses_29578�V�S
L�I
C�@
inputs4������������������������������������
p 
� "H�E
>�;
04������������������������������������
� �
%__inference_SPD_1_layer_call_fn_29545_;�8
1�.
(�%
inputs���������0(@
p
� " ����������0(@�
%__inference_SPD_1_layer_call_fn_29550_;�8
1�.
(�%
inputs���������0(@
p 
� " ����������0(@�
%__inference_SPD_1_layer_call_fn_29583�V�S
L�I
C�@
inputs4������������������������������������
p
� ";�84�������������������������������������
%__inference_SPD_1_layer_call_fn_29588�V�S
L�I
C�@
inputs4������������������������������������
p 
� ";�84�������������������������������������
@__inference_SPD_2_layer_call_and_return_conditional_losses_29651�V�S
L�I
C�@
inputs4������������������������������������
p
� "H�E
>�;
04������������������������������������
� �
@__inference_SPD_2_layer_call_and_return_conditional_losses_29656�V�S
L�I
C�@
inputs4������������������������������������
p 
� "H�E
>�;
04������������������������������������
� �
@__inference_SPD_2_layer_call_and_return_conditional_losses_29689n<�9
2�/
)�&
inputs���������
�
p
� ".�+
$�!
0���������
�
� �
@__inference_SPD_2_layer_call_and_return_conditional_losses_29694n<�9
2�/
)�&
inputs���������
�
p 
� ".�+
$�!
0���������
�
� �
%__inference_SPD_2_layer_call_fn_29661�V�S
L�I
C�@
inputs4������������������������������������
p
� ";�84�������������������������������������
%__inference_SPD_2_layer_call_fn_29666�V�S
L�I
C�@
inputs4������������������������������������
p 
� ";�84�������������������������������������
%__inference_SPD_2_layer_call_fn_29699a<�9
2�/
)�&
inputs���������
�
p
� "!����������
��
%__inference_SPD_2_layer_call_fn_29704a<�9
2�/
)�&
inputs���������
�
p 
� "!����������
��
@__inference_SPD_3_layer_call_and_return_conditional_losses_29747n<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
@__inference_SPD_3_layer_call_and_return_conditional_losses_29752n<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� �
@__inference_SPD_3_layer_call_and_return_conditional_losses_29785�V�S
L�I
C�@
inputs4������������������������������������
p
� "H�E
>�;
04������������������������������������
� �
@__inference_SPD_3_layer_call_and_return_conditional_losses_29790�V�S
L�I
C�@
inputs4������������������������������������
p 
� "H�E
>�;
04������������������������������������
� �
%__inference_SPD_3_layer_call_fn_29757a<�9
2�/
)�&
inputs����������
p
� "!������������
%__inference_SPD_3_layer_call_fn_29762a<�9
2�/
)�&
inputs����������
p 
� "!������������
%__inference_SPD_3_layer_call_fn_29795�V�S
L�I
C�@
inputs4������������������������������������
p
� ";�84�������������������������������������
%__inference_SPD_3_layer_call_fn_29800�V�S
L�I
C�@
inputs4������������������������������������
p 
� ";�84�������������������������������������
@__inference_SPD_4_layer_call_and_return_conditional_losses_29836�V�S
L�I
C�@
inputs4������������������������������������
p
� "H�E
>�;
04������������������������������������
� �
@__inference_SPD_4_layer_call_and_return_conditional_losses_29841�V�S
L�I
C�@
inputs4������������������������������������
p 
� "H�E
>�;
04������������������������������������
� �
@__inference_SPD_4_layer_call_and_return_conditional_losses_29874n<�9
2�/
)�&
inputs���������
�
p
� ".�+
$�!
0���������
�
� �
@__inference_SPD_4_layer_call_and_return_conditional_losses_29879n<�9
2�/
)�&
inputs���������
�
p 
� ".�+
$�!
0���������
�
� �
%__inference_SPD_4_layer_call_fn_29846�V�S
L�I
C�@
inputs4������������������������������������
p
� ";�84�������������������������������������
%__inference_SPD_4_layer_call_fn_29851�V�S
L�I
C�@
inputs4������������������������������������
p 
� ";�84�������������������������������������
%__inference_SPD_4_layer_call_fn_29884a<�9
2�/
)�&
inputs���������
�
p
� "!����������
��
%__inference_SPD_4_layer_call_fn_29889a<�9
2�/
)�&
inputs���������
�
p 
� "!����������
��
@__inference_SPD_5_layer_call_and_return_conditional_losses_29925n<�9
2�/
)�&
inputs���������0(�
p
� ".�+
$�!
0���������0(�
� �
@__inference_SPD_5_layer_call_and_return_conditional_losses_29930n<�9
2�/
)�&
inputs���������0(�
p 
� ".�+
$�!
0���������0(�
� �
@__inference_SPD_5_layer_call_and_return_conditional_losses_29963�V�S
L�I
C�@
inputs4������������������������������������
p
� "H�E
>�;
04������������������������������������
� �
@__inference_SPD_5_layer_call_and_return_conditional_losses_29968�V�S
L�I
C�@
inputs4������������������������������������
p 
� "H�E
>�;
04������������������������������������
� �
%__inference_SPD_5_layer_call_fn_29935a<�9
2�/
)�&
inputs���������0(�
p
� "!����������0(��
%__inference_SPD_5_layer_call_fn_29940a<�9
2�/
)�&
inputs���������0(�
p 
� "!����������0(��
%__inference_SPD_5_layer_call_fn_29973�V�S
L�I
C�@
inputs4������������������������������������
p
� ";�84�������������������������������������
%__inference_SPD_5_layer_call_fn_29978�V�S
L�I
C�@
inputs4������������������������������������
p 
� ";�84�������������������������������������
 __inference__wrapped_model_27525�%&01?@JKYZhiwx~����}�z
s�p
n�k
3�0
Img_Input_Layer�����������
4�1
Mask_Input_Layer�����������
� "=�:
8
Conv_T_5,�)
Conv_T_5������������
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_29983�J�G
@�=
;�8
inputs,����������������������������
� "@�=
6�3
0,����������������������������
� �
-__inference_leaky_re_lu_5_layer_call_fn_29988�J�G
@�=
;�8
inputs,����������������������������
� "3�0,�����������������������������
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_29993�J�G
@�=
;�8
inputs,����������������������������
� "@�=
6�3
0,����������������������������
� �
-__inference_leaky_re_lu_6_layer_call_fn_29998�J�G
@�=
;�8
inputs,����������������������������
� "3�0,�����������������������������
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_30003�I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������@
� �
-__inference_leaky_re_lu_7_layer_call_fn_30008I�F
?�<
:�7
inputs+���������������������������@
� "2�/+���������������������������@�
H__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_30013�I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+��������������������������� 
� �
-__inference_leaky_re_lu_8_layer_call_fn_30018I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+��������������������������� �
#__inference_signature_wrapper_28965�%&01?@JKYZhiwx~�������
� 
���
F
Img_Input_Layer3�0
Img_Input_Layer�����������
H
Mask_Input_Layer4�1
Mask_Input_Layer�����������"=�:
8
Conv_T_5,�)
Conv_T_5�����������