аЌ3
—£
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
dtypetypeИ
Њ
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
executor_typestring И
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.3.02v2.3.0-rc2-23-gb36436b0878ег(
Ж
Gen_Conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameGen_Conv_1/kernel

%Gen_Conv_1/kernel/Read/ReadVariableOpReadVariableOpGen_Conv_1/kernel*&
_output_shapes
:@*
dtype0
v
Gen_Conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameGen_Conv_1/bias
o
#Gen_Conv_1/bias/Read/ReadVariableOpReadVariableOpGen_Conv_1/bias*
_output_shapes
:@*
dtype0
t
Gen_BN_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameGen_BN_1/gamma
m
"Gen_BN_1/gamma/Read/ReadVariableOpReadVariableOpGen_BN_1/gamma*
_output_shapes
:@*
dtype0
r
Gen_BN_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameGen_BN_1/beta
k
!Gen_BN_1/beta/Read/ReadVariableOpReadVariableOpGen_BN_1/beta*
_output_shapes
:@*
dtype0
А
Gen_BN_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameGen_BN_1/moving_mean
y
(Gen_BN_1/moving_mean/Read/ReadVariableOpReadVariableOpGen_BN_1/moving_mean*
_output_shapes
:@*
dtype0
И
Gen_BN_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameGen_BN_1/moving_variance
Б
,Gen_BN_1/moving_variance/Read/ReadVariableOpReadVariableOpGen_BN_1/moving_variance*
_output_shapes
:@*
dtype0
З
Gen_Conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*"
shared_nameGen_Conv_2/kernel
А
%Gen_Conv_2/kernel/Read/ReadVariableOpReadVariableOpGen_Conv_2/kernel*'
_output_shapes
:@А*
dtype0
w
Gen_Conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А* 
shared_nameGen_Conv_2/bias
p
#Gen_Conv_2/bias/Read/ReadVariableOpReadVariableOpGen_Conv_2/bias*
_output_shapes	
:А*
dtype0
u
Gen_BN_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameGen_BN_2/gamma
n
"Gen_BN_2/gamma/Read/ReadVariableOpReadVariableOpGen_BN_2/gamma*
_output_shapes	
:А*
dtype0
s
Gen_BN_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameGen_BN_2/beta
l
!Gen_BN_2/beta/Read/ReadVariableOpReadVariableOpGen_BN_2/beta*
_output_shapes	
:А*
dtype0
Б
Gen_BN_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameGen_BN_2/moving_mean
z
(Gen_BN_2/moving_mean/Read/ReadVariableOpReadVariableOpGen_BN_2/moving_mean*
_output_shapes	
:А*
dtype0
Й
Gen_BN_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_nameGen_BN_2/moving_variance
В
,Gen_BN_2/moving_variance/Read/ReadVariableOpReadVariableOpGen_BN_2/moving_variance*
_output_shapes	
:А*
dtype0
И
Gen_Conv_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*"
shared_nameGen_Conv_3/kernel
Б
%Gen_Conv_3/kernel/Read/ReadVariableOpReadVariableOpGen_Conv_3/kernel*(
_output_shapes
:АА*
dtype0
w
Gen_Conv_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А* 
shared_nameGen_Conv_3/bias
p
#Gen_Conv_3/bias/Read/ReadVariableOpReadVariableOpGen_Conv_3/bias*
_output_shapes	
:А*
dtype0
u
Gen_BN_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameGen_BN_3/gamma
n
"Gen_BN_3/gamma/Read/ReadVariableOpReadVariableOpGen_BN_3/gamma*
_output_shapes	
:А*
dtype0
s
Gen_BN_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameGen_BN_3/beta
l
!Gen_BN_3/beta/Read/ReadVariableOpReadVariableOpGen_BN_3/beta*
_output_shapes	
:А*
dtype0
Б
Gen_BN_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameGen_BN_3/moving_mean
z
(Gen_BN_3/moving_mean/Read/ReadVariableOpReadVariableOpGen_BN_3/moving_mean*
_output_shapes	
:А*
dtype0
Й
Gen_BN_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_nameGen_BN_3/moving_variance
В
,Gen_BN_3/moving_variance/Read/ReadVariableOpReadVariableOpGen_BN_3/moving_variance*
_output_shapes	
:А*
dtype0
И
Gen_Conv_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*"
shared_nameGen_Conv_4/kernel
Б
%Gen_Conv_4/kernel/Read/ReadVariableOpReadVariableOpGen_Conv_4/kernel*(
_output_shapes
:АА*
dtype0
w
Gen_Conv_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А* 
shared_nameGen_Conv_4/bias
p
#Gen_Conv_4/bias/Read/ReadVariableOpReadVariableOpGen_Conv_4/bias*
_output_shapes	
:А*
dtype0
u
Gen_BN_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameGen_BN_4/gamma
n
"Gen_BN_4/gamma/Read/ReadVariableOpReadVariableOpGen_BN_4/gamma*
_output_shapes	
:А*
dtype0
s
Gen_BN_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameGen_BN_4/beta
l
!Gen_BN_4/beta/Read/ReadVariableOpReadVariableOpGen_BN_4/beta*
_output_shapes	
:А*
dtype0
Б
Gen_BN_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameGen_BN_4/moving_mean
z
(Gen_BN_4/moving_mean/Read/ReadVariableOpReadVariableOpGen_BN_4/moving_mean*
_output_shapes	
:А*
dtype0
Й
Gen_BN_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_nameGen_BN_4/moving_variance
В
,Gen_BN_4/moving_variance/Read/ReadVariableOpReadVariableOpGen_BN_4/moving_variance*
_output_shapes	
:А*
dtype0
И
Gen_Conv_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*"
shared_nameGen_Conv_5/kernel
Б
%Gen_Conv_5/kernel/Read/ReadVariableOpReadVariableOpGen_Conv_5/kernel*(
_output_shapes
:АА*
dtype0
w
Gen_Conv_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А* 
shared_nameGen_Conv_5/bias
p
#Gen_Conv_5/bias/Read/ReadVariableOpReadVariableOpGen_Conv_5/bias*
_output_shapes	
:А*
dtype0
u
Gen_BN_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameGen_BN_5/gamma
n
"Gen_BN_5/gamma/Read/ReadVariableOpReadVariableOpGen_BN_5/gamma*
_output_shapes	
:А*
dtype0
s
Gen_BN_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameGen_BN_5/beta
l
!Gen_BN_5/beta/Read/ReadVariableOpReadVariableOpGen_BN_5/beta*
_output_shapes	
:А*
dtype0
Б
Gen_BN_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameGen_BN_5/moving_mean
z
(Gen_BN_5/moving_mean/Read/ReadVariableOpReadVariableOpGen_BN_5/moving_mean*
_output_shapes	
:А*
dtype0
Й
Gen_BN_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_nameGen_BN_5/moving_variance
В
,Gen_BN_5/moving_variance/Read/ReadVariableOpReadVariableOpGen_BN_5/moving_variance*
_output_shapes	
:А*
dtype0
М
Gen_Conv_T_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameGen_Conv_T_1/kernel
Е
'Gen_Conv_T_1/kernel/Read/ReadVariableOpReadVariableOpGen_Conv_T_1/kernel*(
_output_shapes
:АА*
dtype0
{
Gen_Conv_T_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameGen_Conv_T_1/bias
t
%Gen_Conv_T_1/bias/Read/ReadVariableOpReadVariableOpGen_Conv_T_1/bias*
_output_shapes	
:А*
dtype0
u
Gen_BN_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameGen_BN_6/gamma
n
"Gen_BN_6/gamma/Read/ReadVariableOpReadVariableOpGen_BN_6/gamma*
_output_shapes	
:А*
dtype0
s
Gen_BN_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameGen_BN_6/beta
l
!Gen_BN_6/beta/Read/ReadVariableOpReadVariableOpGen_BN_6/beta*
_output_shapes	
:А*
dtype0
Б
Gen_BN_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameGen_BN_6/moving_mean
z
(Gen_BN_6/moving_mean/Read/ReadVariableOpReadVariableOpGen_BN_6/moving_mean*
_output_shapes	
:А*
dtype0
Й
Gen_BN_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_nameGen_BN_6/moving_variance
В
,Gen_BN_6/moving_variance/Read/ReadVariableOpReadVariableOpGen_BN_6/moving_variance*
_output_shapes	
:А*
dtype0
М
Gen_Conv_T_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameGen_Conv_T_2/kernel
Е
'Gen_Conv_T_2/kernel/Read/ReadVariableOpReadVariableOpGen_Conv_T_2/kernel*(
_output_shapes
:АА*
dtype0
{
Gen_Conv_T_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameGen_Conv_T_2/bias
t
%Gen_Conv_T_2/bias/Read/ReadVariableOpReadVariableOpGen_Conv_T_2/bias*
_output_shapes	
:А*
dtype0
u
Gen_BN_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameGen_BN_7/gamma
n
"Gen_BN_7/gamma/Read/ReadVariableOpReadVariableOpGen_BN_7/gamma*
_output_shapes	
:А*
dtype0
s
Gen_BN_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameGen_BN_7/beta
l
!Gen_BN_7/beta/Read/ReadVariableOpReadVariableOpGen_BN_7/beta*
_output_shapes	
:А*
dtype0
Б
Gen_BN_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameGen_BN_7/moving_mean
z
(Gen_BN_7/moving_mean/Read/ReadVariableOpReadVariableOpGen_BN_7/moving_mean*
_output_shapes	
:А*
dtype0
Й
Gen_BN_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_nameGen_BN_7/moving_variance
В
,Gen_BN_7/moving_variance/Read/ReadVariableOpReadVariableOpGen_BN_7/moving_variance*
_output_shapes	
:А*
dtype0
М
Gen_Conv_T_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameGen_Conv_T_3/kernel
Е
'Gen_Conv_T_3/kernel/Read/ReadVariableOpReadVariableOpGen_Conv_T_3/kernel*(
_output_shapes
:АА*
dtype0
{
Gen_Conv_T_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameGen_Conv_T_3/bias
t
%Gen_Conv_T_3/bias/Read/ReadVariableOpReadVariableOpGen_Conv_T_3/bias*
_output_shapes	
:А*
dtype0
u
Gen_BN_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameGen_BN_8/gamma
n
"Gen_BN_8/gamma/Read/ReadVariableOpReadVariableOpGen_BN_8/gamma*
_output_shapes	
:А*
dtype0
s
Gen_BN_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameGen_BN_8/beta
l
!Gen_BN_8/beta/Read/ReadVariableOpReadVariableOpGen_BN_8/beta*
_output_shapes	
:А*
dtype0
Б
Gen_BN_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameGen_BN_8/moving_mean
z
(Gen_BN_8/moving_mean/Read/ReadVariableOpReadVariableOpGen_BN_8/moving_mean*
_output_shapes	
:А*
dtype0
Й
Gen_BN_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_nameGen_BN_8/moving_variance
В
,Gen_BN_8/moving_variance/Read/ReadVariableOpReadVariableOpGen_BN_8/moving_variance*
_output_shapes	
:А*
dtype0
Л
Gen_Conv_T_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*$
shared_nameGen_Conv_T_4/kernel
Д
'Gen_Conv_T_4/kernel/Read/ReadVariableOpReadVariableOpGen_Conv_T_4/kernel*'
_output_shapes
:@А*
dtype0
z
Gen_Conv_T_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameGen_Conv_T_4/bias
s
%Gen_Conv_T_4/bias/Read/ReadVariableOpReadVariableOpGen_Conv_T_4/bias*
_output_shapes
:@*
dtype0
t
Gen_BN_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameGen_BN_9/gamma
m
"Gen_BN_9/gamma/Read/ReadVariableOpReadVariableOpGen_BN_9/gamma*
_output_shapes
:@*
dtype0
r
Gen_BN_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameGen_BN_9/beta
k
!Gen_BN_9/beta/Read/ReadVariableOpReadVariableOpGen_BN_9/beta*
_output_shapes
:@*
dtype0
А
Gen_BN_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameGen_BN_9/moving_mean
y
(Gen_BN_9/moving_mean/Read/ReadVariableOpReadVariableOpGen_BN_9/moving_mean*
_output_shapes
:@*
dtype0
И
Gen_BN_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameGen_BN_9/moving_variance
Б
,Gen_BN_9/moving_variance/Read/ReadVariableOpReadVariableOpGen_BN_9/moving_variance*
_output_shapes
:@*
dtype0
Л
Gen_Conv_T_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*$
shared_nameGen_Conv_T_5/kernel
Д
'Gen_Conv_T_5/kernel/Read/ReadVariableOpReadVariableOpGen_Conv_T_5/kernel*'
_output_shapes
: А*
dtype0
z
Gen_Conv_T_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameGen_Conv_T_5/bias
s
%Gen_Conv_T_5/bias/Read/ReadVariableOpReadVariableOpGen_Conv_T_5/bias*
_output_shapes
: *
dtype0
v
Gen_BN_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameGen_BN_10/gamma
o
#Gen_BN_10/gamma/Read/ReadVariableOpReadVariableOpGen_BN_10/gamma*
_output_shapes
: *
dtype0
t
Gen_BN_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameGen_BN_10/beta
m
"Gen_BN_10/beta/Read/ReadVariableOpReadVariableOpGen_BN_10/beta*
_output_shapes
: *
dtype0
В
Gen_BN_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameGen_BN_10/moving_mean
{
)Gen_BN_10/moving_mean/Read/ReadVariableOpReadVariableOpGen_BN_10/moving_mean*
_output_shapes
: *
dtype0
К
Gen_BN_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameGen_BN_10/moving_variance
Г
-Gen_BN_10/moving_variance/Read/ReadVariableOpReadVariableOpGen_BN_10/moving_variance*
_output_shapes
: *
dtype0
К
Gen_Conv_T_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameGen_Conv_T_6/kernel
Г
'Gen_Conv_T_6/kernel/Read/ReadVariableOpReadVariableOpGen_Conv_T_6/kernel*&
_output_shapes
: *
dtype0
z
Gen_Conv_T_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameGen_Conv_T_6/bias
s
%Gen_Conv_T_6/bias/Read/ReadVariableOpReadVariableOpGen_Conv_T_6/bias*
_output_shapes
:*
dtype0

NoOpNoOp
љЇ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*чє
valueмєBиє Bає
э
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer-14
layer_with_weights-8
layer-15
layer-16
layer_with_weights-9
layer-17
layer-18
layer_with_weights-10
layer-19
layer_with_weights-11
layer-20
layer-21
layer-22
layer_with_weights-12
layer-23
layer_with_weights-13
layer-24
layer-25
layer_with_weights-14
layer-26
layer_with_weights-15
layer-27
layer-28
layer-29
layer_with_weights-16
layer-30
 layer_with_weights-17
 layer-31
!layer-32
"layer_with_weights-18
"layer-33
#layer_with_weights-19
#layer-34
$layer_with_weights-20
$layer-35
%	variables
&regularization_losses
'trainable_variables
(	keras_api
)
signatures
 
x
*
activation

+kernel
,bias
-	variables
.regularization_losses
/trainable_variables
0	keras_api
R
1	variables
2regularization_losses
3trainable_variables
4	keras_api
Ч
5axis
	6gamma
7beta
8moving_mean
9moving_variance
:	variables
;regularization_losses
<trainable_variables
=	keras_api
x
>
activation

?kernel
@bias
A	variables
Bregularization_losses
Ctrainable_variables
D	keras_api
R
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
Ч
Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
R
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
x
V
activation

Wkernel
Xbias
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
R
]	variables
^regularization_losses
_trainable_variables
`	keras_api
Ч
aaxis
	bgamma
cbeta
dmoving_mean
emoving_variance
f	variables
gregularization_losses
htrainable_variables
i	keras_api
x
j
activation

kkernel
lbias
m	variables
nregularization_losses
otrainable_variables
p	keras_api
R
q	variables
rregularization_losses
strainable_variables
t	keras_api
Ч
uaxis
	vgamma
wbeta
xmoving_mean
ymoving_variance
z	variables
{regularization_losses
|trainable_variables
}	keras_api
T
~	variables
regularization_losses
Аtrainable_variables
Б	keras_api

В
activation
Гkernel
	Дbias
Е	variables
Жregularization_losses
Зtrainable_variables
И	keras_api
V
Й	variables
Кregularization_losses
Лtrainable_variables
М	keras_api
†
	Нaxis

Оgamma
	Пbeta
Рmoving_mean
Сmoving_variance
Т	variables
Уregularization_losses
Фtrainable_variables
Х	keras_api
V
Ц	variables
Чregularization_losses
Шtrainable_variables
Щ	keras_api

Ъ
activation
Ыkernel
	Ьbias
Э	variables
Юregularization_losses
Яtrainable_variables
†	keras_api
†
	°axis

Ґgamma
	£beta
§moving_mean
•moving_variance
¶	variables
Іregularization_losses
®trainable_variables
©	keras_api
V
™	variables
Ђregularization_losses
ђtrainable_variables
≠	keras_api
V
Ѓ	variables
ѓregularization_losses
∞trainable_variables
±	keras_api

≤
activation
≥kernel
	іbias
µ	variables
ґregularization_losses
Јtrainable_variables
Є	keras_api
†
	єaxis

Їgamma
	їbeta
Љmoving_mean
љmoving_variance
Њ	variables
њregularization_losses
јtrainable_variables
Ѕ	keras_api
V
¬	variables
√regularization_losses
ƒtrainable_variables
≈	keras_api

∆
activation
«kernel
	»bias
…	variables
 regularization_losses
Ћtrainable_variables
ћ	keras_api
†
	Ќaxis

ќgamma
	ѕbeta
–moving_mean
—moving_variance
“	variables
”regularization_losses
‘trainable_variables
’	keras_api
V
÷	variables
„regularization_losses
Ўtrainable_variables
ў	keras_api
V
Џ	variables
џregularization_losses
№trainable_variables
Ё	keras_api

ё
activation
яkernel
	аbias
б	variables
вregularization_losses
гtrainable_variables
д	keras_api
†
	еaxis

жgamma
	зbeta
иmoving_mean
йmoving_variance
к	variables
лregularization_losses
мtrainable_variables
н	keras_api
V
о	variables
пregularization_losses
рtrainable_variables
с	keras_api

т
activation
уkernel
	фbias
х	variables
цregularization_losses
чtrainable_variables
ш	keras_api
†
	щaxis

ъgamma
	ыbeta
ьmoving_mean
эmoving_variance
ю	variables
€regularization_losses
Аtrainable_variables
Б	keras_api
n
Вkernel
	Гbias
Д	variables
Еregularization_losses
Жtrainable_variables
З	keras_api
М
+0
,1
62
73
84
95
?6
@7
J8
K9
L10
M11
W12
X13
b14
c15
d16
e17
k18
l19
v20
w21
x22
y23
Г24
Д25
О26
П27
Р28
С29
Ы30
Ь31
Ґ32
£33
§34
•35
≥36
і37
Ї38
ї39
Љ40
љ41
«42
»43
ќ44
ѕ45
–46
—47
я48
а49
ж50
з51
и52
й53
у54
ф55
ъ56
ы57
ь58
э59
В60
Г61
 
а
+0
,1
62
73
?4
@5
J6
K7
W8
X9
b10
c11
k12
l13
v14
w15
Г16
Д17
О18
П19
Ы20
Ь21
Ґ22
£23
≥24
і25
Ї26
ї27
«28
»29
ќ30
ѕ31
я32
а33
ж34
з35
у36
ф37
ъ38
ы39
В40
Г41
≤
%	variables
 Иlayer_regularization_losses
Йlayer_metrics
Кmetrics
Лlayers
Мnon_trainable_variables
&regularization_losses
'trainable_variables
 
V
Н	variables
Оregularization_losses
Пtrainable_variables
Р	keras_api
][
VARIABLE_VALUEGen_Conv_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEGen_Conv_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1
 

+0
,1
≤
-	variables
 Сlayer_regularization_losses
Тlayer_metrics
Уmetrics
Фlayers
Хnon_trainable_variables
.regularization_losses
/trainable_variables
 
 
 
≤
1	variables
 Цlayer_regularization_losses
Чlayer_metrics
Шmetrics
Щlayers
Ъnon_trainable_variables
2regularization_losses
3trainable_variables
 
YW
VARIABLE_VALUEGen_BN_1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEGen_BN_1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEGen_BN_1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEGen_BN_1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

60
71
82
93
 

60
71
≤
:	variables
 Ыlayer_regularization_losses
Ьlayer_metrics
Эmetrics
Юlayers
Яnon_trainable_variables
;regularization_losses
<trainable_variables
V
†	variables
°regularization_losses
Ґtrainable_variables
£	keras_api
][
VARIABLE_VALUEGen_Conv_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEGen_Conv_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
@1
 

?0
@1
≤
A	variables
 §layer_regularization_losses
•layer_metrics
¶metrics
Іlayers
®non_trainable_variables
Bregularization_losses
Ctrainable_variables
 
 
 
≤
E	variables
 ©layer_regularization_losses
™layer_metrics
Ђmetrics
ђlayers
≠non_trainable_variables
Fregularization_losses
Gtrainable_variables
 
YW
VARIABLE_VALUEGen_BN_2/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEGen_BN_2/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEGen_BN_2/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEGen_BN_2/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

J0
K1
L2
M3
 

J0
K1
≤
N	variables
 Ѓlayer_regularization_losses
ѓlayer_metrics
∞metrics
±layers
≤non_trainable_variables
Oregularization_losses
Ptrainable_variables
 
 
 
≤
R	variables
 ≥layer_regularization_losses
іlayer_metrics
µmetrics
ґlayers
Јnon_trainable_variables
Sregularization_losses
Ttrainable_variables
V
Є	variables
єregularization_losses
Їtrainable_variables
ї	keras_api
][
VARIABLE_VALUEGen_Conv_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEGen_Conv_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

W0
X1
 

W0
X1
≤
Y	variables
 Љlayer_regularization_losses
љlayer_metrics
Њmetrics
њlayers
јnon_trainable_variables
Zregularization_losses
[trainable_variables
 
 
 
≤
]	variables
 Ѕlayer_regularization_losses
¬layer_metrics
√metrics
ƒlayers
≈non_trainable_variables
^regularization_losses
_trainable_variables
 
YW
VARIABLE_VALUEGen_BN_3/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEGen_BN_3/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEGen_BN_3/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEGen_BN_3/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

b0
c1
d2
e3
 

b0
c1
≤
f	variables
 ∆layer_regularization_losses
«layer_metrics
»metrics
…layers
 non_trainable_variables
gregularization_losses
htrainable_variables
V
Ћ	variables
ћregularization_losses
Ќtrainable_variables
ќ	keras_api
][
VARIABLE_VALUEGen_Conv_4/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEGen_Conv_4/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

k0
l1
 

k0
l1
≤
m	variables
 ѕlayer_regularization_losses
–layer_metrics
—metrics
“layers
”non_trainable_variables
nregularization_losses
otrainable_variables
 
 
 
≤
q	variables
 ‘layer_regularization_losses
’layer_metrics
÷metrics
„layers
Ўnon_trainable_variables
rregularization_losses
strainable_variables
 
YW
VARIABLE_VALUEGen_BN_4/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEGen_BN_4/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEGen_BN_4/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEGen_BN_4/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

v0
w1
x2
y3
 

v0
w1
≤
z	variables
 ўlayer_regularization_losses
Џlayer_metrics
џmetrics
№layers
Ёnon_trainable_variables
{regularization_losses
|trainable_variables
 
 
 
≥
~	variables
 ёlayer_regularization_losses
яlayer_metrics
аmetrics
бlayers
вnon_trainable_variables
regularization_losses
Аtrainable_variables
V
г	variables
дregularization_losses
еtrainable_variables
ж	keras_api
][
VARIABLE_VALUEGen_Conv_5/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEGen_Conv_5/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

Г0
Д1
 

Г0
Д1
µ
Е	variables
 зlayer_regularization_losses
иlayer_metrics
йmetrics
кlayers
лnon_trainable_variables
Жregularization_losses
Зtrainable_variables
 
 
 
µ
Й	variables
 мlayer_regularization_losses
нlayer_metrics
оmetrics
пlayers
рnon_trainable_variables
Кregularization_losses
Лtrainable_variables
 
YW
VARIABLE_VALUEGen_BN_5/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEGen_BN_5/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEGen_BN_5/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEGen_BN_5/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
О0
П1
Р2
С3
 

О0
П1
µ
Т	variables
 сlayer_regularization_losses
тlayer_metrics
уmetrics
фlayers
хnon_trainable_variables
Уregularization_losses
Фtrainable_variables
 
 
 
µ
Ц	variables
 цlayer_regularization_losses
чlayer_metrics
шmetrics
щlayers
ъnon_trainable_variables
Чregularization_losses
Шtrainable_variables
V
ы	variables
ьregularization_losses
эtrainable_variables
ю	keras_api
`^
VARIABLE_VALUEGen_Conv_T_1/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEGen_Conv_T_1/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

Ы0
Ь1
 

Ы0
Ь1
µ
Э	variables
 €layer_regularization_losses
Аlayer_metrics
Бmetrics
Вlayers
Гnon_trainable_variables
Юregularization_losses
Яtrainable_variables
 
ZX
VARIABLE_VALUEGen_BN_6/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEGen_BN_6/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEGen_BN_6/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEGen_BN_6/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
Ґ0
£1
§2
•3
 

Ґ0
£1
µ
¶	variables
 Дlayer_regularization_losses
Еlayer_metrics
Жmetrics
Зlayers
Иnon_trainable_variables
Іregularization_losses
®trainable_variables
 
 
 
µ
™	variables
 Йlayer_regularization_losses
Кlayer_metrics
Лmetrics
Мlayers
Нnon_trainable_variables
Ђregularization_losses
ђtrainable_variables
 
 
 
µ
Ѓ	variables
 Оlayer_regularization_losses
Пlayer_metrics
Рmetrics
Сlayers
Тnon_trainable_variables
ѓregularization_losses
∞trainable_variables
V
У	variables
Фregularization_losses
Хtrainable_variables
Ц	keras_api
`^
VARIABLE_VALUEGen_Conv_T_2/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEGen_Conv_T_2/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

≥0
і1
 

≥0
і1
µ
µ	variables
 Чlayer_regularization_losses
Шlayer_metrics
Щmetrics
Ъlayers
Ыnon_trainable_variables
ґregularization_losses
Јtrainable_variables
 
ZX
VARIABLE_VALUEGen_BN_7/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEGen_BN_7/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEGen_BN_7/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEGen_BN_7/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
Ї0
ї1
Љ2
љ3
 

Ї0
ї1
µ
Њ	variables
 Ьlayer_regularization_losses
Эlayer_metrics
Юmetrics
Яlayers
†non_trainable_variables
њregularization_losses
јtrainable_variables
 
 
 
µ
¬	variables
 °layer_regularization_losses
Ґlayer_metrics
£metrics
§layers
•non_trainable_variables
√regularization_losses
ƒtrainable_variables
V
¶	variables
Іregularization_losses
®trainable_variables
©	keras_api
`^
VARIABLE_VALUEGen_Conv_T_3/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEGen_Conv_T_3/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

«0
»1
 

«0
»1
µ
…	variables
 ™layer_regularization_losses
Ђlayer_metrics
ђmetrics
≠layers
Ѓnon_trainable_variables
 regularization_losses
Ћtrainable_variables
 
ZX
VARIABLE_VALUEGen_BN_8/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEGen_BN_8/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEGen_BN_8/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEGen_BN_8/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
ќ0
ѕ1
–2
—3
 

ќ0
ѕ1
µ
“	variables
 ѓlayer_regularization_losses
∞layer_metrics
±metrics
≤layers
≥non_trainable_variables
”regularization_losses
‘trainable_variables
 
 
 
µ
÷	variables
 іlayer_regularization_losses
µlayer_metrics
ґmetrics
Јlayers
Єnon_trainable_variables
„regularization_losses
Ўtrainable_variables
 
 
 
µ
Џ	variables
 єlayer_regularization_losses
Їlayer_metrics
їmetrics
Љlayers
љnon_trainable_variables
џregularization_losses
№trainable_variables
V
Њ	variables
њregularization_losses
јtrainable_variables
Ѕ	keras_api
`^
VARIABLE_VALUEGen_Conv_T_4/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEGen_Conv_T_4/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

я0
а1
 

я0
а1
µ
б	variables
 ¬layer_regularization_losses
√layer_metrics
ƒmetrics
≈layers
∆non_trainable_variables
вregularization_losses
гtrainable_variables
 
ZX
VARIABLE_VALUEGen_BN_9/gamma6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEGen_BN_9/beta5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEGen_BN_9/moving_mean<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEGen_BN_9/moving_variance@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
ж0
з1
и2
й3
 

ж0
з1
µ
к	variables
 «layer_regularization_losses
»layer_metrics
…metrics
 layers
Ћnon_trainable_variables
лregularization_losses
мtrainable_variables
 
 
 
µ
о	variables
 ћlayer_regularization_losses
Ќlayer_metrics
ќmetrics
ѕlayers
–non_trainable_variables
пregularization_losses
рtrainable_variables
V
—	variables
“regularization_losses
”trainable_variables
‘	keras_api
`^
VARIABLE_VALUEGen_Conv_T_5/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEGen_Conv_T_5/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE

у0
ф1
 

у0
ф1
µ
х	variables
 ’layer_regularization_losses
÷layer_metrics
„metrics
Ўlayers
ўnon_trainable_variables
цregularization_losses
чtrainable_variables
 
[Y
VARIABLE_VALUEGen_BN_10/gamma6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEGen_BN_10/beta5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEGen_BN_10/moving_mean<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEGen_BN_10/moving_variance@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
ъ0
ы1
ь2
э3
 

ъ0
ы1
µ
ю	variables
 Џlayer_regularization_losses
џlayer_metrics
№metrics
Ёlayers
ёnon_trainable_variables
€regularization_losses
Аtrainable_variables
`^
VARIABLE_VALUEGen_Conv_T_6/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEGen_Conv_T_6/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE

В0
Г1
 

В0
Г1
µ
Д	variables
 яlayer_regularization_losses
аlayer_metrics
бmetrics
вlayers
гnon_trainable_variables
Еregularization_losses
Жtrainable_variables
 
 
 
Ц
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
25
26
27
28
29
30
 31
!32
"33
#34
$35
Ґ
80
91
L2
M3
d4
e5
x6
y7
Р8
С9
§10
•11
Љ12
љ13
–14
—15
и16
й17
ь18
э19
 
 
 
µ
Н	variables
 дlayer_regularization_losses
еlayer_metrics
жmetrics
зlayers
иnon_trainable_variables
Оregularization_losses
Пtrainable_variables
 
 
 

*0
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

80
91
 
 
 
µ
†	variables
 йlayer_regularization_losses
кlayer_metrics
лmetrics
мlayers
нnon_trainable_variables
°regularization_losses
Ґtrainable_variables
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
 

L0
M1
 
 
 
 
 
 
 
 
µ
Є	variables
 оlayer_regularization_losses
пlayer_metrics
рmetrics
сlayers
тnon_trainable_variables
єregularization_losses
Їtrainable_variables
 
 
 

V0
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

d0
e1
 
 
 
µ
Ћ	variables
 уlayer_regularization_losses
фlayer_metrics
хmetrics
цlayers
чnon_trainable_variables
ћregularization_losses
Ќtrainable_variables
 
 
 

j0
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

x0
y1
 
 
 
 
 
 
 
 
µ
г	variables
 шlayer_regularization_losses
щlayer_metrics
ъmetrics
ыlayers
ьnon_trainable_variables
дregularization_losses
еtrainable_variables
 
 
 

В0
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

Р0
С1
 
 
 
 
 
 
 
 
µ
ы	variables
 эlayer_regularization_losses
юlayer_metrics
€metrics
Аlayers
Бnon_trainable_variables
ьregularization_losses
эtrainable_variables
 
 
 

Ъ0
 
 
 
 
 

§0
•1
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
µ
У	variables
 Вlayer_regularization_losses
Гlayer_metrics
Дmetrics
Еlayers
Жnon_trainable_variables
Фregularization_losses
Хtrainable_variables
 
 
 

≤0
 
 
 
 
 

Љ0
љ1
 
 
 
 
 
 
 
 
µ
¶	variables
 Зlayer_regularization_losses
Иlayer_metrics
Йmetrics
Кlayers
Лnon_trainable_variables
Іregularization_losses
®trainable_variables
 
 
 

∆0
 
 
 
 
 

–0
—1
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
µ
Њ	variables
 Мlayer_regularization_losses
Нlayer_metrics
Оmetrics
Пlayers
Рnon_trainable_variables
њregularization_losses
јtrainable_variables
 
 
 

ё0
 
 
 
 
 

и0
й1
 
 
 
 
 
 
 
 
µ
—	variables
 Сlayer_regularization_losses
Тlayer_metrics
Уmetrics
Фlayers
Хnon_trainable_variables
“regularization_losses
”trainable_variables
 
 
 

т0
 
 
 
 
 

ь0
э1
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
Р
serving_default_Gen_InputPlaceholder*1
_output_shapes
:€€€€€€€€€ј†*
dtype0*&
shape:€€€€€€€€€ј†
ы
StatefulPartitionedCallStatefulPartitionedCallserving_default_Gen_InputGen_Conv_1/kernelGen_Conv_1/biasGen_BN_1/gammaGen_BN_1/betaGen_BN_1/moving_meanGen_BN_1/moving_varianceGen_Conv_2/kernelGen_Conv_2/biasGen_BN_2/gammaGen_BN_2/betaGen_BN_2/moving_meanGen_BN_2/moving_varianceGen_Conv_3/kernelGen_Conv_3/biasGen_BN_3/gammaGen_BN_3/betaGen_BN_3/moving_meanGen_BN_3/moving_varianceGen_Conv_4/kernelGen_Conv_4/biasGen_BN_4/gammaGen_BN_4/betaGen_BN_4/moving_meanGen_BN_4/moving_varianceGen_Conv_5/kernelGen_Conv_5/biasGen_BN_5/gammaGen_BN_5/betaGen_BN_5/moving_meanGen_BN_5/moving_varianceGen_Conv_T_1/kernelGen_Conv_T_1/biasGen_BN_6/gammaGen_BN_6/betaGen_BN_6/moving_meanGen_BN_6/moving_varianceGen_Conv_T_2/kernelGen_Conv_T_2/biasGen_BN_7/gammaGen_BN_7/betaGen_BN_7/moving_meanGen_BN_7/moving_varianceGen_Conv_T_3/kernelGen_Conv_T_3/biasGen_BN_8/gammaGen_BN_8/betaGen_BN_8/moving_meanGen_BN_8/moving_varianceGen_Conv_T_4/kernelGen_Conv_T_4/biasGen_BN_9/gammaGen_BN_9/betaGen_BN_9/moving_meanGen_BN_9/moving_varianceGen_Conv_T_5/kernelGen_Conv_T_5/biasGen_BN_10/gammaGen_BN_10/betaGen_BN_10/moving_meanGen_BN_10/moving_varianceGen_Conv_T_6/kernelGen_Conv_T_6/bias*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ј†*`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_143758
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
т
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%Gen_Conv_1/kernel/Read/ReadVariableOp#Gen_Conv_1/bias/Read/ReadVariableOp"Gen_BN_1/gamma/Read/ReadVariableOp!Gen_BN_1/beta/Read/ReadVariableOp(Gen_BN_1/moving_mean/Read/ReadVariableOp,Gen_BN_1/moving_variance/Read/ReadVariableOp%Gen_Conv_2/kernel/Read/ReadVariableOp#Gen_Conv_2/bias/Read/ReadVariableOp"Gen_BN_2/gamma/Read/ReadVariableOp!Gen_BN_2/beta/Read/ReadVariableOp(Gen_BN_2/moving_mean/Read/ReadVariableOp,Gen_BN_2/moving_variance/Read/ReadVariableOp%Gen_Conv_3/kernel/Read/ReadVariableOp#Gen_Conv_3/bias/Read/ReadVariableOp"Gen_BN_3/gamma/Read/ReadVariableOp!Gen_BN_3/beta/Read/ReadVariableOp(Gen_BN_3/moving_mean/Read/ReadVariableOp,Gen_BN_3/moving_variance/Read/ReadVariableOp%Gen_Conv_4/kernel/Read/ReadVariableOp#Gen_Conv_4/bias/Read/ReadVariableOp"Gen_BN_4/gamma/Read/ReadVariableOp!Gen_BN_4/beta/Read/ReadVariableOp(Gen_BN_4/moving_mean/Read/ReadVariableOp,Gen_BN_4/moving_variance/Read/ReadVariableOp%Gen_Conv_5/kernel/Read/ReadVariableOp#Gen_Conv_5/bias/Read/ReadVariableOp"Gen_BN_5/gamma/Read/ReadVariableOp!Gen_BN_5/beta/Read/ReadVariableOp(Gen_BN_5/moving_mean/Read/ReadVariableOp,Gen_BN_5/moving_variance/Read/ReadVariableOp'Gen_Conv_T_1/kernel/Read/ReadVariableOp%Gen_Conv_T_1/bias/Read/ReadVariableOp"Gen_BN_6/gamma/Read/ReadVariableOp!Gen_BN_6/beta/Read/ReadVariableOp(Gen_BN_6/moving_mean/Read/ReadVariableOp,Gen_BN_6/moving_variance/Read/ReadVariableOp'Gen_Conv_T_2/kernel/Read/ReadVariableOp%Gen_Conv_T_2/bias/Read/ReadVariableOp"Gen_BN_7/gamma/Read/ReadVariableOp!Gen_BN_7/beta/Read/ReadVariableOp(Gen_BN_7/moving_mean/Read/ReadVariableOp,Gen_BN_7/moving_variance/Read/ReadVariableOp'Gen_Conv_T_3/kernel/Read/ReadVariableOp%Gen_Conv_T_3/bias/Read/ReadVariableOp"Gen_BN_8/gamma/Read/ReadVariableOp!Gen_BN_8/beta/Read/ReadVariableOp(Gen_BN_8/moving_mean/Read/ReadVariableOp,Gen_BN_8/moving_variance/Read/ReadVariableOp'Gen_Conv_T_4/kernel/Read/ReadVariableOp%Gen_Conv_T_4/bias/Read/ReadVariableOp"Gen_BN_9/gamma/Read/ReadVariableOp!Gen_BN_9/beta/Read/ReadVariableOp(Gen_BN_9/moving_mean/Read/ReadVariableOp,Gen_BN_9/moving_variance/Read/ReadVariableOp'Gen_Conv_T_5/kernel/Read/ReadVariableOp%Gen_Conv_T_5/bias/Read/ReadVariableOp#Gen_BN_10/gamma/Read/ReadVariableOp"Gen_BN_10/beta/Read/ReadVariableOp)Gen_BN_10/moving_mean/Read/ReadVariableOp-Gen_BN_10/moving_variance/Read/ReadVariableOp'Gen_Conv_T_6/kernel/Read/ReadVariableOp%Gen_Conv_T_6/bias/Read/ReadVariableOpConst*K
TinD
B2@*
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
GPU2*0J 8В *(
f#R!
__inference__traced_save_146511
Х
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameGen_Conv_1/kernelGen_Conv_1/biasGen_BN_1/gammaGen_BN_1/betaGen_BN_1/moving_meanGen_BN_1/moving_varianceGen_Conv_2/kernelGen_Conv_2/biasGen_BN_2/gammaGen_BN_2/betaGen_BN_2/moving_meanGen_BN_2/moving_varianceGen_Conv_3/kernelGen_Conv_3/biasGen_BN_3/gammaGen_BN_3/betaGen_BN_3/moving_meanGen_BN_3/moving_varianceGen_Conv_4/kernelGen_Conv_4/biasGen_BN_4/gammaGen_BN_4/betaGen_BN_4/moving_meanGen_BN_4/moving_varianceGen_Conv_5/kernelGen_Conv_5/biasGen_BN_5/gammaGen_BN_5/betaGen_BN_5/moving_meanGen_BN_5/moving_varianceGen_Conv_T_1/kernelGen_Conv_T_1/biasGen_BN_6/gammaGen_BN_6/betaGen_BN_6/moving_meanGen_BN_6/moving_varianceGen_Conv_T_2/kernelGen_Conv_T_2/biasGen_BN_7/gammaGen_BN_7/betaGen_BN_7/moving_meanGen_BN_7/moving_varianceGen_Conv_T_3/kernelGen_Conv_T_3/biasGen_BN_8/gammaGen_BN_8/betaGen_BN_8/moving_meanGen_BN_8/moving_varianceGen_Conv_T_4/kernelGen_Conv_T_4/biasGen_BN_9/gammaGen_BN_9/betaGen_BN_9/moving_meanGen_BN_9/moving_varianceGen_Conv_T_5/kernelGen_Conv_T_5/biasGen_BN_10/gammaGen_BN_10/betaGen_BN_10/moving_meanGen_BN_10/moving_varianceGen_Conv_T_6/kernelGen_Conv_T_6/bias*J
TinC
A2?*
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
GPU2*0J 8В *+
f&R$
"__inference__traced_restore_146707П€%
ѕ
э
D__inference_Gen_BN_3_layer_call_and_return_conditional_losses_145254

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ѕ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€А:::::X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ь
Y
-__inference_Gen_Concat_3_layer_call_fn_146035
inputs_0
inputs_1
identityя
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Concat_3_layer_call_and_return_conditional_losses_1427292
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:€€€€€€€€€0(А:l h
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:€€€€€€€€€0(А
"
_user_specified_name
inputs/1
љ
Ґ
E__inference_Gen_BN_10_layer_call_and_return_conditional_losses_141818

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¶
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
ь
Y
-__inference_Gen_Concat_2_layer_call_fn_145958
inputs_0
inputs_1
identityя
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Concat_2_layer_call_and_return_conditional_losses_1426732
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:€€€€€€€€€А:l h
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:€€€€€€€€€А
"
_user_specified_name
inputs/1
ф
°
D__inference_Gen_BN_1_layer_call_and_return_conditional_losses_141956

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ў
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€`P@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Ф
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:€€€€€€€€€`P@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€`P@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:€€€€€€€€€`P@
 
_user_specified_nameinputs
вД
И
!__inference__wrapped_model_140135
	gen_input7
3generator_gen_conv_1_conv2d_readvariableop_resource8
4generator_gen_conv_1_biasadd_readvariableop_resource.
*generator_gen_bn_1_readvariableop_resource0
,generator_gen_bn_1_readvariableop_1_resource?
;generator_gen_bn_1_fusedbatchnormv3_readvariableop_resourceA
=generator_gen_bn_1_fusedbatchnormv3_readvariableop_1_resource7
3generator_gen_conv_2_conv2d_readvariableop_resource8
4generator_gen_conv_2_biasadd_readvariableop_resource.
*generator_gen_bn_2_readvariableop_resource0
,generator_gen_bn_2_readvariableop_1_resource?
;generator_gen_bn_2_fusedbatchnormv3_readvariableop_resourceA
=generator_gen_bn_2_fusedbatchnormv3_readvariableop_1_resource7
3generator_gen_conv_3_conv2d_readvariableop_resource8
4generator_gen_conv_3_biasadd_readvariableop_resource.
*generator_gen_bn_3_readvariableop_resource0
,generator_gen_bn_3_readvariableop_1_resource?
;generator_gen_bn_3_fusedbatchnormv3_readvariableop_resourceA
=generator_gen_bn_3_fusedbatchnormv3_readvariableop_1_resource7
3generator_gen_conv_4_conv2d_readvariableop_resource8
4generator_gen_conv_4_biasadd_readvariableop_resource.
*generator_gen_bn_4_readvariableop_resource0
,generator_gen_bn_4_readvariableop_1_resource?
;generator_gen_bn_4_fusedbatchnormv3_readvariableop_resourceA
=generator_gen_bn_4_fusedbatchnormv3_readvariableop_1_resource7
3generator_gen_conv_5_conv2d_readvariableop_resource8
4generator_gen_conv_5_biasadd_readvariableop_resource.
*generator_gen_bn_5_readvariableop_resource0
,generator_gen_bn_5_readvariableop_1_resource?
;generator_gen_bn_5_fusedbatchnormv3_readvariableop_resourceA
=generator_gen_bn_5_fusedbatchnormv3_readvariableop_1_resourceC
?generator_gen_conv_t_1_conv2d_transpose_readvariableop_resource:
6generator_gen_conv_t_1_biasadd_readvariableop_resource.
*generator_gen_bn_6_readvariableop_resource0
,generator_gen_bn_6_readvariableop_1_resource?
;generator_gen_bn_6_fusedbatchnormv3_readvariableop_resourceA
=generator_gen_bn_6_fusedbatchnormv3_readvariableop_1_resourceC
?generator_gen_conv_t_2_conv2d_transpose_readvariableop_resource:
6generator_gen_conv_t_2_biasadd_readvariableop_resource.
*generator_gen_bn_7_readvariableop_resource0
,generator_gen_bn_7_readvariableop_1_resource?
;generator_gen_bn_7_fusedbatchnormv3_readvariableop_resourceA
=generator_gen_bn_7_fusedbatchnormv3_readvariableop_1_resourceC
?generator_gen_conv_t_3_conv2d_transpose_readvariableop_resource:
6generator_gen_conv_t_3_biasadd_readvariableop_resource.
*generator_gen_bn_8_readvariableop_resource0
,generator_gen_bn_8_readvariableop_1_resource?
;generator_gen_bn_8_fusedbatchnormv3_readvariableop_resourceA
=generator_gen_bn_8_fusedbatchnormv3_readvariableop_1_resourceC
?generator_gen_conv_t_4_conv2d_transpose_readvariableop_resource:
6generator_gen_conv_t_4_biasadd_readvariableop_resource.
*generator_gen_bn_9_readvariableop_resource0
,generator_gen_bn_9_readvariableop_1_resource?
;generator_gen_bn_9_fusedbatchnormv3_readvariableop_resourceA
=generator_gen_bn_9_fusedbatchnormv3_readvariableop_1_resourceC
?generator_gen_conv_t_5_conv2d_transpose_readvariableop_resource:
6generator_gen_conv_t_5_biasadd_readvariableop_resource/
+generator_gen_bn_10_readvariableop_resource1
-generator_gen_bn_10_readvariableop_1_resource@
<generator_gen_bn_10_fusedbatchnormv3_readvariableop_resourceB
>generator_gen_bn_10_fusedbatchnormv3_readvariableop_1_resourceC
?generator_gen_conv_t_6_conv2d_transpose_readvariableop_resource:
6generator_gen_conv_t_6_biasadd_readvariableop_resource
identityИ‘
*Generator/Gen_Conv_1/Conv2D/ReadVariableOpReadVariableOp3generator_gen_conv_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02,
*Generator/Gen_Conv_1/Conv2D/ReadVariableOpз
Generator/Gen_Conv_1/Conv2DConv2D	gen_input2Generator/Gen_Conv_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ј†@*
paddingSAME*
strides
2
Generator/Gen_Conv_1/Conv2DЋ
+Generator/Gen_Conv_1/BiasAdd/ReadVariableOpReadVariableOp4generator_gen_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+Generator/Gen_Conv_1/BiasAdd/ReadVariableOpё
Generator/Gen_Conv_1/BiasAddBiasAdd$Generator/Gen_Conv_1/Conv2D:output:03Generator/Gen_Conv_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ј†@2
Generator/Gen_Conv_1/BiasAddњ
*Generator/Gen_Conv_1/leaky_re_lu/LeakyRelu	LeakyRelu%Generator/Gen_Conv_1/BiasAdd:output:0*1
_output_shapes
:€€€€€€€€€ј†@2,
*Generator/Gen_Conv_1/leaky_re_lu/LeakyReluк
Generator/Gen_MP_1/MaxPoolMaxPool8Generator/Gen_Conv_1/leaky_re_lu/LeakyRelu:activations:0*/
_output_shapes
:€€€€€€€€€`P@*
ksize
*
paddingVALID*
strides
2
Generator/Gen_MP_1/MaxPool≠
!Generator/Gen_BN_1/ReadVariableOpReadVariableOp*generator_gen_bn_1_readvariableop_resource*
_output_shapes
:@*
dtype02#
!Generator/Gen_BN_1/ReadVariableOp≥
#Generator/Gen_BN_1/ReadVariableOp_1ReadVariableOp,generator_gen_bn_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02%
#Generator/Gen_BN_1/ReadVariableOp_1а
2Generator/Gen_BN_1/FusedBatchNormV3/ReadVariableOpReadVariableOp;generator_gen_bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype024
2Generator/Gen_BN_1/FusedBatchNormV3/ReadVariableOpж
4Generator/Gen_BN_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=generator_gen_bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype026
4Generator/Gen_BN_1/FusedBatchNormV3/ReadVariableOp_1ў
#Generator/Gen_BN_1/FusedBatchNormV3FusedBatchNormV3#Generator/Gen_MP_1/MaxPool:output:0)Generator/Gen_BN_1/ReadVariableOp:value:0+Generator/Gen_BN_1/ReadVariableOp_1:value:0:Generator/Gen_BN_1/FusedBatchNormV3/ReadVariableOp:value:0<Generator/Gen_BN_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€`P@:@:@:@:@:*
epsilon%oГ:*
is_training( 2%
#Generator/Gen_BN_1/FusedBatchNormV3’
*Generator/Gen_Conv_2/Conv2D/ReadVariableOpReadVariableOp3generator_gen_conv_2_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02,
*Generator/Gen_Conv_2/Conv2D/ReadVariableOpД
Generator/Gen_Conv_2/Conv2DConv2D'Generator/Gen_BN_1/FusedBatchNormV3:y:02Generator/Gen_Conv_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€`PА*
paddingSAME*
strides
2
Generator/Gen_Conv_2/Conv2Dћ
+Generator/Gen_Conv_2/BiasAdd/ReadVariableOpReadVariableOp4generator_gen_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+Generator/Gen_Conv_2/BiasAdd/ReadVariableOpЁ
Generator/Gen_Conv_2/BiasAddBiasAdd$Generator/Gen_Conv_2/Conv2D:output:03Generator/Gen_Conv_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€`PА2
Generator/Gen_Conv_2/BiasAdd¬
,Generator/Gen_Conv_2/leaky_re_lu_1/LeakyRelu	LeakyRelu%Generator/Gen_Conv_2/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€`PА2.
,Generator/Gen_Conv_2/leaky_re_lu_1/LeakyReluн
Generator/Gen_MP_2/MaxPoolMaxPool:Generator/Gen_Conv_2/leaky_re_lu_1/LeakyRelu:activations:0*0
_output_shapes
:€€€€€€€€€0(А*
ksize
*
paddingVALID*
strides
2
Generator/Gen_MP_2/MaxPoolЃ
!Generator/Gen_BN_2/ReadVariableOpReadVariableOp*generator_gen_bn_2_readvariableop_resource*
_output_shapes	
:А*
dtype02#
!Generator/Gen_BN_2/ReadVariableOpі
#Generator/Gen_BN_2/ReadVariableOp_1ReadVariableOp,generator_gen_bn_2_readvariableop_1_resource*
_output_shapes	
:А*
dtype02%
#Generator/Gen_BN_2/ReadVariableOp_1б
2Generator/Gen_BN_2/FusedBatchNormV3/ReadVariableOpReadVariableOp;generator_gen_bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype024
2Generator/Gen_BN_2/FusedBatchNormV3/ReadVariableOpз
4Generator/Gen_BN_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=generator_gen_bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype026
4Generator/Gen_BN_2/FusedBatchNormV3/ReadVariableOp_1ё
#Generator/Gen_BN_2/FusedBatchNormV3FusedBatchNormV3#Generator/Gen_MP_2/MaxPool:output:0)Generator/Gen_BN_2/ReadVariableOp:value:0+Generator/Gen_BN_2/ReadVariableOp_1:value:0:Generator/Gen_BN_2/FusedBatchNormV3/ReadVariableOp:value:0<Generator/Gen_BN_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€0(А:А:А:А:А:*
epsilon%oГ:*
is_training( 2%
#Generator/Gen_BN_2/FusedBatchNormV3ђ
Generator/Gen_SPD_1/IdentityIdentity'Generator/Gen_BN_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2
Generator/Gen_SPD_1/Identity÷
*Generator/Gen_Conv_3/Conv2D/ReadVariableOpReadVariableOp3generator_gen_conv_3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02,
*Generator/Gen_Conv_3/Conv2D/ReadVariableOpВ
Generator/Gen_Conv_3/Conv2DConv2D%Generator/Gen_SPD_1/Identity:output:02Generator/Gen_Conv_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€0(А*
paddingSAME*
strides
2
Generator/Gen_Conv_3/Conv2Dћ
+Generator/Gen_Conv_3/BiasAdd/ReadVariableOpReadVariableOp4generator_gen_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+Generator/Gen_Conv_3/BiasAdd/ReadVariableOpЁ
Generator/Gen_Conv_3/BiasAddBiasAdd$Generator/Gen_Conv_3/Conv2D:output:03Generator/Gen_Conv_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2
Generator/Gen_Conv_3/BiasAdd¬
,Generator/Gen_Conv_3/leaky_re_lu_2/LeakyRelu	LeakyRelu%Generator/Gen_Conv_3/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€0(А2.
,Generator/Gen_Conv_3/leaky_re_lu_2/LeakyReluн
Generator/Gen_MP_3/MaxPoolMaxPool:Generator/Gen_Conv_3/leaky_re_lu_2/LeakyRelu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
Generator/Gen_MP_3/MaxPoolЃ
!Generator/Gen_BN_3/ReadVariableOpReadVariableOp*generator_gen_bn_3_readvariableop_resource*
_output_shapes	
:А*
dtype02#
!Generator/Gen_BN_3/ReadVariableOpі
#Generator/Gen_BN_3/ReadVariableOp_1ReadVariableOp,generator_gen_bn_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02%
#Generator/Gen_BN_3/ReadVariableOp_1б
2Generator/Gen_BN_3/FusedBatchNormV3/ReadVariableOpReadVariableOp;generator_gen_bn_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype024
2Generator/Gen_BN_3/FusedBatchNormV3/ReadVariableOpз
4Generator/Gen_BN_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=generator_gen_bn_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype026
4Generator/Gen_BN_3/FusedBatchNormV3/ReadVariableOp_1ё
#Generator/Gen_BN_3/FusedBatchNormV3FusedBatchNormV3#Generator/Gen_MP_3/MaxPool:output:0)Generator/Gen_BN_3/ReadVariableOp:value:0+Generator/Gen_BN_3/ReadVariableOp_1:value:0:Generator/Gen_BN_3/FusedBatchNormV3/ReadVariableOp:value:0<Generator/Gen_BN_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2%
#Generator/Gen_BN_3/FusedBatchNormV3÷
*Generator/Gen_Conv_4/Conv2D/ReadVariableOpReadVariableOp3generator_gen_conv_4_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02,
*Generator/Gen_Conv_4/Conv2D/ReadVariableOpД
Generator/Gen_Conv_4/Conv2DConv2D'Generator/Gen_BN_3/FusedBatchNormV3:y:02Generator/Gen_Conv_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Generator/Gen_Conv_4/Conv2Dћ
+Generator/Gen_Conv_4/BiasAdd/ReadVariableOpReadVariableOp4generator_gen_conv_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+Generator/Gen_Conv_4/BiasAdd/ReadVariableOpЁ
Generator/Gen_Conv_4/BiasAddBiasAdd$Generator/Gen_Conv_4/Conv2D:output:03Generator/Gen_Conv_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Generator/Gen_Conv_4/BiasAdd¬
,Generator/Gen_Conv_4/leaky_re_lu_3/LeakyRelu	LeakyRelu%Generator/Gen_Conv_4/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€А2.
,Generator/Gen_Conv_4/leaky_re_lu_3/LeakyReluн
Generator/Gen_MP_4/MaxPoolMaxPool:Generator/Gen_Conv_4/leaky_re_lu_3/LeakyRelu:activations:0*0
_output_shapes
:€€€€€€€€€
А*
ksize
*
paddingVALID*
strides
2
Generator/Gen_MP_4/MaxPoolЃ
!Generator/Gen_BN_4/ReadVariableOpReadVariableOp*generator_gen_bn_4_readvariableop_resource*
_output_shapes	
:А*
dtype02#
!Generator/Gen_BN_4/ReadVariableOpі
#Generator/Gen_BN_4/ReadVariableOp_1ReadVariableOp,generator_gen_bn_4_readvariableop_1_resource*
_output_shapes	
:А*
dtype02%
#Generator/Gen_BN_4/ReadVariableOp_1б
2Generator/Gen_BN_4/FusedBatchNormV3/ReadVariableOpReadVariableOp;generator_gen_bn_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype024
2Generator/Gen_BN_4/FusedBatchNormV3/ReadVariableOpз
4Generator/Gen_BN_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=generator_gen_bn_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype026
4Generator/Gen_BN_4/FusedBatchNormV3/ReadVariableOp_1ё
#Generator/Gen_BN_4/FusedBatchNormV3FusedBatchNormV3#Generator/Gen_MP_4/MaxPool:output:0)Generator/Gen_BN_4/ReadVariableOp:value:0+Generator/Gen_BN_4/ReadVariableOp_1:value:0:Generator/Gen_BN_4/FusedBatchNormV3/ReadVariableOp:value:0<Generator/Gen_BN_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€
А:А:А:А:А:*
epsilon%oГ:*
is_training( 2%
#Generator/Gen_BN_4/FusedBatchNormV3ђ
Generator/Gen_SPD_2/IdentityIdentity'Generator/Gen_BN_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€
А2
Generator/Gen_SPD_2/Identity÷
*Generator/Gen_Conv_5/Conv2D/ReadVariableOpReadVariableOp3generator_gen_conv_5_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02,
*Generator/Gen_Conv_5/Conv2D/ReadVariableOpВ
Generator/Gen_Conv_5/Conv2DConv2D%Generator/Gen_SPD_2/Identity:output:02Generator/Gen_Conv_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€
А*
paddingSAME*
strides
2
Generator/Gen_Conv_5/Conv2Dћ
+Generator/Gen_Conv_5/BiasAdd/ReadVariableOpReadVariableOp4generator_gen_conv_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+Generator/Gen_Conv_5/BiasAdd/ReadVariableOpЁ
Generator/Gen_Conv_5/BiasAddBiasAdd$Generator/Gen_Conv_5/Conv2D:output:03Generator/Gen_Conv_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€
А2
Generator/Gen_Conv_5/BiasAdd¬
,Generator/Gen_Conv_5/leaky_re_lu_4/LeakyRelu	LeakyRelu%Generator/Gen_Conv_5/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€
А2.
,Generator/Gen_Conv_5/leaky_re_lu_4/LeakyReluн
Generator/Gen_MP_5/MaxPoolMaxPool:Generator/Gen_Conv_5/leaky_re_lu_4/LeakyRelu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
Generator/Gen_MP_5/MaxPoolЃ
!Generator/Gen_BN_5/ReadVariableOpReadVariableOp*generator_gen_bn_5_readvariableop_resource*
_output_shapes	
:А*
dtype02#
!Generator/Gen_BN_5/ReadVariableOpі
#Generator/Gen_BN_5/ReadVariableOp_1ReadVariableOp,generator_gen_bn_5_readvariableop_1_resource*
_output_shapes	
:А*
dtype02%
#Generator/Gen_BN_5/ReadVariableOp_1б
2Generator/Gen_BN_5/FusedBatchNormV3/ReadVariableOpReadVariableOp;generator_gen_bn_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype024
2Generator/Gen_BN_5/FusedBatchNormV3/ReadVariableOpз
4Generator/Gen_BN_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=generator_gen_bn_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype026
4Generator/Gen_BN_5/FusedBatchNormV3/ReadVariableOp_1ё
#Generator/Gen_BN_5/FusedBatchNormV3FusedBatchNormV3#Generator/Gen_MP_5/MaxPool:output:0)Generator/Gen_BN_5/ReadVariableOp:value:0+Generator/Gen_BN_5/ReadVariableOp_1:value:0:Generator/Gen_BN_5/FusedBatchNormV3/ReadVariableOp:value:0<Generator/Gen_BN_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2%
#Generator/Gen_BN_5/FusedBatchNormV3ђ
Generator/Gen_SPD_3/IdentityIdentity'Generator/Gen_BN_5/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Generator/Gen_SPD_3/IdentityС
Generator/Gen_Conv_T_1/ShapeShape%Generator/Gen_SPD_3/Identity:output:0*
T0*
_output_shapes
:2
Generator/Gen_Conv_T_1/ShapeҐ
*Generator/Gen_Conv_T_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*Generator/Gen_Conv_T_1/strided_slice/stack¶
,Generator/Gen_Conv_T_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,Generator/Gen_Conv_T_1/strided_slice/stack_1¶
,Generator/Gen_Conv_T_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Generator/Gen_Conv_T_1/strided_slice/stack_2м
$Generator/Gen_Conv_T_1/strided_sliceStridedSlice%Generator/Gen_Conv_T_1/Shape:output:03Generator/Gen_Conv_T_1/strided_slice/stack:output:05Generator/Gen_Conv_T_1/strided_slice/stack_1:output:05Generator/Gen_Conv_T_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$Generator/Gen_Conv_T_1/strided_sliceВ
Generator/Gen_Conv_T_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2 
Generator/Gen_Conv_T_1/stack/1В
Generator/Gen_Conv_T_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :
2 
Generator/Gen_Conv_T_1/stack/2Г
Generator/Gen_Conv_T_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :А2 
Generator/Gen_Conv_T_1/stack/3Ь
Generator/Gen_Conv_T_1/stackPack-Generator/Gen_Conv_T_1/strided_slice:output:0'Generator/Gen_Conv_T_1/stack/1:output:0'Generator/Gen_Conv_T_1/stack/2:output:0'Generator/Gen_Conv_T_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/Gen_Conv_T_1/stack¶
,Generator/Gen_Conv_T_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,Generator/Gen_Conv_T_1/strided_slice_1/stack™
.Generator/Gen_Conv_T_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.Generator/Gen_Conv_T_1/strided_slice_1/stack_1™
.Generator/Gen_Conv_T_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.Generator/Gen_Conv_T_1/strided_slice_1/stack_2ц
&Generator/Gen_Conv_T_1/strided_slice_1StridedSlice%Generator/Gen_Conv_T_1/stack:output:05Generator/Gen_Conv_T_1/strided_slice_1/stack:output:07Generator/Gen_Conv_T_1/strided_slice_1/stack_1:output:07Generator/Gen_Conv_T_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&Generator/Gen_Conv_T_1/strided_slice_1ъ
6Generator/Gen_Conv_T_1/conv2d_transpose/ReadVariableOpReadVariableOp?generator_gen_conv_t_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:АА*
dtype028
6Generator/Gen_Conv_T_1/conv2d_transpose/ReadVariableOpЏ
'Generator/Gen_Conv_T_1/conv2d_transposeConv2DBackpropInput%Generator/Gen_Conv_T_1/stack:output:0>Generator/Gen_Conv_T_1/conv2d_transpose/ReadVariableOp:value:0%Generator/Gen_SPD_3/Identity:output:0*
T0*0
_output_shapes
:€€€€€€€€€
А*
paddingSAME*
strides
2)
'Generator/Gen_Conv_T_1/conv2d_transpose“
-Generator/Gen_Conv_T_1/BiasAdd/ReadVariableOpReadVariableOp6generator_gen_conv_t_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-Generator/Gen_Conv_T_1/BiasAdd/ReadVariableOpп
Generator/Gen_Conv_T_1/BiasAddBiasAdd0Generator/Gen_Conv_T_1/conv2d_transpose:output:05Generator/Gen_Conv_T_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€
А2 
Generator/Gen_Conv_T_1/BiasAdd»
.Generator/Gen_Conv_T_1/leaky_re_lu_5/LeakyRelu	LeakyRelu'Generator/Gen_Conv_T_1/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€
А20
.Generator/Gen_Conv_T_1/leaky_re_lu_5/LeakyReluЃ
!Generator/Gen_BN_6/ReadVariableOpReadVariableOp*generator_gen_bn_6_readvariableop_resource*
_output_shapes	
:А*
dtype02#
!Generator/Gen_BN_6/ReadVariableOpі
#Generator/Gen_BN_6/ReadVariableOp_1ReadVariableOp,generator_gen_bn_6_readvariableop_1_resource*
_output_shapes	
:А*
dtype02%
#Generator/Gen_BN_6/ReadVariableOp_1б
2Generator/Gen_BN_6/FusedBatchNormV3/ReadVariableOpReadVariableOp;generator_gen_bn_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype024
2Generator/Gen_BN_6/FusedBatchNormV3/ReadVariableOpз
4Generator/Gen_BN_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=generator_gen_bn_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype026
4Generator/Gen_BN_6/FusedBatchNormV3/ReadVariableOp_1ч
#Generator/Gen_BN_6/FusedBatchNormV3FusedBatchNormV3<Generator/Gen_Conv_T_1/leaky_re_lu_5/LeakyRelu:activations:0)Generator/Gen_BN_6/ReadVariableOp:value:0+Generator/Gen_BN_6/ReadVariableOp_1:value:0:Generator/Gen_BN_6/FusedBatchNormV3/ReadVariableOp:value:0<Generator/Gen_BN_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€
А:А:А:А:А:*
epsilon%oГ:*
is_training( 2%
#Generator/Gen_BN_6/FusedBatchNormV3К
"Generator/Gen_Concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"Generator/Gen_Concat_1/concat/axisН
Generator/Gen_Concat_1/concatConcatV2'Generator/Gen_BN_6/FusedBatchNormV3:y:0'Generator/Gen_BN_4/FusedBatchNormV3:y:0+Generator/Gen_Concat_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€
А2
Generator/Gen_Concat_1/concatЂ
Generator/Gen_SPD_4/IdentityIdentity&Generator/Gen_Concat_1/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€
А2
Generator/Gen_SPD_4/IdentityС
Generator/Gen_Conv_T_2/ShapeShape%Generator/Gen_SPD_4/Identity:output:0*
T0*
_output_shapes
:2
Generator/Gen_Conv_T_2/ShapeҐ
*Generator/Gen_Conv_T_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*Generator/Gen_Conv_T_2/strided_slice/stack¶
,Generator/Gen_Conv_T_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,Generator/Gen_Conv_T_2/strided_slice/stack_1¶
,Generator/Gen_Conv_T_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Generator/Gen_Conv_T_2/strided_slice/stack_2м
$Generator/Gen_Conv_T_2/strided_sliceStridedSlice%Generator/Gen_Conv_T_2/Shape:output:03Generator/Gen_Conv_T_2/strided_slice/stack:output:05Generator/Gen_Conv_T_2/strided_slice/stack_1:output:05Generator/Gen_Conv_T_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$Generator/Gen_Conv_T_2/strided_sliceВ
Generator/Gen_Conv_T_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2 
Generator/Gen_Conv_T_2/stack/1В
Generator/Gen_Conv_T_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2 
Generator/Gen_Conv_T_2/stack/2Г
Generator/Gen_Conv_T_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :А2 
Generator/Gen_Conv_T_2/stack/3Ь
Generator/Gen_Conv_T_2/stackPack-Generator/Gen_Conv_T_2/strided_slice:output:0'Generator/Gen_Conv_T_2/stack/1:output:0'Generator/Gen_Conv_T_2/stack/2:output:0'Generator/Gen_Conv_T_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/Gen_Conv_T_2/stack¶
,Generator/Gen_Conv_T_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,Generator/Gen_Conv_T_2/strided_slice_1/stack™
.Generator/Gen_Conv_T_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.Generator/Gen_Conv_T_2/strided_slice_1/stack_1™
.Generator/Gen_Conv_T_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.Generator/Gen_Conv_T_2/strided_slice_1/stack_2ц
&Generator/Gen_Conv_T_2/strided_slice_1StridedSlice%Generator/Gen_Conv_T_2/stack:output:05Generator/Gen_Conv_T_2/strided_slice_1/stack:output:07Generator/Gen_Conv_T_2/strided_slice_1/stack_1:output:07Generator/Gen_Conv_T_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&Generator/Gen_Conv_T_2/strided_slice_1ъ
6Generator/Gen_Conv_T_2/conv2d_transpose/ReadVariableOpReadVariableOp?generator_gen_conv_t_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:АА*
dtype028
6Generator/Gen_Conv_T_2/conv2d_transpose/ReadVariableOpЏ
'Generator/Gen_Conv_T_2/conv2d_transposeConv2DBackpropInput%Generator/Gen_Conv_T_2/stack:output:0>Generator/Gen_Conv_T_2/conv2d_transpose/ReadVariableOp:value:0%Generator/Gen_SPD_4/Identity:output:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2)
'Generator/Gen_Conv_T_2/conv2d_transpose“
-Generator/Gen_Conv_T_2/BiasAdd/ReadVariableOpReadVariableOp6generator_gen_conv_t_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-Generator/Gen_Conv_T_2/BiasAdd/ReadVariableOpп
Generator/Gen_Conv_T_2/BiasAddBiasAdd0Generator/Gen_Conv_T_2/conv2d_transpose:output:05Generator/Gen_Conv_T_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2 
Generator/Gen_Conv_T_2/BiasAdd»
.Generator/Gen_Conv_T_2/leaky_re_lu_6/LeakyRelu	LeakyRelu'Generator/Gen_Conv_T_2/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€А20
.Generator/Gen_Conv_T_2/leaky_re_lu_6/LeakyReluЃ
!Generator/Gen_BN_7/ReadVariableOpReadVariableOp*generator_gen_bn_7_readvariableop_resource*
_output_shapes	
:А*
dtype02#
!Generator/Gen_BN_7/ReadVariableOpі
#Generator/Gen_BN_7/ReadVariableOp_1ReadVariableOp,generator_gen_bn_7_readvariableop_1_resource*
_output_shapes	
:А*
dtype02%
#Generator/Gen_BN_7/ReadVariableOp_1б
2Generator/Gen_BN_7/FusedBatchNormV3/ReadVariableOpReadVariableOp;generator_gen_bn_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype024
2Generator/Gen_BN_7/FusedBatchNormV3/ReadVariableOpз
4Generator/Gen_BN_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=generator_gen_bn_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype026
4Generator/Gen_BN_7/FusedBatchNormV3/ReadVariableOp_1ч
#Generator/Gen_BN_7/FusedBatchNormV3FusedBatchNormV3<Generator/Gen_Conv_T_2/leaky_re_lu_6/LeakyRelu:activations:0)Generator/Gen_BN_7/ReadVariableOp:value:0+Generator/Gen_BN_7/ReadVariableOp_1:value:0:Generator/Gen_BN_7/FusedBatchNormV3/ReadVariableOp:value:0<Generator/Gen_BN_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2%
#Generator/Gen_BN_7/FusedBatchNormV3К
"Generator/Gen_Concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"Generator/Gen_Concat_2/concat/axisН
Generator/Gen_Concat_2/concatConcatV2'Generator/Gen_BN_7/FusedBatchNormV3:y:0'Generator/Gen_BN_3/FusedBatchNormV3:y:0+Generator/Gen_Concat_2/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А2
Generator/Gen_Concat_2/concatТ
Generator/Gen_Conv_T_3/ShapeShape&Generator/Gen_Concat_2/concat:output:0*
T0*
_output_shapes
:2
Generator/Gen_Conv_T_3/ShapeҐ
*Generator/Gen_Conv_T_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*Generator/Gen_Conv_T_3/strided_slice/stack¶
,Generator/Gen_Conv_T_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,Generator/Gen_Conv_T_3/strided_slice/stack_1¶
,Generator/Gen_Conv_T_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Generator/Gen_Conv_T_3/strided_slice/stack_2м
$Generator/Gen_Conv_T_3/strided_sliceStridedSlice%Generator/Gen_Conv_T_3/Shape:output:03Generator/Gen_Conv_T_3/strided_slice/stack:output:05Generator/Gen_Conv_T_3/strided_slice/stack_1:output:05Generator/Gen_Conv_T_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$Generator/Gen_Conv_T_3/strided_sliceВ
Generator/Gen_Conv_T_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :02 
Generator/Gen_Conv_T_3/stack/1В
Generator/Gen_Conv_T_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :(2 
Generator/Gen_Conv_T_3/stack/2Г
Generator/Gen_Conv_T_3/stack/3Const*
_output_shapes
: *
dtype0*
value
B :А2 
Generator/Gen_Conv_T_3/stack/3Ь
Generator/Gen_Conv_T_3/stackPack-Generator/Gen_Conv_T_3/strided_slice:output:0'Generator/Gen_Conv_T_3/stack/1:output:0'Generator/Gen_Conv_T_3/stack/2:output:0'Generator/Gen_Conv_T_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/Gen_Conv_T_3/stack¶
,Generator/Gen_Conv_T_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,Generator/Gen_Conv_T_3/strided_slice_1/stack™
.Generator/Gen_Conv_T_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.Generator/Gen_Conv_T_3/strided_slice_1/stack_1™
.Generator/Gen_Conv_T_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.Generator/Gen_Conv_T_3/strided_slice_1/stack_2ц
&Generator/Gen_Conv_T_3/strided_slice_1StridedSlice%Generator/Gen_Conv_T_3/stack:output:05Generator/Gen_Conv_T_3/strided_slice_1/stack:output:07Generator/Gen_Conv_T_3/strided_slice_1/stack_1:output:07Generator/Gen_Conv_T_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&Generator/Gen_Conv_T_3/strided_slice_1ъ
6Generator/Gen_Conv_T_3/conv2d_transpose/ReadVariableOpReadVariableOp?generator_gen_conv_t_3_conv2d_transpose_readvariableop_resource*(
_output_shapes
:АА*
dtype028
6Generator/Gen_Conv_T_3/conv2d_transpose/ReadVariableOpџ
'Generator/Gen_Conv_T_3/conv2d_transposeConv2DBackpropInput%Generator/Gen_Conv_T_3/stack:output:0>Generator/Gen_Conv_T_3/conv2d_transpose/ReadVariableOp:value:0&Generator/Gen_Concat_2/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€0(А*
paddingSAME*
strides
2)
'Generator/Gen_Conv_T_3/conv2d_transpose“
-Generator/Gen_Conv_T_3/BiasAdd/ReadVariableOpReadVariableOp6generator_gen_conv_t_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-Generator/Gen_Conv_T_3/BiasAdd/ReadVariableOpп
Generator/Gen_Conv_T_3/BiasAddBiasAdd0Generator/Gen_Conv_T_3/conv2d_transpose:output:05Generator/Gen_Conv_T_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2 
Generator/Gen_Conv_T_3/BiasAdd»
.Generator/Gen_Conv_T_3/leaky_re_lu_7/LeakyRelu	LeakyRelu'Generator/Gen_Conv_T_3/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€0(А20
.Generator/Gen_Conv_T_3/leaky_re_lu_7/LeakyReluЃ
!Generator/Gen_BN_8/ReadVariableOpReadVariableOp*generator_gen_bn_8_readvariableop_resource*
_output_shapes	
:А*
dtype02#
!Generator/Gen_BN_8/ReadVariableOpі
#Generator/Gen_BN_8/ReadVariableOp_1ReadVariableOp,generator_gen_bn_8_readvariableop_1_resource*
_output_shapes	
:А*
dtype02%
#Generator/Gen_BN_8/ReadVariableOp_1б
2Generator/Gen_BN_8/FusedBatchNormV3/ReadVariableOpReadVariableOp;generator_gen_bn_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype024
2Generator/Gen_BN_8/FusedBatchNormV3/ReadVariableOpз
4Generator/Gen_BN_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=generator_gen_bn_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype026
4Generator/Gen_BN_8/FusedBatchNormV3/ReadVariableOp_1ч
#Generator/Gen_BN_8/FusedBatchNormV3FusedBatchNormV3<Generator/Gen_Conv_T_3/leaky_re_lu_7/LeakyRelu:activations:0)Generator/Gen_BN_8/ReadVariableOp:value:0+Generator/Gen_BN_8/ReadVariableOp_1:value:0:Generator/Gen_BN_8/FusedBatchNormV3/ReadVariableOp:value:0<Generator/Gen_BN_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€0(А:А:А:А:А:*
epsilon%oГ:*
is_training( 2%
#Generator/Gen_BN_8/FusedBatchNormV3К
"Generator/Gen_Concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"Generator/Gen_Concat_3/concat/axisН
Generator/Gen_Concat_3/concatConcatV2'Generator/Gen_BN_8/FusedBatchNormV3:y:0'Generator/Gen_BN_2/FusedBatchNormV3:y:0+Generator/Gen_Concat_3/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€0(А2
Generator/Gen_Concat_3/concatЂ
Generator/Gen_SPD_5/IdentityIdentity&Generator/Gen_Concat_3/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2
Generator/Gen_SPD_5/IdentityС
Generator/Gen_Conv_T_4/ShapeShape%Generator/Gen_SPD_5/Identity:output:0*
T0*
_output_shapes
:2
Generator/Gen_Conv_T_4/ShapeҐ
*Generator/Gen_Conv_T_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*Generator/Gen_Conv_T_4/strided_slice/stack¶
,Generator/Gen_Conv_T_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,Generator/Gen_Conv_T_4/strided_slice/stack_1¶
,Generator/Gen_Conv_T_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Generator/Gen_Conv_T_4/strided_slice/stack_2м
$Generator/Gen_Conv_T_4/strided_sliceStridedSlice%Generator/Gen_Conv_T_4/Shape:output:03Generator/Gen_Conv_T_4/strided_slice/stack:output:05Generator/Gen_Conv_T_4/strided_slice/stack_1:output:05Generator/Gen_Conv_T_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$Generator/Gen_Conv_T_4/strided_sliceВ
Generator/Gen_Conv_T_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :`2 
Generator/Gen_Conv_T_4/stack/1В
Generator/Gen_Conv_T_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :P2 
Generator/Gen_Conv_T_4/stack/2В
Generator/Gen_Conv_T_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2 
Generator/Gen_Conv_T_4/stack/3Ь
Generator/Gen_Conv_T_4/stackPack-Generator/Gen_Conv_T_4/strided_slice:output:0'Generator/Gen_Conv_T_4/stack/1:output:0'Generator/Gen_Conv_T_4/stack/2:output:0'Generator/Gen_Conv_T_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/Gen_Conv_T_4/stack¶
,Generator/Gen_Conv_T_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,Generator/Gen_Conv_T_4/strided_slice_1/stack™
.Generator/Gen_Conv_T_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.Generator/Gen_Conv_T_4/strided_slice_1/stack_1™
.Generator/Gen_Conv_T_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.Generator/Gen_Conv_T_4/strided_slice_1/stack_2ц
&Generator/Gen_Conv_T_4/strided_slice_1StridedSlice%Generator/Gen_Conv_T_4/stack:output:05Generator/Gen_Conv_T_4/strided_slice_1/stack:output:07Generator/Gen_Conv_T_4/strided_slice_1/stack_1:output:07Generator/Gen_Conv_T_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&Generator/Gen_Conv_T_4/strided_slice_1щ
6Generator/Gen_Conv_T_4/conv2d_transpose/ReadVariableOpReadVariableOp?generator_gen_conv_t_4_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@А*
dtype028
6Generator/Gen_Conv_T_4/conv2d_transpose/ReadVariableOpў
'Generator/Gen_Conv_T_4/conv2d_transposeConv2DBackpropInput%Generator/Gen_Conv_T_4/stack:output:0>Generator/Gen_Conv_T_4/conv2d_transpose/ReadVariableOp:value:0%Generator/Gen_SPD_5/Identity:output:0*
T0*/
_output_shapes
:€€€€€€€€€`P@*
paddingSAME*
strides
2)
'Generator/Gen_Conv_T_4/conv2d_transpose—
-Generator/Gen_Conv_T_4/BiasAdd/ReadVariableOpReadVariableOp6generator_gen_conv_t_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-Generator/Gen_Conv_T_4/BiasAdd/ReadVariableOpо
Generator/Gen_Conv_T_4/BiasAddBiasAdd0Generator/Gen_Conv_T_4/conv2d_transpose:output:05Generator/Gen_Conv_T_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€`P@2 
Generator/Gen_Conv_T_4/BiasAdd«
.Generator/Gen_Conv_T_4/leaky_re_lu_8/LeakyRelu	LeakyRelu'Generator/Gen_Conv_T_4/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€`P@20
.Generator/Gen_Conv_T_4/leaky_re_lu_8/LeakyRelu≠
!Generator/Gen_BN_9/ReadVariableOpReadVariableOp*generator_gen_bn_9_readvariableop_resource*
_output_shapes
:@*
dtype02#
!Generator/Gen_BN_9/ReadVariableOp≥
#Generator/Gen_BN_9/ReadVariableOp_1ReadVariableOp,generator_gen_bn_9_readvariableop_1_resource*
_output_shapes
:@*
dtype02%
#Generator/Gen_BN_9/ReadVariableOp_1а
2Generator/Gen_BN_9/FusedBatchNormV3/ReadVariableOpReadVariableOp;generator_gen_bn_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype024
2Generator/Gen_BN_9/FusedBatchNormV3/ReadVariableOpж
4Generator/Gen_BN_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=generator_gen_bn_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype026
4Generator/Gen_BN_9/FusedBatchNormV3/ReadVariableOp_1т
#Generator/Gen_BN_9/FusedBatchNormV3FusedBatchNormV3<Generator/Gen_Conv_T_4/leaky_re_lu_8/LeakyRelu:activations:0)Generator/Gen_BN_9/ReadVariableOp:value:0+Generator/Gen_BN_9/ReadVariableOp_1:value:0:Generator/Gen_BN_9/FusedBatchNormV3/ReadVariableOp:value:0<Generator/Gen_BN_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€`P@:@:@:@:@:*
epsilon%oГ:*
is_training( 2%
#Generator/Gen_BN_9/FusedBatchNormV3К
"Generator/Gen_Concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"Generator/Gen_Concat_4/concat/axisН
Generator/Gen_Concat_4/concatConcatV2'Generator/Gen_BN_9/FusedBatchNormV3:y:0'Generator/Gen_BN_1/FusedBatchNormV3:y:0+Generator/Gen_Concat_4/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€`PА2
Generator/Gen_Concat_4/concatТ
Generator/Gen_Conv_T_5/ShapeShape&Generator/Gen_Concat_4/concat:output:0*
T0*
_output_shapes
:2
Generator/Gen_Conv_T_5/ShapeҐ
*Generator/Gen_Conv_T_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*Generator/Gen_Conv_T_5/strided_slice/stack¶
,Generator/Gen_Conv_T_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,Generator/Gen_Conv_T_5/strided_slice/stack_1¶
,Generator/Gen_Conv_T_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Generator/Gen_Conv_T_5/strided_slice/stack_2м
$Generator/Gen_Conv_T_5/strided_sliceStridedSlice%Generator/Gen_Conv_T_5/Shape:output:03Generator/Gen_Conv_T_5/strided_slice/stack:output:05Generator/Gen_Conv_T_5/strided_slice/stack_1:output:05Generator/Gen_Conv_T_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$Generator/Gen_Conv_T_5/strided_sliceГ
Generator/Gen_Conv_T_5/stack/1Const*
_output_shapes
: *
dtype0*
value
B :ј2 
Generator/Gen_Conv_T_5/stack/1Г
Generator/Gen_Conv_T_5/stack/2Const*
_output_shapes
: *
dtype0*
value
B :†2 
Generator/Gen_Conv_T_5/stack/2В
Generator/Gen_Conv_T_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2 
Generator/Gen_Conv_T_5/stack/3Ь
Generator/Gen_Conv_T_5/stackPack-Generator/Gen_Conv_T_5/strided_slice:output:0'Generator/Gen_Conv_T_5/stack/1:output:0'Generator/Gen_Conv_T_5/stack/2:output:0'Generator/Gen_Conv_T_5/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/Gen_Conv_T_5/stack¶
,Generator/Gen_Conv_T_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,Generator/Gen_Conv_T_5/strided_slice_1/stack™
.Generator/Gen_Conv_T_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.Generator/Gen_Conv_T_5/strided_slice_1/stack_1™
.Generator/Gen_Conv_T_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.Generator/Gen_Conv_T_5/strided_slice_1/stack_2ц
&Generator/Gen_Conv_T_5/strided_slice_1StridedSlice%Generator/Gen_Conv_T_5/stack:output:05Generator/Gen_Conv_T_5/strided_slice_1/stack:output:07Generator/Gen_Conv_T_5/strided_slice_1/stack_1:output:07Generator/Gen_Conv_T_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&Generator/Gen_Conv_T_5/strided_slice_1щ
6Generator/Gen_Conv_T_5/conv2d_transpose/ReadVariableOpReadVariableOp?generator_gen_conv_t_5_conv2d_transpose_readvariableop_resource*'
_output_shapes
: А*
dtype028
6Generator/Gen_Conv_T_5/conv2d_transpose/ReadVariableOp№
'Generator/Gen_Conv_T_5/conv2d_transposeConv2DBackpropInput%Generator/Gen_Conv_T_5/stack:output:0>Generator/Gen_Conv_T_5/conv2d_transpose/ReadVariableOp:value:0&Generator/Gen_Concat_4/concat:output:0*
T0*1
_output_shapes
:€€€€€€€€€ј† *
paddingSAME*
strides
2)
'Generator/Gen_Conv_T_5/conv2d_transpose—
-Generator/Gen_Conv_T_5/BiasAdd/ReadVariableOpReadVariableOp6generator_gen_conv_t_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-Generator/Gen_Conv_T_5/BiasAdd/ReadVariableOpр
Generator/Gen_Conv_T_5/BiasAddBiasAdd0Generator/Gen_Conv_T_5/conv2d_transpose:output:05Generator/Gen_Conv_T_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ј† 2 
Generator/Gen_Conv_T_5/BiasAdd…
.Generator/Gen_Conv_T_5/leaky_re_lu_9/LeakyRelu	LeakyRelu'Generator/Gen_Conv_T_5/BiasAdd:output:0*1
_output_shapes
:€€€€€€€€€ј† 20
.Generator/Gen_Conv_T_5/leaky_re_lu_9/LeakyRelu∞
"Generator/Gen_BN_10/ReadVariableOpReadVariableOp+generator_gen_bn_10_readvariableop_resource*
_output_shapes
: *
dtype02$
"Generator/Gen_BN_10/ReadVariableOpґ
$Generator/Gen_BN_10/ReadVariableOp_1ReadVariableOp-generator_gen_bn_10_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$Generator/Gen_BN_10/ReadVariableOp_1г
3Generator/Gen_BN_10/FusedBatchNormV3/ReadVariableOpReadVariableOp<generator_gen_bn_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3Generator/Gen_BN_10/FusedBatchNormV3/ReadVariableOpй
5Generator/Gen_BN_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>generator_gen_bn_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5Generator/Gen_BN_10/FusedBatchNormV3/ReadVariableOp_1ъ
$Generator/Gen_BN_10/FusedBatchNormV3FusedBatchNormV3<Generator/Gen_Conv_T_5/leaky_re_lu_9/LeakyRelu:activations:0*Generator/Gen_BN_10/ReadVariableOp:value:0,Generator/Gen_BN_10/ReadVariableOp_1:value:0;Generator/Gen_BN_10/FusedBatchNormV3/ReadVariableOp:value:0=Generator/Gen_BN_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:€€€€€€€€€ј† : : : : :*
epsilon%oГ:*
is_training( 2&
$Generator/Gen_BN_10/FusedBatchNormV3Ф
Generator/Gen_Conv_T_6/ShapeShape(Generator/Gen_BN_10/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Generator/Gen_Conv_T_6/ShapeҐ
*Generator/Gen_Conv_T_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*Generator/Gen_Conv_T_6/strided_slice/stack¶
,Generator/Gen_Conv_T_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,Generator/Gen_Conv_T_6/strided_slice/stack_1¶
,Generator/Gen_Conv_T_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Generator/Gen_Conv_T_6/strided_slice/stack_2м
$Generator/Gen_Conv_T_6/strided_sliceStridedSlice%Generator/Gen_Conv_T_6/Shape:output:03Generator/Gen_Conv_T_6/strided_slice/stack:output:05Generator/Gen_Conv_T_6/strided_slice/stack_1:output:05Generator/Gen_Conv_T_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$Generator/Gen_Conv_T_6/strided_sliceГ
Generator/Gen_Conv_T_6/stack/1Const*
_output_shapes
: *
dtype0*
value
B :ј2 
Generator/Gen_Conv_T_6/stack/1Г
Generator/Gen_Conv_T_6/stack/2Const*
_output_shapes
: *
dtype0*
value
B :†2 
Generator/Gen_Conv_T_6/stack/2В
Generator/Gen_Conv_T_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2 
Generator/Gen_Conv_T_6/stack/3Ь
Generator/Gen_Conv_T_6/stackPack-Generator/Gen_Conv_T_6/strided_slice:output:0'Generator/Gen_Conv_T_6/stack/1:output:0'Generator/Gen_Conv_T_6/stack/2:output:0'Generator/Gen_Conv_T_6/stack/3:output:0*
N*
T0*
_output_shapes
:2
Generator/Gen_Conv_T_6/stack¶
,Generator/Gen_Conv_T_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,Generator/Gen_Conv_T_6/strided_slice_1/stack™
.Generator/Gen_Conv_T_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.Generator/Gen_Conv_T_6/strided_slice_1/stack_1™
.Generator/Gen_Conv_T_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.Generator/Gen_Conv_T_6/strided_slice_1/stack_2ц
&Generator/Gen_Conv_T_6/strided_slice_1StridedSlice%Generator/Gen_Conv_T_6/stack:output:05Generator/Gen_Conv_T_6/strided_slice_1/stack:output:07Generator/Gen_Conv_T_6/strided_slice_1/stack_1:output:07Generator/Gen_Conv_T_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&Generator/Gen_Conv_T_6/strided_slice_1ш
6Generator/Gen_Conv_T_6/conv2d_transpose/ReadVariableOpReadVariableOp?generator_gen_conv_t_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype028
6Generator/Gen_Conv_T_6/conv2d_transpose/ReadVariableOpё
'Generator/Gen_Conv_T_6/conv2d_transposeConv2DBackpropInput%Generator/Gen_Conv_T_6/stack:output:0>Generator/Gen_Conv_T_6/conv2d_transpose/ReadVariableOp:value:0(Generator/Gen_BN_10/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:€€€€€€€€€ј†*
paddingSAME*
strides
2)
'Generator/Gen_Conv_T_6/conv2d_transpose—
-Generator/Gen_Conv_T_6/BiasAdd/ReadVariableOpReadVariableOp6generator_gen_conv_t_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-Generator/Gen_Conv_T_6/BiasAdd/ReadVariableOpр
Generator/Gen_Conv_T_6/BiasAddBiasAdd0Generator/Gen_Conv_T_6/conv2d_transpose:output:05Generator/Gen_Conv_T_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ј†2 
Generator/Gen_Conv_T_6/BiasAddІ
Generator/Gen_Conv_T_6/TanhTanh'Generator/Gen_Conv_T_6/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€ј†2
Generator/Gen_Conv_T_6/Tanh}
IdentityIdentityGenerator/Gen_Conv_T_6/Tanh:y:0*
T0*1
_output_shapes
:€€€€€€€€€ј†2

Identity"
identityIdentity:output:0*™
_input_shapesШ
Х:€€€€€€€€€ј†:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\ X
1
_output_shapes
:€€€€€€€€€ј†
#
_user_specified_name	Gen_Input
М
Ь
)__inference_Gen_BN_5_layer_call_fn_145588

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_5_layer_call_and_return_conditional_losses_1408402
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
м
c
E__inference_Gen_SPD_5_layer_call_and_return_conditional_losses_146101

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:€€€€€€€€€0(А:X T
0
_output_shapes
:€€€€€€€€€0(А
 
_user_specified_nameinputs
І%
Ї
H__inference_Gen_Conv_T_3_layer_call_and_return_conditional_losses_141356

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
strided_slice_2/stack_2м
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
B :А2	
stack/3В
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
strided_slice_3/stack_2м
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3µ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_transpose/ReadVariableOpс
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
2
conv2d_transposeН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOp•
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2	
BiasAddЛ
leaky_re_lu_7/PartitionedCallPartitionedCallBiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_1413472
leaky_re_lu_7/PartitionedCallХ
IdentityIdentity&leaky_re_lu_7/PartitionedCall:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
М
J
.__inference_leaky_re_lu_5_layer_call_fn_146262

inputs
identityе
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_1409572
PartitionedCallЗ
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ґ
E
)__inference_Gen_MP_4_layer_call_fn_140563

inputs
identityи
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_MP_4_layer_call_and_return_conditional_losses_1405572
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ї
F
*__inference_Gen_SPD_2_layer_call_fn_145466

inputs
identityѕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_2_layer_call_and_return_conditional_losses_1423792
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€
А:X T
0
_output_shapes
:€€€€€€€€€
А
 
_user_specified_nameinputs
Ч
t
H__inference_Gen_Concat_2_layer_call_and_return_conditional_losses_145952
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisК
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:€€€€€€€€€А:l h
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:€€€€€€€€€А
"
_user_specified_name
inputs/1
И
Ь
)__inference_Gen_BN_1_layer_call_fn_144908

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_1_layer_call_and_return_conditional_losses_1402402
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ј
Ь
)__inference_Gen_BN_1_layer_call_fn_144844

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`P@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_1_layer_call_and_return_conditional_losses_1419742
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€`P@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€`P@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€`P@
 
_user_specified_nameinputs
А
°
D__inference_Gen_BN_3_layer_call_and_return_conditional_losses_145236

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ё
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Х
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Љ
°
D__inference_Gen_BN_9_layer_call_and_return_conditional_losses_146131

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¶
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
’
c
E__inference_Gen_SPD_1_layer_call_and_return_conditional_losses_145122

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

IdentityМ

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
»
°
D__inference_Gen_BN_8_layer_call_and_return_conditional_losses_141428

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1п
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1І
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
м
c
E__inference_Gen_SPD_1_layer_call_and_return_conditional_losses_145084

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:€€€€€€€€€0(А:X T
0
_output_shapes
:€€€€€€€€€0(А
 
_user_specified_nameinputs
Э
d
E__inference_Gen_SPD_5_layer_call_and_return_conditional_losses_146058

inputs
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
 *  †?2
dropout/ConstЦ
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2
dropout/MulВ
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1В
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ц
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape‘
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/yѕ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/GreaterEqualР
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/CastЭ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2
dropout/Mul_1И
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ъ
`
D__inference_Gen_MP_1_layer_call_and_return_conditional_losses_140141

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
’
c
E__inference_Gen_SPD_4_layer_call_and_return_conditional_losses_141145

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

IdentityМ

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
М
Ь
)__inference_Gen_BN_8_layer_call_fn_146022

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_8_layer_call_and_return_conditional_losses_1414592
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
У
t
H__inference_Gen_Concat_4_layer_call_and_return_conditional_losses_146182
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisК
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€`PА2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:€€€€€€€€€`PА2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:€€€€€€€€€`P@:k g
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:€€€€€€€€€`P@
"
_user_specified_name
inputs/1
ъ
`
D__inference_Gen_MP_4_layer_call_and_return_conditional_losses_140557

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ѓ
c
*__inference_Gen_SPD_4_layer_call_fn_145838

inputs
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_4_layer_call_and_return_conditional_losses_1411352
StatefulPartitionedCall±
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
К
Э
*__inference_Gen_BN_10_layer_call_fn_146252

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_BN_10_layer_call_and_return_conditional_losses_1418492
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
И
Э
*__inference_Gen_BN_10_layer_call_fn_146239

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_BN_10_layer_call_and_return_conditional_losses_1418182
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
«
c
*__inference_Gen_SPD_1_layer_call_fn_145089

inputs
identityИҐStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_1_layer_call_and_return_conditional_losses_1421332
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€0(А22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€0(А
 
_user_specified_nameinputs
Ш
э
D__inference_Gen_BN_4_layer_call_and_return_conditional_losses_145338

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1б
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
К
Ь
)__inference_Gen_BN_4_layer_call_fn_145351

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_4_layer_call_and_return_conditional_losses_1406252
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
К
Ь
)__inference_Gen_BN_3_layer_call_fn_145203

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_3_layer_call_and_return_conditional_losses_1405092
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
м
c
E__inference_Gen_SPD_1_layer_call_and_return_conditional_losses_142138

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:€€€€€€€€€0(А:X T
0
_output_shapes
:€€€€€€€€€0(А
 
_user_specified_nameinputs
м
c
E__inference_Gen_SPD_2_layer_call_and_return_conditional_losses_145456

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:€€€€€€€€€
А:X T
0
_output_shapes
:€€€€€€€€€
А
 
_user_specified_nameinputs
К
Ь
)__inference_Gen_BN_6_layer_call_fn_145779

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_6_layer_call_and_return_conditional_losses_1410382
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
¬
Ь
)__inference_Gen_BN_5_layer_call_fn_145639

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_5_layer_call_and_return_conditional_losses_1424382
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Нђ
ш
E__inference_Generator_layer_call_and_return_conditional_losses_143208

inputs
gen_conv_1_143048
gen_conv_1_143050
gen_bn_1_143054
gen_bn_1_143056
gen_bn_1_143058
gen_bn_1_143060
gen_conv_2_143063
gen_conv_2_143065
gen_bn_2_143069
gen_bn_2_143071
gen_bn_2_143073
gen_bn_2_143075
gen_conv_3_143079
gen_conv_3_143081
gen_bn_3_143085
gen_bn_3_143087
gen_bn_3_143089
gen_bn_3_143091
gen_conv_4_143094
gen_conv_4_143096
gen_bn_4_143100
gen_bn_4_143102
gen_bn_4_143104
gen_bn_4_143106
gen_conv_5_143110
gen_conv_5_143112
gen_bn_5_143116
gen_bn_5_143118
gen_bn_5_143120
gen_bn_5_143122
gen_conv_t_1_143126
gen_conv_t_1_143128
gen_bn_6_143131
gen_bn_6_143133
gen_bn_6_143135
gen_bn_6_143137
gen_conv_t_2_143142
gen_conv_t_2_143144
gen_bn_7_143147
gen_bn_7_143149
gen_bn_7_143151
gen_bn_7_143153
gen_conv_t_3_143157
gen_conv_t_3_143159
gen_bn_8_143162
gen_bn_8_143164
gen_bn_8_143166
gen_bn_8_143168
gen_conv_t_4_143173
gen_conv_t_4_143175
gen_bn_9_143178
gen_bn_9_143180
gen_bn_9_143182
gen_bn_9_143184
gen_conv_t_5_143188
gen_conv_t_5_143190
gen_bn_10_143193
gen_bn_10_143195
gen_bn_10_143197
gen_bn_10_143199
gen_conv_t_6_143202
gen_conv_t_6_143204
identityИҐ Gen_BN_1/StatefulPartitionedCallҐ!Gen_BN_10/StatefulPartitionedCallҐ Gen_BN_2/StatefulPartitionedCallҐ Gen_BN_3/StatefulPartitionedCallҐ Gen_BN_4/StatefulPartitionedCallҐ Gen_BN_5/StatefulPartitionedCallҐ Gen_BN_6/StatefulPartitionedCallҐ Gen_BN_7/StatefulPartitionedCallҐ Gen_BN_8/StatefulPartitionedCallҐ Gen_BN_9/StatefulPartitionedCallҐ"Gen_Conv_1/StatefulPartitionedCallҐ"Gen_Conv_2/StatefulPartitionedCallҐ"Gen_Conv_3/StatefulPartitionedCallҐ"Gen_Conv_4/StatefulPartitionedCallҐ"Gen_Conv_5/StatefulPartitionedCallҐ$Gen_Conv_T_1/StatefulPartitionedCallҐ$Gen_Conv_T_2/StatefulPartitionedCallҐ$Gen_Conv_T_3/StatefulPartitionedCallҐ$Gen_Conv_T_4/StatefulPartitionedCallҐ$Gen_Conv_T_5/StatefulPartitionedCallҐ$Gen_Conv_T_6/StatefulPartitionedCallҐ!Gen_SPD_1/StatefulPartitionedCallҐ!Gen_SPD_2/StatefulPartitionedCallҐ!Gen_SPD_3/StatefulPartitionedCallҐ!Gen_SPD_4/StatefulPartitionedCallҐ!Gen_SPD_5/StatefulPartitionedCallЂ
"Gen_Conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsgen_conv_1_143048gen_conv_1_143050*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ј†@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_Gen_Conv_1_layer_call_and_return_conditional_losses_1419202$
"Gen_Conv_1/StatefulPartitionedCallД
Gen_MP_1/PartitionedCallPartitionedCall+Gen_Conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`P@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_MP_1_layer_call_and_return_conditional_losses_1401412
Gen_MP_1/PartitionedCallё
 Gen_BN_1/StatefulPartitionedCallStatefulPartitionedCall!Gen_MP_1/PartitionedCall:output:0gen_bn_1_143054gen_bn_1_143056gen_bn_1_143058gen_bn_1_143060*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`P@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_1_layer_call_and_return_conditional_losses_1419562"
 Gen_BN_1/StatefulPartitionedCallЌ
"Gen_Conv_2/StatefulPartitionedCallStatefulPartitionedCall)Gen_BN_1/StatefulPartitionedCall:output:0gen_conv_2_143063gen_conv_2_143065*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€`PА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_Gen_Conv_2_layer_call_and_return_conditional_losses_1420212$
"Gen_Conv_2/StatefulPartitionedCallЕ
Gen_MP_2/PartitionedCallPartitionedCall+Gen_Conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_MP_2_layer_call_and_return_conditional_losses_1402572
Gen_MP_2/PartitionedCallя
 Gen_BN_2/StatefulPartitionedCallStatefulPartitionedCall!Gen_MP_2/PartitionedCall:output:0gen_bn_2_143069gen_bn_2_143071gen_bn_2_143073gen_bn_2_143075*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_2_layer_call_and_return_conditional_losses_1420572"
 Gen_BN_2/StatefulPartitionedCallЮ
!Gen_SPD_1/StatefulPartitionedCallStatefulPartitionedCall)Gen_BN_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_1_layer_call_and_return_conditional_losses_1421332#
!Gen_SPD_1/StatefulPartitionedCallќ
"Gen_Conv_3/StatefulPartitionedCallStatefulPartitionedCall*Gen_SPD_1/StatefulPartitionedCall:output:0gen_conv_3_143079gen_conv_3_143081*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_Gen_Conv_3_layer_call_and_return_conditional_losses_1421612$
"Gen_Conv_3/StatefulPartitionedCallЕ
Gen_MP_3/PartitionedCallPartitionedCall+Gen_Conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_MP_3_layer_call_and_return_conditional_losses_1404412
Gen_MP_3/PartitionedCallя
 Gen_BN_3/StatefulPartitionedCallStatefulPartitionedCall!Gen_MP_3/PartitionedCall:output:0gen_bn_3_143085gen_bn_3_143087gen_bn_3_143089gen_bn_3_143091*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_3_layer_call_and_return_conditional_losses_1421972"
 Gen_BN_3/StatefulPartitionedCallЌ
"Gen_Conv_4/StatefulPartitionedCallStatefulPartitionedCall)Gen_BN_3/StatefulPartitionedCall:output:0gen_conv_4_143094gen_conv_4_143096*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_Gen_Conv_4_layer_call_and_return_conditional_losses_1422622$
"Gen_Conv_4/StatefulPartitionedCallЕ
Gen_MP_4/PartitionedCallPartitionedCall+Gen_Conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_MP_4_layer_call_and_return_conditional_losses_1405572
Gen_MP_4/PartitionedCallя
 Gen_BN_4/StatefulPartitionedCallStatefulPartitionedCall!Gen_MP_4/PartitionedCall:output:0gen_bn_4_143100gen_bn_4_143102gen_bn_4_143104gen_bn_4_143106*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_4_layer_call_and_return_conditional_losses_1422982"
 Gen_BN_4/StatefulPartitionedCall¬
!Gen_SPD_2/StatefulPartitionedCallStatefulPartitionedCall)Gen_BN_4/StatefulPartitionedCall:output:0"^Gen_SPD_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_2_layer_call_and_return_conditional_losses_1423742#
!Gen_SPD_2/StatefulPartitionedCallќ
"Gen_Conv_5/StatefulPartitionedCallStatefulPartitionedCall*Gen_SPD_2/StatefulPartitionedCall:output:0gen_conv_5_143110gen_conv_5_143112*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_Gen_Conv_5_layer_call_and_return_conditional_losses_1424022$
"Gen_Conv_5/StatefulPartitionedCallЕ
Gen_MP_5/PartitionedCallPartitionedCall+Gen_Conv_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_MP_5_layer_call_and_return_conditional_losses_1407412
Gen_MP_5/PartitionedCallя
 Gen_BN_5/StatefulPartitionedCallStatefulPartitionedCall!Gen_MP_5/PartitionedCall:output:0gen_bn_5_143116gen_bn_5_143118gen_bn_5_143120gen_bn_5_143122*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_5_layer_call_and_return_conditional_losses_1424382"
 Gen_BN_5/StatefulPartitionedCall¬
!Gen_SPD_3/StatefulPartitionedCallStatefulPartitionedCall)Gen_BN_5/StatefulPartitionedCall:output:0"^Gen_SPD_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_3_layer_call_and_return_conditional_losses_1425142#
!Gen_SPD_3/StatefulPartitionedCallк
$Gen_Conv_T_1/StatefulPartitionedCallStatefulPartitionedCall*Gen_SPD_3/StatefulPartitionedCall:output:0gen_conv_t_1_143126gen_conv_t_1_143128*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_1_layer_call_and_return_conditional_losses_1409662&
$Gen_Conv_T_1/StatefulPartitionedCallэ
 Gen_BN_6/StatefulPartitionedCallStatefulPartitionedCall-Gen_Conv_T_1/StatefulPartitionedCall:output:0gen_bn_6_143131gen_bn_6_143133gen_bn_6_143135gen_bn_6_143137*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_6_layer_call_and_return_conditional_losses_1410382"
 Gen_BN_6/StatefulPartitionedCallї
Gen_Concat_1/PartitionedCallPartitionedCall)Gen_BN_6/StatefulPartitionedCall:output:0)Gen_BN_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Concat_1_layer_call_and_return_conditional_losses_1425782
Gen_Concat_1/PartitionedCallЊ
!Gen_SPD_4/StatefulPartitionedCallStatefulPartitionedCall%Gen_Concat_1/PartitionedCall:output:0"^Gen_SPD_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_4_layer_call_and_return_conditional_losses_1426092#
!Gen_SPD_4/StatefulPartitionedCallк
$Gen_Conv_T_2/StatefulPartitionedCallStatefulPartitionedCall*Gen_SPD_4/StatefulPartitionedCall:output:0gen_conv_t_2_143142gen_conv_t_2_143144*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_2_layer_call_and_return_conditional_losses_1411952&
$Gen_Conv_T_2/StatefulPartitionedCallэ
 Gen_BN_7/StatefulPartitionedCallStatefulPartitionedCall-Gen_Conv_T_2/StatefulPartitionedCall:output:0gen_bn_7_143147gen_bn_7_143149gen_bn_7_143151gen_bn_7_143153*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_7_layer_call_and_return_conditional_losses_1412672"
 Gen_BN_7/StatefulPartitionedCallї
Gen_Concat_2/PartitionedCallPartitionedCall)Gen_BN_7/StatefulPartitionedCall:output:0)Gen_BN_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Concat_2_layer_call_and_return_conditional_losses_1426732
Gen_Concat_2/PartitionedCallе
$Gen_Conv_T_3/StatefulPartitionedCallStatefulPartitionedCall%Gen_Concat_2/PartitionedCall:output:0gen_conv_t_3_143157gen_conv_t_3_143159*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_3_layer_call_and_return_conditional_losses_1413562&
$Gen_Conv_T_3/StatefulPartitionedCallэ
 Gen_BN_8/StatefulPartitionedCallStatefulPartitionedCall-Gen_Conv_T_3/StatefulPartitionedCall:output:0gen_bn_8_143162gen_bn_8_143164gen_bn_8_143166gen_bn_8_143168*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_8_layer_call_and_return_conditional_losses_1414282"
 Gen_BN_8/StatefulPartitionedCallї
Gen_Concat_3/PartitionedCallPartitionedCall)Gen_BN_8/StatefulPartitionedCall:output:0)Gen_BN_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Concat_3_layer_call_and_return_conditional_losses_1427292
Gen_Concat_3/PartitionedCallЊ
!Gen_SPD_5/StatefulPartitionedCallStatefulPartitionedCall%Gen_Concat_3/PartitionedCall:output:0"^Gen_SPD_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_5_layer_call_and_return_conditional_losses_1427602#
!Gen_SPD_5/StatefulPartitionedCallй
$Gen_Conv_T_4/StatefulPartitionedCallStatefulPartitionedCall*Gen_SPD_5/StatefulPartitionedCall:output:0gen_conv_t_4_143173gen_conv_t_4_143175*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_4_layer_call_and_return_conditional_losses_1415852&
$Gen_Conv_T_4/StatefulPartitionedCallь
 Gen_BN_9/StatefulPartitionedCallStatefulPartitionedCall-Gen_Conv_T_4/StatefulPartitionedCall:output:0gen_bn_9_143178gen_bn_9_143180gen_bn_9_143182gen_bn_9_143184*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_9_layer_call_and_return_conditional_losses_1416572"
 Gen_BN_9/StatefulPartitionedCallї
Gen_Concat_4/PartitionedCallPartitionedCall)Gen_BN_9/StatefulPartitionedCall:output:0)Gen_BN_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€`PА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Concat_4_layer_call_and_return_conditional_losses_1428242
Gen_Concat_4/PartitionedCallд
$Gen_Conv_T_5/StatefulPartitionedCallStatefulPartitionedCall%Gen_Concat_4/PartitionedCall:output:0gen_conv_t_5_143188gen_conv_t_5_143190*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_5_layer_call_and_return_conditional_losses_1417462&
$Gen_Conv_T_5/StatefulPartitionedCallГ
!Gen_BN_10/StatefulPartitionedCallStatefulPartitionedCall-Gen_Conv_T_5/StatefulPartitionedCall:output:0gen_bn_10_143193gen_bn_10_143195gen_bn_10_143197gen_bn_10_143199*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_BN_10_layer_call_and_return_conditional_losses_1418182#
!Gen_BN_10/StatefulPartitionedCallй
$Gen_Conv_T_6/StatefulPartitionedCallStatefulPartitionedCall*Gen_BN_10/StatefulPartitionedCall:output:0gen_conv_t_6_143202gen_conv_t_6_143204*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_6_layer_call_and_return_conditional_losses_1418952&
$Gen_Conv_T_6/StatefulPartitionedCall—
IdentityIdentity-Gen_Conv_T_6/StatefulPartitionedCall:output:0!^Gen_BN_1/StatefulPartitionedCall"^Gen_BN_10/StatefulPartitionedCall!^Gen_BN_2/StatefulPartitionedCall!^Gen_BN_3/StatefulPartitionedCall!^Gen_BN_4/StatefulPartitionedCall!^Gen_BN_5/StatefulPartitionedCall!^Gen_BN_6/StatefulPartitionedCall!^Gen_BN_7/StatefulPartitionedCall!^Gen_BN_8/StatefulPartitionedCall!^Gen_BN_9/StatefulPartitionedCall#^Gen_Conv_1/StatefulPartitionedCall#^Gen_Conv_2/StatefulPartitionedCall#^Gen_Conv_3/StatefulPartitionedCall#^Gen_Conv_4/StatefulPartitionedCall#^Gen_Conv_5/StatefulPartitionedCall%^Gen_Conv_T_1/StatefulPartitionedCall%^Gen_Conv_T_2/StatefulPartitionedCall%^Gen_Conv_T_3/StatefulPartitionedCall%^Gen_Conv_T_4/StatefulPartitionedCall%^Gen_Conv_T_5/StatefulPartitionedCall%^Gen_Conv_T_6/StatefulPartitionedCall"^Gen_SPD_1/StatefulPartitionedCall"^Gen_SPD_2/StatefulPartitionedCall"^Gen_SPD_3/StatefulPartitionedCall"^Gen_SPD_4/StatefulPartitionedCall"^Gen_SPD_5/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*™
_input_shapesШ
Х:€€€€€€€€€ј†::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2D
 Gen_BN_1/StatefulPartitionedCall Gen_BN_1/StatefulPartitionedCall2F
!Gen_BN_10/StatefulPartitionedCall!Gen_BN_10/StatefulPartitionedCall2D
 Gen_BN_2/StatefulPartitionedCall Gen_BN_2/StatefulPartitionedCall2D
 Gen_BN_3/StatefulPartitionedCall Gen_BN_3/StatefulPartitionedCall2D
 Gen_BN_4/StatefulPartitionedCall Gen_BN_4/StatefulPartitionedCall2D
 Gen_BN_5/StatefulPartitionedCall Gen_BN_5/StatefulPartitionedCall2D
 Gen_BN_6/StatefulPartitionedCall Gen_BN_6/StatefulPartitionedCall2D
 Gen_BN_7/StatefulPartitionedCall Gen_BN_7/StatefulPartitionedCall2D
 Gen_BN_8/StatefulPartitionedCall Gen_BN_8/StatefulPartitionedCall2D
 Gen_BN_9/StatefulPartitionedCall Gen_BN_9/StatefulPartitionedCall2H
"Gen_Conv_1/StatefulPartitionedCall"Gen_Conv_1/StatefulPartitionedCall2H
"Gen_Conv_2/StatefulPartitionedCall"Gen_Conv_2/StatefulPartitionedCall2H
"Gen_Conv_3/StatefulPartitionedCall"Gen_Conv_3/StatefulPartitionedCall2H
"Gen_Conv_4/StatefulPartitionedCall"Gen_Conv_4/StatefulPartitionedCall2H
"Gen_Conv_5/StatefulPartitionedCall"Gen_Conv_5/StatefulPartitionedCall2L
$Gen_Conv_T_1/StatefulPartitionedCall$Gen_Conv_T_1/StatefulPartitionedCall2L
$Gen_Conv_T_2/StatefulPartitionedCall$Gen_Conv_T_2/StatefulPartitionedCall2L
$Gen_Conv_T_3/StatefulPartitionedCall$Gen_Conv_T_3/StatefulPartitionedCall2L
$Gen_Conv_T_4/StatefulPartitionedCall$Gen_Conv_T_4/StatefulPartitionedCall2L
$Gen_Conv_T_5/StatefulPartitionedCall$Gen_Conv_T_5/StatefulPartitionedCall2L
$Gen_Conv_T_6/StatefulPartitionedCall$Gen_Conv_T_6/StatefulPartitionedCall2F
!Gen_SPD_1/StatefulPartitionedCall!Gen_SPD_1/StatefulPartitionedCall2F
!Gen_SPD_2/StatefulPartitionedCall!Gen_SPD_2/StatefulPartitionedCall2F
!Gen_SPD_3/StatefulPartitionedCall!Gen_SPD_3/StatefulPartitionedCall2F
!Gen_SPD_4/StatefulPartitionedCall!Gen_SPD_4/StatefulPartitionedCall2F
!Gen_SPD_5/StatefulPartitionedCall!Gen_SPD_5/StatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€ј†
 
_user_specified_nameinputs
А
°
D__inference_Gen_BN_3_layer_call_and_return_conditional_losses_142197

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ё
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Х
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ш
э
D__inference_Gen_BN_3_layer_call_and_return_conditional_losses_140540

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1б
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
»
°
D__inference_Gen_BN_6_layer_call_and_return_conditional_losses_145748

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1п
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1І
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
К
Ь
)__inference_Gen_BN_2_layer_call_fn_145043

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_2_layer_call_and_return_conditional_losses_1403252
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
И
А
+__inference_Gen_Conv_5_layer_call_fn_145524

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_Gen_Conv_5_layer_call_and_return_conditional_losses_1424022
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€
А::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€
А
 
_user_specified_nameinputs
Э
d
E__inference_Gen_SPD_1_layer_call_and_return_conditional_losses_140422

inputs
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
 *  †?2
dropout/ConstЦ
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2
dropout/MulВ
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1В
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ц
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape‘
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/yѕ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/GreaterEqualР
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/CastЭ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2
dropout/Mul_1И
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ї
F
*__inference_Gen_SPD_4_layer_call_fn_145881

inputs
identityѕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_4_layer_call_and_return_conditional_losses_1426142
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€
А:X T
0
_output_shapes
:€€€€€€€€€
А
 
_user_specified_nameinputs
І%
Ї
H__inference_Gen_Conv_T_2_layer_call_and_return_conditional_losses_141195

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
strided_slice_2/stack_2м
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
B :А2	
stack/3В
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
strided_slice_3/stack_2м
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3µ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_transpose/ReadVariableOpс
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
2
conv2d_transposeН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOp•
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2	
BiasAddЛ
leaky_re_lu_6/PartitionedCallPartitionedCallBiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_1411862
leaky_re_lu_6/PartitionedCallХ
IdentityIdentity&leaky_re_lu_6/PartitionedCall:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ш
э
D__inference_Gen_BN_8_layer_call_and_return_conditional_losses_145996

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1б
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
ѓ
c
*__inference_Gen_SPD_2_layer_call_fn_145499

inputs
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_2_layer_call_and_return_conditional_losses_1407222
StatefulPartitionedCall±
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
А
°
D__inference_Gen_BN_4_layer_call_and_return_conditional_losses_142298

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ё
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€
А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Х
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€
А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:€€€€€€€€€
А
 
_user_specified_nameinputs
ЧЋ
љ
E__inference_Generator_layer_call_and_return_conditional_losses_144502

inputs-
)gen_conv_1_conv2d_readvariableop_resource.
*gen_conv_1_biasadd_readvariableop_resource$
 gen_bn_1_readvariableop_resource&
"gen_bn_1_readvariableop_1_resource5
1gen_bn_1_fusedbatchnormv3_readvariableop_resource7
3gen_bn_1_fusedbatchnormv3_readvariableop_1_resource-
)gen_conv_2_conv2d_readvariableop_resource.
*gen_conv_2_biasadd_readvariableop_resource$
 gen_bn_2_readvariableop_resource&
"gen_bn_2_readvariableop_1_resource5
1gen_bn_2_fusedbatchnormv3_readvariableop_resource7
3gen_bn_2_fusedbatchnormv3_readvariableop_1_resource-
)gen_conv_3_conv2d_readvariableop_resource.
*gen_conv_3_biasadd_readvariableop_resource$
 gen_bn_3_readvariableop_resource&
"gen_bn_3_readvariableop_1_resource5
1gen_bn_3_fusedbatchnormv3_readvariableop_resource7
3gen_bn_3_fusedbatchnormv3_readvariableop_1_resource-
)gen_conv_4_conv2d_readvariableop_resource.
*gen_conv_4_biasadd_readvariableop_resource$
 gen_bn_4_readvariableop_resource&
"gen_bn_4_readvariableop_1_resource5
1gen_bn_4_fusedbatchnormv3_readvariableop_resource7
3gen_bn_4_fusedbatchnormv3_readvariableop_1_resource-
)gen_conv_5_conv2d_readvariableop_resource.
*gen_conv_5_biasadd_readvariableop_resource$
 gen_bn_5_readvariableop_resource&
"gen_bn_5_readvariableop_1_resource5
1gen_bn_5_fusedbatchnormv3_readvariableop_resource7
3gen_bn_5_fusedbatchnormv3_readvariableop_1_resource9
5gen_conv_t_1_conv2d_transpose_readvariableop_resource0
,gen_conv_t_1_biasadd_readvariableop_resource$
 gen_bn_6_readvariableop_resource&
"gen_bn_6_readvariableop_1_resource5
1gen_bn_6_fusedbatchnormv3_readvariableop_resource7
3gen_bn_6_fusedbatchnormv3_readvariableop_1_resource9
5gen_conv_t_2_conv2d_transpose_readvariableop_resource0
,gen_conv_t_2_biasadd_readvariableop_resource$
 gen_bn_7_readvariableop_resource&
"gen_bn_7_readvariableop_1_resource5
1gen_bn_7_fusedbatchnormv3_readvariableop_resource7
3gen_bn_7_fusedbatchnormv3_readvariableop_1_resource9
5gen_conv_t_3_conv2d_transpose_readvariableop_resource0
,gen_conv_t_3_biasadd_readvariableop_resource$
 gen_bn_8_readvariableop_resource&
"gen_bn_8_readvariableop_1_resource5
1gen_bn_8_fusedbatchnormv3_readvariableop_resource7
3gen_bn_8_fusedbatchnormv3_readvariableop_1_resource9
5gen_conv_t_4_conv2d_transpose_readvariableop_resource0
,gen_conv_t_4_biasadd_readvariableop_resource$
 gen_bn_9_readvariableop_resource&
"gen_bn_9_readvariableop_1_resource5
1gen_bn_9_fusedbatchnormv3_readvariableop_resource7
3gen_bn_9_fusedbatchnormv3_readvariableop_1_resource9
5gen_conv_t_5_conv2d_transpose_readvariableop_resource0
,gen_conv_t_5_biasadd_readvariableop_resource%
!gen_bn_10_readvariableop_resource'
#gen_bn_10_readvariableop_1_resource6
2gen_bn_10_fusedbatchnormv3_readvariableop_resource8
4gen_bn_10_fusedbatchnormv3_readvariableop_1_resource9
5gen_conv_t_6_conv2d_transpose_readvariableop_resource0
,gen_conv_t_6_biasadd_readvariableop_resource
identityИґ
 Gen_Conv_1/Conv2D/ReadVariableOpReadVariableOp)gen_conv_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02"
 Gen_Conv_1/Conv2D/ReadVariableOp∆
Gen_Conv_1/Conv2DConv2Dinputs(Gen_Conv_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ј†@*
paddingSAME*
strides
2
Gen_Conv_1/Conv2D≠
!Gen_Conv_1/BiasAdd/ReadVariableOpReadVariableOp*gen_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!Gen_Conv_1/BiasAdd/ReadVariableOpґ
Gen_Conv_1/BiasAddBiasAddGen_Conv_1/Conv2D:output:0)Gen_Conv_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ј†@2
Gen_Conv_1/BiasAdd°
 Gen_Conv_1/leaky_re_lu/LeakyRelu	LeakyReluGen_Conv_1/BiasAdd:output:0*1
_output_shapes
:€€€€€€€€€ј†@2"
 Gen_Conv_1/leaky_re_lu/LeakyReluћ
Gen_MP_1/MaxPoolMaxPool.Gen_Conv_1/leaky_re_lu/LeakyRelu:activations:0*/
_output_shapes
:€€€€€€€€€`P@*
ksize
*
paddingVALID*
strides
2
Gen_MP_1/MaxPoolП
Gen_BN_1/ReadVariableOpReadVariableOp gen_bn_1_readvariableop_resource*
_output_shapes
:@*
dtype02
Gen_BN_1/ReadVariableOpХ
Gen_BN_1/ReadVariableOp_1ReadVariableOp"gen_bn_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02
Gen_BN_1/ReadVariableOp_1¬
(Gen_BN_1/FusedBatchNormV3/ReadVariableOpReadVariableOp1gen_bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02*
(Gen_BN_1/FusedBatchNormV3/ReadVariableOp»
*Gen_BN_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3gen_bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02,
*Gen_BN_1/FusedBatchNormV3/ReadVariableOp_1У
Gen_BN_1/FusedBatchNormV3FusedBatchNormV3Gen_MP_1/MaxPool:output:0Gen_BN_1/ReadVariableOp:value:0!Gen_BN_1/ReadVariableOp_1:value:00Gen_BN_1/FusedBatchNormV3/ReadVariableOp:value:02Gen_BN_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€`P@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
Gen_BN_1/FusedBatchNormV3Ј
 Gen_Conv_2/Conv2D/ReadVariableOpReadVariableOp)gen_conv_2_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02"
 Gen_Conv_2/Conv2D/ReadVariableOp№
Gen_Conv_2/Conv2DConv2DGen_BN_1/FusedBatchNormV3:y:0(Gen_Conv_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€`PА*
paddingSAME*
strides
2
Gen_Conv_2/Conv2DЃ
!Gen_Conv_2/BiasAdd/ReadVariableOpReadVariableOp*gen_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02#
!Gen_Conv_2/BiasAdd/ReadVariableOpµ
Gen_Conv_2/BiasAddBiasAddGen_Conv_2/Conv2D:output:0)Gen_Conv_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€`PА2
Gen_Conv_2/BiasAdd§
"Gen_Conv_2/leaky_re_lu_1/LeakyRelu	LeakyReluGen_Conv_2/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€`PА2$
"Gen_Conv_2/leaky_re_lu_1/LeakyReluѕ
Gen_MP_2/MaxPoolMaxPool0Gen_Conv_2/leaky_re_lu_1/LeakyRelu:activations:0*0
_output_shapes
:€€€€€€€€€0(А*
ksize
*
paddingVALID*
strides
2
Gen_MP_2/MaxPoolР
Gen_BN_2/ReadVariableOpReadVariableOp gen_bn_2_readvariableop_resource*
_output_shapes	
:А*
dtype02
Gen_BN_2/ReadVariableOpЦ
Gen_BN_2/ReadVariableOp_1ReadVariableOp"gen_bn_2_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
Gen_BN_2/ReadVariableOp_1√
(Gen_BN_2/FusedBatchNormV3/ReadVariableOpReadVariableOp1gen_bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(Gen_BN_2/FusedBatchNormV3/ReadVariableOp…
*Gen_BN_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3gen_bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02,
*Gen_BN_2/FusedBatchNormV3/ReadVariableOp_1Ш
Gen_BN_2/FusedBatchNormV3FusedBatchNormV3Gen_MP_2/MaxPool:output:0Gen_BN_2/ReadVariableOp:value:0!Gen_BN_2/ReadVariableOp_1:value:00Gen_BN_2/FusedBatchNormV3/ReadVariableOp:value:02Gen_BN_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€0(А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
Gen_BN_2/FusedBatchNormV3О
Gen_SPD_1/IdentityIdentityGen_BN_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2
Gen_SPD_1/IdentityЄ
 Gen_Conv_3/Conv2D/ReadVariableOpReadVariableOp)gen_conv_3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02"
 Gen_Conv_3/Conv2D/ReadVariableOpЏ
Gen_Conv_3/Conv2DConv2DGen_SPD_1/Identity:output:0(Gen_Conv_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€0(А*
paddingSAME*
strides
2
Gen_Conv_3/Conv2DЃ
!Gen_Conv_3/BiasAdd/ReadVariableOpReadVariableOp*gen_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02#
!Gen_Conv_3/BiasAdd/ReadVariableOpµ
Gen_Conv_3/BiasAddBiasAddGen_Conv_3/Conv2D:output:0)Gen_Conv_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2
Gen_Conv_3/BiasAdd§
"Gen_Conv_3/leaky_re_lu_2/LeakyRelu	LeakyReluGen_Conv_3/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€0(А2$
"Gen_Conv_3/leaky_re_lu_2/LeakyReluѕ
Gen_MP_3/MaxPoolMaxPool0Gen_Conv_3/leaky_re_lu_2/LeakyRelu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
Gen_MP_3/MaxPoolР
Gen_BN_3/ReadVariableOpReadVariableOp gen_bn_3_readvariableop_resource*
_output_shapes	
:А*
dtype02
Gen_BN_3/ReadVariableOpЦ
Gen_BN_3/ReadVariableOp_1ReadVariableOp"gen_bn_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
Gen_BN_3/ReadVariableOp_1√
(Gen_BN_3/FusedBatchNormV3/ReadVariableOpReadVariableOp1gen_bn_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(Gen_BN_3/FusedBatchNormV3/ReadVariableOp…
*Gen_BN_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3gen_bn_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02,
*Gen_BN_3/FusedBatchNormV3/ReadVariableOp_1Ш
Gen_BN_3/FusedBatchNormV3FusedBatchNormV3Gen_MP_3/MaxPool:output:0Gen_BN_3/ReadVariableOp:value:0!Gen_BN_3/ReadVariableOp_1:value:00Gen_BN_3/FusedBatchNormV3/ReadVariableOp:value:02Gen_BN_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
Gen_BN_3/FusedBatchNormV3Є
 Gen_Conv_4/Conv2D/ReadVariableOpReadVariableOp)gen_conv_4_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02"
 Gen_Conv_4/Conv2D/ReadVariableOp№
Gen_Conv_4/Conv2DConv2DGen_BN_3/FusedBatchNormV3:y:0(Gen_Conv_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Gen_Conv_4/Conv2DЃ
!Gen_Conv_4/BiasAdd/ReadVariableOpReadVariableOp*gen_conv_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02#
!Gen_Conv_4/BiasAdd/ReadVariableOpµ
Gen_Conv_4/BiasAddBiasAddGen_Conv_4/Conv2D:output:0)Gen_Conv_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Gen_Conv_4/BiasAdd§
"Gen_Conv_4/leaky_re_lu_3/LeakyRelu	LeakyReluGen_Conv_4/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€А2$
"Gen_Conv_4/leaky_re_lu_3/LeakyReluѕ
Gen_MP_4/MaxPoolMaxPool0Gen_Conv_4/leaky_re_lu_3/LeakyRelu:activations:0*0
_output_shapes
:€€€€€€€€€
А*
ksize
*
paddingVALID*
strides
2
Gen_MP_4/MaxPoolР
Gen_BN_4/ReadVariableOpReadVariableOp gen_bn_4_readvariableop_resource*
_output_shapes	
:А*
dtype02
Gen_BN_4/ReadVariableOpЦ
Gen_BN_4/ReadVariableOp_1ReadVariableOp"gen_bn_4_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
Gen_BN_4/ReadVariableOp_1√
(Gen_BN_4/FusedBatchNormV3/ReadVariableOpReadVariableOp1gen_bn_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(Gen_BN_4/FusedBatchNormV3/ReadVariableOp…
*Gen_BN_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3gen_bn_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02,
*Gen_BN_4/FusedBatchNormV3/ReadVariableOp_1Ш
Gen_BN_4/FusedBatchNormV3FusedBatchNormV3Gen_MP_4/MaxPool:output:0Gen_BN_4/ReadVariableOp:value:0!Gen_BN_4/ReadVariableOp_1:value:00Gen_BN_4/FusedBatchNormV3/ReadVariableOp:value:02Gen_BN_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€
А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
Gen_BN_4/FusedBatchNormV3О
Gen_SPD_2/IdentityIdentityGen_BN_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€
А2
Gen_SPD_2/IdentityЄ
 Gen_Conv_5/Conv2D/ReadVariableOpReadVariableOp)gen_conv_5_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02"
 Gen_Conv_5/Conv2D/ReadVariableOpЏ
Gen_Conv_5/Conv2DConv2DGen_SPD_2/Identity:output:0(Gen_Conv_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€
А*
paddingSAME*
strides
2
Gen_Conv_5/Conv2DЃ
!Gen_Conv_5/BiasAdd/ReadVariableOpReadVariableOp*gen_conv_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02#
!Gen_Conv_5/BiasAdd/ReadVariableOpµ
Gen_Conv_5/BiasAddBiasAddGen_Conv_5/Conv2D:output:0)Gen_Conv_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€
А2
Gen_Conv_5/BiasAdd§
"Gen_Conv_5/leaky_re_lu_4/LeakyRelu	LeakyReluGen_Conv_5/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€
А2$
"Gen_Conv_5/leaky_re_lu_4/LeakyReluѕ
Gen_MP_5/MaxPoolMaxPool0Gen_Conv_5/leaky_re_lu_4/LeakyRelu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
Gen_MP_5/MaxPoolР
Gen_BN_5/ReadVariableOpReadVariableOp gen_bn_5_readvariableop_resource*
_output_shapes	
:А*
dtype02
Gen_BN_5/ReadVariableOpЦ
Gen_BN_5/ReadVariableOp_1ReadVariableOp"gen_bn_5_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
Gen_BN_5/ReadVariableOp_1√
(Gen_BN_5/FusedBatchNormV3/ReadVariableOpReadVariableOp1gen_bn_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(Gen_BN_5/FusedBatchNormV3/ReadVariableOp…
*Gen_BN_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3gen_bn_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02,
*Gen_BN_5/FusedBatchNormV3/ReadVariableOp_1Ш
Gen_BN_5/FusedBatchNormV3FusedBatchNormV3Gen_MP_5/MaxPool:output:0Gen_BN_5/ReadVariableOp:value:0!Gen_BN_5/ReadVariableOp_1:value:00Gen_BN_5/FusedBatchNormV3/ReadVariableOp:value:02Gen_BN_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
Gen_BN_5/FusedBatchNormV3О
Gen_SPD_3/IdentityIdentityGen_BN_5/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Gen_SPD_3/Identitys
Gen_Conv_T_1/ShapeShapeGen_SPD_3/Identity:output:0*
T0*
_output_shapes
:2
Gen_Conv_T_1/ShapeО
 Gen_Conv_T_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 Gen_Conv_T_1/strided_slice/stackТ
"Gen_Conv_T_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"Gen_Conv_T_1/strided_slice/stack_1Т
"Gen_Conv_T_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"Gen_Conv_T_1/strided_slice/stack_2∞
Gen_Conv_T_1/strided_sliceStridedSliceGen_Conv_T_1/Shape:output:0)Gen_Conv_T_1/strided_slice/stack:output:0+Gen_Conv_T_1/strided_slice/stack_1:output:0+Gen_Conv_T_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_Conv_T_1/strided_slicen
Gen_Conv_T_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
Gen_Conv_T_1/stack/1n
Gen_Conv_T_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :
2
Gen_Conv_T_1/stack/2o
Gen_Conv_T_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :А2
Gen_Conv_T_1/stack/3а
Gen_Conv_T_1/stackPack#Gen_Conv_T_1/strided_slice:output:0Gen_Conv_T_1/stack/1:output:0Gen_Conv_T_1/stack/2:output:0Gen_Conv_T_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
Gen_Conv_T_1/stackТ
"Gen_Conv_T_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"Gen_Conv_T_1/strided_slice_1/stackЦ
$Gen_Conv_T_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$Gen_Conv_T_1/strided_slice_1/stack_1Ц
$Gen_Conv_T_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$Gen_Conv_T_1/strided_slice_1/stack_2Ї
Gen_Conv_T_1/strided_slice_1StridedSliceGen_Conv_T_1/stack:output:0+Gen_Conv_T_1/strided_slice_1/stack:output:0-Gen_Conv_T_1/strided_slice_1/stack_1:output:0-Gen_Conv_T_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_Conv_T_1/strided_slice_1№
,Gen_Conv_T_1/conv2d_transpose/ReadVariableOpReadVariableOp5gen_conv_t_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,Gen_Conv_T_1/conv2d_transpose/ReadVariableOp®
Gen_Conv_T_1/conv2d_transposeConv2DBackpropInputGen_Conv_T_1/stack:output:04Gen_Conv_T_1/conv2d_transpose/ReadVariableOp:value:0Gen_SPD_3/Identity:output:0*
T0*0
_output_shapes
:€€€€€€€€€
А*
paddingSAME*
strides
2
Gen_Conv_T_1/conv2d_transposeі
#Gen_Conv_T_1/BiasAdd/ReadVariableOpReadVariableOp,gen_conv_t_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#Gen_Conv_T_1/BiasAdd/ReadVariableOp«
Gen_Conv_T_1/BiasAddBiasAdd&Gen_Conv_T_1/conv2d_transpose:output:0+Gen_Conv_T_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€
А2
Gen_Conv_T_1/BiasAdd™
$Gen_Conv_T_1/leaky_re_lu_5/LeakyRelu	LeakyReluGen_Conv_T_1/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€
А2&
$Gen_Conv_T_1/leaky_re_lu_5/LeakyReluР
Gen_BN_6/ReadVariableOpReadVariableOp gen_bn_6_readvariableop_resource*
_output_shapes	
:А*
dtype02
Gen_BN_6/ReadVariableOpЦ
Gen_BN_6/ReadVariableOp_1ReadVariableOp"gen_bn_6_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
Gen_BN_6/ReadVariableOp_1√
(Gen_BN_6/FusedBatchNormV3/ReadVariableOpReadVariableOp1gen_bn_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(Gen_BN_6/FusedBatchNormV3/ReadVariableOp…
*Gen_BN_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3gen_bn_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02,
*Gen_BN_6/FusedBatchNormV3/ReadVariableOp_1±
Gen_BN_6/FusedBatchNormV3FusedBatchNormV32Gen_Conv_T_1/leaky_re_lu_5/LeakyRelu:activations:0Gen_BN_6/ReadVariableOp:value:0!Gen_BN_6/ReadVariableOp_1:value:00Gen_BN_6/FusedBatchNormV3/ReadVariableOp:value:02Gen_BN_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€
А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
Gen_BN_6/FusedBatchNormV3v
Gen_Concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Gen_Concat_1/concat/axisџ
Gen_Concat_1/concatConcatV2Gen_BN_6/FusedBatchNormV3:y:0Gen_BN_4/FusedBatchNormV3:y:0!Gen_Concat_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€
А2
Gen_Concat_1/concatН
Gen_SPD_4/IdentityIdentityGen_Concat_1/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€
А2
Gen_SPD_4/Identitys
Gen_Conv_T_2/ShapeShapeGen_SPD_4/Identity:output:0*
T0*
_output_shapes
:2
Gen_Conv_T_2/ShapeО
 Gen_Conv_T_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 Gen_Conv_T_2/strided_slice/stackТ
"Gen_Conv_T_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"Gen_Conv_T_2/strided_slice/stack_1Т
"Gen_Conv_T_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"Gen_Conv_T_2/strided_slice/stack_2∞
Gen_Conv_T_2/strided_sliceStridedSliceGen_Conv_T_2/Shape:output:0)Gen_Conv_T_2/strided_slice/stack:output:0+Gen_Conv_T_2/strided_slice/stack_1:output:0+Gen_Conv_T_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_Conv_T_2/strided_slicen
Gen_Conv_T_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
Gen_Conv_T_2/stack/1n
Gen_Conv_T_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
Gen_Conv_T_2/stack/2o
Gen_Conv_T_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :А2
Gen_Conv_T_2/stack/3а
Gen_Conv_T_2/stackPack#Gen_Conv_T_2/strided_slice:output:0Gen_Conv_T_2/stack/1:output:0Gen_Conv_T_2/stack/2:output:0Gen_Conv_T_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
Gen_Conv_T_2/stackТ
"Gen_Conv_T_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"Gen_Conv_T_2/strided_slice_1/stackЦ
$Gen_Conv_T_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$Gen_Conv_T_2/strided_slice_1/stack_1Ц
$Gen_Conv_T_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$Gen_Conv_T_2/strided_slice_1/stack_2Ї
Gen_Conv_T_2/strided_slice_1StridedSliceGen_Conv_T_2/stack:output:0+Gen_Conv_T_2/strided_slice_1/stack:output:0-Gen_Conv_T_2/strided_slice_1/stack_1:output:0-Gen_Conv_T_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_Conv_T_2/strided_slice_1№
,Gen_Conv_T_2/conv2d_transpose/ReadVariableOpReadVariableOp5gen_conv_t_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,Gen_Conv_T_2/conv2d_transpose/ReadVariableOp®
Gen_Conv_T_2/conv2d_transposeConv2DBackpropInputGen_Conv_T_2/stack:output:04Gen_Conv_T_2/conv2d_transpose/ReadVariableOp:value:0Gen_SPD_4/Identity:output:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Gen_Conv_T_2/conv2d_transposeі
#Gen_Conv_T_2/BiasAdd/ReadVariableOpReadVariableOp,gen_conv_t_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#Gen_Conv_T_2/BiasAdd/ReadVariableOp«
Gen_Conv_T_2/BiasAddBiasAdd&Gen_Conv_T_2/conv2d_transpose:output:0+Gen_Conv_T_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Gen_Conv_T_2/BiasAdd™
$Gen_Conv_T_2/leaky_re_lu_6/LeakyRelu	LeakyReluGen_Conv_T_2/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€А2&
$Gen_Conv_T_2/leaky_re_lu_6/LeakyReluР
Gen_BN_7/ReadVariableOpReadVariableOp gen_bn_7_readvariableop_resource*
_output_shapes	
:А*
dtype02
Gen_BN_7/ReadVariableOpЦ
Gen_BN_7/ReadVariableOp_1ReadVariableOp"gen_bn_7_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
Gen_BN_7/ReadVariableOp_1√
(Gen_BN_7/FusedBatchNormV3/ReadVariableOpReadVariableOp1gen_bn_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(Gen_BN_7/FusedBatchNormV3/ReadVariableOp…
*Gen_BN_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3gen_bn_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02,
*Gen_BN_7/FusedBatchNormV3/ReadVariableOp_1±
Gen_BN_7/FusedBatchNormV3FusedBatchNormV32Gen_Conv_T_2/leaky_re_lu_6/LeakyRelu:activations:0Gen_BN_7/ReadVariableOp:value:0!Gen_BN_7/ReadVariableOp_1:value:00Gen_BN_7/FusedBatchNormV3/ReadVariableOp:value:02Gen_BN_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
Gen_BN_7/FusedBatchNormV3v
Gen_Concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Gen_Concat_2/concat/axisџ
Gen_Concat_2/concatConcatV2Gen_BN_7/FusedBatchNormV3:y:0Gen_BN_3/FusedBatchNormV3:y:0!Gen_Concat_2/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А2
Gen_Concat_2/concatt
Gen_Conv_T_3/ShapeShapeGen_Concat_2/concat:output:0*
T0*
_output_shapes
:2
Gen_Conv_T_3/ShapeО
 Gen_Conv_T_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 Gen_Conv_T_3/strided_slice/stackТ
"Gen_Conv_T_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"Gen_Conv_T_3/strided_slice/stack_1Т
"Gen_Conv_T_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"Gen_Conv_T_3/strided_slice/stack_2∞
Gen_Conv_T_3/strided_sliceStridedSliceGen_Conv_T_3/Shape:output:0)Gen_Conv_T_3/strided_slice/stack:output:0+Gen_Conv_T_3/strided_slice/stack_1:output:0+Gen_Conv_T_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_Conv_T_3/strided_slicen
Gen_Conv_T_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :02
Gen_Conv_T_3/stack/1n
Gen_Conv_T_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :(2
Gen_Conv_T_3/stack/2o
Gen_Conv_T_3/stack/3Const*
_output_shapes
: *
dtype0*
value
B :А2
Gen_Conv_T_3/stack/3а
Gen_Conv_T_3/stackPack#Gen_Conv_T_3/strided_slice:output:0Gen_Conv_T_3/stack/1:output:0Gen_Conv_T_3/stack/2:output:0Gen_Conv_T_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
Gen_Conv_T_3/stackТ
"Gen_Conv_T_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"Gen_Conv_T_3/strided_slice_1/stackЦ
$Gen_Conv_T_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$Gen_Conv_T_3/strided_slice_1/stack_1Ц
$Gen_Conv_T_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$Gen_Conv_T_3/strided_slice_1/stack_2Ї
Gen_Conv_T_3/strided_slice_1StridedSliceGen_Conv_T_3/stack:output:0+Gen_Conv_T_3/strided_slice_1/stack:output:0-Gen_Conv_T_3/strided_slice_1/stack_1:output:0-Gen_Conv_T_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_Conv_T_3/strided_slice_1№
,Gen_Conv_T_3/conv2d_transpose/ReadVariableOpReadVariableOp5gen_conv_t_3_conv2d_transpose_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,Gen_Conv_T_3/conv2d_transpose/ReadVariableOp©
Gen_Conv_T_3/conv2d_transposeConv2DBackpropInputGen_Conv_T_3/stack:output:04Gen_Conv_T_3/conv2d_transpose/ReadVariableOp:value:0Gen_Concat_2/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€0(А*
paddingSAME*
strides
2
Gen_Conv_T_3/conv2d_transposeі
#Gen_Conv_T_3/BiasAdd/ReadVariableOpReadVariableOp,gen_conv_t_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#Gen_Conv_T_3/BiasAdd/ReadVariableOp«
Gen_Conv_T_3/BiasAddBiasAdd&Gen_Conv_T_3/conv2d_transpose:output:0+Gen_Conv_T_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2
Gen_Conv_T_3/BiasAdd™
$Gen_Conv_T_3/leaky_re_lu_7/LeakyRelu	LeakyReluGen_Conv_T_3/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€0(А2&
$Gen_Conv_T_3/leaky_re_lu_7/LeakyReluР
Gen_BN_8/ReadVariableOpReadVariableOp gen_bn_8_readvariableop_resource*
_output_shapes	
:А*
dtype02
Gen_BN_8/ReadVariableOpЦ
Gen_BN_8/ReadVariableOp_1ReadVariableOp"gen_bn_8_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
Gen_BN_8/ReadVariableOp_1√
(Gen_BN_8/FusedBatchNormV3/ReadVariableOpReadVariableOp1gen_bn_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(Gen_BN_8/FusedBatchNormV3/ReadVariableOp…
*Gen_BN_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3gen_bn_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02,
*Gen_BN_8/FusedBatchNormV3/ReadVariableOp_1±
Gen_BN_8/FusedBatchNormV3FusedBatchNormV32Gen_Conv_T_3/leaky_re_lu_7/LeakyRelu:activations:0Gen_BN_8/ReadVariableOp:value:0!Gen_BN_8/ReadVariableOp_1:value:00Gen_BN_8/FusedBatchNormV3/ReadVariableOp:value:02Gen_BN_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€0(А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
Gen_BN_8/FusedBatchNormV3v
Gen_Concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Gen_Concat_3/concat/axisџ
Gen_Concat_3/concatConcatV2Gen_BN_8/FusedBatchNormV3:y:0Gen_BN_2/FusedBatchNormV3:y:0!Gen_Concat_3/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€0(А2
Gen_Concat_3/concatН
Gen_SPD_5/IdentityIdentityGen_Concat_3/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2
Gen_SPD_5/Identitys
Gen_Conv_T_4/ShapeShapeGen_SPD_5/Identity:output:0*
T0*
_output_shapes
:2
Gen_Conv_T_4/ShapeО
 Gen_Conv_T_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 Gen_Conv_T_4/strided_slice/stackТ
"Gen_Conv_T_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"Gen_Conv_T_4/strided_slice/stack_1Т
"Gen_Conv_T_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"Gen_Conv_T_4/strided_slice/stack_2∞
Gen_Conv_T_4/strided_sliceStridedSliceGen_Conv_T_4/Shape:output:0)Gen_Conv_T_4/strided_slice/stack:output:0+Gen_Conv_T_4/strided_slice/stack_1:output:0+Gen_Conv_T_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_Conv_T_4/strided_slicen
Gen_Conv_T_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :`2
Gen_Conv_T_4/stack/1n
Gen_Conv_T_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :P2
Gen_Conv_T_4/stack/2n
Gen_Conv_T_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
Gen_Conv_T_4/stack/3а
Gen_Conv_T_4/stackPack#Gen_Conv_T_4/strided_slice:output:0Gen_Conv_T_4/stack/1:output:0Gen_Conv_T_4/stack/2:output:0Gen_Conv_T_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
Gen_Conv_T_4/stackТ
"Gen_Conv_T_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"Gen_Conv_T_4/strided_slice_1/stackЦ
$Gen_Conv_T_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$Gen_Conv_T_4/strided_slice_1/stack_1Ц
$Gen_Conv_T_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$Gen_Conv_T_4/strided_slice_1/stack_2Ї
Gen_Conv_T_4/strided_slice_1StridedSliceGen_Conv_T_4/stack:output:0+Gen_Conv_T_4/strided_slice_1/stack:output:0-Gen_Conv_T_4/strided_slice_1/stack_1:output:0-Gen_Conv_T_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_Conv_T_4/strided_slice_1џ
,Gen_Conv_T_4/conv2d_transpose/ReadVariableOpReadVariableOp5gen_conv_t_4_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@А*
dtype02.
,Gen_Conv_T_4/conv2d_transpose/ReadVariableOpІ
Gen_Conv_T_4/conv2d_transposeConv2DBackpropInputGen_Conv_T_4/stack:output:04Gen_Conv_T_4/conv2d_transpose/ReadVariableOp:value:0Gen_SPD_5/Identity:output:0*
T0*/
_output_shapes
:€€€€€€€€€`P@*
paddingSAME*
strides
2
Gen_Conv_T_4/conv2d_transpose≥
#Gen_Conv_T_4/BiasAdd/ReadVariableOpReadVariableOp,gen_conv_t_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#Gen_Conv_T_4/BiasAdd/ReadVariableOp∆
Gen_Conv_T_4/BiasAddBiasAdd&Gen_Conv_T_4/conv2d_transpose:output:0+Gen_Conv_T_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€`P@2
Gen_Conv_T_4/BiasAdd©
$Gen_Conv_T_4/leaky_re_lu_8/LeakyRelu	LeakyReluGen_Conv_T_4/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€`P@2&
$Gen_Conv_T_4/leaky_re_lu_8/LeakyReluП
Gen_BN_9/ReadVariableOpReadVariableOp gen_bn_9_readvariableop_resource*
_output_shapes
:@*
dtype02
Gen_BN_9/ReadVariableOpХ
Gen_BN_9/ReadVariableOp_1ReadVariableOp"gen_bn_9_readvariableop_1_resource*
_output_shapes
:@*
dtype02
Gen_BN_9/ReadVariableOp_1¬
(Gen_BN_9/FusedBatchNormV3/ReadVariableOpReadVariableOp1gen_bn_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02*
(Gen_BN_9/FusedBatchNormV3/ReadVariableOp»
*Gen_BN_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3gen_bn_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02,
*Gen_BN_9/FusedBatchNormV3/ReadVariableOp_1ђ
Gen_BN_9/FusedBatchNormV3FusedBatchNormV32Gen_Conv_T_4/leaky_re_lu_8/LeakyRelu:activations:0Gen_BN_9/ReadVariableOp:value:0!Gen_BN_9/ReadVariableOp_1:value:00Gen_BN_9/FusedBatchNormV3/ReadVariableOp:value:02Gen_BN_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€`P@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
Gen_BN_9/FusedBatchNormV3v
Gen_Concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Gen_Concat_4/concat/axisџ
Gen_Concat_4/concatConcatV2Gen_BN_9/FusedBatchNormV3:y:0Gen_BN_1/FusedBatchNormV3:y:0!Gen_Concat_4/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€`PА2
Gen_Concat_4/concatt
Gen_Conv_T_5/ShapeShapeGen_Concat_4/concat:output:0*
T0*
_output_shapes
:2
Gen_Conv_T_5/ShapeО
 Gen_Conv_T_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 Gen_Conv_T_5/strided_slice/stackТ
"Gen_Conv_T_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"Gen_Conv_T_5/strided_slice/stack_1Т
"Gen_Conv_T_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"Gen_Conv_T_5/strided_slice/stack_2∞
Gen_Conv_T_5/strided_sliceStridedSliceGen_Conv_T_5/Shape:output:0)Gen_Conv_T_5/strided_slice/stack:output:0+Gen_Conv_T_5/strided_slice/stack_1:output:0+Gen_Conv_T_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_Conv_T_5/strided_sliceo
Gen_Conv_T_5/stack/1Const*
_output_shapes
: *
dtype0*
value
B :ј2
Gen_Conv_T_5/stack/1o
Gen_Conv_T_5/stack/2Const*
_output_shapes
: *
dtype0*
value
B :†2
Gen_Conv_T_5/stack/2n
Gen_Conv_T_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Gen_Conv_T_5/stack/3а
Gen_Conv_T_5/stackPack#Gen_Conv_T_5/strided_slice:output:0Gen_Conv_T_5/stack/1:output:0Gen_Conv_T_5/stack/2:output:0Gen_Conv_T_5/stack/3:output:0*
N*
T0*
_output_shapes
:2
Gen_Conv_T_5/stackТ
"Gen_Conv_T_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"Gen_Conv_T_5/strided_slice_1/stackЦ
$Gen_Conv_T_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$Gen_Conv_T_5/strided_slice_1/stack_1Ц
$Gen_Conv_T_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$Gen_Conv_T_5/strided_slice_1/stack_2Ї
Gen_Conv_T_5/strided_slice_1StridedSliceGen_Conv_T_5/stack:output:0+Gen_Conv_T_5/strided_slice_1/stack:output:0-Gen_Conv_T_5/strided_slice_1/stack_1:output:0-Gen_Conv_T_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_Conv_T_5/strided_slice_1џ
,Gen_Conv_T_5/conv2d_transpose/ReadVariableOpReadVariableOp5gen_conv_t_5_conv2d_transpose_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,Gen_Conv_T_5/conv2d_transpose/ReadVariableOp™
Gen_Conv_T_5/conv2d_transposeConv2DBackpropInputGen_Conv_T_5/stack:output:04Gen_Conv_T_5/conv2d_transpose/ReadVariableOp:value:0Gen_Concat_4/concat:output:0*
T0*1
_output_shapes
:€€€€€€€€€ј† *
paddingSAME*
strides
2
Gen_Conv_T_5/conv2d_transpose≥
#Gen_Conv_T_5/BiasAdd/ReadVariableOpReadVariableOp,gen_conv_t_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#Gen_Conv_T_5/BiasAdd/ReadVariableOp»
Gen_Conv_T_5/BiasAddBiasAdd&Gen_Conv_T_5/conv2d_transpose:output:0+Gen_Conv_T_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ј† 2
Gen_Conv_T_5/BiasAddЂ
$Gen_Conv_T_5/leaky_re_lu_9/LeakyRelu	LeakyReluGen_Conv_T_5/BiasAdd:output:0*1
_output_shapes
:€€€€€€€€€ј† 2&
$Gen_Conv_T_5/leaky_re_lu_9/LeakyReluТ
Gen_BN_10/ReadVariableOpReadVariableOp!gen_bn_10_readvariableop_resource*
_output_shapes
: *
dtype02
Gen_BN_10/ReadVariableOpШ
Gen_BN_10/ReadVariableOp_1ReadVariableOp#gen_bn_10_readvariableop_1_resource*
_output_shapes
: *
dtype02
Gen_BN_10/ReadVariableOp_1≈
)Gen_BN_10/FusedBatchNormV3/ReadVariableOpReadVariableOp2gen_bn_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02+
)Gen_BN_10/FusedBatchNormV3/ReadVariableOpЋ
+Gen_BN_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp4gen_bn_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02-
+Gen_BN_10/FusedBatchNormV3/ReadVariableOp_1і
Gen_BN_10/FusedBatchNormV3FusedBatchNormV32Gen_Conv_T_5/leaky_re_lu_9/LeakyRelu:activations:0 Gen_BN_10/ReadVariableOp:value:0"Gen_BN_10/ReadVariableOp_1:value:01Gen_BN_10/FusedBatchNormV3/ReadVariableOp:value:03Gen_BN_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:€€€€€€€€€ј† : : : : :*
epsilon%oГ:*
is_training( 2
Gen_BN_10/FusedBatchNormV3v
Gen_Conv_T_6/ShapeShapeGen_BN_10/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Gen_Conv_T_6/ShapeО
 Gen_Conv_T_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 Gen_Conv_T_6/strided_slice/stackТ
"Gen_Conv_T_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"Gen_Conv_T_6/strided_slice/stack_1Т
"Gen_Conv_T_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"Gen_Conv_T_6/strided_slice/stack_2∞
Gen_Conv_T_6/strided_sliceStridedSliceGen_Conv_T_6/Shape:output:0)Gen_Conv_T_6/strided_slice/stack:output:0+Gen_Conv_T_6/strided_slice/stack_1:output:0+Gen_Conv_T_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_Conv_T_6/strided_sliceo
Gen_Conv_T_6/stack/1Const*
_output_shapes
: *
dtype0*
value
B :ј2
Gen_Conv_T_6/stack/1o
Gen_Conv_T_6/stack/2Const*
_output_shapes
: *
dtype0*
value
B :†2
Gen_Conv_T_6/stack/2n
Gen_Conv_T_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
Gen_Conv_T_6/stack/3а
Gen_Conv_T_6/stackPack#Gen_Conv_T_6/strided_slice:output:0Gen_Conv_T_6/stack/1:output:0Gen_Conv_T_6/stack/2:output:0Gen_Conv_T_6/stack/3:output:0*
N*
T0*
_output_shapes
:2
Gen_Conv_T_6/stackТ
"Gen_Conv_T_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"Gen_Conv_T_6/strided_slice_1/stackЦ
$Gen_Conv_T_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$Gen_Conv_T_6/strided_slice_1/stack_1Ц
$Gen_Conv_T_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$Gen_Conv_T_6/strided_slice_1/stack_2Ї
Gen_Conv_T_6/strided_slice_1StridedSliceGen_Conv_T_6/stack:output:0+Gen_Conv_T_6/strided_slice_1/stack:output:0-Gen_Conv_T_6/strided_slice_1/stack_1:output:0-Gen_Conv_T_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_Conv_T_6/strided_slice_1Џ
,Gen_Conv_T_6/conv2d_transpose/ReadVariableOpReadVariableOp5gen_conv_t_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02.
,Gen_Conv_T_6/conv2d_transpose/ReadVariableOpђ
Gen_Conv_T_6/conv2d_transposeConv2DBackpropInputGen_Conv_T_6/stack:output:04Gen_Conv_T_6/conv2d_transpose/ReadVariableOp:value:0Gen_BN_10/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:€€€€€€€€€ј†*
paddingSAME*
strides
2
Gen_Conv_T_6/conv2d_transpose≥
#Gen_Conv_T_6/BiasAdd/ReadVariableOpReadVariableOp,gen_conv_t_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#Gen_Conv_T_6/BiasAdd/ReadVariableOp»
Gen_Conv_T_6/BiasAddBiasAdd&Gen_Conv_T_6/conv2d_transpose:output:0+Gen_Conv_T_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ј†2
Gen_Conv_T_6/BiasAddЙ
Gen_Conv_T_6/TanhTanhGen_Conv_T_6/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€ј†2
Gen_Conv_T_6/Tanhs
IdentityIdentityGen_Conv_T_6/Tanh:y:0*
T0*1
_output_shapes
:€€€€€€€€€ј†2

Identity"
identityIdentity:output:0*™
_input_shapesШ
Х:€€€€€€€€€ј†:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::Y U
1
_output_shapes
:€€€€€€€€€ј†
 
_user_specified_nameinputs
ѓ
e
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_141186

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2
	LeakyReluЖ
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ш
э
D__inference_Gen_BN_2_layer_call_and_return_conditional_losses_145030

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1б
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
»
°
D__inference_Gen_BN_7_layer_call_and_return_conditional_losses_145901

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1п
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1І
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
А
°
D__inference_Gen_BN_5_layer_call_and_return_conditional_losses_145608

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ё
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Х
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
м
c
E__inference_Gen_SPD_2_layer_call_and_return_conditional_losses_142379

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:€€€€€€€€€
А:X T
0
_output_shapes
:€€€€€€€€€
А
 
_user_specified_nameinputs
М
э
D__inference_Gen_BN_1_layer_call_and_return_conditional_losses_140240

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
’
c
E__inference_Gen_SPD_2_layer_call_and_return_conditional_losses_140732

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

IdentityМ

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
К
Ь
)__inference_Gen_BN_5_layer_call_fn_145575

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_5_layer_call_and_return_conditional_losses_1408092
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Щ
d
E__inference_Gen_SPD_4_layer_call_and_return_conditional_losses_142609

inputs
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
 *  †?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€
А2
dropout/MulВ
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1В
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ц
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape‘
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/yѕ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/GreaterEqualР
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€
А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€
А:X T
0
_output_shapes
:€€€€€€€€€
А
 
_user_specified_nameinputs
Ћ	
Ѓ
F__inference_Gen_Conv_5_layer_call_and_return_conditional_losses_142402

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€
А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€
А2	
BiasAddГ
leaky_re_lu_4/LeakyRelu	LeakyReluBiasAdd:output:0*0
_output_shapes
:€€€€€€€€€
А2
leaky_re_lu_4/LeakyReluВ
IdentityIdentity%leaky_re_lu_4/LeakyRelu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€
А:::X T
0
_output_shapes
:€€€€€€€€€
А
 
_user_specified_nameinputs
Ш
э
D__inference_Gen_BN_4_layer_call_and_return_conditional_losses_140656

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1б
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
§
F
*__inference_Gen_SPD_4_layer_call_fn_145843

inputs
identityй
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_4_layer_call_and_return_conditional_losses_1411452
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Н
ю
E__inference_Gen_BN_10_layer_call_and_return_conditional_losses_141849

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
Ђ
e
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_141576

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Љ
°
D__inference_Gen_BN_9_layer_call_and_return_conditional_losses_141657

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¶
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
І%
Ї
H__inference_Gen_Conv_T_1_layer_call_and_return_conditional_losses_140966

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
strided_slice_2/stack_2м
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
B :А2	
stack/3В
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
strided_slice_3/stack_2м
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3µ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_transpose/ReadVariableOpс
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
2
conv2d_transposeН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOp•
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2	
BiasAddЛ
leaky_re_lu_5/PartitionedCallPartitionedCallBiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_1409572
leaky_re_lu_5/PartitionedCallХ
IdentityIdentity&leaky_re_lu_5/PartitionedCall:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ђ
e
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_146287

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Л
r
H__inference_Gen_Concat_4_layer_call_and_return_conditional_losses_142824

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisИ
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€`PА2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:€€€€€€€€€`PА2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:€€€€€€€€€`P@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:WS
/
_output_shapes
:€€€€€€€€€`P@
 
_user_specified_nameinputs
†%
Ї
H__inference_Gen_Conv_T_5_layer_call_and_return_conditional_losses_141746

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
strided_slice_2/stack_2м
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
stack/3В
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
strided_slice_3/stack_2м
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3і
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_transpose/ReadVariableOpр
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
paddingSAME*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp§
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2	
BiasAddК
leaky_re_lu_9/PartitionedCallPartitionedCallBiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_1417372
leaky_re_lu_9/PartitionedCallФ
IdentityIdentity&leaky_re_lu_9/PartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ш
э
D__inference_Gen_BN_5_layer_call_and_return_conditional_losses_140840

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1б
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
И
А
+__inference_Gen_Conv_3_layer_call_fn_145152

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_Gen_Conv_3_layer_call_and_return_conditional_losses_1421612
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€0(А::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€0(А
 
_user_specified_nameinputs
К
Ь
)__inference_Gen_BN_7_layer_call_fn_145932

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_7_layer_call_and_return_conditional_losses_1412672
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
√
э
D__inference_Gen_BN_1_layer_call_and_return_conditional_losses_141974

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€`P@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€`P@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€`P@:::::W S
/
_output_shapes
:€€€€€€€€€`P@
 
_user_specified_nameinputs
Ж
Ь
)__inference_Gen_BN_1_layer_call_fn_144895

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_1_layer_call_and_return_conditional_losses_1402092
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
М
э
D__inference_Gen_BN_1_layer_call_and_return_conditional_losses_144882

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ѓ
e
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_146257

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2
	LeakyReluЖ
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
м
c
E__inference_Gen_SPD_3_layer_call_and_return_conditional_losses_142519

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:€€€€€€€€€А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
И
є
*__inference_Generator_layer_call_fn_143335
	gen_input
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

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60
identityИҐStatefulPartitionedCallј	
StatefulPartitionedCallStatefulPartitionedCall	gen_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*L
_read_only_resource_inputs.
,*	
 !"%&'(+,-.1234789:=>*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Generator_layer_call_and_return_conditional_losses_1432082
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*™
_input_shapesШ
Х:€€€€€€€€€ј†::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
1
_output_shapes
:€€€€€€€€€ј†
#
_user_specified_name	Gen_Input
’
c
E__inference_Gen_SPD_5_layer_call_and_return_conditional_losses_141535

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

IdentityМ

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
П
r
H__inference_Gen_Concat_2_layer_call_and_return_conditional_losses_142673

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisИ
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:€€€€€€€€€А:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs:XT
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ґ
E
)__inference_Gen_MP_5_layer_call_fn_140747

inputs
identityи
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_MP_5_layer_call_and_return_conditional_losses_1407412
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
И
J
.__inference_leaky_re_lu_8_layer_call_fn_146292

inputs
identityд
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_1415762
PartitionedCallЖ
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
«
c
*__inference_Gen_SPD_5_layer_call_fn_146106

inputs
identityИҐStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_5_layer_call_and_return_conditional_losses_1427602
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€0(А22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€0(А
 
_user_specified_nameinputs
Щ
d
E__inference_Gen_SPD_1_layer_call_and_return_conditional_losses_142133

inputs
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
 *  †?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2
dropout/MulВ
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1В
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ц
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape‘
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/yѕ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/GreaterEqualР
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€0(А:X T
0
_output_shapes
:€€€€€€€€€0(А
 
_user_specified_nameinputs
™§
«
E__inference_Generator_layer_call_and_return_conditional_losses_143042
	gen_input
gen_conv_1_142882
gen_conv_1_142884
gen_bn_1_142888
gen_bn_1_142890
gen_bn_1_142892
gen_bn_1_142894
gen_conv_2_142897
gen_conv_2_142899
gen_bn_2_142903
gen_bn_2_142905
gen_bn_2_142907
gen_bn_2_142909
gen_conv_3_142913
gen_conv_3_142915
gen_bn_3_142919
gen_bn_3_142921
gen_bn_3_142923
gen_bn_3_142925
gen_conv_4_142928
gen_conv_4_142930
gen_bn_4_142934
gen_bn_4_142936
gen_bn_4_142938
gen_bn_4_142940
gen_conv_5_142944
gen_conv_5_142946
gen_bn_5_142950
gen_bn_5_142952
gen_bn_5_142954
gen_bn_5_142956
gen_conv_t_1_142960
gen_conv_t_1_142962
gen_bn_6_142965
gen_bn_6_142967
gen_bn_6_142969
gen_bn_6_142971
gen_conv_t_2_142976
gen_conv_t_2_142978
gen_bn_7_142981
gen_bn_7_142983
gen_bn_7_142985
gen_bn_7_142987
gen_conv_t_3_142991
gen_conv_t_3_142993
gen_bn_8_142996
gen_bn_8_142998
gen_bn_8_143000
gen_bn_8_143002
gen_conv_t_4_143007
gen_conv_t_4_143009
gen_bn_9_143012
gen_bn_9_143014
gen_bn_9_143016
gen_bn_9_143018
gen_conv_t_5_143022
gen_conv_t_5_143024
gen_bn_10_143027
gen_bn_10_143029
gen_bn_10_143031
gen_bn_10_143033
gen_conv_t_6_143036
gen_conv_t_6_143038
identityИҐ Gen_BN_1/StatefulPartitionedCallҐ!Gen_BN_10/StatefulPartitionedCallҐ Gen_BN_2/StatefulPartitionedCallҐ Gen_BN_3/StatefulPartitionedCallҐ Gen_BN_4/StatefulPartitionedCallҐ Gen_BN_5/StatefulPartitionedCallҐ Gen_BN_6/StatefulPartitionedCallҐ Gen_BN_7/StatefulPartitionedCallҐ Gen_BN_8/StatefulPartitionedCallҐ Gen_BN_9/StatefulPartitionedCallҐ"Gen_Conv_1/StatefulPartitionedCallҐ"Gen_Conv_2/StatefulPartitionedCallҐ"Gen_Conv_3/StatefulPartitionedCallҐ"Gen_Conv_4/StatefulPartitionedCallҐ"Gen_Conv_5/StatefulPartitionedCallҐ$Gen_Conv_T_1/StatefulPartitionedCallҐ$Gen_Conv_T_2/StatefulPartitionedCallҐ$Gen_Conv_T_3/StatefulPartitionedCallҐ$Gen_Conv_T_4/StatefulPartitionedCallҐ$Gen_Conv_T_5/StatefulPartitionedCallҐ$Gen_Conv_T_6/StatefulPartitionedCallЃ
"Gen_Conv_1/StatefulPartitionedCallStatefulPartitionedCall	gen_inputgen_conv_1_142882gen_conv_1_142884*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ј†@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_Gen_Conv_1_layer_call_and_return_conditional_losses_1419202$
"Gen_Conv_1/StatefulPartitionedCallД
Gen_MP_1/PartitionedCallPartitionedCall+Gen_Conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`P@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_MP_1_layer_call_and_return_conditional_losses_1401412
Gen_MP_1/PartitionedCallа
 Gen_BN_1/StatefulPartitionedCallStatefulPartitionedCall!Gen_MP_1/PartitionedCall:output:0gen_bn_1_142888gen_bn_1_142890gen_bn_1_142892gen_bn_1_142894*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`P@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_1_layer_call_and_return_conditional_losses_1419742"
 Gen_BN_1/StatefulPartitionedCallЌ
"Gen_Conv_2/StatefulPartitionedCallStatefulPartitionedCall)Gen_BN_1/StatefulPartitionedCall:output:0gen_conv_2_142897gen_conv_2_142899*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€`PА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_Gen_Conv_2_layer_call_and_return_conditional_losses_1420212$
"Gen_Conv_2/StatefulPartitionedCallЕ
Gen_MP_2/PartitionedCallPartitionedCall+Gen_Conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_MP_2_layer_call_and_return_conditional_losses_1402572
Gen_MP_2/PartitionedCallб
 Gen_BN_2/StatefulPartitionedCallStatefulPartitionedCall!Gen_MP_2/PartitionedCall:output:0gen_bn_2_142903gen_bn_2_142905gen_bn_2_142907gen_bn_2_142909*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_2_layer_call_and_return_conditional_losses_1420752"
 Gen_BN_2/StatefulPartitionedCallЖ
Gen_SPD_1/PartitionedCallPartitionedCall)Gen_BN_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_1_layer_call_and_return_conditional_losses_1421382
Gen_SPD_1/PartitionedCall∆
"Gen_Conv_3/StatefulPartitionedCallStatefulPartitionedCall"Gen_SPD_1/PartitionedCall:output:0gen_conv_3_142913gen_conv_3_142915*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_Gen_Conv_3_layer_call_and_return_conditional_losses_1421612$
"Gen_Conv_3/StatefulPartitionedCallЕ
Gen_MP_3/PartitionedCallPartitionedCall+Gen_Conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_MP_3_layer_call_and_return_conditional_losses_1404412
Gen_MP_3/PartitionedCallб
 Gen_BN_3/StatefulPartitionedCallStatefulPartitionedCall!Gen_MP_3/PartitionedCall:output:0gen_bn_3_142919gen_bn_3_142921gen_bn_3_142923gen_bn_3_142925*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_3_layer_call_and_return_conditional_losses_1422152"
 Gen_BN_3/StatefulPartitionedCallЌ
"Gen_Conv_4/StatefulPartitionedCallStatefulPartitionedCall)Gen_BN_3/StatefulPartitionedCall:output:0gen_conv_4_142928gen_conv_4_142930*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_Gen_Conv_4_layer_call_and_return_conditional_losses_1422622$
"Gen_Conv_4/StatefulPartitionedCallЕ
Gen_MP_4/PartitionedCallPartitionedCall+Gen_Conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_MP_4_layer_call_and_return_conditional_losses_1405572
Gen_MP_4/PartitionedCallб
 Gen_BN_4/StatefulPartitionedCallStatefulPartitionedCall!Gen_MP_4/PartitionedCall:output:0gen_bn_4_142934gen_bn_4_142936gen_bn_4_142938gen_bn_4_142940*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_4_layer_call_and_return_conditional_losses_1423162"
 Gen_BN_4/StatefulPartitionedCallЖ
Gen_SPD_2/PartitionedCallPartitionedCall)Gen_BN_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_2_layer_call_and_return_conditional_losses_1423792
Gen_SPD_2/PartitionedCall∆
"Gen_Conv_5/StatefulPartitionedCallStatefulPartitionedCall"Gen_SPD_2/PartitionedCall:output:0gen_conv_5_142944gen_conv_5_142946*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_Gen_Conv_5_layer_call_and_return_conditional_losses_1424022$
"Gen_Conv_5/StatefulPartitionedCallЕ
Gen_MP_5/PartitionedCallPartitionedCall+Gen_Conv_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_MP_5_layer_call_and_return_conditional_losses_1407412
Gen_MP_5/PartitionedCallб
 Gen_BN_5/StatefulPartitionedCallStatefulPartitionedCall!Gen_MP_5/PartitionedCall:output:0gen_bn_5_142950gen_bn_5_142952gen_bn_5_142954gen_bn_5_142956*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_5_layer_call_and_return_conditional_losses_1424562"
 Gen_BN_5/StatefulPartitionedCallЖ
Gen_SPD_3/PartitionedCallPartitionedCall)Gen_BN_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_3_layer_call_and_return_conditional_losses_1425192
Gen_SPD_3/PartitionedCallв
$Gen_Conv_T_1/StatefulPartitionedCallStatefulPartitionedCall"Gen_SPD_3/PartitionedCall:output:0gen_conv_t_1_142960gen_conv_t_1_142962*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_1_layer_call_and_return_conditional_losses_1409662&
$Gen_Conv_T_1/StatefulPartitionedCall€
 Gen_BN_6/StatefulPartitionedCallStatefulPartitionedCall-Gen_Conv_T_1/StatefulPartitionedCall:output:0gen_bn_6_142965gen_bn_6_142967gen_bn_6_142969gen_bn_6_142971*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_6_layer_call_and_return_conditional_losses_1410692"
 Gen_BN_6/StatefulPartitionedCallї
Gen_Concat_1/PartitionedCallPartitionedCall)Gen_BN_6/StatefulPartitionedCall:output:0)Gen_BN_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Concat_1_layer_call_and_return_conditional_losses_1425782
Gen_Concat_1/PartitionedCallВ
Gen_SPD_4/PartitionedCallPartitionedCall%Gen_Concat_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_4_layer_call_and_return_conditional_losses_1426142
Gen_SPD_4/PartitionedCallв
$Gen_Conv_T_2/StatefulPartitionedCallStatefulPartitionedCall"Gen_SPD_4/PartitionedCall:output:0gen_conv_t_2_142976gen_conv_t_2_142978*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_2_layer_call_and_return_conditional_losses_1411952&
$Gen_Conv_T_2/StatefulPartitionedCall€
 Gen_BN_7/StatefulPartitionedCallStatefulPartitionedCall-Gen_Conv_T_2/StatefulPartitionedCall:output:0gen_bn_7_142981gen_bn_7_142983gen_bn_7_142985gen_bn_7_142987*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_7_layer_call_and_return_conditional_losses_1412982"
 Gen_BN_7/StatefulPartitionedCallї
Gen_Concat_2/PartitionedCallPartitionedCall)Gen_BN_7/StatefulPartitionedCall:output:0)Gen_BN_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Concat_2_layer_call_and_return_conditional_losses_1426732
Gen_Concat_2/PartitionedCallе
$Gen_Conv_T_3/StatefulPartitionedCallStatefulPartitionedCall%Gen_Concat_2/PartitionedCall:output:0gen_conv_t_3_142991gen_conv_t_3_142993*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_3_layer_call_and_return_conditional_losses_1413562&
$Gen_Conv_T_3/StatefulPartitionedCall€
 Gen_BN_8/StatefulPartitionedCallStatefulPartitionedCall-Gen_Conv_T_3/StatefulPartitionedCall:output:0gen_bn_8_142996gen_bn_8_142998gen_bn_8_143000gen_bn_8_143002*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_8_layer_call_and_return_conditional_losses_1414592"
 Gen_BN_8/StatefulPartitionedCallї
Gen_Concat_3/PartitionedCallPartitionedCall)Gen_BN_8/StatefulPartitionedCall:output:0)Gen_BN_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Concat_3_layer_call_and_return_conditional_losses_1427292
Gen_Concat_3/PartitionedCallВ
Gen_SPD_5/PartitionedCallPartitionedCall%Gen_Concat_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_5_layer_call_and_return_conditional_losses_1427652
Gen_SPD_5/PartitionedCallб
$Gen_Conv_T_4/StatefulPartitionedCallStatefulPartitionedCall"Gen_SPD_5/PartitionedCall:output:0gen_conv_t_4_143007gen_conv_t_4_143009*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_4_layer_call_and_return_conditional_losses_1415852&
$Gen_Conv_T_4/StatefulPartitionedCallю
 Gen_BN_9/StatefulPartitionedCallStatefulPartitionedCall-Gen_Conv_T_4/StatefulPartitionedCall:output:0gen_bn_9_143012gen_bn_9_143014gen_bn_9_143016gen_bn_9_143018*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_9_layer_call_and_return_conditional_losses_1416882"
 Gen_BN_9/StatefulPartitionedCallї
Gen_Concat_4/PartitionedCallPartitionedCall)Gen_BN_9/StatefulPartitionedCall:output:0)Gen_BN_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€`PА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Concat_4_layer_call_and_return_conditional_losses_1428242
Gen_Concat_4/PartitionedCallд
$Gen_Conv_T_5/StatefulPartitionedCallStatefulPartitionedCall%Gen_Concat_4/PartitionedCall:output:0gen_conv_t_5_143022gen_conv_t_5_143024*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_5_layer_call_and_return_conditional_losses_1417462&
$Gen_Conv_T_5/StatefulPartitionedCallЕ
!Gen_BN_10/StatefulPartitionedCallStatefulPartitionedCall-Gen_Conv_T_5/StatefulPartitionedCall:output:0gen_bn_10_143027gen_bn_10_143029gen_bn_10_143031gen_bn_10_143033*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_BN_10_layer_call_and_return_conditional_losses_1418492#
!Gen_BN_10/StatefulPartitionedCallй
$Gen_Conv_T_6/StatefulPartitionedCallStatefulPartitionedCall*Gen_BN_10/StatefulPartitionedCall:output:0gen_conv_t_6_143036gen_conv_t_6_143038*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_6_layer_call_and_return_conditional_losses_1418952&
$Gen_Conv_T_6/StatefulPartitionedCallЭ
IdentityIdentity-Gen_Conv_T_6/StatefulPartitionedCall:output:0!^Gen_BN_1/StatefulPartitionedCall"^Gen_BN_10/StatefulPartitionedCall!^Gen_BN_2/StatefulPartitionedCall!^Gen_BN_3/StatefulPartitionedCall!^Gen_BN_4/StatefulPartitionedCall!^Gen_BN_5/StatefulPartitionedCall!^Gen_BN_6/StatefulPartitionedCall!^Gen_BN_7/StatefulPartitionedCall!^Gen_BN_8/StatefulPartitionedCall!^Gen_BN_9/StatefulPartitionedCall#^Gen_Conv_1/StatefulPartitionedCall#^Gen_Conv_2/StatefulPartitionedCall#^Gen_Conv_3/StatefulPartitionedCall#^Gen_Conv_4/StatefulPartitionedCall#^Gen_Conv_5/StatefulPartitionedCall%^Gen_Conv_T_1/StatefulPartitionedCall%^Gen_Conv_T_2/StatefulPartitionedCall%^Gen_Conv_T_3/StatefulPartitionedCall%^Gen_Conv_T_4/StatefulPartitionedCall%^Gen_Conv_T_5/StatefulPartitionedCall%^Gen_Conv_T_6/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*™
_input_shapesШ
Х:€€€€€€€€€ј†::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2D
 Gen_BN_1/StatefulPartitionedCall Gen_BN_1/StatefulPartitionedCall2F
!Gen_BN_10/StatefulPartitionedCall!Gen_BN_10/StatefulPartitionedCall2D
 Gen_BN_2/StatefulPartitionedCall Gen_BN_2/StatefulPartitionedCall2D
 Gen_BN_3/StatefulPartitionedCall Gen_BN_3/StatefulPartitionedCall2D
 Gen_BN_4/StatefulPartitionedCall Gen_BN_4/StatefulPartitionedCall2D
 Gen_BN_5/StatefulPartitionedCall Gen_BN_5/StatefulPartitionedCall2D
 Gen_BN_6/StatefulPartitionedCall Gen_BN_6/StatefulPartitionedCall2D
 Gen_BN_7/StatefulPartitionedCall Gen_BN_7/StatefulPartitionedCall2D
 Gen_BN_8/StatefulPartitionedCall Gen_BN_8/StatefulPartitionedCall2D
 Gen_BN_9/StatefulPartitionedCall Gen_BN_9/StatefulPartitionedCall2H
"Gen_Conv_1/StatefulPartitionedCall"Gen_Conv_1/StatefulPartitionedCall2H
"Gen_Conv_2/StatefulPartitionedCall"Gen_Conv_2/StatefulPartitionedCall2H
"Gen_Conv_3/StatefulPartitionedCall"Gen_Conv_3/StatefulPartitionedCall2H
"Gen_Conv_4/StatefulPartitionedCall"Gen_Conv_4/StatefulPartitionedCall2H
"Gen_Conv_5/StatefulPartitionedCall"Gen_Conv_5/StatefulPartitionedCall2L
$Gen_Conv_T_1/StatefulPartitionedCall$Gen_Conv_T_1/StatefulPartitionedCall2L
$Gen_Conv_T_2/StatefulPartitionedCall$Gen_Conv_T_2/StatefulPartitionedCall2L
$Gen_Conv_T_3/StatefulPartitionedCall$Gen_Conv_T_3/StatefulPartitionedCall2L
$Gen_Conv_T_4/StatefulPartitionedCall$Gen_Conv_T_4/StatefulPartitionedCall2L
$Gen_Conv_T_5/StatefulPartitionedCall$Gen_Conv_T_5/StatefulPartitionedCall2L
$Gen_Conv_T_6/StatefulPartitionedCall$Gen_Conv_T_6/StatefulPartitionedCall:\ X
1
_output_shapes
:€€€€€€€€€ј†
#
_user_specified_name	Gen_Input
Ш
э
D__inference_Gen_BN_6_layer_call_and_return_conditional_losses_141069

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1б
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
П
r
H__inference_Gen_Concat_1_layer_call_and_return_conditional_losses_142578

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisИ
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€
А2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:€€€€€€€€€
А:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs:XT
0
_output_shapes
:€€€€€€€€€
А
 
_user_specified_nameinputs
ѓ
c
*__inference_Gen_SPD_3_layer_call_fn_145723

inputs
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_3_layer_call_and_return_conditional_losses_1409062
StatefulPartitionedCall±
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
»
°
D__inference_Gen_BN_2_layer_call_and_return_conditional_losses_140325

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1п
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1І
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
‘
В
-__inference_Gen_Conv_T_2_layer_call_fn_141205

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_2_layer_call_and_return_conditional_losses_1411952
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
“
В
-__inference_Gen_Conv_T_5_layer_call_fn_141756

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_5_layer_call_and_return_conditional_losses_1417462
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Э
d
E__inference_Gen_SPD_5_layer_call_and_return_conditional_losses_141525

inputs
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
 *  †?2
dropout/ConstЦ
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2
dropout/MulВ
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1В
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ц
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape‘
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/yѕ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/GreaterEqualР
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/CastЭ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2
dropout/Mul_1И
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
И
J
.__inference_leaky_re_lu_9_layer_call_fn_146302

inputs
identityд
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_1417372
PartitionedCallЖ
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
А
°
D__inference_Gen_BN_5_layer_call_and_return_conditional_losses_142438

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ё
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Х
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Щ
d
E__inference_Gen_SPD_2_layer_call_and_return_conditional_losses_145451

inputs
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
 *  †?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€
А2
dropout/MulВ
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1В
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ц
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape‘
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/yѕ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/GreaterEqualР
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€
А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€
А:X T
0
_output_shapes
:€€€€€€€€€
А
 
_user_specified_nameinputs
м
c
E__inference_Gen_SPD_5_layer_call_and_return_conditional_losses_142765

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:€€€€€€€€€0(А:X T
0
_output_shapes
:€€€€€€€€€0(А
 
_user_specified_nameinputs
°§
ƒ
E__inference_Generator_layer_call_and_return_conditional_losses_143500

inputs
gen_conv_1_143340
gen_conv_1_143342
gen_bn_1_143346
gen_bn_1_143348
gen_bn_1_143350
gen_bn_1_143352
gen_conv_2_143355
gen_conv_2_143357
gen_bn_2_143361
gen_bn_2_143363
gen_bn_2_143365
gen_bn_2_143367
gen_conv_3_143371
gen_conv_3_143373
gen_bn_3_143377
gen_bn_3_143379
gen_bn_3_143381
gen_bn_3_143383
gen_conv_4_143386
gen_conv_4_143388
gen_bn_4_143392
gen_bn_4_143394
gen_bn_4_143396
gen_bn_4_143398
gen_conv_5_143402
gen_conv_5_143404
gen_bn_5_143408
gen_bn_5_143410
gen_bn_5_143412
gen_bn_5_143414
gen_conv_t_1_143418
gen_conv_t_1_143420
gen_bn_6_143423
gen_bn_6_143425
gen_bn_6_143427
gen_bn_6_143429
gen_conv_t_2_143434
gen_conv_t_2_143436
gen_bn_7_143439
gen_bn_7_143441
gen_bn_7_143443
gen_bn_7_143445
gen_conv_t_3_143449
gen_conv_t_3_143451
gen_bn_8_143454
gen_bn_8_143456
gen_bn_8_143458
gen_bn_8_143460
gen_conv_t_4_143465
gen_conv_t_4_143467
gen_bn_9_143470
gen_bn_9_143472
gen_bn_9_143474
gen_bn_9_143476
gen_conv_t_5_143480
gen_conv_t_5_143482
gen_bn_10_143485
gen_bn_10_143487
gen_bn_10_143489
gen_bn_10_143491
gen_conv_t_6_143494
gen_conv_t_6_143496
identityИҐ Gen_BN_1/StatefulPartitionedCallҐ!Gen_BN_10/StatefulPartitionedCallҐ Gen_BN_2/StatefulPartitionedCallҐ Gen_BN_3/StatefulPartitionedCallҐ Gen_BN_4/StatefulPartitionedCallҐ Gen_BN_5/StatefulPartitionedCallҐ Gen_BN_6/StatefulPartitionedCallҐ Gen_BN_7/StatefulPartitionedCallҐ Gen_BN_8/StatefulPartitionedCallҐ Gen_BN_9/StatefulPartitionedCallҐ"Gen_Conv_1/StatefulPartitionedCallҐ"Gen_Conv_2/StatefulPartitionedCallҐ"Gen_Conv_3/StatefulPartitionedCallҐ"Gen_Conv_4/StatefulPartitionedCallҐ"Gen_Conv_5/StatefulPartitionedCallҐ$Gen_Conv_T_1/StatefulPartitionedCallҐ$Gen_Conv_T_2/StatefulPartitionedCallҐ$Gen_Conv_T_3/StatefulPartitionedCallҐ$Gen_Conv_T_4/StatefulPartitionedCallҐ$Gen_Conv_T_5/StatefulPartitionedCallҐ$Gen_Conv_T_6/StatefulPartitionedCallЂ
"Gen_Conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsgen_conv_1_143340gen_conv_1_143342*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ј†@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_Gen_Conv_1_layer_call_and_return_conditional_losses_1419202$
"Gen_Conv_1/StatefulPartitionedCallД
Gen_MP_1/PartitionedCallPartitionedCall+Gen_Conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`P@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_MP_1_layer_call_and_return_conditional_losses_1401412
Gen_MP_1/PartitionedCallа
 Gen_BN_1/StatefulPartitionedCallStatefulPartitionedCall!Gen_MP_1/PartitionedCall:output:0gen_bn_1_143346gen_bn_1_143348gen_bn_1_143350gen_bn_1_143352*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`P@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_1_layer_call_and_return_conditional_losses_1419742"
 Gen_BN_1/StatefulPartitionedCallЌ
"Gen_Conv_2/StatefulPartitionedCallStatefulPartitionedCall)Gen_BN_1/StatefulPartitionedCall:output:0gen_conv_2_143355gen_conv_2_143357*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€`PА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_Gen_Conv_2_layer_call_and_return_conditional_losses_1420212$
"Gen_Conv_2/StatefulPartitionedCallЕ
Gen_MP_2/PartitionedCallPartitionedCall+Gen_Conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_MP_2_layer_call_and_return_conditional_losses_1402572
Gen_MP_2/PartitionedCallб
 Gen_BN_2/StatefulPartitionedCallStatefulPartitionedCall!Gen_MP_2/PartitionedCall:output:0gen_bn_2_143361gen_bn_2_143363gen_bn_2_143365gen_bn_2_143367*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_2_layer_call_and_return_conditional_losses_1420752"
 Gen_BN_2/StatefulPartitionedCallЖ
Gen_SPD_1/PartitionedCallPartitionedCall)Gen_BN_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_1_layer_call_and_return_conditional_losses_1421382
Gen_SPD_1/PartitionedCall∆
"Gen_Conv_3/StatefulPartitionedCallStatefulPartitionedCall"Gen_SPD_1/PartitionedCall:output:0gen_conv_3_143371gen_conv_3_143373*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_Gen_Conv_3_layer_call_and_return_conditional_losses_1421612$
"Gen_Conv_3/StatefulPartitionedCallЕ
Gen_MP_3/PartitionedCallPartitionedCall+Gen_Conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_MP_3_layer_call_and_return_conditional_losses_1404412
Gen_MP_3/PartitionedCallб
 Gen_BN_3/StatefulPartitionedCallStatefulPartitionedCall!Gen_MP_3/PartitionedCall:output:0gen_bn_3_143377gen_bn_3_143379gen_bn_3_143381gen_bn_3_143383*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_3_layer_call_and_return_conditional_losses_1422152"
 Gen_BN_3/StatefulPartitionedCallЌ
"Gen_Conv_4/StatefulPartitionedCallStatefulPartitionedCall)Gen_BN_3/StatefulPartitionedCall:output:0gen_conv_4_143386gen_conv_4_143388*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_Gen_Conv_4_layer_call_and_return_conditional_losses_1422622$
"Gen_Conv_4/StatefulPartitionedCallЕ
Gen_MP_4/PartitionedCallPartitionedCall+Gen_Conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_MP_4_layer_call_and_return_conditional_losses_1405572
Gen_MP_4/PartitionedCallб
 Gen_BN_4/StatefulPartitionedCallStatefulPartitionedCall!Gen_MP_4/PartitionedCall:output:0gen_bn_4_143392gen_bn_4_143394gen_bn_4_143396gen_bn_4_143398*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_4_layer_call_and_return_conditional_losses_1423162"
 Gen_BN_4/StatefulPartitionedCallЖ
Gen_SPD_2/PartitionedCallPartitionedCall)Gen_BN_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_2_layer_call_and_return_conditional_losses_1423792
Gen_SPD_2/PartitionedCall∆
"Gen_Conv_5/StatefulPartitionedCallStatefulPartitionedCall"Gen_SPD_2/PartitionedCall:output:0gen_conv_5_143402gen_conv_5_143404*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_Gen_Conv_5_layer_call_and_return_conditional_losses_1424022$
"Gen_Conv_5/StatefulPartitionedCallЕ
Gen_MP_5/PartitionedCallPartitionedCall+Gen_Conv_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_MP_5_layer_call_and_return_conditional_losses_1407412
Gen_MP_5/PartitionedCallб
 Gen_BN_5/StatefulPartitionedCallStatefulPartitionedCall!Gen_MP_5/PartitionedCall:output:0gen_bn_5_143408gen_bn_5_143410gen_bn_5_143412gen_bn_5_143414*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_5_layer_call_and_return_conditional_losses_1424562"
 Gen_BN_5/StatefulPartitionedCallЖ
Gen_SPD_3/PartitionedCallPartitionedCall)Gen_BN_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_3_layer_call_and_return_conditional_losses_1425192
Gen_SPD_3/PartitionedCallв
$Gen_Conv_T_1/StatefulPartitionedCallStatefulPartitionedCall"Gen_SPD_3/PartitionedCall:output:0gen_conv_t_1_143418gen_conv_t_1_143420*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_1_layer_call_and_return_conditional_losses_1409662&
$Gen_Conv_T_1/StatefulPartitionedCall€
 Gen_BN_6/StatefulPartitionedCallStatefulPartitionedCall-Gen_Conv_T_1/StatefulPartitionedCall:output:0gen_bn_6_143423gen_bn_6_143425gen_bn_6_143427gen_bn_6_143429*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_6_layer_call_and_return_conditional_losses_1410692"
 Gen_BN_6/StatefulPartitionedCallї
Gen_Concat_1/PartitionedCallPartitionedCall)Gen_BN_6/StatefulPartitionedCall:output:0)Gen_BN_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Concat_1_layer_call_and_return_conditional_losses_1425782
Gen_Concat_1/PartitionedCallВ
Gen_SPD_4/PartitionedCallPartitionedCall%Gen_Concat_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_4_layer_call_and_return_conditional_losses_1426142
Gen_SPD_4/PartitionedCallв
$Gen_Conv_T_2/StatefulPartitionedCallStatefulPartitionedCall"Gen_SPD_4/PartitionedCall:output:0gen_conv_t_2_143434gen_conv_t_2_143436*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_2_layer_call_and_return_conditional_losses_1411952&
$Gen_Conv_T_2/StatefulPartitionedCall€
 Gen_BN_7/StatefulPartitionedCallStatefulPartitionedCall-Gen_Conv_T_2/StatefulPartitionedCall:output:0gen_bn_7_143439gen_bn_7_143441gen_bn_7_143443gen_bn_7_143445*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_7_layer_call_and_return_conditional_losses_1412982"
 Gen_BN_7/StatefulPartitionedCallї
Gen_Concat_2/PartitionedCallPartitionedCall)Gen_BN_7/StatefulPartitionedCall:output:0)Gen_BN_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Concat_2_layer_call_and_return_conditional_losses_1426732
Gen_Concat_2/PartitionedCallе
$Gen_Conv_T_3/StatefulPartitionedCallStatefulPartitionedCall%Gen_Concat_2/PartitionedCall:output:0gen_conv_t_3_143449gen_conv_t_3_143451*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_3_layer_call_and_return_conditional_losses_1413562&
$Gen_Conv_T_3/StatefulPartitionedCall€
 Gen_BN_8/StatefulPartitionedCallStatefulPartitionedCall-Gen_Conv_T_3/StatefulPartitionedCall:output:0gen_bn_8_143454gen_bn_8_143456gen_bn_8_143458gen_bn_8_143460*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_8_layer_call_and_return_conditional_losses_1414592"
 Gen_BN_8/StatefulPartitionedCallї
Gen_Concat_3/PartitionedCallPartitionedCall)Gen_BN_8/StatefulPartitionedCall:output:0)Gen_BN_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Concat_3_layer_call_and_return_conditional_losses_1427292
Gen_Concat_3/PartitionedCallВ
Gen_SPD_5/PartitionedCallPartitionedCall%Gen_Concat_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_5_layer_call_and_return_conditional_losses_1427652
Gen_SPD_5/PartitionedCallб
$Gen_Conv_T_4/StatefulPartitionedCallStatefulPartitionedCall"Gen_SPD_5/PartitionedCall:output:0gen_conv_t_4_143465gen_conv_t_4_143467*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_4_layer_call_and_return_conditional_losses_1415852&
$Gen_Conv_T_4/StatefulPartitionedCallю
 Gen_BN_9/StatefulPartitionedCallStatefulPartitionedCall-Gen_Conv_T_4/StatefulPartitionedCall:output:0gen_bn_9_143470gen_bn_9_143472gen_bn_9_143474gen_bn_9_143476*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_9_layer_call_and_return_conditional_losses_1416882"
 Gen_BN_9/StatefulPartitionedCallї
Gen_Concat_4/PartitionedCallPartitionedCall)Gen_BN_9/StatefulPartitionedCall:output:0)Gen_BN_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€`PА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Concat_4_layer_call_and_return_conditional_losses_1428242
Gen_Concat_4/PartitionedCallд
$Gen_Conv_T_5/StatefulPartitionedCallStatefulPartitionedCall%Gen_Concat_4/PartitionedCall:output:0gen_conv_t_5_143480gen_conv_t_5_143482*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_5_layer_call_and_return_conditional_losses_1417462&
$Gen_Conv_T_5/StatefulPartitionedCallЕ
!Gen_BN_10/StatefulPartitionedCallStatefulPartitionedCall-Gen_Conv_T_5/StatefulPartitionedCall:output:0gen_bn_10_143485gen_bn_10_143487gen_bn_10_143489gen_bn_10_143491*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_BN_10_layer_call_and_return_conditional_losses_1418492#
!Gen_BN_10/StatefulPartitionedCallй
$Gen_Conv_T_6/StatefulPartitionedCallStatefulPartitionedCall*Gen_BN_10/StatefulPartitionedCall:output:0gen_conv_t_6_143494gen_conv_t_6_143496*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_6_layer_call_and_return_conditional_losses_1418952&
$Gen_Conv_T_6/StatefulPartitionedCallЭ
IdentityIdentity-Gen_Conv_T_6/StatefulPartitionedCall:output:0!^Gen_BN_1/StatefulPartitionedCall"^Gen_BN_10/StatefulPartitionedCall!^Gen_BN_2/StatefulPartitionedCall!^Gen_BN_3/StatefulPartitionedCall!^Gen_BN_4/StatefulPartitionedCall!^Gen_BN_5/StatefulPartitionedCall!^Gen_BN_6/StatefulPartitionedCall!^Gen_BN_7/StatefulPartitionedCall!^Gen_BN_8/StatefulPartitionedCall!^Gen_BN_9/StatefulPartitionedCall#^Gen_Conv_1/StatefulPartitionedCall#^Gen_Conv_2/StatefulPartitionedCall#^Gen_Conv_3/StatefulPartitionedCall#^Gen_Conv_4/StatefulPartitionedCall#^Gen_Conv_5/StatefulPartitionedCall%^Gen_Conv_T_1/StatefulPartitionedCall%^Gen_Conv_T_2/StatefulPartitionedCall%^Gen_Conv_T_3/StatefulPartitionedCall%^Gen_Conv_T_4/StatefulPartitionedCall%^Gen_Conv_T_5/StatefulPartitionedCall%^Gen_Conv_T_6/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*™
_input_shapesШ
Х:€€€€€€€€€ј†::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2D
 Gen_BN_1/StatefulPartitionedCall Gen_BN_1/StatefulPartitionedCall2F
!Gen_BN_10/StatefulPartitionedCall!Gen_BN_10/StatefulPartitionedCall2D
 Gen_BN_2/StatefulPartitionedCall Gen_BN_2/StatefulPartitionedCall2D
 Gen_BN_3/StatefulPartitionedCall Gen_BN_3/StatefulPartitionedCall2D
 Gen_BN_4/StatefulPartitionedCall Gen_BN_4/StatefulPartitionedCall2D
 Gen_BN_5/StatefulPartitionedCall Gen_BN_5/StatefulPartitionedCall2D
 Gen_BN_6/StatefulPartitionedCall Gen_BN_6/StatefulPartitionedCall2D
 Gen_BN_7/StatefulPartitionedCall Gen_BN_7/StatefulPartitionedCall2D
 Gen_BN_8/StatefulPartitionedCall Gen_BN_8/StatefulPartitionedCall2D
 Gen_BN_9/StatefulPartitionedCall Gen_BN_9/StatefulPartitionedCall2H
"Gen_Conv_1/StatefulPartitionedCall"Gen_Conv_1/StatefulPartitionedCall2H
"Gen_Conv_2/StatefulPartitionedCall"Gen_Conv_2/StatefulPartitionedCall2H
"Gen_Conv_3/StatefulPartitionedCall"Gen_Conv_3/StatefulPartitionedCall2H
"Gen_Conv_4/StatefulPartitionedCall"Gen_Conv_4/StatefulPartitionedCall2H
"Gen_Conv_5/StatefulPartitionedCall"Gen_Conv_5/StatefulPartitionedCall2L
$Gen_Conv_T_1/StatefulPartitionedCall$Gen_Conv_T_1/StatefulPartitionedCall2L
$Gen_Conv_T_2/StatefulPartitionedCall$Gen_Conv_T_2/StatefulPartitionedCall2L
$Gen_Conv_T_3/StatefulPartitionedCall$Gen_Conv_T_3/StatefulPartitionedCall2L
$Gen_Conv_T_4/StatefulPartitionedCall$Gen_Conv_T_4/StatefulPartitionedCall2L
$Gen_Conv_T_5/StatefulPartitionedCall$Gen_Conv_T_5/StatefulPartitionedCall2L
$Gen_Conv_T_6/StatefulPartitionedCall$Gen_Conv_T_6/StatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€ј†
 
_user_specified_nameinputs
Э
d
E__inference_Gen_SPD_2_layer_call_and_return_conditional_losses_145489

inputs
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
 *  †?2
dropout/ConstЦ
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2
dropout/MulВ
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1В
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ц
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape‘
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/yѕ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/GreaterEqualР
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/CastЭ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2
dropout/Mul_1И
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Щ
d
E__inference_Gen_SPD_5_layer_call_and_return_conditional_losses_142760

inputs
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
 *  †?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2
dropout/MulВ
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1В
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ц
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape‘
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/yѕ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/GreaterEqualР
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€0(А:X T
0
_output_shapes
:€€€€€€€€€0(А
 
_user_specified_nameinputs
Ш
э
D__inference_Gen_BN_5_layer_call_and_return_conditional_losses_145562

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1б
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
§
F
*__inference_Gen_SPD_3_layer_call_fn_145728

inputs
identityй
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_3_layer_call_and_return_conditional_losses_1409162
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ґ
E
)__inference_Gen_MP_1_layer_call_fn_140147

inputs
identityи
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_MP_1_layer_call_and_return_conditional_losses_1401412
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ь
Y
-__inference_Gen_Concat_1_layer_call_fn_145805
inputs_0
inputs_1
identityя
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Concat_1_layer_call_and_return_conditional_losses_1425782
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:€€€€€€€€€
А:l h
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:€€€€€€€€€
А
"
_user_specified_name
inputs/1
ш
Y
-__inference_Gen_Concat_4_layer_call_fn_146188
inputs_0
inputs_1
identityя
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€`PА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Concat_4_layer_call_and_return_conditional_losses_1428242
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€`PА2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:€€€€€€€€€`P@:k g
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:€€€€€€€€€`P@
"
_user_specified_name
inputs/1
»
°
D__inference_Gen_BN_5_layer_call_and_return_conditional_losses_145544

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1п
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1І
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
’
c
E__inference_Gen_SPD_2_layer_call_and_return_conditional_losses_145494

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

IdentityМ

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
М
J
.__inference_leaky_re_lu_6_layer_call_fn_146272

inputs
identityе
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_1411862
PartitionedCallЗ
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ш
э
D__inference_Gen_BN_3_layer_call_and_return_conditional_losses_145190

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1б
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
¬
Ь
)__inference_Gen_BN_3_layer_call_fn_145267

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_3_layer_call_and_return_conditional_losses_1421972
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Щ
d
E__inference_Gen_SPD_5_layer_call_and_return_conditional_losses_146096

inputs
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
 *  †?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2
dropout/MulВ
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1В
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ц
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape‘
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/yѕ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/GreaterEqualР
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€0(А:X T
0
_output_shapes
:€€€€€€€€€0(А
 
_user_specified_nameinputs
√
э
D__inference_Gen_BN_1_layer_call_and_return_conditional_losses_144818

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€`P@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€`P@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€`P@:::::W S
/
_output_shapes
:€€€€€€€€€`P@
 
_user_specified_nameinputs
«
c
*__inference_Gen_SPD_3_layer_call_fn_145685

inputs
identityИҐStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_3_layer_call_and_return_conditional_losses_1425142
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
М
э
D__inference_Gen_BN_9_layer_call_and_return_conditional_losses_146149

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
«
c
*__inference_Gen_SPD_2_layer_call_fn_145461

inputs
identityИҐStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_2_layer_call_and_return_conditional_losses_1423742
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€
А22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€
А
 
_user_specified_nameinputs
Э
d
E__inference_Gen_SPD_4_layer_call_and_return_conditional_losses_141135

inputs
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
 *  †?2
dropout/ConstЦ
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2
dropout/MulВ
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1В
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ц
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape‘
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/yѕ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/GreaterEqualР
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/CastЭ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2
dropout/Mul_1И
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ћ	
Ѓ
F__inference_Gen_Conv_3_layer_call_and_return_conditional_losses_142161

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€0(А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2	
BiasAddГ
leaky_re_lu_2/LeakyRelu	LeakyReluBiasAdd:output:0*0
_output_shapes
:€€€€€€€€€0(А2
leaky_re_lu_2/LeakyReluВ
IdentityIdentity%leaky_re_lu_2/LeakyRelu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€0(А:::X T
0
_output_shapes
:€€€€€€€€€0(А
 
_user_specified_nameinputs
И
А
+__inference_Gen_Conv_4_layer_call_fn_145300

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_Gen_Conv_4_layer_call_and_return_conditional_losses_1422622
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
‘
В
-__inference_Gen_Conv_T_1_layer_call_fn_140976

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_1_layer_call_and_return_conditional_losses_1409662
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
љ
Ґ
E__inference_Gen_BN_10_layer_call_and_return_conditional_losses_146208

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¶
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
м
c
E__inference_Gen_SPD_4_layer_call_and_return_conditional_losses_142614

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:€€€€€€€€€
А:X T
0
_output_shapes
:€€€€€€€€€
А
 
_user_specified_nameinputs
’
c
E__inference_Gen_SPD_5_layer_call_and_return_conditional_losses_146063

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

IdentityМ

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
“
≥
$__inference_signature_wrapper_143758
	gen_input
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

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60
identityИҐStatefulPartitionedCall†	
StatefulPartitionedCallStatefulPartitionedCall	gen_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ј†*`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_1401352
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€ј†2

Identity"
identityIdentity:output:0*™
_input_shapesШ
Х:€€€€€€€€€ј†::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
1
_output_shapes
:€€€€€€€€€ј†
#
_user_specified_name	Gen_Input
м
c
E__inference_Gen_SPD_3_layer_call_and_return_conditional_losses_145680

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:€€€€€€€€€А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ѓ
e
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_146267

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2
	LeakyReluЖ
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Э
d
E__inference_Gen_SPD_4_layer_call_and_return_conditional_losses_145828

inputs
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
 *  †?2
dropout/ConstЦ
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2
dropout/MulВ
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1В
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ц
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape‘
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/yѕ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/GreaterEqualР
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/CastЭ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2
dropout/Mul_1И
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
»
°
D__inference_Gen_BN_4_layer_call_and_return_conditional_losses_140625

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1п
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1І
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
е"
Ї
H__inference_Gen_Conv_T_6_layer_call_and_return_conditional_losses_141895

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
strided_slice_2/stack_2м
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
value	B :2
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
value	B :2	
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
stack/3В
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
strided_slice_3/stack_2м
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3≥
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOpр
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp§
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2	
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2
Tanhv
IdentityIdentityTanh:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
Э
d
E__inference_Gen_SPD_3_layer_call_and_return_conditional_losses_140906

inputs
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
 *  †?2
dropout/ConstЦ
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2
dropout/MulВ
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1В
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ц
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape‘
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/yѕ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/GreaterEqualР
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/CastЭ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2
dropout/Mul_1И
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
»
°
D__inference_Gen_BN_2_layer_call_and_return_conditional_losses_145012

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1п
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1І
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
»
°
D__inference_Gen_BN_3_layer_call_and_return_conditional_losses_145172

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1п
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1І
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
ѕ
э
D__inference_Gen_BN_3_layer_call_and_return_conditional_losses_142215

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ѕ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€А:::::X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ѓ
e
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_146277

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2
	LeakyReluЖ
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Цђ
ы
E__inference_Generator_layer_call_and_return_conditional_losses_142879
	gen_input
gen_conv_1_141931
gen_conv_1_141933
gen_bn_1_142001
gen_bn_1_142003
gen_bn_1_142005
gen_bn_1_142007
gen_conv_2_142032
gen_conv_2_142034
gen_bn_2_142102
gen_bn_2_142104
gen_bn_2_142106
gen_bn_2_142108
gen_conv_3_142172
gen_conv_3_142174
gen_bn_3_142242
gen_bn_3_142244
gen_bn_3_142246
gen_bn_3_142248
gen_conv_4_142273
gen_conv_4_142275
gen_bn_4_142343
gen_bn_4_142345
gen_bn_4_142347
gen_bn_4_142349
gen_conv_5_142413
gen_conv_5_142415
gen_bn_5_142483
gen_bn_5_142485
gen_bn_5_142487
gen_bn_5_142489
gen_conv_t_1_142531
gen_conv_t_1_142533
gen_bn_6_142562
gen_bn_6_142564
gen_bn_6_142566
gen_bn_6_142568
gen_conv_t_2_142626
gen_conv_t_2_142628
gen_bn_7_142657
gen_bn_7_142659
gen_bn_7_142661
gen_bn_7_142663
gen_conv_t_3_142682
gen_conv_t_3_142684
gen_bn_8_142713
gen_bn_8_142715
gen_bn_8_142717
gen_bn_8_142719
gen_conv_t_4_142777
gen_conv_t_4_142779
gen_bn_9_142808
gen_bn_9_142810
gen_bn_9_142812
gen_bn_9_142814
gen_conv_t_5_142833
gen_conv_t_5_142835
gen_bn_10_142864
gen_bn_10_142866
gen_bn_10_142868
gen_bn_10_142870
gen_conv_t_6_142873
gen_conv_t_6_142875
identityИҐ Gen_BN_1/StatefulPartitionedCallҐ!Gen_BN_10/StatefulPartitionedCallҐ Gen_BN_2/StatefulPartitionedCallҐ Gen_BN_3/StatefulPartitionedCallҐ Gen_BN_4/StatefulPartitionedCallҐ Gen_BN_5/StatefulPartitionedCallҐ Gen_BN_6/StatefulPartitionedCallҐ Gen_BN_7/StatefulPartitionedCallҐ Gen_BN_8/StatefulPartitionedCallҐ Gen_BN_9/StatefulPartitionedCallҐ"Gen_Conv_1/StatefulPartitionedCallҐ"Gen_Conv_2/StatefulPartitionedCallҐ"Gen_Conv_3/StatefulPartitionedCallҐ"Gen_Conv_4/StatefulPartitionedCallҐ"Gen_Conv_5/StatefulPartitionedCallҐ$Gen_Conv_T_1/StatefulPartitionedCallҐ$Gen_Conv_T_2/StatefulPartitionedCallҐ$Gen_Conv_T_3/StatefulPartitionedCallҐ$Gen_Conv_T_4/StatefulPartitionedCallҐ$Gen_Conv_T_5/StatefulPartitionedCallҐ$Gen_Conv_T_6/StatefulPartitionedCallҐ!Gen_SPD_1/StatefulPartitionedCallҐ!Gen_SPD_2/StatefulPartitionedCallҐ!Gen_SPD_3/StatefulPartitionedCallҐ!Gen_SPD_4/StatefulPartitionedCallҐ!Gen_SPD_5/StatefulPartitionedCallЃ
"Gen_Conv_1/StatefulPartitionedCallStatefulPartitionedCall	gen_inputgen_conv_1_141931gen_conv_1_141933*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ј†@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_Gen_Conv_1_layer_call_and_return_conditional_losses_1419202$
"Gen_Conv_1/StatefulPartitionedCallД
Gen_MP_1/PartitionedCallPartitionedCall+Gen_Conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`P@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_MP_1_layer_call_and_return_conditional_losses_1401412
Gen_MP_1/PartitionedCallё
 Gen_BN_1/StatefulPartitionedCallStatefulPartitionedCall!Gen_MP_1/PartitionedCall:output:0gen_bn_1_142001gen_bn_1_142003gen_bn_1_142005gen_bn_1_142007*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`P@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_1_layer_call_and_return_conditional_losses_1419562"
 Gen_BN_1/StatefulPartitionedCallЌ
"Gen_Conv_2/StatefulPartitionedCallStatefulPartitionedCall)Gen_BN_1/StatefulPartitionedCall:output:0gen_conv_2_142032gen_conv_2_142034*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€`PА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_Gen_Conv_2_layer_call_and_return_conditional_losses_1420212$
"Gen_Conv_2/StatefulPartitionedCallЕ
Gen_MP_2/PartitionedCallPartitionedCall+Gen_Conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_MP_2_layer_call_and_return_conditional_losses_1402572
Gen_MP_2/PartitionedCallя
 Gen_BN_2/StatefulPartitionedCallStatefulPartitionedCall!Gen_MP_2/PartitionedCall:output:0gen_bn_2_142102gen_bn_2_142104gen_bn_2_142106gen_bn_2_142108*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_2_layer_call_and_return_conditional_losses_1420572"
 Gen_BN_2/StatefulPartitionedCallЮ
!Gen_SPD_1/StatefulPartitionedCallStatefulPartitionedCall)Gen_BN_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_1_layer_call_and_return_conditional_losses_1421332#
!Gen_SPD_1/StatefulPartitionedCallќ
"Gen_Conv_3/StatefulPartitionedCallStatefulPartitionedCall*Gen_SPD_1/StatefulPartitionedCall:output:0gen_conv_3_142172gen_conv_3_142174*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_Gen_Conv_3_layer_call_and_return_conditional_losses_1421612$
"Gen_Conv_3/StatefulPartitionedCallЕ
Gen_MP_3/PartitionedCallPartitionedCall+Gen_Conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_MP_3_layer_call_and_return_conditional_losses_1404412
Gen_MP_3/PartitionedCallя
 Gen_BN_3/StatefulPartitionedCallStatefulPartitionedCall!Gen_MP_3/PartitionedCall:output:0gen_bn_3_142242gen_bn_3_142244gen_bn_3_142246gen_bn_3_142248*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_3_layer_call_and_return_conditional_losses_1421972"
 Gen_BN_3/StatefulPartitionedCallЌ
"Gen_Conv_4/StatefulPartitionedCallStatefulPartitionedCall)Gen_BN_3/StatefulPartitionedCall:output:0gen_conv_4_142273gen_conv_4_142275*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_Gen_Conv_4_layer_call_and_return_conditional_losses_1422622$
"Gen_Conv_4/StatefulPartitionedCallЕ
Gen_MP_4/PartitionedCallPartitionedCall+Gen_Conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_MP_4_layer_call_and_return_conditional_losses_1405572
Gen_MP_4/PartitionedCallя
 Gen_BN_4/StatefulPartitionedCallStatefulPartitionedCall!Gen_MP_4/PartitionedCall:output:0gen_bn_4_142343gen_bn_4_142345gen_bn_4_142347gen_bn_4_142349*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_4_layer_call_and_return_conditional_losses_1422982"
 Gen_BN_4/StatefulPartitionedCall¬
!Gen_SPD_2/StatefulPartitionedCallStatefulPartitionedCall)Gen_BN_4/StatefulPartitionedCall:output:0"^Gen_SPD_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_2_layer_call_and_return_conditional_losses_1423742#
!Gen_SPD_2/StatefulPartitionedCallќ
"Gen_Conv_5/StatefulPartitionedCallStatefulPartitionedCall*Gen_SPD_2/StatefulPartitionedCall:output:0gen_conv_5_142413gen_conv_5_142415*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_Gen_Conv_5_layer_call_and_return_conditional_losses_1424022$
"Gen_Conv_5/StatefulPartitionedCallЕ
Gen_MP_5/PartitionedCallPartitionedCall+Gen_Conv_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_MP_5_layer_call_and_return_conditional_losses_1407412
Gen_MP_5/PartitionedCallя
 Gen_BN_5/StatefulPartitionedCallStatefulPartitionedCall!Gen_MP_5/PartitionedCall:output:0gen_bn_5_142483gen_bn_5_142485gen_bn_5_142487gen_bn_5_142489*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_5_layer_call_and_return_conditional_losses_1424382"
 Gen_BN_5/StatefulPartitionedCall¬
!Gen_SPD_3/StatefulPartitionedCallStatefulPartitionedCall)Gen_BN_5/StatefulPartitionedCall:output:0"^Gen_SPD_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_3_layer_call_and_return_conditional_losses_1425142#
!Gen_SPD_3/StatefulPartitionedCallк
$Gen_Conv_T_1/StatefulPartitionedCallStatefulPartitionedCall*Gen_SPD_3/StatefulPartitionedCall:output:0gen_conv_t_1_142531gen_conv_t_1_142533*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_1_layer_call_and_return_conditional_losses_1409662&
$Gen_Conv_T_1/StatefulPartitionedCallэ
 Gen_BN_6/StatefulPartitionedCallStatefulPartitionedCall-Gen_Conv_T_1/StatefulPartitionedCall:output:0gen_bn_6_142562gen_bn_6_142564gen_bn_6_142566gen_bn_6_142568*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_6_layer_call_and_return_conditional_losses_1410382"
 Gen_BN_6/StatefulPartitionedCallї
Gen_Concat_1/PartitionedCallPartitionedCall)Gen_BN_6/StatefulPartitionedCall:output:0)Gen_BN_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Concat_1_layer_call_and_return_conditional_losses_1425782
Gen_Concat_1/PartitionedCallЊ
!Gen_SPD_4/StatefulPartitionedCallStatefulPartitionedCall%Gen_Concat_1/PartitionedCall:output:0"^Gen_SPD_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_4_layer_call_and_return_conditional_losses_1426092#
!Gen_SPD_4/StatefulPartitionedCallк
$Gen_Conv_T_2/StatefulPartitionedCallStatefulPartitionedCall*Gen_SPD_4/StatefulPartitionedCall:output:0gen_conv_t_2_142626gen_conv_t_2_142628*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_2_layer_call_and_return_conditional_losses_1411952&
$Gen_Conv_T_2/StatefulPartitionedCallэ
 Gen_BN_7/StatefulPartitionedCallStatefulPartitionedCall-Gen_Conv_T_2/StatefulPartitionedCall:output:0gen_bn_7_142657gen_bn_7_142659gen_bn_7_142661gen_bn_7_142663*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_7_layer_call_and_return_conditional_losses_1412672"
 Gen_BN_7/StatefulPartitionedCallї
Gen_Concat_2/PartitionedCallPartitionedCall)Gen_BN_7/StatefulPartitionedCall:output:0)Gen_BN_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Concat_2_layer_call_and_return_conditional_losses_1426732
Gen_Concat_2/PartitionedCallе
$Gen_Conv_T_3/StatefulPartitionedCallStatefulPartitionedCall%Gen_Concat_2/PartitionedCall:output:0gen_conv_t_3_142682gen_conv_t_3_142684*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_3_layer_call_and_return_conditional_losses_1413562&
$Gen_Conv_T_3/StatefulPartitionedCallэ
 Gen_BN_8/StatefulPartitionedCallStatefulPartitionedCall-Gen_Conv_T_3/StatefulPartitionedCall:output:0gen_bn_8_142713gen_bn_8_142715gen_bn_8_142717gen_bn_8_142719*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_8_layer_call_and_return_conditional_losses_1414282"
 Gen_BN_8/StatefulPartitionedCallї
Gen_Concat_3/PartitionedCallPartitionedCall)Gen_BN_8/StatefulPartitionedCall:output:0)Gen_BN_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Concat_3_layer_call_and_return_conditional_losses_1427292
Gen_Concat_3/PartitionedCallЊ
!Gen_SPD_5/StatefulPartitionedCallStatefulPartitionedCall%Gen_Concat_3/PartitionedCall:output:0"^Gen_SPD_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_5_layer_call_and_return_conditional_losses_1427602#
!Gen_SPD_5/StatefulPartitionedCallй
$Gen_Conv_T_4/StatefulPartitionedCallStatefulPartitionedCall*Gen_SPD_5/StatefulPartitionedCall:output:0gen_conv_t_4_142777gen_conv_t_4_142779*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_4_layer_call_and_return_conditional_losses_1415852&
$Gen_Conv_T_4/StatefulPartitionedCallь
 Gen_BN_9/StatefulPartitionedCallStatefulPartitionedCall-Gen_Conv_T_4/StatefulPartitionedCall:output:0gen_bn_9_142808gen_bn_9_142810gen_bn_9_142812gen_bn_9_142814*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_9_layer_call_and_return_conditional_losses_1416572"
 Gen_BN_9/StatefulPartitionedCallї
Gen_Concat_4/PartitionedCallPartitionedCall)Gen_BN_9/StatefulPartitionedCall:output:0)Gen_BN_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€`PА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Concat_4_layer_call_and_return_conditional_losses_1428242
Gen_Concat_4/PartitionedCallд
$Gen_Conv_T_5/StatefulPartitionedCallStatefulPartitionedCall%Gen_Concat_4/PartitionedCall:output:0gen_conv_t_5_142833gen_conv_t_5_142835*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_5_layer_call_and_return_conditional_losses_1417462&
$Gen_Conv_T_5/StatefulPartitionedCallГ
!Gen_BN_10/StatefulPartitionedCallStatefulPartitionedCall-Gen_Conv_T_5/StatefulPartitionedCall:output:0gen_bn_10_142864gen_bn_10_142866gen_bn_10_142868gen_bn_10_142870*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_BN_10_layer_call_and_return_conditional_losses_1418182#
!Gen_BN_10/StatefulPartitionedCallй
$Gen_Conv_T_6/StatefulPartitionedCallStatefulPartitionedCall*Gen_BN_10/StatefulPartitionedCall:output:0gen_conv_t_6_142873gen_conv_t_6_142875*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_6_layer_call_and_return_conditional_losses_1418952&
$Gen_Conv_T_6/StatefulPartitionedCall—
IdentityIdentity-Gen_Conv_T_6/StatefulPartitionedCall:output:0!^Gen_BN_1/StatefulPartitionedCall"^Gen_BN_10/StatefulPartitionedCall!^Gen_BN_2/StatefulPartitionedCall!^Gen_BN_3/StatefulPartitionedCall!^Gen_BN_4/StatefulPartitionedCall!^Gen_BN_5/StatefulPartitionedCall!^Gen_BN_6/StatefulPartitionedCall!^Gen_BN_7/StatefulPartitionedCall!^Gen_BN_8/StatefulPartitionedCall!^Gen_BN_9/StatefulPartitionedCall#^Gen_Conv_1/StatefulPartitionedCall#^Gen_Conv_2/StatefulPartitionedCall#^Gen_Conv_3/StatefulPartitionedCall#^Gen_Conv_4/StatefulPartitionedCall#^Gen_Conv_5/StatefulPartitionedCall%^Gen_Conv_T_1/StatefulPartitionedCall%^Gen_Conv_T_2/StatefulPartitionedCall%^Gen_Conv_T_3/StatefulPartitionedCall%^Gen_Conv_T_4/StatefulPartitionedCall%^Gen_Conv_T_5/StatefulPartitionedCall%^Gen_Conv_T_6/StatefulPartitionedCall"^Gen_SPD_1/StatefulPartitionedCall"^Gen_SPD_2/StatefulPartitionedCall"^Gen_SPD_3/StatefulPartitionedCall"^Gen_SPD_4/StatefulPartitionedCall"^Gen_SPD_5/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*™
_input_shapesШ
Х:€€€€€€€€€ј†::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2D
 Gen_BN_1/StatefulPartitionedCall Gen_BN_1/StatefulPartitionedCall2F
!Gen_BN_10/StatefulPartitionedCall!Gen_BN_10/StatefulPartitionedCall2D
 Gen_BN_2/StatefulPartitionedCall Gen_BN_2/StatefulPartitionedCall2D
 Gen_BN_3/StatefulPartitionedCall Gen_BN_3/StatefulPartitionedCall2D
 Gen_BN_4/StatefulPartitionedCall Gen_BN_4/StatefulPartitionedCall2D
 Gen_BN_5/StatefulPartitionedCall Gen_BN_5/StatefulPartitionedCall2D
 Gen_BN_6/StatefulPartitionedCall Gen_BN_6/StatefulPartitionedCall2D
 Gen_BN_7/StatefulPartitionedCall Gen_BN_7/StatefulPartitionedCall2D
 Gen_BN_8/StatefulPartitionedCall Gen_BN_8/StatefulPartitionedCall2D
 Gen_BN_9/StatefulPartitionedCall Gen_BN_9/StatefulPartitionedCall2H
"Gen_Conv_1/StatefulPartitionedCall"Gen_Conv_1/StatefulPartitionedCall2H
"Gen_Conv_2/StatefulPartitionedCall"Gen_Conv_2/StatefulPartitionedCall2H
"Gen_Conv_3/StatefulPartitionedCall"Gen_Conv_3/StatefulPartitionedCall2H
"Gen_Conv_4/StatefulPartitionedCall"Gen_Conv_4/StatefulPartitionedCall2H
"Gen_Conv_5/StatefulPartitionedCall"Gen_Conv_5/StatefulPartitionedCall2L
$Gen_Conv_T_1/StatefulPartitionedCall$Gen_Conv_T_1/StatefulPartitionedCall2L
$Gen_Conv_T_2/StatefulPartitionedCall$Gen_Conv_T_2/StatefulPartitionedCall2L
$Gen_Conv_T_3/StatefulPartitionedCall$Gen_Conv_T_3/StatefulPartitionedCall2L
$Gen_Conv_T_4/StatefulPartitionedCall$Gen_Conv_T_4/StatefulPartitionedCall2L
$Gen_Conv_T_5/StatefulPartitionedCall$Gen_Conv_T_5/StatefulPartitionedCall2L
$Gen_Conv_T_6/StatefulPartitionedCall$Gen_Conv_T_6/StatefulPartitionedCall2F
!Gen_SPD_1/StatefulPartitionedCall!Gen_SPD_1/StatefulPartitionedCall2F
!Gen_SPD_2/StatefulPartitionedCall!Gen_SPD_2/StatefulPartitionedCall2F
!Gen_SPD_3/StatefulPartitionedCall!Gen_SPD_3/StatefulPartitionedCall2F
!Gen_SPD_4/StatefulPartitionedCall!Gen_SPD_4/StatefulPartitionedCall2F
!Gen_SPD_5/StatefulPartitionedCall!Gen_SPD_5/StatefulPartitionedCall:\ X
1
_output_shapes
:€€€€€€€€€ј†
#
_user_specified_name	Gen_Input
»	
Ѓ
F__inference_Gen_Conv_2_layer_call_and_return_conditional_losses_144919

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€`PА*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€`PА2	
BiasAddГ
leaky_re_lu_1/LeakyRelu	LeakyReluBiasAdd:output:0*0
_output_shapes
:€€€€€€€€€`PА2
leaky_re_lu_1/LeakyReluВ
IdentityIdentity%leaky_re_lu_1/LeakyRelu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€`PА2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€`P@:::W S
/
_output_shapes
:€€€€€€€€€`P@
 
_user_specified_nameinputs
ф
°
D__inference_Gen_BN_1_layer_call_and_return_conditional_losses_144800

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ў
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€`P@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Ф
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:€€€€€€€€€`P@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€`P@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:€€€€€€€€€`P@
 
_user_specified_nameinputs
М
Ь
)__inference_Gen_BN_2_layer_call_fn_145056

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_2_layer_call_and_return_conditional_losses_1403562
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ч
t
H__inference_Gen_Concat_1_layer_call_and_return_conditional_losses_145799
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisК
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€
А2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:€€€€€€€€€
А:l h
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:€€€€€€€€€
А
"
_user_specified_name
inputs/1
ѓ
c
*__inference_Gen_SPD_1_layer_call_fn_145127

inputs
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_1_layer_call_and_return_conditional_losses_1404222
StatefulPartitionedCall±
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
М
Ь
)__inference_Gen_BN_7_layer_call_fn_145945

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_7_layer_call_and_return_conditional_losses_1412982
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ж
Ь
)__inference_Gen_BN_9_layer_call_fn_146162

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_9_layer_call_and_return_conditional_losses_1416572
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
’
c
E__inference_Gen_SPD_3_layer_call_and_return_conditional_losses_140916

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

IdentityМ

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ш
э
D__inference_Gen_BN_8_layer_call_and_return_conditional_losses_141459

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1б
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
А
°
D__inference_Gen_BN_2_layer_call_and_return_conditional_losses_144948

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ё
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€0(А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Х
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€0(А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:€€€€€€€€€0(А
 
_user_specified_nameinputs
Э
d
E__inference_Gen_SPD_2_layer_call_and_return_conditional_losses_140722

inputs
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
 *  †?2
dropout/ConstЦ
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2
dropout/MulВ
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1В
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ц
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape‘
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/yѕ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/GreaterEqualР
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/CastЭ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2
dropout/Mul_1И
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ѓ
c
*__inference_Gen_SPD_5_layer_call_fn_146068

inputs
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_5_layer_call_and_return_conditional_losses_1415252
StatefulPartitionedCall±
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ћ	
Ѓ
F__inference_Gen_Conv_4_layer_call_and_return_conditional_losses_142262

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2	
BiasAddГ
leaky_re_lu_3/LeakyRelu	LeakyReluBiasAdd:output:0*0
_output_shapes
:€€€€€€€€€А2
leaky_re_lu_3/LeakyReluВ
IdentityIdentity%leaky_re_lu_3/LeakyRelu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€А:::X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ш
э
D__inference_Gen_BN_7_layer_call_and_return_conditional_losses_141298

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1б
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
ѕ
э
D__inference_Gen_BN_2_layer_call_and_return_conditional_losses_144966

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ѕ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€0(А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€0(А:::::X T
0
_output_shapes
:€€€€€€€€€0(А
 
_user_specified_nameinputs
Ђ
e
I__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_146297

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
ѕ
э
D__inference_Gen_BN_2_layer_call_and_return_conditional_losses_142075

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ѕ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€0(А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€0(А:::::X T
0
_output_shapes
:€€€€€€€€€0(А
 
_user_specified_nameinputs
А
°
D__inference_Gen_BN_4_layer_call_and_return_conditional_losses_145384

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ё
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€
А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Х
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€
А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:€€€€€€€€€
А
 
_user_specified_nameinputs
ъ
`
D__inference_Gen_MP_3_layer_call_and_return_conditional_losses_140441

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
†%
Ї
H__inference_Gen_Conv_T_4_layer_call_and_return_conditional_losses_141585

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
strided_slice_2/stack_2м
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
stack/3В
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
strided_slice_3/stack_2м
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3і
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@А*
dtype02!
conv2d_transpose/ReadVariableOpр
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp§
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2	
BiasAddК
leaky_re_lu_8/PartitionedCallPartitionedCallBiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_1415762
leaky_re_lu_8/PartitionedCallФ
IdentityIdentity&leaky_re_lu_8/PartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ш
э
D__inference_Gen_BN_6_layer_call_and_return_conditional_losses_145766

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1б
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
ї
F
*__inference_Gen_SPD_3_layer_call_fn_145690

inputs
identityѕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_3_layer_call_and_return_conditional_losses_1425192
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
§
F
*__inference_Gen_SPD_2_layer_call_fn_145504

inputs
identityй
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_2_layer_call_and_return_conditional_losses_1407322
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
И
Ь
)__inference_Gen_BN_9_layer_call_fn_146175

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_9_layer_call_and_return_conditional_losses_1416882
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
»	
Ѓ
F__inference_Gen_Conv_2_layer_call_and_return_conditional_losses_142021

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€`PА*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€`PА2	
BiasAddГ
leaky_re_lu_1/LeakyRelu	LeakyReluBiasAdd:output:0*0
_output_shapes
:€€€€€€€€€`PА2
leaky_re_lu_1/LeakyReluВ
IdentityIdentity%leaky_re_lu_1/LeakyRelu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€`PА2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€`P@:::W S
/
_output_shapes
:€€€€€€€€€`P@
 
_user_specified_nameinputs
Љ
°
D__inference_Gen_BN_1_layer_call_and_return_conditional_losses_144864

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¶
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
»
°
D__inference_Gen_BN_4_layer_call_and_return_conditional_losses_145320

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1п
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1І
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
ѕ
э
D__inference_Gen_BN_4_layer_call_and_return_conditional_losses_145402

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ѕ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€
А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€
А:::::X T
0
_output_shapes
:€€€€€€€€€
А
 
_user_specified_nameinputs
ъ
`
D__inference_Gen_MP_5_layer_call_and_return_conditional_losses_140741

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
М
Ь
)__inference_Gen_BN_3_layer_call_fn_145216

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_3_layer_call_and_return_conditional_losses_1405402
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
«
c
*__inference_Gen_SPD_4_layer_call_fn_145876

inputs
identityИҐStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_4_layer_call_and_return_conditional_losses_1426092
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€
А22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€
А
 
_user_specified_nameinputs
§
F
*__inference_Gen_SPD_5_layer_call_fn_146073

inputs
identityй
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_5_layer_call_and_return_conditional_losses_1415352
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ьэ
У 
"__inference__traced_restore_146707
file_prefix&
"assignvariableop_gen_conv_1_kernel&
"assignvariableop_1_gen_conv_1_bias%
!assignvariableop_2_gen_bn_1_gamma$
 assignvariableop_3_gen_bn_1_beta+
'assignvariableop_4_gen_bn_1_moving_mean/
+assignvariableop_5_gen_bn_1_moving_variance(
$assignvariableop_6_gen_conv_2_kernel&
"assignvariableop_7_gen_conv_2_bias%
!assignvariableop_8_gen_bn_2_gamma$
 assignvariableop_9_gen_bn_2_beta,
(assignvariableop_10_gen_bn_2_moving_mean0
,assignvariableop_11_gen_bn_2_moving_variance)
%assignvariableop_12_gen_conv_3_kernel'
#assignvariableop_13_gen_conv_3_bias&
"assignvariableop_14_gen_bn_3_gamma%
!assignvariableop_15_gen_bn_3_beta,
(assignvariableop_16_gen_bn_3_moving_mean0
,assignvariableop_17_gen_bn_3_moving_variance)
%assignvariableop_18_gen_conv_4_kernel'
#assignvariableop_19_gen_conv_4_bias&
"assignvariableop_20_gen_bn_4_gamma%
!assignvariableop_21_gen_bn_4_beta,
(assignvariableop_22_gen_bn_4_moving_mean0
,assignvariableop_23_gen_bn_4_moving_variance)
%assignvariableop_24_gen_conv_5_kernel'
#assignvariableop_25_gen_conv_5_bias&
"assignvariableop_26_gen_bn_5_gamma%
!assignvariableop_27_gen_bn_5_beta,
(assignvariableop_28_gen_bn_5_moving_mean0
,assignvariableop_29_gen_bn_5_moving_variance+
'assignvariableop_30_gen_conv_t_1_kernel)
%assignvariableop_31_gen_conv_t_1_bias&
"assignvariableop_32_gen_bn_6_gamma%
!assignvariableop_33_gen_bn_6_beta,
(assignvariableop_34_gen_bn_6_moving_mean0
,assignvariableop_35_gen_bn_6_moving_variance+
'assignvariableop_36_gen_conv_t_2_kernel)
%assignvariableop_37_gen_conv_t_2_bias&
"assignvariableop_38_gen_bn_7_gamma%
!assignvariableop_39_gen_bn_7_beta,
(assignvariableop_40_gen_bn_7_moving_mean0
,assignvariableop_41_gen_bn_7_moving_variance+
'assignvariableop_42_gen_conv_t_3_kernel)
%assignvariableop_43_gen_conv_t_3_bias&
"assignvariableop_44_gen_bn_8_gamma%
!assignvariableop_45_gen_bn_8_beta,
(assignvariableop_46_gen_bn_8_moving_mean0
,assignvariableop_47_gen_bn_8_moving_variance+
'assignvariableop_48_gen_conv_t_4_kernel)
%assignvariableop_49_gen_conv_t_4_bias&
"assignvariableop_50_gen_bn_9_gamma%
!assignvariableop_51_gen_bn_9_beta,
(assignvariableop_52_gen_bn_9_moving_mean0
,assignvariableop_53_gen_bn_9_moving_variance+
'assignvariableop_54_gen_conv_t_5_kernel)
%assignvariableop_55_gen_conv_t_5_bias'
#assignvariableop_56_gen_bn_10_gamma&
"assignvariableop_57_gen_bn_10_beta-
)assignvariableop_58_gen_bn_10_moving_mean1
-assignvariableop_59_gen_bn_10_moving_variance+
'assignvariableop_60_gen_conv_t_6_kernel)
%assignvariableop_61_gen_conv_t_6_bias
identity_63ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9ѓ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*ї
value±BЃ?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesП
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*У
valueЙBЖ?B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesй
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Т
_output_shapes€
ь:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*M
dtypesC
A2?2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity°
AssignVariableOpAssignVariableOp"assignvariableop_gen_conv_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1І
AssignVariableOp_1AssignVariableOp"assignvariableop_1_gen_conv_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¶
AssignVariableOp_2AssignVariableOp!assignvariableop_2_gen_bn_1_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3•
AssignVariableOp_3AssignVariableOp assignvariableop_3_gen_bn_1_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ђ
AssignVariableOp_4AssignVariableOp'assignvariableop_4_gen_bn_1_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5∞
AssignVariableOp_5AssignVariableOp+assignvariableop_5_gen_bn_1_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6©
AssignVariableOp_6AssignVariableOp$assignvariableop_6_gen_conv_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7І
AssignVariableOp_7AssignVariableOp"assignvariableop_7_gen_conv_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¶
AssignVariableOp_8AssignVariableOp!assignvariableop_8_gen_bn_2_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9•
AssignVariableOp_9AssignVariableOp assignvariableop_9_gen_bn_2_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10∞
AssignVariableOp_10AssignVariableOp(assignvariableop_10_gen_bn_2_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11і
AssignVariableOp_11AssignVariableOp,assignvariableop_11_gen_bn_2_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12≠
AssignVariableOp_12AssignVariableOp%assignvariableop_12_gen_conv_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ђ
AssignVariableOp_13AssignVariableOp#assignvariableop_13_gen_conv_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14™
AssignVariableOp_14AssignVariableOp"assignvariableop_14_gen_bn_3_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15©
AssignVariableOp_15AssignVariableOp!assignvariableop_15_gen_bn_3_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16∞
AssignVariableOp_16AssignVariableOp(assignvariableop_16_gen_bn_3_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17і
AssignVariableOp_17AssignVariableOp,assignvariableop_17_gen_bn_3_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18≠
AssignVariableOp_18AssignVariableOp%assignvariableop_18_gen_conv_4_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ђ
AssignVariableOp_19AssignVariableOp#assignvariableop_19_gen_conv_4_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20™
AssignVariableOp_20AssignVariableOp"assignvariableop_20_gen_bn_4_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21©
AssignVariableOp_21AssignVariableOp!assignvariableop_21_gen_bn_4_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22∞
AssignVariableOp_22AssignVariableOp(assignvariableop_22_gen_bn_4_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23і
AssignVariableOp_23AssignVariableOp,assignvariableop_23_gen_bn_4_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24≠
AssignVariableOp_24AssignVariableOp%assignvariableop_24_gen_conv_5_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ђ
AssignVariableOp_25AssignVariableOp#assignvariableop_25_gen_conv_5_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26™
AssignVariableOp_26AssignVariableOp"assignvariableop_26_gen_bn_5_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27©
AssignVariableOp_27AssignVariableOp!assignvariableop_27_gen_bn_5_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28∞
AssignVariableOp_28AssignVariableOp(assignvariableop_28_gen_bn_5_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29і
AssignVariableOp_29AssignVariableOp,assignvariableop_29_gen_bn_5_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30ѓ
AssignVariableOp_30AssignVariableOp'assignvariableop_30_gen_conv_t_1_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31≠
AssignVariableOp_31AssignVariableOp%assignvariableop_31_gen_conv_t_1_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32™
AssignVariableOp_32AssignVariableOp"assignvariableop_32_gen_bn_6_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33©
AssignVariableOp_33AssignVariableOp!assignvariableop_33_gen_bn_6_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34∞
AssignVariableOp_34AssignVariableOp(assignvariableop_34_gen_bn_6_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35і
AssignVariableOp_35AssignVariableOp,assignvariableop_35_gen_bn_6_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36ѓ
AssignVariableOp_36AssignVariableOp'assignvariableop_36_gen_conv_t_2_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37≠
AssignVariableOp_37AssignVariableOp%assignvariableop_37_gen_conv_t_2_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38™
AssignVariableOp_38AssignVariableOp"assignvariableop_38_gen_bn_7_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39©
AssignVariableOp_39AssignVariableOp!assignvariableop_39_gen_bn_7_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40∞
AssignVariableOp_40AssignVariableOp(assignvariableop_40_gen_bn_7_moving_meanIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41і
AssignVariableOp_41AssignVariableOp,assignvariableop_41_gen_bn_7_moving_varianceIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42ѓ
AssignVariableOp_42AssignVariableOp'assignvariableop_42_gen_conv_t_3_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43≠
AssignVariableOp_43AssignVariableOp%assignvariableop_43_gen_conv_t_3_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44™
AssignVariableOp_44AssignVariableOp"assignvariableop_44_gen_bn_8_gammaIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45©
AssignVariableOp_45AssignVariableOp!assignvariableop_45_gen_bn_8_betaIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46∞
AssignVariableOp_46AssignVariableOp(assignvariableop_46_gen_bn_8_moving_meanIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47і
AssignVariableOp_47AssignVariableOp,assignvariableop_47_gen_bn_8_moving_varianceIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48ѓ
AssignVariableOp_48AssignVariableOp'assignvariableop_48_gen_conv_t_4_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49≠
AssignVariableOp_49AssignVariableOp%assignvariableop_49_gen_conv_t_4_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50™
AssignVariableOp_50AssignVariableOp"assignvariableop_50_gen_bn_9_gammaIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51©
AssignVariableOp_51AssignVariableOp!assignvariableop_51_gen_bn_9_betaIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52∞
AssignVariableOp_52AssignVariableOp(assignvariableop_52_gen_bn_9_moving_meanIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53і
AssignVariableOp_53AssignVariableOp,assignvariableop_53_gen_bn_9_moving_varianceIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54ѓ
AssignVariableOp_54AssignVariableOp'assignvariableop_54_gen_conv_t_5_kernelIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55≠
AssignVariableOp_55AssignVariableOp%assignvariableop_55_gen_conv_t_5_biasIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56Ђ
AssignVariableOp_56AssignVariableOp#assignvariableop_56_gen_bn_10_gammaIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57™
AssignVariableOp_57AssignVariableOp"assignvariableop_57_gen_bn_10_betaIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58±
AssignVariableOp_58AssignVariableOp)assignvariableop_58_gen_bn_10_moving_meanIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59µ
AssignVariableOp_59AssignVariableOp-assignvariableop_59_gen_bn_10_moving_varianceIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60ѓ
AssignVariableOp_60AssignVariableOp'assignvariableop_60_gen_conv_t_6_kernelIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61≠
AssignVariableOp_61AssignVariableOp%assignvariableop_61_gen_conv_t_6_biasIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_619
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp≤
Identity_62Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_62•
Identity_63IdentityIdentity_62:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_63"#
identity_63Identity_63:output:0*П
_input_shapesэ
ъ: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_61AssignVariableOp_612(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
’
c
E__inference_Gen_SPD_3_layer_call_and_return_conditional_losses_145718

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

IdentityМ

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
€
ґ
*__inference_Generator_layer_call_fn_144631

inputs
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

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60
identityИҐStatefulPartitionedCallљ	
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*L
_read_only_resource_inputs.
,*	
 !"%&'(+,-.1234789:=>*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Generator_layer_call_and_return_conditional_losses_1432082
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*™
_input_shapesШ
Х:€€€€€€€€€ј†::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€ј†
 
_user_specified_nameinputs
Щ
d
E__inference_Gen_SPD_1_layer_call_and_return_conditional_losses_145079

inputs
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
 *  †?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2
dropout/MulВ
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1В
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ц
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape‘
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/yѕ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/GreaterEqualР
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€0(А:X T
0
_output_shapes
:€€€€€€€€€0(А
 
_user_specified_nameinputs
»
°
D__inference_Gen_BN_6_layer_call_and_return_conditional_losses_141038

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1п
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1І
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Щ
d
E__inference_Gen_SPD_3_layer_call_and_return_conditional_losses_142514

inputs
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
 *  †?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout/MulВ
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1В
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ц
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape‘
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/yѕ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/GreaterEqualР
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ґ
E
)__inference_Gen_MP_3_layer_call_fn_140447

inputs
identityи
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_MP_3_layer_call_and_return_conditional_losses_1404412
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Э
d
E__inference_Gen_SPD_3_layer_call_and_return_conditional_losses_145713

inputs
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
 *  †?2
dropout/ConstЦ
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2
dropout/MulВ
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1В
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ц
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape‘
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/yѕ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/GreaterEqualР
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/CastЭ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2
dropout/Mul_1И
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
м
c
E__inference_Gen_SPD_4_layer_call_and_return_conditional_losses_145871

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:€€€€€€€€€
А:X T
0
_output_shapes
:€€€€€€€€€
А
 
_user_specified_nameinputs
ї
F
*__inference_Gen_SPD_1_layer_call_fn_145094

inputs
identityѕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_1_layer_call_and_return_conditional_losses_1421382
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€0(А:X T
0
_output_shapes
:€€€€€€€€€0(А
 
_user_specified_nameinputs
А
°
D__inference_Gen_BN_2_layer_call_and_return_conditional_losses_142057

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ё
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€0(А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Х
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€0(А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:€€€€€€€€€0(А
 
_user_specified_nameinputs
»
°
D__inference_Gen_BN_5_layer_call_and_return_conditional_losses_140809

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1п
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1І
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
’
c
E__inference_Gen_SPD_1_layer_call_and_return_conditional_losses_140432

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

IdentityМ

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Љ
°
D__inference_Gen_BN_1_layer_call_and_return_conditional_losses_140209

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¶
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Н
ю
E__inference_Gen_BN_10_layer_call_and_return_conditional_losses_146226

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
ѕ
э
D__inference_Gen_BN_4_layer_call_and_return_conditional_losses_142316

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ѕ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€
А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€
А:::::X T
0
_output_shapes
:€€€€€€€€€
А
 
_user_specified_nameinputs
Ь
є
*__inference_Generator_layer_call_fn_143627
	gen_input
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

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60
identityИҐStatefulPartitionedCall‘	
StatefulPartitionedCallStatefulPartitionedCall	gen_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Generator_layer_call_and_return_conditional_losses_1435002
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*™
_input_shapesШ
Х:€€€€€€€€€ј†::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
1
_output_shapes
:€€€€€€€€€ј†
#
_user_specified_name	Gen_Input
ƒ
Ь
)__inference_Gen_BN_2_layer_call_fn_144992

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_2_layer_call_and_return_conditional_losses_1420752
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€0(А::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€0(А
 
_user_specified_nameinputs
Щ
d
E__inference_Gen_SPD_3_layer_call_and_return_conditional_losses_145675

inputs
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
 *  †?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout/MulВ
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1В
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ц
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape‘
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/yѕ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/GreaterEqualР
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Э
d
E__inference_Gen_SPD_1_layer_call_and_return_conditional_losses_145117

inputs
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
 *  †?2
dropout/ConstЦ
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2
dropout/MulВ
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1В
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ц
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape‘
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/yѕ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/GreaterEqualР
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/CastЭ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2
dropout/Mul_1И
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ъ
`
D__inference_Gen_MP_2_layer_call_and_return_conditional_losses_140257

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
У
ґ
*__inference_Generator_layer_call_fn_144760

inputs
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

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60
identityИҐStatefulPartitionedCall—	
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Generator_layer_call_and_return_conditional_losses_1435002
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*™
_input_shapesШ
Х:€€€€€€€€€ј†::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€ј†
 
_user_specified_nameinputs
Щ
d
E__inference_Gen_SPD_4_layer_call_and_return_conditional_losses_145866

inputs
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
 *  †?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€
А2
dropout/MulВ
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1В
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ц
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape‘
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/yѕ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/GreaterEqualР
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€
А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€
А:X T
0
_output_shapes
:€€€€€€€€€
А
 
_user_specified_nameinputs
»
°
D__inference_Gen_BN_8_layer_call_and_return_conditional_losses_145978

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1п
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1І
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
М
А
+__inference_Gen_Conv_1_layer_call_fn_144780

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ј†@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_Gen_Conv_1_layer_call_and_return_conditional_losses_1419202
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€ј†@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€ј†::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€ј†
 
_user_specified_nameinputs
Ћ	
Ѓ
F__inference_Gen_Conv_5_layer_call_and_return_conditional_losses_145515

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€
А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€
А2	
BiasAddГ
leaky_re_lu_4/LeakyRelu	LeakyReluBiasAdd:output:0*0
_output_shapes
:€€€€€€€€€
А2
leaky_re_lu_4/LeakyReluВ
IdentityIdentity%leaky_re_lu_4/LeakyRelu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€
А:::X T
0
_output_shapes
:€€€€€€€€€
А
 
_user_specified_nameinputs
–
В
-__inference_Gen_Conv_T_6_layer_call_fn_141905

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_6_layer_call_and_return_conditional_losses_1418952
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
М
э
D__inference_Gen_BN_9_layer_call_and_return_conditional_losses_141688

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ш
э
D__inference_Gen_BN_2_layer_call_and_return_conditional_losses_140356

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1б
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
»
°
D__inference_Gen_BN_3_layer_call_and_return_conditional_losses_140509

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1п
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1І
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ђ
e
I__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_141737

inputs
identityn
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2
	LeakyReluЕ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
»
°
D__inference_Gen_BN_7_layer_call_and_return_conditional_losses_141267

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1п
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3€
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1І
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ћ	
Ѓ
F__inference_Gen_Conv_3_layer_call_and_return_conditional_losses_145143

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€0(А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2	
BiasAddГ
leaky_re_lu_2/LeakyRelu	LeakyReluBiasAdd:output:0*0
_output_shapes
:€€€€€€€€€0(А2
leaky_re_lu_2/LeakyReluВ
IdentityIdentity%leaky_re_lu_2/LeakyRelu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€0(А:::X T
0
_output_shapes
:€€€€€€€€€0(А
 
_user_specified_nameinputs
М
Ь
)__inference_Gen_BN_6_layer_call_fn_145792

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_6_layer_call_and_return_conditional_losses_1410692
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
ѕ
э
D__inference_Gen_BN_5_layer_call_and_return_conditional_losses_142456

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ѕ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€А:::::X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
§
F
*__inference_Gen_SPD_1_layer_call_fn_145132

inputs
identityй
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_1_layer_call_and_return_conditional_losses_1404322
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ш
э
D__inference_Gen_BN_7_layer_call_and_return_conditional_losses_145919

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1б
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
М
Ь
)__inference_Gen_BN_4_layer_call_fn_145364

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_4_layer_call_and_return_conditional_losses_1406562
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
кР
џ
E__inference_Generator_layer_call_and_return_conditional_losses_144185

inputs-
)gen_conv_1_conv2d_readvariableop_resource.
*gen_conv_1_biasadd_readvariableop_resource$
 gen_bn_1_readvariableop_resource&
"gen_bn_1_readvariableop_1_resource5
1gen_bn_1_fusedbatchnormv3_readvariableop_resource7
3gen_bn_1_fusedbatchnormv3_readvariableop_1_resource-
)gen_conv_2_conv2d_readvariableop_resource.
*gen_conv_2_biasadd_readvariableop_resource$
 gen_bn_2_readvariableop_resource&
"gen_bn_2_readvariableop_1_resource5
1gen_bn_2_fusedbatchnormv3_readvariableop_resource7
3gen_bn_2_fusedbatchnormv3_readvariableop_1_resource-
)gen_conv_3_conv2d_readvariableop_resource.
*gen_conv_3_biasadd_readvariableop_resource$
 gen_bn_3_readvariableop_resource&
"gen_bn_3_readvariableop_1_resource5
1gen_bn_3_fusedbatchnormv3_readvariableop_resource7
3gen_bn_3_fusedbatchnormv3_readvariableop_1_resource-
)gen_conv_4_conv2d_readvariableop_resource.
*gen_conv_4_biasadd_readvariableop_resource$
 gen_bn_4_readvariableop_resource&
"gen_bn_4_readvariableop_1_resource5
1gen_bn_4_fusedbatchnormv3_readvariableop_resource7
3gen_bn_4_fusedbatchnormv3_readvariableop_1_resource-
)gen_conv_5_conv2d_readvariableop_resource.
*gen_conv_5_biasadd_readvariableop_resource$
 gen_bn_5_readvariableop_resource&
"gen_bn_5_readvariableop_1_resource5
1gen_bn_5_fusedbatchnormv3_readvariableop_resource7
3gen_bn_5_fusedbatchnormv3_readvariableop_1_resource9
5gen_conv_t_1_conv2d_transpose_readvariableop_resource0
,gen_conv_t_1_biasadd_readvariableop_resource$
 gen_bn_6_readvariableop_resource&
"gen_bn_6_readvariableop_1_resource5
1gen_bn_6_fusedbatchnormv3_readvariableop_resource7
3gen_bn_6_fusedbatchnormv3_readvariableop_1_resource9
5gen_conv_t_2_conv2d_transpose_readvariableop_resource0
,gen_conv_t_2_biasadd_readvariableop_resource$
 gen_bn_7_readvariableop_resource&
"gen_bn_7_readvariableop_1_resource5
1gen_bn_7_fusedbatchnormv3_readvariableop_resource7
3gen_bn_7_fusedbatchnormv3_readvariableop_1_resource9
5gen_conv_t_3_conv2d_transpose_readvariableop_resource0
,gen_conv_t_3_biasadd_readvariableop_resource$
 gen_bn_8_readvariableop_resource&
"gen_bn_8_readvariableop_1_resource5
1gen_bn_8_fusedbatchnormv3_readvariableop_resource7
3gen_bn_8_fusedbatchnormv3_readvariableop_1_resource9
5gen_conv_t_4_conv2d_transpose_readvariableop_resource0
,gen_conv_t_4_biasadd_readvariableop_resource$
 gen_bn_9_readvariableop_resource&
"gen_bn_9_readvariableop_1_resource5
1gen_bn_9_fusedbatchnormv3_readvariableop_resource7
3gen_bn_9_fusedbatchnormv3_readvariableop_1_resource9
5gen_conv_t_5_conv2d_transpose_readvariableop_resource0
,gen_conv_t_5_biasadd_readvariableop_resource%
!gen_bn_10_readvariableop_resource'
#gen_bn_10_readvariableop_1_resource6
2gen_bn_10_fusedbatchnormv3_readvariableop_resource8
4gen_bn_10_fusedbatchnormv3_readvariableop_1_resource9
5gen_conv_t_6_conv2d_transpose_readvariableop_resource0
,gen_conv_t_6_biasadd_readvariableop_resource
identityИҐGen_BN_1/AssignNewValueҐGen_BN_1/AssignNewValue_1ҐGen_BN_10/AssignNewValueҐGen_BN_10/AssignNewValue_1ҐGen_BN_2/AssignNewValueҐGen_BN_2/AssignNewValue_1ҐGen_BN_3/AssignNewValueҐGen_BN_3/AssignNewValue_1ҐGen_BN_4/AssignNewValueҐGen_BN_4/AssignNewValue_1ҐGen_BN_5/AssignNewValueҐGen_BN_5/AssignNewValue_1ҐGen_BN_6/AssignNewValueҐGen_BN_6/AssignNewValue_1ҐGen_BN_7/AssignNewValueҐGen_BN_7/AssignNewValue_1ҐGen_BN_8/AssignNewValueҐGen_BN_8/AssignNewValue_1ҐGen_BN_9/AssignNewValueҐGen_BN_9/AssignNewValue_1ґ
 Gen_Conv_1/Conv2D/ReadVariableOpReadVariableOp)gen_conv_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02"
 Gen_Conv_1/Conv2D/ReadVariableOp∆
Gen_Conv_1/Conv2DConv2Dinputs(Gen_Conv_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ј†@*
paddingSAME*
strides
2
Gen_Conv_1/Conv2D≠
!Gen_Conv_1/BiasAdd/ReadVariableOpReadVariableOp*gen_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!Gen_Conv_1/BiasAdd/ReadVariableOpґ
Gen_Conv_1/BiasAddBiasAddGen_Conv_1/Conv2D:output:0)Gen_Conv_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ј†@2
Gen_Conv_1/BiasAdd°
 Gen_Conv_1/leaky_re_lu/LeakyRelu	LeakyReluGen_Conv_1/BiasAdd:output:0*1
_output_shapes
:€€€€€€€€€ј†@2"
 Gen_Conv_1/leaky_re_lu/LeakyReluћ
Gen_MP_1/MaxPoolMaxPool.Gen_Conv_1/leaky_re_lu/LeakyRelu:activations:0*/
_output_shapes
:€€€€€€€€€`P@*
ksize
*
paddingVALID*
strides
2
Gen_MP_1/MaxPoolП
Gen_BN_1/ReadVariableOpReadVariableOp gen_bn_1_readvariableop_resource*
_output_shapes
:@*
dtype02
Gen_BN_1/ReadVariableOpХ
Gen_BN_1/ReadVariableOp_1ReadVariableOp"gen_bn_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02
Gen_BN_1/ReadVariableOp_1¬
(Gen_BN_1/FusedBatchNormV3/ReadVariableOpReadVariableOp1gen_bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02*
(Gen_BN_1/FusedBatchNormV3/ReadVariableOp»
*Gen_BN_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3gen_bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02,
*Gen_BN_1/FusedBatchNormV3/ReadVariableOp_1°
Gen_BN_1/FusedBatchNormV3FusedBatchNormV3Gen_MP_1/MaxPool:output:0Gen_BN_1/ReadVariableOp:value:0!Gen_BN_1/ReadVariableOp_1:value:00Gen_BN_1/FusedBatchNormV3/ReadVariableOp:value:02Gen_BN_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€`P@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
Gen_BN_1/FusedBatchNormV3µ
Gen_BN_1/AssignNewValueAssignVariableOp1gen_bn_1_fusedbatchnormv3_readvariableop_resource&Gen_BN_1/FusedBatchNormV3:batch_mean:0)^Gen_BN_1/FusedBatchNormV3/ReadVariableOp*D
_class:
86loc:@Gen_BN_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
Gen_BN_1/AssignNewValue√
Gen_BN_1/AssignNewValue_1AssignVariableOp3gen_bn_1_fusedbatchnormv3_readvariableop_1_resource*Gen_BN_1/FusedBatchNormV3:batch_variance:0+^Gen_BN_1/FusedBatchNormV3/ReadVariableOp_1*F
_class<
:8loc:@Gen_BN_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
Gen_BN_1/AssignNewValue_1Ј
 Gen_Conv_2/Conv2D/ReadVariableOpReadVariableOp)gen_conv_2_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02"
 Gen_Conv_2/Conv2D/ReadVariableOp№
Gen_Conv_2/Conv2DConv2DGen_BN_1/FusedBatchNormV3:y:0(Gen_Conv_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€`PА*
paddingSAME*
strides
2
Gen_Conv_2/Conv2DЃ
!Gen_Conv_2/BiasAdd/ReadVariableOpReadVariableOp*gen_conv_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02#
!Gen_Conv_2/BiasAdd/ReadVariableOpµ
Gen_Conv_2/BiasAddBiasAddGen_Conv_2/Conv2D:output:0)Gen_Conv_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€`PА2
Gen_Conv_2/BiasAdd§
"Gen_Conv_2/leaky_re_lu_1/LeakyRelu	LeakyReluGen_Conv_2/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€`PА2$
"Gen_Conv_2/leaky_re_lu_1/LeakyReluѕ
Gen_MP_2/MaxPoolMaxPool0Gen_Conv_2/leaky_re_lu_1/LeakyRelu:activations:0*0
_output_shapes
:€€€€€€€€€0(А*
ksize
*
paddingVALID*
strides
2
Gen_MP_2/MaxPoolР
Gen_BN_2/ReadVariableOpReadVariableOp gen_bn_2_readvariableop_resource*
_output_shapes	
:А*
dtype02
Gen_BN_2/ReadVariableOpЦ
Gen_BN_2/ReadVariableOp_1ReadVariableOp"gen_bn_2_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
Gen_BN_2/ReadVariableOp_1√
(Gen_BN_2/FusedBatchNormV3/ReadVariableOpReadVariableOp1gen_bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(Gen_BN_2/FusedBatchNormV3/ReadVariableOp…
*Gen_BN_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3gen_bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02,
*Gen_BN_2/FusedBatchNormV3/ReadVariableOp_1¶
Gen_BN_2/FusedBatchNormV3FusedBatchNormV3Gen_MP_2/MaxPool:output:0Gen_BN_2/ReadVariableOp:value:0!Gen_BN_2/ReadVariableOp_1:value:00Gen_BN_2/FusedBatchNormV3/ReadVariableOp:value:02Gen_BN_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€0(А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
Gen_BN_2/FusedBatchNormV3µ
Gen_BN_2/AssignNewValueAssignVariableOp1gen_bn_2_fusedbatchnormv3_readvariableop_resource&Gen_BN_2/FusedBatchNormV3:batch_mean:0)^Gen_BN_2/FusedBatchNormV3/ReadVariableOp*D
_class:
86loc:@Gen_BN_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
Gen_BN_2/AssignNewValue√
Gen_BN_2/AssignNewValue_1AssignVariableOp3gen_bn_2_fusedbatchnormv3_readvariableop_1_resource*Gen_BN_2/FusedBatchNormV3:batch_variance:0+^Gen_BN_2/FusedBatchNormV3/ReadVariableOp_1*F
_class<
:8loc:@Gen_BN_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
Gen_BN_2/AssignNewValue_1o
Gen_SPD_1/ShapeShapeGen_BN_2/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Gen_SPD_1/ShapeИ
Gen_SPD_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Gen_SPD_1/strided_slice/stackМ
Gen_SPD_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
Gen_SPD_1/strided_slice/stack_1М
Gen_SPD_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
Gen_SPD_1/strided_slice/stack_2Ю
Gen_SPD_1/strided_sliceStridedSliceGen_SPD_1/Shape:output:0&Gen_SPD_1/strided_slice/stack:output:0(Gen_SPD_1/strided_slice/stack_1:output:0(Gen_SPD_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_SPD_1/strided_sliceМ
Gen_SPD_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2!
Gen_SPD_1/strided_slice_1/stackР
!Gen_SPD_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!Gen_SPD_1/strided_slice_1/stack_1Р
!Gen_SPD_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!Gen_SPD_1/strided_slice_1/stack_2®
Gen_SPD_1/strided_slice_1StridedSliceGen_SPD_1/Shape:output:0(Gen_SPD_1/strided_slice_1/stack:output:0*Gen_SPD_1/strided_slice_1/stack_1:output:0*Gen_SPD_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_SPD_1/strided_slice_1w
Gen_SPD_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
Gen_SPD_1/dropout/Const±
Gen_SPD_1/dropout/MulMulGen_BN_2/FusedBatchNormV3:y:0 Gen_SPD_1/dropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2
Gen_SPD_1/dropout/MulЦ
(Gen_SPD_1/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(Gen_SPD_1/dropout/random_uniform/shape/1Ц
(Gen_SPD_1/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(Gen_SPD_1/dropout/random_uniform/shape/2≤
&Gen_SPD_1/dropout/random_uniform/shapePack Gen_SPD_1/strided_slice:output:01Gen_SPD_1/dropout/random_uniform/shape/1:output:01Gen_SPD_1/dropout/random_uniform/shape/2:output:0"Gen_SPD_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2(
&Gen_SPD_1/dropout/random_uniform/shapeт
.Gen_SPD_1/dropout/random_uniform/RandomUniformRandomUniform/Gen_SPD_1/dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
dtype020
.Gen_SPD_1/dropout/random_uniform/RandomUniformЙ
 Gen_SPD_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2"
 Gen_SPD_1/dropout/GreaterEqual/yч
Gen_SPD_1/dropout/GreaterEqualGreaterEqual7Gen_SPD_1/dropout/random_uniform/RandomUniform:output:0)Gen_SPD_1/dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2 
Gen_SPD_1/dropout/GreaterEqualЃ
Gen_SPD_1/dropout/CastCast"Gen_SPD_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
Gen_SPD_1/dropout/CastЂ
Gen_SPD_1/dropout/Mul_1MulGen_SPD_1/dropout/Mul:z:0Gen_SPD_1/dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2
Gen_SPD_1/dropout/Mul_1Є
 Gen_Conv_3/Conv2D/ReadVariableOpReadVariableOp)gen_conv_3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02"
 Gen_Conv_3/Conv2D/ReadVariableOpЏ
Gen_Conv_3/Conv2DConv2DGen_SPD_1/dropout/Mul_1:z:0(Gen_Conv_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€0(А*
paddingSAME*
strides
2
Gen_Conv_3/Conv2DЃ
!Gen_Conv_3/BiasAdd/ReadVariableOpReadVariableOp*gen_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02#
!Gen_Conv_3/BiasAdd/ReadVariableOpµ
Gen_Conv_3/BiasAddBiasAddGen_Conv_3/Conv2D:output:0)Gen_Conv_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2
Gen_Conv_3/BiasAdd§
"Gen_Conv_3/leaky_re_lu_2/LeakyRelu	LeakyReluGen_Conv_3/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€0(А2$
"Gen_Conv_3/leaky_re_lu_2/LeakyReluѕ
Gen_MP_3/MaxPoolMaxPool0Gen_Conv_3/leaky_re_lu_2/LeakyRelu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
Gen_MP_3/MaxPoolР
Gen_BN_3/ReadVariableOpReadVariableOp gen_bn_3_readvariableop_resource*
_output_shapes	
:А*
dtype02
Gen_BN_3/ReadVariableOpЦ
Gen_BN_3/ReadVariableOp_1ReadVariableOp"gen_bn_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
Gen_BN_3/ReadVariableOp_1√
(Gen_BN_3/FusedBatchNormV3/ReadVariableOpReadVariableOp1gen_bn_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(Gen_BN_3/FusedBatchNormV3/ReadVariableOp…
*Gen_BN_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3gen_bn_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02,
*Gen_BN_3/FusedBatchNormV3/ReadVariableOp_1¶
Gen_BN_3/FusedBatchNormV3FusedBatchNormV3Gen_MP_3/MaxPool:output:0Gen_BN_3/ReadVariableOp:value:0!Gen_BN_3/ReadVariableOp_1:value:00Gen_BN_3/FusedBatchNormV3/ReadVariableOp:value:02Gen_BN_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
Gen_BN_3/FusedBatchNormV3µ
Gen_BN_3/AssignNewValueAssignVariableOp1gen_bn_3_fusedbatchnormv3_readvariableop_resource&Gen_BN_3/FusedBatchNormV3:batch_mean:0)^Gen_BN_3/FusedBatchNormV3/ReadVariableOp*D
_class:
86loc:@Gen_BN_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
Gen_BN_3/AssignNewValue√
Gen_BN_3/AssignNewValue_1AssignVariableOp3gen_bn_3_fusedbatchnormv3_readvariableop_1_resource*Gen_BN_3/FusedBatchNormV3:batch_variance:0+^Gen_BN_3/FusedBatchNormV3/ReadVariableOp_1*F
_class<
:8loc:@Gen_BN_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
Gen_BN_3/AssignNewValue_1Є
 Gen_Conv_4/Conv2D/ReadVariableOpReadVariableOp)gen_conv_4_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02"
 Gen_Conv_4/Conv2D/ReadVariableOp№
Gen_Conv_4/Conv2DConv2DGen_BN_3/FusedBatchNormV3:y:0(Gen_Conv_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Gen_Conv_4/Conv2DЃ
!Gen_Conv_4/BiasAdd/ReadVariableOpReadVariableOp*gen_conv_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02#
!Gen_Conv_4/BiasAdd/ReadVariableOpµ
Gen_Conv_4/BiasAddBiasAddGen_Conv_4/Conv2D:output:0)Gen_Conv_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Gen_Conv_4/BiasAdd§
"Gen_Conv_4/leaky_re_lu_3/LeakyRelu	LeakyReluGen_Conv_4/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€А2$
"Gen_Conv_4/leaky_re_lu_3/LeakyReluѕ
Gen_MP_4/MaxPoolMaxPool0Gen_Conv_4/leaky_re_lu_3/LeakyRelu:activations:0*0
_output_shapes
:€€€€€€€€€
А*
ksize
*
paddingVALID*
strides
2
Gen_MP_4/MaxPoolР
Gen_BN_4/ReadVariableOpReadVariableOp gen_bn_4_readvariableop_resource*
_output_shapes	
:А*
dtype02
Gen_BN_4/ReadVariableOpЦ
Gen_BN_4/ReadVariableOp_1ReadVariableOp"gen_bn_4_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
Gen_BN_4/ReadVariableOp_1√
(Gen_BN_4/FusedBatchNormV3/ReadVariableOpReadVariableOp1gen_bn_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(Gen_BN_4/FusedBatchNormV3/ReadVariableOp…
*Gen_BN_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3gen_bn_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02,
*Gen_BN_4/FusedBatchNormV3/ReadVariableOp_1¶
Gen_BN_4/FusedBatchNormV3FusedBatchNormV3Gen_MP_4/MaxPool:output:0Gen_BN_4/ReadVariableOp:value:0!Gen_BN_4/ReadVariableOp_1:value:00Gen_BN_4/FusedBatchNormV3/ReadVariableOp:value:02Gen_BN_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€
А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
Gen_BN_4/FusedBatchNormV3µ
Gen_BN_4/AssignNewValueAssignVariableOp1gen_bn_4_fusedbatchnormv3_readvariableop_resource&Gen_BN_4/FusedBatchNormV3:batch_mean:0)^Gen_BN_4/FusedBatchNormV3/ReadVariableOp*D
_class:
86loc:@Gen_BN_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
Gen_BN_4/AssignNewValue√
Gen_BN_4/AssignNewValue_1AssignVariableOp3gen_bn_4_fusedbatchnormv3_readvariableop_1_resource*Gen_BN_4/FusedBatchNormV3:batch_variance:0+^Gen_BN_4/FusedBatchNormV3/ReadVariableOp_1*F
_class<
:8loc:@Gen_BN_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
Gen_BN_4/AssignNewValue_1o
Gen_SPD_2/ShapeShapeGen_BN_4/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Gen_SPD_2/ShapeИ
Gen_SPD_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Gen_SPD_2/strided_slice/stackМ
Gen_SPD_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
Gen_SPD_2/strided_slice/stack_1М
Gen_SPD_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
Gen_SPD_2/strided_slice/stack_2Ю
Gen_SPD_2/strided_sliceStridedSliceGen_SPD_2/Shape:output:0&Gen_SPD_2/strided_slice/stack:output:0(Gen_SPD_2/strided_slice/stack_1:output:0(Gen_SPD_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_SPD_2/strided_sliceМ
Gen_SPD_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2!
Gen_SPD_2/strided_slice_1/stackР
!Gen_SPD_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!Gen_SPD_2/strided_slice_1/stack_1Р
!Gen_SPD_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!Gen_SPD_2/strided_slice_1/stack_2®
Gen_SPD_2/strided_slice_1StridedSliceGen_SPD_2/Shape:output:0(Gen_SPD_2/strided_slice_1/stack:output:0*Gen_SPD_2/strided_slice_1/stack_1:output:0*Gen_SPD_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_SPD_2/strided_slice_1w
Gen_SPD_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
Gen_SPD_2/dropout/Const±
Gen_SPD_2/dropout/MulMulGen_BN_4/FusedBatchNormV3:y:0 Gen_SPD_2/dropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€
А2
Gen_SPD_2/dropout/MulЦ
(Gen_SPD_2/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(Gen_SPD_2/dropout/random_uniform/shape/1Ц
(Gen_SPD_2/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(Gen_SPD_2/dropout/random_uniform/shape/2≤
&Gen_SPD_2/dropout/random_uniform/shapePack Gen_SPD_2/strided_slice:output:01Gen_SPD_2/dropout/random_uniform/shape/1:output:01Gen_SPD_2/dropout/random_uniform/shape/2:output:0"Gen_SPD_2/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2(
&Gen_SPD_2/dropout/random_uniform/shapeт
.Gen_SPD_2/dropout/random_uniform/RandomUniformRandomUniform/Gen_SPD_2/dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
dtype020
.Gen_SPD_2/dropout/random_uniform/RandomUniformЙ
 Gen_SPD_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2"
 Gen_SPD_2/dropout/GreaterEqual/yч
Gen_SPD_2/dropout/GreaterEqualGreaterEqual7Gen_SPD_2/dropout/random_uniform/RandomUniform:output:0)Gen_SPD_2/dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2 
Gen_SPD_2/dropout/GreaterEqualЃ
Gen_SPD_2/dropout/CastCast"Gen_SPD_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
Gen_SPD_2/dropout/CastЂ
Gen_SPD_2/dropout/Mul_1MulGen_SPD_2/dropout/Mul:z:0Gen_SPD_2/dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€
А2
Gen_SPD_2/dropout/Mul_1Є
 Gen_Conv_5/Conv2D/ReadVariableOpReadVariableOp)gen_conv_5_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02"
 Gen_Conv_5/Conv2D/ReadVariableOpЏ
Gen_Conv_5/Conv2DConv2DGen_SPD_2/dropout/Mul_1:z:0(Gen_Conv_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€
А*
paddingSAME*
strides
2
Gen_Conv_5/Conv2DЃ
!Gen_Conv_5/BiasAdd/ReadVariableOpReadVariableOp*gen_conv_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02#
!Gen_Conv_5/BiasAdd/ReadVariableOpµ
Gen_Conv_5/BiasAddBiasAddGen_Conv_5/Conv2D:output:0)Gen_Conv_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€
А2
Gen_Conv_5/BiasAdd§
"Gen_Conv_5/leaky_re_lu_4/LeakyRelu	LeakyReluGen_Conv_5/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€
А2$
"Gen_Conv_5/leaky_re_lu_4/LeakyReluѕ
Gen_MP_5/MaxPoolMaxPool0Gen_Conv_5/leaky_re_lu_4/LeakyRelu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
Gen_MP_5/MaxPoolР
Gen_BN_5/ReadVariableOpReadVariableOp gen_bn_5_readvariableop_resource*
_output_shapes	
:А*
dtype02
Gen_BN_5/ReadVariableOpЦ
Gen_BN_5/ReadVariableOp_1ReadVariableOp"gen_bn_5_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
Gen_BN_5/ReadVariableOp_1√
(Gen_BN_5/FusedBatchNormV3/ReadVariableOpReadVariableOp1gen_bn_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(Gen_BN_5/FusedBatchNormV3/ReadVariableOp…
*Gen_BN_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3gen_bn_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02,
*Gen_BN_5/FusedBatchNormV3/ReadVariableOp_1¶
Gen_BN_5/FusedBatchNormV3FusedBatchNormV3Gen_MP_5/MaxPool:output:0Gen_BN_5/ReadVariableOp:value:0!Gen_BN_5/ReadVariableOp_1:value:00Gen_BN_5/FusedBatchNormV3/ReadVariableOp:value:02Gen_BN_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
Gen_BN_5/FusedBatchNormV3µ
Gen_BN_5/AssignNewValueAssignVariableOp1gen_bn_5_fusedbatchnormv3_readvariableop_resource&Gen_BN_5/FusedBatchNormV3:batch_mean:0)^Gen_BN_5/FusedBatchNormV3/ReadVariableOp*D
_class:
86loc:@Gen_BN_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
Gen_BN_5/AssignNewValue√
Gen_BN_5/AssignNewValue_1AssignVariableOp3gen_bn_5_fusedbatchnormv3_readvariableop_1_resource*Gen_BN_5/FusedBatchNormV3:batch_variance:0+^Gen_BN_5/FusedBatchNormV3/ReadVariableOp_1*F
_class<
:8loc:@Gen_BN_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
Gen_BN_5/AssignNewValue_1o
Gen_SPD_3/ShapeShapeGen_BN_5/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Gen_SPD_3/ShapeИ
Gen_SPD_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Gen_SPD_3/strided_slice/stackМ
Gen_SPD_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
Gen_SPD_3/strided_slice/stack_1М
Gen_SPD_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
Gen_SPD_3/strided_slice/stack_2Ю
Gen_SPD_3/strided_sliceStridedSliceGen_SPD_3/Shape:output:0&Gen_SPD_3/strided_slice/stack:output:0(Gen_SPD_3/strided_slice/stack_1:output:0(Gen_SPD_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_SPD_3/strided_sliceМ
Gen_SPD_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2!
Gen_SPD_3/strided_slice_1/stackР
!Gen_SPD_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!Gen_SPD_3/strided_slice_1/stack_1Р
!Gen_SPD_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!Gen_SPD_3/strided_slice_1/stack_2®
Gen_SPD_3/strided_slice_1StridedSliceGen_SPD_3/Shape:output:0(Gen_SPD_3/strided_slice_1/stack:output:0*Gen_SPD_3/strided_slice_1/stack_1:output:0*Gen_SPD_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_SPD_3/strided_slice_1w
Gen_SPD_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
Gen_SPD_3/dropout/Const±
Gen_SPD_3/dropout/MulMulGen_BN_5/FusedBatchNormV3:y:0 Gen_SPD_3/dropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Gen_SPD_3/dropout/MulЦ
(Gen_SPD_3/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(Gen_SPD_3/dropout/random_uniform/shape/1Ц
(Gen_SPD_3/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(Gen_SPD_3/dropout/random_uniform/shape/2≤
&Gen_SPD_3/dropout/random_uniform/shapePack Gen_SPD_3/strided_slice:output:01Gen_SPD_3/dropout/random_uniform/shape/1:output:01Gen_SPD_3/dropout/random_uniform/shape/2:output:0"Gen_SPD_3/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2(
&Gen_SPD_3/dropout/random_uniform/shapeт
.Gen_SPD_3/dropout/random_uniform/RandomUniformRandomUniform/Gen_SPD_3/dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
dtype020
.Gen_SPD_3/dropout/random_uniform/RandomUniformЙ
 Gen_SPD_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2"
 Gen_SPD_3/dropout/GreaterEqual/yч
Gen_SPD_3/dropout/GreaterEqualGreaterEqual7Gen_SPD_3/dropout/random_uniform/RandomUniform:output:0)Gen_SPD_3/dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2 
Gen_SPD_3/dropout/GreaterEqualЃ
Gen_SPD_3/dropout/CastCast"Gen_SPD_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
Gen_SPD_3/dropout/CastЂ
Gen_SPD_3/dropout/Mul_1MulGen_SPD_3/dropout/Mul:z:0Gen_SPD_3/dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Gen_SPD_3/dropout/Mul_1s
Gen_Conv_T_1/ShapeShapeGen_SPD_3/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
Gen_Conv_T_1/ShapeО
 Gen_Conv_T_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 Gen_Conv_T_1/strided_slice/stackТ
"Gen_Conv_T_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"Gen_Conv_T_1/strided_slice/stack_1Т
"Gen_Conv_T_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"Gen_Conv_T_1/strided_slice/stack_2∞
Gen_Conv_T_1/strided_sliceStridedSliceGen_Conv_T_1/Shape:output:0)Gen_Conv_T_1/strided_slice/stack:output:0+Gen_Conv_T_1/strided_slice/stack_1:output:0+Gen_Conv_T_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_Conv_T_1/strided_slicen
Gen_Conv_T_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
Gen_Conv_T_1/stack/1n
Gen_Conv_T_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :
2
Gen_Conv_T_1/stack/2o
Gen_Conv_T_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :А2
Gen_Conv_T_1/stack/3а
Gen_Conv_T_1/stackPack#Gen_Conv_T_1/strided_slice:output:0Gen_Conv_T_1/stack/1:output:0Gen_Conv_T_1/stack/2:output:0Gen_Conv_T_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
Gen_Conv_T_1/stackТ
"Gen_Conv_T_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"Gen_Conv_T_1/strided_slice_1/stackЦ
$Gen_Conv_T_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$Gen_Conv_T_1/strided_slice_1/stack_1Ц
$Gen_Conv_T_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$Gen_Conv_T_1/strided_slice_1/stack_2Ї
Gen_Conv_T_1/strided_slice_1StridedSliceGen_Conv_T_1/stack:output:0+Gen_Conv_T_1/strided_slice_1/stack:output:0-Gen_Conv_T_1/strided_slice_1/stack_1:output:0-Gen_Conv_T_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_Conv_T_1/strided_slice_1№
,Gen_Conv_T_1/conv2d_transpose/ReadVariableOpReadVariableOp5gen_conv_t_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,Gen_Conv_T_1/conv2d_transpose/ReadVariableOp®
Gen_Conv_T_1/conv2d_transposeConv2DBackpropInputGen_Conv_T_1/stack:output:04Gen_Conv_T_1/conv2d_transpose/ReadVariableOp:value:0Gen_SPD_3/dropout/Mul_1:z:0*
T0*0
_output_shapes
:€€€€€€€€€
А*
paddingSAME*
strides
2
Gen_Conv_T_1/conv2d_transposeі
#Gen_Conv_T_1/BiasAdd/ReadVariableOpReadVariableOp,gen_conv_t_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#Gen_Conv_T_1/BiasAdd/ReadVariableOp«
Gen_Conv_T_1/BiasAddBiasAdd&Gen_Conv_T_1/conv2d_transpose:output:0+Gen_Conv_T_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€
А2
Gen_Conv_T_1/BiasAdd™
$Gen_Conv_T_1/leaky_re_lu_5/LeakyRelu	LeakyReluGen_Conv_T_1/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€
А2&
$Gen_Conv_T_1/leaky_re_lu_5/LeakyReluР
Gen_BN_6/ReadVariableOpReadVariableOp gen_bn_6_readvariableop_resource*
_output_shapes	
:А*
dtype02
Gen_BN_6/ReadVariableOpЦ
Gen_BN_6/ReadVariableOp_1ReadVariableOp"gen_bn_6_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
Gen_BN_6/ReadVariableOp_1√
(Gen_BN_6/FusedBatchNormV3/ReadVariableOpReadVariableOp1gen_bn_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(Gen_BN_6/FusedBatchNormV3/ReadVariableOp…
*Gen_BN_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3gen_bn_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02,
*Gen_BN_6/FusedBatchNormV3/ReadVariableOp_1њ
Gen_BN_6/FusedBatchNormV3FusedBatchNormV32Gen_Conv_T_1/leaky_re_lu_5/LeakyRelu:activations:0Gen_BN_6/ReadVariableOp:value:0!Gen_BN_6/ReadVariableOp_1:value:00Gen_BN_6/FusedBatchNormV3/ReadVariableOp:value:02Gen_BN_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€
А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
Gen_BN_6/FusedBatchNormV3µ
Gen_BN_6/AssignNewValueAssignVariableOp1gen_bn_6_fusedbatchnormv3_readvariableop_resource&Gen_BN_6/FusedBatchNormV3:batch_mean:0)^Gen_BN_6/FusedBatchNormV3/ReadVariableOp*D
_class:
86loc:@Gen_BN_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
Gen_BN_6/AssignNewValue√
Gen_BN_6/AssignNewValue_1AssignVariableOp3gen_bn_6_fusedbatchnormv3_readvariableop_1_resource*Gen_BN_6/FusedBatchNormV3:batch_variance:0+^Gen_BN_6/FusedBatchNormV3/ReadVariableOp_1*F
_class<
:8loc:@Gen_BN_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
Gen_BN_6/AssignNewValue_1v
Gen_Concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Gen_Concat_1/concat/axisџ
Gen_Concat_1/concatConcatV2Gen_BN_6/FusedBatchNormV3:y:0Gen_BN_4/FusedBatchNormV3:y:0!Gen_Concat_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€
А2
Gen_Concat_1/concatn
Gen_SPD_4/ShapeShapeGen_Concat_1/concat:output:0*
T0*
_output_shapes
:2
Gen_SPD_4/ShapeИ
Gen_SPD_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Gen_SPD_4/strided_slice/stackМ
Gen_SPD_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
Gen_SPD_4/strided_slice/stack_1М
Gen_SPD_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
Gen_SPD_4/strided_slice/stack_2Ю
Gen_SPD_4/strided_sliceStridedSliceGen_SPD_4/Shape:output:0&Gen_SPD_4/strided_slice/stack:output:0(Gen_SPD_4/strided_slice/stack_1:output:0(Gen_SPD_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_SPD_4/strided_sliceМ
Gen_SPD_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2!
Gen_SPD_4/strided_slice_1/stackР
!Gen_SPD_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!Gen_SPD_4/strided_slice_1/stack_1Р
!Gen_SPD_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!Gen_SPD_4/strided_slice_1/stack_2®
Gen_SPD_4/strided_slice_1StridedSliceGen_SPD_4/Shape:output:0(Gen_SPD_4/strided_slice_1/stack:output:0*Gen_SPD_4/strided_slice_1/stack_1:output:0*Gen_SPD_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_SPD_4/strided_slice_1w
Gen_SPD_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
Gen_SPD_4/dropout/Const∞
Gen_SPD_4/dropout/MulMulGen_Concat_1/concat:output:0 Gen_SPD_4/dropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€
А2
Gen_SPD_4/dropout/MulЦ
(Gen_SPD_4/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(Gen_SPD_4/dropout/random_uniform/shape/1Ц
(Gen_SPD_4/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(Gen_SPD_4/dropout/random_uniform/shape/2≤
&Gen_SPD_4/dropout/random_uniform/shapePack Gen_SPD_4/strided_slice:output:01Gen_SPD_4/dropout/random_uniform/shape/1:output:01Gen_SPD_4/dropout/random_uniform/shape/2:output:0"Gen_SPD_4/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2(
&Gen_SPD_4/dropout/random_uniform/shapeт
.Gen_SPD_4/dropout/random_uniform/RandomUniformRandomUniform/Gen_SPD_4/dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
dtype020
.Gen_SPD_4/dropout/random_uniform/RandomUniformЙ
 Gen_SPD_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2"
 Gen_SPD_4/dropout/GreaterEqual/yч
Gen_SPD_4/dropout/GreaterEqualGreaterEqual7Gen_SPD_4/dropout/random_uniform/RandomUniform:output:0)Gen_SPD_4/dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2 
Gen_SPD_4/dropout/GreaterEqualЃ
Gen_SPD_4/dropout/CastCast"Gen_SPD_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
Gen_SPD_4/dropout/CastЂ
Gen_SPD_4/dropout/Mul_1MulGen_SPD_4/dropout/Mul:z:0Gen_SPD_4/dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€
А2
Gen_SPD_4/dropout/Mul_1s
Gen_Conv_T_2/ShapeShapeGen_SPD_4/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
Gen_Conv_T_2/ShapeО
 Gen_Conv_T_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 Gen_Conv_T_2/strided_slice/stackТ
"Gen_Conv_T_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"Gen_Conv_T_2/strided_slice/stack_1Т
"Gen_Conv_T_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"Gen_Conv_T_2/strided_slice/stack_2∞
Gen_Conv_T_2/strided_sliceStridedSliceGen_Conv_T_2/Shape:output:0)Gen_Conv_T_2/strided_slice/stack:output:0+Gen_Conv_T_2/strided_slice/stack_1:output:0+Gen_Conv_T_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_Conv_T_2/strided_slicen
Gen_Conv_T_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
Gen_Conv_T_2/stack/1n
Gen_Conv_T_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
Gen_Conv_T_2/stack/2o
Gen_Conv_T_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :А2
Gen_Conv_T_2/stack/3а
Gen_Conv_T_2/stackPack#Gen_Conv_T_2/strided_slice:output:0Gen_Conv_T_2/stack/1:output:0Gen_Conv_T_2/stack/2:output:0Gen_Conv_T_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
Gen_Conv_T_2/stackТ
"Gen_Conv_T_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"Gen_Conv_T_2/strided_slice_1/stackЦ
$Gen_Conv_T_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$Gen_Conv_T_2/strided_slice_1/stack_1Ц
$Gen_Conv_T_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$Gen_Conv_T_2/strided_slice_1/stack_2Ї
Gen_Conv_T_2/strided_slice_1StridedSliceGen_Conv_T_2/stack:output:0+Gen_Conv_T_2/strided_slice_1/stack:output:0-Gen_Conv_T_2/strided_slice_1/stack_1:output:0-Gen_Conv_T_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_Conv_T_2/strided_slice_1№
,Gen_Conv_T_2/conv2d_transpose/ReadVariableOpReadVariableOp5gen_conv_t_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,Gen_Conv_T_2/conv2d_transpose/ReadVariableOp®
Gen_Conv_T_2/conv2d_transposeConv2DBackpropInputGen_Conv_T_2/stack:output:04Gen_Conv_T_2/conv2d_transpose/ReadVariableOp:value:0Gen_SPD_4/dropout/Mul_1:z:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Gen_Conv_T_2/conv2d_transposeі
#Gen_Conv_T_2/BiasAdd/ReadVariableOpReadVariableOp,gen_conv_t_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#Gen_Conv_T_2/BiasAdd/ReadVariableOp«
Gen_Conv_T_2/BiasAddBiasAdd&Gen_Conv_T_2/conv2d_transpose:output:0+Gen_Conv_T_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Gen_Conv_T_2/BiasAdd™
$Gen_Conv_T_2/leaky_re_lu_6/LeakyRelu	LeakyReluGen_Conv_T_2/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€А2&
$Gen_Conv_T_2/leaky_re_lu_6/LeakyReluР
Gen_BN_7/ReadVariableOpReadVariableOp gen_bn_7_readvariableop_resource*
_output_shapes	
:А*
dtype02
Gen_BN_7/ReadVariableOpЦ
Gen_BN_7/ReadVariableOp_1ReadVariableOp"gen_bn_7_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
Gen_BN_7/ReadVariableOp_1√
(Gen_BN_7/FusedBatchNormV3/ReadVariableOpReadVariableOp1gen_bn_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(Gen_BN_7/FusedBatchNormV3/ReadVariableOp…
*Gen_BN_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3gen_bn_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02,
*Gen_BN_7/FusedBatchNormV3/ReadVariableOp_1њ
Gen_BN_7/FusedBatchNormV3FusedBatchNormV32Gen_Conv_T_2/leaky_re_lu_6/LeakyRelu:activations:0Gen_BN_7/ReadVariableOp:value:0!Gen_BN_7/ReadVariableOp_1:value:00Gen_BN_7/FusedBatchNormV3/ReadVariableOp:value:02Gen_BN_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
Gen_BN_7/FusedBatchNormV3µ
Gen_BN_7/AssignNewValueAssignVariableOp1gen_bn_7_fusedbatchnormv3_readvariableop_resource&Gen_BN_7/FusedBatchNormV3:batch_mean:0)^Gen_BN_7/FusedBatchNormV3/ReadVariableOp*D
_class:
86loc:@Gen_BN_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
Gen_BN_7/AssignNewValue√
Gen_BN_7/AssignNewValue_1AssignVariableOp3gen_bn_7_fusedbatchnormv3_readvariableop_1_resource*Gen_BN_7/FusedBatchNormV3:batch_variance:0+^Gen_BN_7/FusedBatchNormV3/ReadVariableOp_1*F
_class<
:8loc:@Gen_BN_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
Gen_BN_7/AssignNewValue_1v
Gen_Concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Gen_Concat_2/concat/axisџ
Gen_Concat_2/concatConcatV2Gen_BN_7/FusedBatchNormV3:y:0Gen_BN_3/FusedBatchNormV3:y:0!Gen_Concat_2/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А2
Gen_Concat_2/concatt
Gen_Conv_T_3/ShapeShapeGen_Concat_2/concat:output:0*
T0*
_output_shapes
:2
Gen_Conv_T_3/ShapeО
 Gen_Conv_T_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 Gen_Conv_T_3/strided_slice/stackТ
"Gen_Conv_T_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"Gen_Conv_T_3/strided_slice/stack_1Т
"Gen_Conv_T_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"Gen_Conv_T_3/strided_slice/stack_2∞
Gen_Conv_T_3/strided_sliceStridedSliceGen_Conv_T_3/Shape:output:0)Gen_Conv_T_3/strided_slice/stack:output:0+Gen_Conv_T_3/strided_slice/stack_1:output:0+Gen_Conv_T_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_Conv_T_3/strided_slicen
Gen_Conv_T_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :02
Gen_Conv_T_3/stack/1n
Gen_Conv_T_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :(2
Gen_Conv_T_3/stack/2o
Gen_Conv_T_3/stack/3Const*
_output_shapes
: *
dtype0*
value
B :А2
Gen_Conv_T_3/stack/3а
Gen_Conv_T_3/stackPack#Gen_Conv_T_3/strided_slice:output:0Gen_Conv_T_3/stack/1:output:0Gen_Conv_T_3/stack/2:output:0Gen_Conv_T_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
Gen_Conv_T_3/stackТ
"Gen_Conv_T_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"Gen_Conv_T_3/strided_slice_1/stackЦ
$Gen_Conv_T_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$Gen_Conv_T_3/strided_slice_1/stack_1Ц
$Gen_Conv_T_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$Gen_Conv_T_3/strided_slice_1/stack_2Ї
Gen_Conv_T_3/strided_slice_1StridedSliceGen_Conv_T_3/stack:output:0+Gen_Conv_T_3/strided_slice_1/stack:output:0-Gen_Conv_T_3/strided_slice_1/stack_1:output:0-Gen_Conv_T_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_Conv_T_3/strided_slice_1№
,Gen_Conv_T_3/conv2d_transpose/ReadVariableOpReadVariableOp5gen_conv_t_3_conv2d_transpose_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,Gen_Conv_T_3/conv2d_transpose/ReadVariableOp©
Gen_Conv_T_3/conv2d_transposeConv2DBackpropInputGen_Conv_T_3/stack:output:04Gen_Conv_T_3/conv2d_transpose/ReadVariableOp:value:0Gen_Concat_2/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€0(А*
paddingSAME*
strides
2
Gen_Conv_T_3/conv2d_transposeі
#Gen_Conv_T_3/BiasAdd/ReadVariableOpReadVariableOp,gen_conv_t_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#Gen_Conv_T_3/BiasAdd/ReadVariableOp«
Gen_Conv_T_3/BiasAddBiasAdd&Gen_Conv_T_3/conv2d_transpose:output:0+Gen_Conv_T_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2
Gen_Conv_T_3/BiasAdd™
$Gen_Conv_T_3/leaky_re_lu_7/LeakyRelu	LeakyReluGen_Conv_T_3/BiasAdd:output:0*0
_output_shapes
:€€€€€€€€€0(А2&
$Gen_Conv_T_3/leaky_re_lu_7/LeakyReluР
Gen_BN_8/ReadVariableOpReadVariableOp gen_bn_8_readvariableop_resource*
_output_shapes	
:А*
dtype02
Gen_BN_8/ReadVariableOpЦ
Gen_BN_8/ReadVariableOp_1ReadVariableOp"gen_bn_8_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
Gen_BN_8/ReadVariableOp_1√
(Gen_BN_8/FusedBatchNormV3/ReadVariableOpReadVariableOp1gen_bn_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(Gen_BN_8/FusedBatchNormV3/ReadVariableOp…
*Gen_BN_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3gen_bn_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02,
*Gen_BN_8/FusedBatchNormV3/ReadVariableOp_1њ
Gen_BN_8/FusedBatchNormV3FusedBatchNormV32Gen_Conv_T_3/leaky_re_lu_7/LeakyRelu:activations:0Gen_BN_8/ReadVariableOp:value:0!Gen_BN_8/ReadVariableOp_1:value:00Gen_BN_8/FusedBatchNormV3/ReadVariableOp:value:02Gen_BN_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€0(А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
Gen_BN_8/FusedBatchNormV3µ
Gen_BN_8/AssignNewValueAssignVariableOp1gen_bn_8_fusedbatchnormv3_readvariableop_resource&Gen_BN_8/FusedBatchNormV3:batch_mean:0)^Gen_BN_8/FusedBatchNormV3/ReadVariableOp*D
_class:
86loc:@Gen_BN_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
Gen_BN_8/AssignNewValue√
Gen_BN_8/AssignNewValue_1AssignVariableOp3gen_bn_8_fusedbatchnormv3_readvariableop_1_resource*Gen_BN_8/FusedBatchNormV3:batch_variance:0+^Gen_BN_8/FusedBatchNormV3/ReadVariableOp_1*F
_class<
:8loc:@Gen_BN_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
Gen_BN_8/AssignNewValue_1v
Gen_Concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Gen_Concat_3/concat/axisџ
Gen_Concat_3/concatConcatV2Gen_BN_8/FusedBatchNormV3:y:0Gen_BN_2/FusedBatchNormV3:y:0!Gen_Concat_3/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€0(А2
Gen_Concat_3/concatn
Gen_SPD_5/ShapeShapeGen_Concat_3/concat:output:0*
T0*
_output_shapes
:2
Gen_SPD_5/ShapeИ
Gen_SPD_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Gen_SPD_5/strided_slice/stackМ
Gen_SPD_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
Gen_SPD_5/strided_slice/stack_1М
Gen_SPD_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
Gen_SPD_5/strided_slice/stack_2Ю
Gen_SPD_5/strided_sliceStridedSliceGen_SPD_5/Shape:output:0&Gen_SPD_5/strided_slice/stack:output:0(Gen_SPD_5/strided_slice/stack_1:output:0(Gen_SPD_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_SPD_5/strided_sliceМ
Gen_SPD_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2!
Gen_SPD_5/strided_slice_1/stackР
!Gen_SPD_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!Gen_SPD_5/strided_slice_1/stack_1Р
!Gen_SPD_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!Gen_SPD_5/strided_slice_1/stack_2®
Gen_SPD_5/strided_slice_1StridedSliceGen_SPD_5/Shape:output:0(Gen_SPD_5/strided_slice_1/stack:output:0*Gen_SPD_5/strided_slice_1/stack_1:output:0*Gen_SPD_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_SPD_5/strided_slice_1w
Gen_SPD_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
Gen_SPD_5/dropout/Const∞
Gen_SPD_5/dropout/MulMulGen_Concat_3/concat:output:0 Gen_SPD_5/dropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2
Gen_SPD_5/dropout/MulЦ
(Gen_SPD_5/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(Gen_SPD_5/dropout/random_uniform/shape/1Ц
(Gen_SPD_5/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(Gen_SPD_5/dropout/random_uniform/shape/2≤
&Gen_SPD_5/dropout/random_uniform/shapePack Gen_SPD_5/strided_slice:output:01Gen_SPD_5/dropout/random_uniform/shape/1:output:01Gen_SPD_5/dropout/random_uniform/shape/2:output:0"Gen_SPD_5/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2(
&Gen_SPD_5/dropout/random_uniform/shapeт
.Gen_SPD_5/dropout/random_uniform/RandomUniformRandomUniform/Gen_SPD_5/dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
dtype020
.Gen_SPD_5/dropout/random_uniform/RandomUniformЙ
 Gen_SPD_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2"
 Gen_SPD_5/dropout/GreaterEqual/yч
Gen_SPD_5/dropout/GreaterEqualGreaterEqual7Gen_SPD_5/dropout/random_uniform/RandomUniform:output:0)Gen_SPD_5/dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2 
Gen_SPD_5/dropout/GreaterEqualЃ
Gen_SPD_5/dropout/CastCast"Gen_SPD_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
Gen_SPD_5/dropout/CastЂ
Gen_SPD_5/dropout/Mul_1MulGen_SPD_5/dropout/Mul:z:0Gen_SPD_5/dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2
Gen_SPD_5/dropout/Mul_1s
Gen_Conv_T_4/ShapeShapeGen_SPD_5/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
Gen_Conv_T_4/ShapeО
 Gen_Conv_T_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 Gen_Conv_T_4/strided_slice/stackТ
"Gen_Conv_T_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"Gen_Conv_T_4/strided_slice/stack_1Т
"Gen_Conv_T_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"Gen_Conv_T_4/strided_slice/stack_2∞
Gen_Conv_T_4/strided_sliceStridedSliceGen_Conv_T_4/Shape:output:0)Gen_Conv_T_4/strided_slice/stack:output:0+Gen_Conv_T_4/strided_slice/stack_1:output:0+Gen_Conv_T_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_Conv_T_4/strided_slicen
Gen_Conv_T_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :`2
Gen_Conv_T_4/stack/1n
Gen_Conv_T_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :P2
Gen_Conv_T_4/stack/2n
Gen_Conv_T_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
Gen_Conv_T_4/stack/3а
Gen_Conv_T_4/stackPack#Gen_Conv_T_4/strided_slice:output:0Gen_Conv_T_4/stack/1:output:0Gen_Conv_T_4/stack/2:output:0Gen_Conv_T_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
Gen_Conv_T_4/stackТ
"Gen_Conv_T_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"Gen_Conv_T_4/strided_slice_1/stackЦ
$Gen_Conv_T_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$Gen_Conv_T_4/strided_slice_1/stack_1Ц
$Gen_Conv_T_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$Gen_Conv_T_4/strided_slice_1/stack_2Ї
Gen_Conv_T_4/strided_slice_1StridedSliceGen_Conv_T_4/stack:output:0+Gen_Conv_T_4/strided_slice_1/stack:output:0-Gen_Conv_T_4/strided_slice_1/stack_1:output:0-Gen_Conv_T_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_Conv_T_4/strided_slice_1џ
,Gen_Conv_T_4/conv2d_transpose/ReadVariableOpReadVariableOp5gen_conv_t_4_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@А*
dtype02.
,Gen_Conv_T_4/conv2d_transpose/ReadVariableOpІ
Gen_Conv_T_4/conv2d_transposeConv2DBackpropInputGen_Conv_T_4/stack:output:04Gen_Conv_T_4/conv2d_transpose/ReadVariableOp:value:0Gen_SPD_5/dropout/Mul_1:z:0*
T0*/
_output_shapes
:€€€€€€€€€`P@*
paddingSAME*
strides
2
Gen_Conv_T_4/conv2d_transpose≥
#Gen_Conv_T_4/BiasAdd/ReadVariableOpReadVariableOp,gen_conv_t_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#Gen_Conv_T_4/BiasAdd/ReadVariableOp∆
Gen_Conv_T_4/BiasAddBiasAdd&Gen_Conv_T_4/conv2d_transpose:output:0+Gen_Conv_T_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€`P@2
Gen_Conv_T_4/BiasAdd©
$Gen_Conv_T_4/leaky_re_lu_8/LeakyRelu	LeakyReluGen_Conv_T_4/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€`P@2&
$Gen_Conv_T_4/leaky_re_lu_8/LeakyReluП
Gen_BN_9/ReadVariableOpReadVariableOp gen_bn_9_readvariableop_resource*
_output_shapes
:@*
dtype02
Gen_BN_9/ReadVariableOpХ
Gen_BN_9/ReadVariableOp_1ReadVariableOp"gen_bn_9_readvariableop_1_resource*
_output_shapes
:@*
dtype02
Gen_BN_9/ReadVariableOp_1¬
(Gen_BN_9/FusedBatchNormV3/ReadVariableOpReadVariableOp1gen_bn_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02*
(Gen_BN_9/FusedBatchNormV3/ReadVariableOp»
*Gen_BN_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3gen_bn_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02,
*Gen_BN_9/FusedBatchNormV3/ReadVariableOp_1Ї
Gen_BN_9/FusedBatchNormV3FusedBatchNormV32Gen_Conv_T_4/leaky_re_lu_8/LeakyRelu:activations:0Gen_BN_9/ReadVariableOp:value:0!Gen_BN_9/ReadVariableOp_1:value:00Gen_BN_9/FusedBatchNormV3/ReadVariableOp:value:02Gen_BN_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€`P@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<2
Gen_BN_9/FusedBatchNormV3µ
Gen_BN_9/AssignNewValueAssignVariableOp1gen_bn_9_fusedbatchnormv3_readvariableop_resource&Gen_BN_9/FusedBatchNormV3:batch_mean:0)^Gen_BN_9/FusedBatchNormV3/ReadVariableOp*D
_class:
86loc:@Gen_BN_9/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
Gen_BN_9/AssignNewValue√
Gen_BN_9/AssignNewValue_1AssignVariableOp3gen_bn_9_fusedbatchnormv3_readvariableop_1_resource*Gen_BN_9/FusedBatchNormV3:batch_variance:0+^Gen_BN_9/FusedBatchNormV3/ReadVariableOp_1*F
_class<
:8loc:@Gen_BN_9/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
Gen_BN_9/AssignNewValue_1v
Gen_Concat_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Gen_Concat_4/concat/axisџ
Gen_Concat_4/concatConcatV2Gen_BN_9/FusedBatchNormV3:y:0Gen_BN_1/FusedBatchNormV3:y:0!Gen_Concat_4/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€`PА2
Gen_Concat_4/concatt
Gen_Conv_T_5/ShapeShapeGen_Concat_4/concat:output:0*
T0*
_output_shapes
:2
Gen_Conv_T_5/ShapeО
 Gen_Conv_T_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 Gen_Conv_T_5/strided_slice/stackТ
"Gen_Conv_T_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"Gen_Conv_T_5/strided_slice/stack_1Т
"Gen_Conv_T_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"Gen_Conv_T_5/strided_slice/stack_2∞
Gen_Conv_T_5/strided_sliceStridedSliceGen_Conv_T_5/Shape:output:0)Gen_Conv_T_5/strided_slice/stack:output:0+Gen_Conv_T_5/strided_slice/stack_1:output:0+Gen_Conv_T_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_Conv_T_5/strided_sliceo
Gen_Conv_T_5/stack/1Const*
_output_shapes
: *
dtype0*
value
B :ј2
Gen_Conv_T_5/stack/1o
Gen_Conv_T_5/stack/2Const*
_output_shapes
: *
dtype0*
value
B :†2
Gen_Conv_T_5/stack/2n
Gen_Conv_T_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Gen_Conv_T_5/stack/3а
Gen_Conv_T_5/stackPack#Gen_Conv_T_5/strided_slice:output:0Gen_Conv_T_5/stack/1:output:0Gen_Conv_T_5/stack/2:output:0Gen_Conv_T_5/stack/3:output:0*
N*
T0*
_output_shapes
:2
Gen_Conv_T_5/stackТ
"Gen_Conv_T_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"Gen_Conv_T_5/strided_slice_1/stackЦ
$Gen_Conv_T_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$Gen_Conv_T_5/strided_slice_1/stack_1Ц
$Gen_Conv_T_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$Gen_Conv_T_5/strided_slice_1/stack_2Ї
Gen_Conv_T_5/strided_slice_1StridedSliceGen_Conv_T_5/stack:output:0+Gen_Conv_T_5/strided_slice_1/stack:output:0-Gen_Conv_T_5/strided_slice_1/stack_1:output:0-Gen_Conv_T_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_Conv_T_5/strided_slice_1џ
,Gen_Conv_T_5/conv2d_transpose/ReadVariableOpReadVariableOp5gen_conv_t_5_conv2d_transpose_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,Gen_Conv_T_5/conv2d_transpose/ReadVariableOp™
Gen_Conv_T_5/conv2d_transposeConv2DBackpropInputGen_Conv_T_5/stack:output:04Gen_Conv_T_5/conv2d_transpose/ReadVariableOp:value:0Gen_Concat_4/concat:output:0*
T0*1
_output_shapes
:€€€€€€€€€ј† *
paddingSAME*
strides
2
Gen_Conv_T_5/conv2d_transpose≥
#Gen_Conv_T_5/BiasAdd/ReadVariableOpReadVariableOp,gen_conv_t_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#Gen_Conv_T_5/BiasAdd/ReadVariableOp»
Gen_Conv_T_5/BiasAddBiasAdd&Gen_Conv_T_5/conv2d_transpose:output:0+Gen_Conv_T_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ј† 2
Gen_Conv_T_5/BiasAddЂ
$Gen_Conv_T_5/leaky_re_lu_9/LeakyRelu	LeakyReluGen_Conv_T_5/BiasAdd:output:0*1
_output_shapes
:€€€€€€€€€ј† 2&
$Gen_Conv_T_5/leaky_re_lu_9/LeakyReluТ
Gen_BN_10/ReadVariableOpReadVariableOp!gen_bn_10_readvariableop_resource*
_output_shapes
: *
dtype02
Gen_BN_10/ReadVariableOpШ
Gen_BN_10/ReadVariableOp_1ReadVariableOp#gen_bn_10_readvariableop_1_resource*
_output_shapes
: *
dtype02
Gen_BN_10/ReadVariableOp_1≈
)Gen_BN_10/FusedBatchNormV3/ReadVariableOpReadVariableOp2gen_bn_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02+
)Gen_BN_10/FusedBatchNormV3/ReadVariableOpЋ
+Gen_BN_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp4gen_bn_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02-
+Gen_BN_10/FusedBatchNormV3/ReadVariableOp_1¬
Gen_BN_10/FusedBatchNormV3FusedBatchNormV32Gen_Conv_T_5/leaky_re_lu_9/LeakyRelu:activations:0 Gen_BN_10/ReadVariableOp:value:0"Gen_BN_10/ReadVariableOp_1:value:01Gen_BN_10/FusedBatchNormV3/ReadVariableOp:value:03Gen_BN_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:€€€€€€€€€ј† : : : : :*
epsilon%oГ:*
exponential_avg_factor%
„#<2
Gen_BN_10/FusedBatchNormV3ї
Gen_BN_10/AssignNewValueAssignVariableOp2gen_bn_10_fusedbatchnormv3_readvariableop_resource'Gen_BN_10/FusedBatchNormV3:batch_mean:0*^Gen_BN_10/FusedBatchNormV3/ReadVariableOp*E
_class;
97loc:@Gen_BN_10/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
Gen_BN_10/AssignNewValue…
Gen_BN_10/AssignNewValue_1AssignVariableOp4gen_bn_10_fusedbatchnormv3_readvariableop_1_resource+Gen_BN_10/FusedBatchNormV3:batch_variance:0,^Gen_BN_10/FusedBatchNormV3/ReadVariableOp_1*G
_class=
;9loc:@Gen_BN_10/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
Gen_BN_10/AssignNewValue_1v
Gen_Conv_T_6/ShapeShapeGen_BN_10/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
Gen_Conv_T_6/ShapeО
 Gen_Conv_T_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 Gen_Conv_T_6/strided_slice/stackТ
"Gen_Conv_T_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"Gen_Conv_T_6/strided_slice/stack_1Т
"Gen_Conv_T_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"Gen_Conv_T_6/strided_slice/stack_2∞
Gen_Conv_T_6/strided_sliceStridedSliceGen_Conv_T_6/Shape:output:0)Gen_Conv_T_6/strided_slice/stack:output:0+Gen_Conv_T_6/strided_slice/stack_1:output:0+Gen_Conv_T_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_Conv_T_6/strided_sliceo
Gen_Conv_T_6/stack/1Const*
_output_shapes
: *
dtype0*
value
B :ј2
Gen_Conv_T_6/stack/1o
Gen_Conv_T_6/stack/2Const*
_output_shapes
: *
dtype0*
value
B :†2
Gen_Conv_T_6/stack/2n
Gen_Conv_T_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
Gen_Conv_T_6/stack/3а
Gen_Conv_T_6/stackPack#Gen_Conv_T_6/strided_slice:output:0Gen_Conv_T_6/stack/1:output:0Gen_Conv_T_6/stack/2:output:0Gen_Conv_T_6/stack/3:output:0*
N*
T0*
_output_shapes
:2
Gen_Conv_T_6/stackТ
"Gen_Conv_T_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"Gen_Conv_T_6/strided_slice_1/stackЦ
$Gen_Conv_T_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$Gen_Conv_T_6/strided_slice_1/stack_1Ц
$Gen_Conv_T_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$Gen_Conv_T_6/strided_slice_1/stack_2Ї
Gen_Conv_T_6/strided_slice_1StridedSliceGen_Conv_T_6/stack:output:0+Gen_Conv_T_6/strided_slice_1/stack:output:0-Gen_Conv_T_6/strided_slice_1/stack_1:output:0-Gen_Conv_T_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Gen_Conv_T_6/strided_slice_1Џ
,Gen_Conv_T_6/conv2d_transpose/ReadVariableOpReadVariableOp5gen_conv_t_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02.
,Gen_Conv_T_6/conv2d_transpose/ReadVariableOpђ
Gen_Conv_T_6/conv2d_transposeConv2DBackpropInputGen_Conv_T_6/stack:output:04Gen_Conv_T_6/conv2d_transpose/ReadVariableOp:value:0Gen_BN_10/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:€€€€€€€€€ј†*
paddingSAME*
strides
2
Gen_Conv_T_6/conv2d_transpose≥
#Gen_Conv_T_6/BiasAdd/ReadVariableOpReadVariableOp,gen_conv_t_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#Gen_Conv_T_6/BiasAdd/ReadVariableOp»
Gen_Conv_T_6/BiasAddBiasAdd&Gen_Conv_T_6/conv2d_transpose:output:0+Gen_Conv_T_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ј†2
Gen_Conv_T_6/BiasAddЙ
Gen_Conv_T_6/TanhTanhGen_Conv_T_6/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€ј†2
Gen_Conv_T_6/TanhС
IdentityIdentityGen_Conv_T_6/Tanh:y:0^Gen_BN_1/AssignNewValue^Gen_BN_1/AssignNewValue_1^Gen_BN_10/AssignNewValue^Gen_BN_10/AssignNewValue_1^Gen_BN_2/AssignNewValue^Gen_BN_2/AssignNewValue_1^Gen_BN_3/AssignNewValue^Gen_BN_3/AssignNewValue_1^Gen_BN_4/AssignNewValue^Gen_BN_4/AssignNewValue_1^Gen_BN_5/AssignNewValue^Gen_BN_5/AssignNewValue_1^Gen_BN_6/AssignNewValue^Gen_BN_6/AssignNewValue_1^Gen_BN_7/AssignNewValue^Gen_BN_7/AssignNewValue_1^Gen_BN_8/AssignNewValue^Gen_BN_8/AssignNewValue_1^Gen_BN_9/AssignNewValue^Gen_BN_9/AssignNewValue_1*
T0*1
_output_shapes
:€€€€€€€€€ј†2

Identity"
identityIdentity:output:0*™
_input_shapesШ
Х:€€€€€€€€€ј†::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
Gen_BN_1/AssignNewValueGen_BN_1/AssignNewValue26
Gen_BN_1/AssignNewValue_1Gen_BN_1/AssignNewValue_124
Gen_BN_10/AssignNewValueGen_BN_10/AssignNewValue28
Gen_BN_10/AssignNewValue_1Gen_BN_10/AssignNewValue_122
Gen_BN_2/AssignNewValueGen_BN_2/AssignNewValue26
Gen_BN_2/AssignNewValue_1Gen_BN_2/AssignNewValue_122
Gen_BN_3/AssignNewValueGen_BN_3/AssignNewValue26
Gen_BN_3/AssignNewValue_1Gen_BN_3/AssignNewValue_122
Gen_BN_4/AssignNewValueGen_BN_4/AssignNewValue26
Gen_BN_4/AssignNewValue_1Gen_BN_4/AssignNewValue_122
Gen_BN_5/AssignNewValueGen_BN_5/AssignNewValue26
Gen_BN_5/AssignNewValue_1Gen_BN_5/AssignNewValue_122
Gen_BN_6/AssignNewValueGen_BN_6/AssignNewValue26
Gen_BN_6/AssignNewValue_1Gen_BN_6/AssignNewValue_122
Gen_BN_7/AssignNewValueGen_BN_7/AssignNewValue26
Gen_BN_7/AssignNewValue_1Gen_BN_7/AssignNewValue_122
Gen_BN_8/AssignNewValueGen_BN_8/AssignNewValue26
Gen_BN_8/AssignNewValue_1Gen_BN_8/AssignNewValue_122
Gen_BN_9/AssignNewValueGen_BN_9/AssignNewValue26
Gen_BN_9/AssignNewValue_1Gen_BN_9/AssignNewValue_1:Y U
1
_output_shapes
:€€€€€€€€€ј†
 
_user_specified_nameinputs
ї
F
*__inference_Gen_SPD_5_layer_call_fn_146111

inputs
identityѕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Gen_SPD_5_layer_call_and_return_conditional_losses_1427652
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€0(А:X T
0
_output_shapes
:€€€€€€€€€0(А
 
_user_specified_nameinputs
Щ
d
E__inference_Gen_SPD_2_layer_call_and_return_conditional_losses_142374

inputs
identityИD
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
strided_slice/stack_2в
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
strided_slice_1/stack_2м
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
 *  †?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€
А2
dropout/MulВ
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1В
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ц
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape‘
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/yѕ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/GreaterEqualР
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€
А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€
А:X T
0
_output_shapes
:€€€€€€€€€
А
 
_user_specified_nameinputs
Ч
t
H__inference_Gen_Concat_3_layer_call_and_return_conditional_losses_146029
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisК
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€0(А2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:€€€€€€€€€0(А:l h
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:€€€€€€€€€0(А
"
_user_specified_name
inputs/1
‘
В
-__inference_Gen_Conv_T_3_layer_call_fn_141366

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_3_layer_call_and_return_conditional_losses_1413562
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
К
Ь
)__inference_Gen_BN_8_layer_call_fn_146009

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_8_layer_call_and_return_conditional_losses_1414282
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
М
J
.__inference_leaky_re_lu_7_layer_call_fn_146282

inputs
identityе
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_1413472
PartitionedCallЗ
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
ƒ
Ь
)__inference_Gen_BN_5_layer_call_fn_145652

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_5_layer_call_and_return_conditional_losses_1424562
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Њ
Ь
)__inference_Gen_BN_1_layer_call_fn_144831

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€`P@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_1_layer_call_and_return_conditional_losses_1419562
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€`P@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€`P@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€`P@
 
_user_specified_nameinputs
зu
ђ
__inference__traced_save_146511
file_prefix0
,savev2_gen_conv_1_kernel_read_readvariableop.
*savev2_gen_conv_1_bias_read_readvariableop-
)savev2_gen_bn_1_gamma_read_readvariableop,
(savev2_gen_bn_1_beta_read_readvariableop3
/savev2_gen_bn_1_moving_mean_read_readvariableop7
3savev2_gen_bn_1_moving_variance_read_readvariableop0
,savev2_gen_conv_2_kernel_read_readvariableop.
*savev2_gen_conv_2_bias_read_readvariableop-
)savev2_gen_bn_2_gamma_read_readvariableop,
(savev2_gen_bn_2_beta_read_readvariableop3
/savev2_gen_bn_2_moving_mean_read_readvariableop7
3savev2_gen_bn_2_moving_variance_read_readvariableop0
,savev2_gen_conv_3_kernel_read_readvariableop.
*savev2_gen_conv_3_bias_read_readvariableop-
)savev2_gen_bn_3_gamma_read_readvariableop,
(savev2_gen_bn_3_beta_read_readvariableop3
/savev2_gen_bn_3_moving_mean_read_readvariableop7
3savev2_gen_bn_3_moving_variance_read_readvariableop0
,savev2_gen_conv_4_kernel_read_readvariableop.
*savev2_gen_conv_4_bias_read_readvariableop-
)savev2_gen_bn_4_gamma_read_readvariableop,
(savev2_gen_bn_4_beta_read_readvariableop3
/savev2_gen_bn_4_moving_mean_read_readvariableop7
3savev2_gen_bn_4_moving_variance_read_readvariableop0
,savev2_gen_conv_5_kernel_read_readvariableop.
*savev2_gen_conv_5_bias_read_readvariableop-
)savev2_gen_bn_5_gamma_read_readvariableop,
(savev2_gen_bn_5_beta_read_readvariableop3
/savev2_gen_bn_5_moving_mean_read_readvariableop7
3savev2_gen_bn_5_moving_variance_read_readvariableop2
.savev2_gen_conv_t_1_kernel_read_readvariableop0
,savev2_gen_conv_t_1_bias_read_readvariableop-
)savev2_gen_bn_6_gamma_read_readvariableop,
(savev2_gen_bn_6_beta_read_readvariableop3
/savev2_gen_bn_6_moving_mean_read_readvariableop7
3savev2_gen_bn_6_moving_variance_read_readvariableop2
.savev2_gen_conv_t_2_kernel_read_readvariableop0
,savev2_gen_conv_t_2_bias_read_readvariableop-
)savev2_gen_bn_7_gamma_read_readvariableop,
(savev2_gen_bn_7_beta_read_readvariableop3
/savev2_gen_bn_7_moving_mean_read_readvariableop7
3savev2_gen_bn_7_moving_variance_read_readvariableop2
.savev2_gen_conv_t_3_kernel_read_readvariableop0
,savev2_gen_conv_t_3_bias_read_readvariableop-
)savev2_gen_bn_8_gamma_read_readvariableop,
(savev2_gen_bn_8_beta_read_readvariableop3
/savev2_gen_bn_8_moving_mean_read_readvariableop7
3savev2_gen_bn_8_moving_variance_read_readvariableop2
.savev2_gen_conv_t_4_kernel_read_readvariableop0
,savev2_gen_conv_t_4_bias_read_readvariableop-
)savev2_gen_bn_9_gamma_read_readvariableop,
(savev2_gen_bn_9_beta_read_readvariableop3
/savev2_gen_bn_9_moving_mean_read_readvariableop7
3savev2_gen_bn_9_moving_variance_read_readvariableop2
.savev2_gen_conv_t_5_kernel_read_readvariableop0
,savev2_gen_conv_t_5_bias_read_readvariableop.
*savev2_gen_bn_10_gamma_read_readvariableop-
)savev2_gen_bn_10_beta_read_readvariableop4
0savev2_gen_bn_10_moving_mean_read_readvariableop8
4savev2_gen_bn_10_moving_variance_read_readvariableop2
.savev2_gen_conv_t_6_kernel_read_readvariableop0
,savev2_gen_conv_t_6_bias_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
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
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_3d0aaeaea33b462494cbfb23df99def8/part2	
Const_1Л
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename©
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*ї
value±BЃ?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЙ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*У
valueЙBЖ?B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesј
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_gen_conv_1_kernel_read_readvariableop*savev2_gen_conv_1_bias_read_readvariableop)savev2_gen_bn_1_gamma_read_readvariableop(savev2_gen_bn_1_beta_read_readvariableop/savev2_gen_bn_1_moving_mean_read_readvariableop3savev2_gen_bn_1_moving_variance_read_readvariableop,savev2_gen_conv_2_kernel_read_readvariableop*savev2_gen_conv_2_bias_read_readvariableop)savev2_gen_bn_2_gamma_read_readvariableop(savev2_gen_bn_2_beta_read_readvariableop/savev2_gen_bn_2_moving_mean_read_readvariableop3savev2_gen_bn_2_moving_variance_read_readvariableop,savev2_gen_conv_3_kernel_read_readvariableop*savev2_gen_conv_3_bias_read_readvariableop)savev2_gen_bn_3_gamma_read_readvariableop(savev2_gen_bn_3_beta_read_readvariableop/savev2_gen_bn_3_moving_mean_read_readvariableop3savev2_gen_bn_3_moving_variance_read_readvariableop,savev2_gen_conv_4_kernel_read_readvariableop*savev2_gen_conv_4_bias_read_readvariableop)savev2_gen_bn_4_gamma_read_readvariableop(savev2_gen_bn_4_beta_read_readvariableop/savev2_gen_bn_4_moving_mean_read_readvariableop3savev2_gen_bn_4_moving_variance_read_readvariableop,savev2_gen_conv_5_kernel_read_readvariableop*savev2_gen_conv_5_bias_read_readvariableop)savev2_gen_bn_5_gamma_read_readvariableop(savev2_gen_bn_5_beta_read_readvariableop/savev2_gen_bn_5_moving_mean_read_readvariableop3savev2_gen_bn_5_moving_variance_read_readvariableop.savev2_gen_conv_t_1_kernel_read_readvariableop,savev2_gen_conv_t_1_bias_read_readvariableop)savev2_gen_bn_6_gamma_read_readvariableop(savev2_gen_bn_6_beta_read_readvariableop/savev2_gen_bn_6_moving_mean_read_readvariableop3savev2_gen_bn_6_moving_variance_read_readvariableop.savev2_gen_conv_t_2_kernel_read_readvariableop,savev2_gen_conv_t_2_bias_read_readvariableop)savev2_gen_bn_7_gamma_read_readvariableop(savev2_gen_bn_7_beta_read_readvariableop/savev2_gen_bn_7_moving_mean_read_readvariableop3savev2_gen_bn_7_moving_variance_read_readvariableop.savev2_gen_conv_t_3_kernel_read_readvariableop,savev2_gen_conv_t_3_bias_read_readvariableop)savev2_gen_bn_8_gamma_read_readvariableop(savev2_gen_bn_8_beta_read_readvariableop/savev2_gen_bn_8_moving_mean_read_readvariableop3savev2_gen_bn_8_moving_variance_read_readvariableop.savev2_gen_conv_t_4_kernel_read_readvariableop,savev2_gen_conv_t_4_bias_read_readvariableop)savev2_gen_bn_9_gamma_read_readvariableop(savev2_gen_bn_9_beta_read_readvariableop/savev2_gen_bn_9_moving_mean_read_readvariableop3savev2_gen_bn_9_moving_variance_read_readvariableop.savev2_gen_conv_t_5_kernel_read_readvariableop,savev2_gen_conv_t_5_bias_read_readvariableop*savev2_gen_bn_10_gamma_read_readvariableop)savev2_gen_bn_10_beta_read_readvariableop0savev2_gen_bn_10_moving_mean_read_readvariableop4savev2_gen_bn_10_moving_variance_read_readvariableop.savev2_gen_conv_t_6_kernel_read_readvariableop,savev2_gen_conv_t_6_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *M
dtypesC
A2?2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
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

identity_1Identity_1:output:0*√
_input_shapes±
Ѓ: :@:@:@:@:@:@:@А:А:А:А:А:А:АА:А:А:А:А:А:АА:А:А:А:А:А:АА:А:А:А:А:А:АА:А:А:А:А:А:АА:А:А:А:А:А:АА:А:А:А:А:А:@А:@:@:@:@:@: А: : : : : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@А:!

_output_shapes	
:А:!	

_output_shapes	
:А:!


_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:! 

_output_shapes	
:А:!!

_output_shapes	
:А:!"

_output_shapes	
:А:!#

_output_shapes	
:А:!$

_output_shapes	
:А:.%*
(
_output_shapes
:АА:!&

_output_shapes	
:А:!'

_output_shapes	
:А:!(

_output_shapes	
:А:!)

_output_shapes	
:А:!*

_output_shapes	
:А:.+*
(
_output_shapes
:АА:!,

_output_shapes	
:А:!-

_output_shapes	
:А:!.

_output_shapes	
:А:!/

_output_shapes	
:А:!0

_output_shapes	
:А:-1)
'
_output_shapes
:@А: 2

_output_shapes
:@: 3

_output_shapes
:@: 4

_output_shapes
:@: 5

_output_shapes
:@: 6

_output_shapes
:@:-7)
'
_output_shapes
: А: 8

_output_shapes
: : 9

_output_shapes
: : :

_output_shapes
: : ;

_output_shapes
: : <

_output_shapes
: :,=(
&
_output_shapes
: : >

_output_shapes
::?

_output_shapes
: 
Ж
А
+__inference_Gen_Conv_2_layer_call_fn_144928

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€`PА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_Gen_Conv_2_layer_call_and_return_conditional_losses_1420212
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€`PА2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€`P@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€`P@
 
_user_specified_nameinputs
Ґ
E
)__inference_Gen_MP_2_layer_call_fn_140263

inputs
identityи
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_MP_2_layer_call_and_return_conditional_losses_1402572
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
’
c
E__inference_Gen_SPD_4_layer_call_and_return_conditional_losses_145833

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

IdentityМ

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
»	
Ѓ
F__inference_Gen_Conv_1_layer_call_and_return_conditional_losses_144771

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp•
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ј†@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ј†@2	
BiasAddА
leaky_re_lu/LeakyRelu	LeakyReluBiasAdd:output:0*1
_output_shapes
:€€€€€€€€€ј†@2
leaky_re_lu/LeakyReluБ
IdentityIdentity#leaky_re_lu/LeakyRelu:activations:0*
T0*1
_output_shapes
:€€€€€€€€€ј†@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€ј†:::Y U
1
_output_shapes
:€€€€€€€€€ј†
 
_user_specified_nameinputs
ƒ
Ь
)__inference_Gen_BN_3_layer_call_fn_145280

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_3_layer_call_and_return_conditional_losses_1422152
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ѕ
э
D__inference_Gen_BN_5_layer_call_and_return_conditional_losses_145626

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ѕ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€А:::::X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ѓ
e
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_140957

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2
	LeakyReluЖ
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
ƒ
Ь
)__inference_Gen_BN_4_layer_call_fn_145428

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_4_layer_call_and_return_conditional_losses_1423162
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€
А::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€
А
 
_user_specified_nameinputs
ѓ
e
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_141347

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2
	LeakyReluЖ
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
¬
Ь
)__inference_Gen_BN_2_layer_call_fn_144979

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€0(А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_2_layer_call_and_return_conditional_losses_1420572
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€0(А::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€0(А
 
_user_specified_nameinputs
Ћ	
Ѓ
F__inference_Gen_Conv_4_layer_call_and_return_conditional_losses_145291

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2	
BiasAddГ
leaky_re_lu_3/LeakyRelu	LeakyReluBiasAdd:output:0*0
_output_shapes
:€€€€€€€€€А2
leaky_re_lu_3/LeakyReluВ
IdentityIdentity%leaky_re_lu_3/LeakyRelu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€А:::X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
П
r
H__inference_Gen_Concat_3_layer_call_and_return_conditional_losses_142729

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisИ
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€0(А2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:€€€€€€€€€0(А2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:€€€€€€€€€0(А:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs:XT
0
_output_shapes
:€€€€€€€€€0(А
 
_user_specified_nameinputs
»	
Ѓ
F__inference_Gen_Conv_1_layer_call_and_return_conditional_losses_141920

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp•
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ј†@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ј†@2	
BiasAddА
leaky_re_lu/LeakyRelu	LeakyReluBiasAdd:output:0*1
_output_shapes
:€€€€€€€€€ј†@2
leaky_re_lu/LeakyReluБ
IdentityIdentity#leaky_re_lu/LeakyRelu:activations:0*
T0*1
_output_shapes
:€€€€€€€€€ј†@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€ј†:::Y U
1
_output_shapes
:€€€€€€€€€ј†
 
_user_specified_nameinputs
“
В
-__inference_Gen_Conv_T_4_layer_call_fn_141595

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_Gen_Conv_T_4_layer_call_and_return_conditional_losses_1415852
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
¬
Ь
)__inference_Gen_BN_4_layer_call_fn_145415

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€
А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_Gen_BN_4_layer_call_and_return_conditional_losses_1422982
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€
А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€
А::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€
А
 
_user_specified_nameinputs"ЄL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*«
serving_default≥
I
	Gen_Input<
serving_default_Gen_Input:0€€€€€€€€€ј†J
Gen_Conv_T_6:
StatefulPartitionedCall:0€€€€€€€€€ј†tensorflow/serving/predict:¶в

Г»
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer-14
layer_with_weights-8
layer-15
layer-16
layer_with_weights-9
layer-17
layer-18
layer_with_weights-10
layer-19
layer_with_weights-11
layer-20
layer-21
layer-22
layer_with_weights-12
layer-23
layer_with_weights-13
layer-24
layer-25
layer_with_weights-14
layer-26
layer_with_weights-15
layer-27
layer-28
layer-29
layer_with_weights-16
layer-30
 layer_with_weights-17
 layer-31
!layer-32
"layer_with_weights-18
"layer-33
#layer_with_weights-19
#layer-34
$layer_with_weights-20
$layer-35
%	variables
&regularization_losses
'trainable_variables
(	keras_api
)
signatures
Ц_default_save_signature
Ч__call__
+Ш&call_and_return_all_conditional_losses"®Њ
_tf_keras_networkЛЊ{"class_name": "Functional", "name": "Generator", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Generator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 160, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Gen_Input"}, "name": "Gen_Input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "Gen_Conv_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Gen_Conv_1", "inbound_nodes": [[["Gen_Input", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Gen_MP_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Gen_MP_1", "inbound_nodes": [[["Gen_Conv_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Gen_BN_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Gen_BN_1", "inbound_nodes": [[["Gen_MP_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Gen_Conv_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Gen_Conv_2", "inbound_nodes": [[["Gen_BN_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Gen_MP_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Gen_MP_2", "inbound_nodes": [[["Gen_Conv_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Gen_BN_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Gen_BN_2", "inbound_nodes": [[["Gen_MP_2", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "Gen_SPD_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "Gen_SPD_1", "inbound_nodes": [[["Gen_BN_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Gen_Conv_3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Gen_Conv_3", "inbound_nodes": [[["Gen_SPD_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Gen_MP_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Gen_MP_3", "inbound_nodes": [[["Gen_Conv_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Gen_BN_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Gen_BN_3", "inbound_nodes": [[["Gen_MP_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Gen_Conv_4", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Gen_Conv_4", "inbound_nodes": [[["Gen_BN_3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Gen_MP_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Gen_MP_4", "inbound_nodes": [[["Gen_Conv_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Gen_BN_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Gen_BN_4", "inbound_nodes": [[["Gen_MP_4", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "Gen_SPD_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "Gen_SPD_2", "inbound_nodes": [[["Gen_BN_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Gen_Conv_5", "trainable": true, "dtype": "float32", "filters": 1024, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Gen_Conv_5", "inbound_nodes": [[["Gen_SPD_2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Gen_MP_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Gen_MP_5", "inbound_nodes": [[["Gen_Conv_5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Gen_BN_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Gen_BN_5", "inbound_nodes": [[["Gen_MP_5", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "Gen_SPD_3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "Gen_SPD_3", "inbound_nodes": [[["Gen_BN_5", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Gen_Conv_T_1", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Gen_Conv_T_1", "inbound_nodes": [[["Gen_SPD_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Gen_BN_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Gen_BN_6", "inbound_nodes": [[["Gen_Conv_T_1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Gen_Concat_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Gen_Concat_1", "inbound_nodes": [[["Gen_BN_6", 0, 0, {}], ["Gen_BN_4", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "Gen_SPD_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "Gen_SPD_4", "inbound_nodes": [[["Gen_Concat_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Gen_Conv_T_2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Gen_Conv_T_2", "inbound_nodes": [[["Gen_SPD_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Gen_BN_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Gen_BN_7", "inbound_nodes": [[["Gen_Conv_T_2", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Gen_Concat_2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Gen_Concat_2", "inbound_nodes": [[["Gen_BN_7", 0, 0, {}], ["Gen_BN_3", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Gen_Conv_T_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Gen_Conv_T_3", "inbound_nodes": [[["Gen_Concat_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Gen_BN_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Gen_BN_8", "inbound_nodes": [[["Gen_Conv_T_3", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Gen_Concat_3", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Gen_Concat_3", "inbound_nodes": [[["Gen_BN_8", 0, 0, {}], ["Gen_BN_2", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "Gen_SPD_5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "Gen_SPD_5", "inbound_nodes": [[["Gen_Concat_3", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Gen_Conv_T_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Gen_Conv_T_4", "inbound_nodes": [[["Gen_SPD_5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Gen_BN_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Gen_BN_9", "inbound_nodes": [[["Gen_Conv_T_4", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Gen_Concat_4", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Gen_Concat_4", "inbound_nodes": [[["Gen_BN_9", 0, 0, {}], ["Gen_BN_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Gen_Conv_T_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_9", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Gen_Conv_T_5", "inbound_nodes": [[["Gen_Concat_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Gen_BN_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Gen_BN_10", "inbound_nodes": [[["Gen_Conv_T_5", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Gen_Conv_T_6", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Gen_Conv_T_6", "inbound_nodes": [[["Gen_BN_10", 0, 0, {}]]]}], "input_layers": [["Gen_Input", 0, 0]], "output_layers": [["Gen_Conv_T_6", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 192, 160, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Generator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 160, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Gen_Input"}, "name": "Gen_Input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "Gen_Conv_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Gen_Conv_1", "inbound_nodes": [[["Gen_Input", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Gen_MP_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Gen_MP_1", "inbound_nodes": [[["Gen_Conv_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Gen_BN_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Gen_BN_1", "inbound_nodes": [[["Gen_MP_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Gen_Conv_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Gen_Conv_2", "inbound_nodes": [[["Gen_BN_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Gen_MP_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Gen_MP_2", "inbound_nodes": [[["Gen_Conv_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Gen_BN_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Gen_BN_2", "inbound_nodes": [[["Gen_MP_2", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "Gen_SPD_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "Gen_SPD_1", "inbound_nodes": [[["Gen_BN_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Gen_Conv_3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Gen_Conv_3", "inbound_nodes": [[["Gen_SPD_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Gen_MP_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Gen_MP_3", "inbound_nodes": [[["Gen_Conv_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Gen_BN_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Gen_BN_3", "inbound_nodes": [[["Gen_MP_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Gen_Conv_4", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Gen_Conv_4", "inbound_nodes": [[["Gen_BN_3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Gen_MP_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Gen_MP_4", "inbound_nodes": [[["Gen_Conv_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Gen_BN_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Gen_BN_4", "inbound_nodes": [[["Gen_MP_4", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "Gen_SPD_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "Gen_SPD_2", "inbound_nodes": [[["Gen_BN_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Gen_Conv_5", "trainable": true, "dtype": "float32", "filters": 1024, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Gen_Conv_5", "inbound_nodes": [[["Gen_SPD_2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Gen_MP_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Gen_MP_5", "inbound_nodes": [[["Gen_Conv_5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Gen_BN_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Gen_BN_5", "inbound_nodes": [[["Gen_MP_5", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "Gen_SPD_3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "Gen_SPD_3", "inbound_nodes": [[["Gen_BN_5", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Gen_Conv_T_1", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Gen_Conv_T_1", "inbound_nodes": [[["Gen_SPD_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Gen_BN_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Gen_BN_6", "inbound_nodes": [[["Gen_Conv_T_1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Gen_Concat_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Gen_Concat_1", "inbound_nodes": [[["Gen_BN_6", 0, 0, {}], ["Gen_BN_4", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "Gen_SPD_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "Gen_SPD_4", "inbound_nodes": [[["Gen_Concat_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Gen_Conv_T_2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Gen_Conv_T_2", "inbound_nodes": [[["Gen_SPD_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Gen_BN_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Gen_BN_7", "inbound_nodes": [[["Gen_Conv_T_2", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Gen_Concat_2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Gen_Concat_2", "inbound_nodes": [[["Gen_BN_7", 0, 0, {}], ["Gen_BN_3", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Gen_Conv_T_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Gen_Conv_T_3", "inbound_nodes": [[["Gen_Concat_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Gen_BN_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Gen_BN_8", "inbound_nodes": [[["Gen_Conv_T_3", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Gen_Concat_3", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Gen_Concat_3", "inbound_nodes": [[["Gen_BN_8", 0, 0, {}], ["Gen_BN_2", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "Gen_SPD_5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "Gen_SPD_5", "inbound_nodes": [[["Gen_Concat_3", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Gen_Conv_T_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Gen_Conv_T_4", "inbound_nodes": [[["Gen_SPD_5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Gen_BN_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Gen_BN_9", "inbound_nodes": [[["Gen_Conv_T_4", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Gen_Concat_4", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Gen_Concat_4", "inbound_nodes": [[["Gen_BN_9", 0, 0, {}], ["Gen_BN_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Gen_Conv_T_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_9", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Gen_Conv_T_5", "inbound_nodes": [[["Gen_Concat_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Gen_BN_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Gen_BN_10", "inbound_nodes": [[["Gen_Conv_T_5", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Gen_Conv_T_6", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Gen_Conv_T_6", "inbound_nodes": [[["Gen_BN_10", 0, 0, {}]]]}], "input_layers": [["Gen_Input", 0, 0]], "output_layers": [["Gen_Conv_T_6", 0, 0]]}}}
Б"ю
_tf_keras_input_layerё{"class_name": "InputLayer", "name": "Gen_Input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 160, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 160, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Gen_Input"}}
§
*
activation

+kernel
,bias
-	variables
.regularization_losses
/trainable_variables
0	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses"н	
_tf_keras_layer”	{"class_name": "Conv2D", "name": "Gen_Conv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_Conv_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 192, 160, 3]}}
у
1	variables
2regularization_losses
3trainable_variables
4	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses"в
_tf_keras_layer»{"class_name": "MaxPooling2D", "name": "Gen_MP_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_MP_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ґ	
5axis
	6gamma
7beta
8moving_mean
9moving_variance
:	variables
;regularization_losses
<trainable_variables
=	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses"ћ
_tf_keras_layer≤{"class_name": "BatchNormalization", "name": "Gen_BN_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_BN_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 80, 64]}}
І
>
activation

?kernel
@bias
A	variables
Bregularization_losses
Ctrainable_variables
D	keras_api
Я__call__
+†&call_and_return_all_conditional_losses"р	
_tf_keras_layer÷	{"class_name": "Conv2D", "name": "Gen_Conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_Conv_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 80, 64]}}
у
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
°__call__
+Ґ&call_and_return_all_conditional_losses"в
_tf_keras_layer»{"class_name": "MaxPooling2D", "name": "Gen_MP_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_MP_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
§	
Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
£__call__
+§&call_and_return_all_conditional_losses"ќ
_tf_keras_layerі{"class_name": "BatchNormalization", "name": "Gen_BN_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_BN_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 40, 128]}}
А
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
•__call__
+¶&call_and_return_all_conditional_losses"п
_tf_keras_layer’{"class_name": "SpatialDropout2D", "name": "Gen_SPD_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_SPD_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
©
V
activation

Wkernel
Xbias
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
І__call__
+®&call_and_return_all_conditional_losses"т	
_tf_keras_layerЎ	{"class_name": "Conv2D", "name": "Gen_Conv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_Conv_3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 40, 128]}}
у
]	variables
^regularization_losses
_trainable_variables
`	keras_api
©__call__
+™&call_and_return_all_conditional_losses"в
_tf_keras_layer»{"class_name": "MaxPooling2D", "name": "Gen_MP_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_MP_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
§	
aaxis
	bgamma
cbeta
dmoving_mean
emoving_variance
f	variables
gregularization_losses
htrainable_variables
i	keras_api
Ђ__call__
+ђ&call_and_return_all_conditional_losses"ќ
_tf_keras_layerі{"class_name": "BatchNormalization", "name": "Gen_BN_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_BN_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 20, 256]}}
©
j
activation

kkernel
lbias
m	variables
nregularization_losses
otrainable_variables
p	keras_api
≠__call__
+Ѓ&call_and_return_all_conditional_losses"т	
_tf_keras_layerЎ	{"class_name": "Conv2D", "name": "Gen_Conv_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_Conv_4", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 20, 256]}}
у
q	variables
rregularization_losses
strainable_variables
t	keras_api
ѓ__call__
+∞&call_and_return_all_conditional_losses"в
_tf_keras_layer»{"class_name": "MaxPooling2D", "name": "Gen_MP_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_MP_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
§	
uaxis
	vgamma
wbeta
xmoving_mean
ymoving_variance
z	variables
{regularization_losses
|trainable_variables
}	keras_api
±__call__
+≤&call_and_return_all_conditional_losses"ќ
_tf_keras_layerі{"class_name": "BatchNormalization", "name": "Gen_BN_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_BN_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 10, 512]}}
В
~	variables
regularization_losses
Аtrainable_variables
Б	keras_api
≥__call__
+і&call_and_return_all_conditional_losses"п
_tf_keras_layer’{"class_name": "SpatialDropout2D", "name": "Gen_SPD_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_SPD_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
±
В
activation
Гkernel
	Дbias
Е	variables
Жregularization_losses
Зtrainable_variables
И	keras_api
µ__call__
+ґ&call_and_return_all_conditional_losses"у	
_tf_keras_layerў	{"class_name": "Conv2D", "name": "Gen_Conv_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_Conv_5", "trainable": true, "dtype": "float32", "filters": 1024, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 10, 512]}}
ч
Й	variables
Кregularization_losses
Лtrainable_variables
М	keras_api
Ј__call__
+Є&call_and_return_all_conditional_losses"в
_tf_keras_layer»{"class_name": "MaxPooling2D", "name": "Gen_MP_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_MP_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
≠	
	Нaxis

Оgamma
	Пbeta
Рmoving_mean
Сmoving_variance
Т	variables
Уregularization_losses
Фtrainable_variables
Х	keras_api
є__call__
+Ї&call_and_return_all_conditional_losses"ќ
_tf_keras_layerі{"class_name": "BatchNormalization", "name": "Gen_BN_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_BN_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 5, 1024]}}
Д
Ц	variables
Чregularization_losses
Шtrainable_variables
Щ	keras_api
ї__call__
+Љ&call_and_return_all_conditional_losses"п
_tf_keras_layer’{"class_name": "SpatialDropout2D", "name": "Gen_SPD_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_SPD_3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
’
Ъ
activation
Ыkernel
	Ьbias
Э	variables
Юregularization_losses
Яtrainable_variables
†	keras_api
љ__call__
+Њ&call_and_return_all_conditional_losses"Ч

_tf_keras_layerэ	{"class_name": "Conv2DTranspose", "name": "Gen_Conv_T_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_Conv_T_1", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 5, 1024]}}
≠	
	°axis

Ґgamma
	£beta
§moving_mean
•moving_variance
¶	variables
Іregularization_losses
®trainable_variables
©	keras_api
њ__call__
+ј&call_and_return_all_conditional_losses"ќ
_tf_keras_layerі{"class_name": "BatchNormalization", "name": "Gen_BN_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_BN_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 10, 512]}}
г
™	variables
Ђregularization_losses
ђtrainable_variables
≠	keras_api
Ѕ__call__
+¬&call_and_return_all_conditional_losses"ќ
_tf_keras_layerі{"class_name": "Concatenate", "name": "Gen_Concat_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_Concat_1", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 12, 10, 512]}, {"class_name": "TensorShape", "items": [null, 12, 10, 512]}]}
Д
Ѓ	variables
ѓregularization_losses
∞trainable_variables
±	keras_api
√__call__
+ƒ&call_and_return_all_conditional_losses"п
_tf_keras_layer’{"class_name": "SpatialDropout2D", "name": "Gen_SPD_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_SPD_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
„
≤
activation
≥kernel
	іbias
µ	variables
ґregularization_losses
Јtrainable_variables
Є	keras_api
≈__call__
+∆&call_and_return_all_conditional_losses"Щ

_tf_keras_layer€	{"class_name": "Conv2DTranspose", "name": "Gen_Conv_T_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_Conv_T_2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 10, 1024]}}
≠	
	єaxis

Їgamma
	їbeta
Љmoving_mean
љmoving_variance
Њ	variables
њregularization_losses
јtrainable_variables
Ѕ	keras_api
«__call__
+»&call_and_return_all_conditional_losses"ќ
_tf_keras_layerі{"class_name": "BatchNormalization", "name": "Gen_BN_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_BN_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 20, 256]}}
г
¬	variables
√regularization_losses
ƒtrainable_variables
≈	keras_api
…__call__
+ &call_and_return_all_conditional_losses"ќ
_tf_keras_layerі{"class_name": "Concatenate", "name": "Gen_Concat_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_Concat_2", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 24, 20, 256]}, {"class_name": "TensorShape", "items": [null, 24, 20, 256]}]}
’
∆
activation
«kernel
	»bias
…	variables
 regularization_losses
Ћtrainable_variables
ћ	keras_api
Ћ__call__
+ћ&call_and_return_all_conditional_losses"Ч

_tf_keras_layerэ	{"class_name": "Conv2DTranspose", "name": "Gen_Conv_T_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_Conv_T_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 20, 512]}}
≠	
	Ќaxis

ќgamma
	ѕbeta
–moving_mean
—moving_variance
“	variables
”regularization_losses
‘trainable_variables
’	keras_api
Ќ__call__
+ќ&call_and_return_all_conditional_losses"ќ
_tf_keras_layerі{"class_name": "BatchNormalization", "name": "Gen_BN_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_BN_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 40, 128]}}
г
÷	variables
„regularization_losses
Ўtrainable_variables
ў	keras_api
ѕ__call__
+–&call_and_return_all_conditional_losses"ќ
_tf_keras_layerі{"class_name": "Concatenate", "name": "Gen_Concat_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_Concat_3", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 48, 40, 128]}, {"class_name": "TensorShape", "items": [null, 48, 40, 128]}]}
Д
Џ	variables
џregularization_losses
№trainable_variables
Ё	keras_api
—__call__
+“&call_and_return_all_conditional_losses"п
_tf_keras_layer’{"class_name": "SpatialDropout2D", "name": "Gen_SPD_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_SPD_5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
‘
ё
activation
яkernel
	аbias
б	variables
вregularization_losses
гtrainable_variables
д	keras_api
”__call__
+‘&call_and_return_all_conditional_losses"Ц

_tf_keras_layerь	{"class_name": "Conv2DTranspose", "name": "Gen_Conv_T_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_Conv_T_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 40, 256]}}
Ђ	
	еaxis

жgamma
	зbeta
иmoving_mean
йmoving_variance
к	variables
лregularization_losses
мtrainable_variables
н	keras_api
’__call__
+÷&call_and_return_all_conditional_losses"ћ
_tf_keras_layer≤{"class_name": "BatchNormalization", "name": "Gen_BN_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_BN_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 80, 64]}}
б
о	variables
пregularization_losses
рtrainable_variables
с	keras_api
„__call__
+Ў&call_and_return_all_conditional_losses"ћ
_tf_keras_layer≤{"class_name": "Concatenate", "name": "Gen_Concat_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_Concat_4", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 96, 80, 64]}, {"class_name": "TensorShape", "items": [null, 96, 80, 64]}]}
‘
т
activation
уkernel
	фbias
х	variables
цregularization_losses
чtrainable_variables
ш	keras_api
ў__call__
+Џ&call_and_return_all_conditional_losses"Ц

_tf_keras_layerь	{"class_name": "Conv2DTranspose", "name": "Gen_Conv_T_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_Conv_T_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_9", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}, "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 80, 128]}}
ѓ	
	щaxis

ъgamma
	ыbeta
ьmoving_mean
эmoving_variance
ю	variables
€regularization_losses
Аtrainable_variables
Б	keras_api
џ__call__
+№&call_and_return_all_conditional_losses"–
_tf_keras_layerґ{"class_name": "BatchNormalization", "name": "Gen_BN_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_BN_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 192, 160, 32]}}
√

Вkernel
	Гbias
Д	variables
Еregularization_losses
Жtrainable_variables
З	keras_api
Ё__call__
+ё&call_and_return_all_conditional_losses"Ц	
_tf_keras_layerь{"class_name": "Conv2DTranspose", "name": "Gen_Conv_T_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Gen_Conv_T_6", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "TruncatedNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 192, 160, 32]}}
ђ
+0
,1
62
73
84
95
?6
@7
J8
K9
L10
M11
W12
X13
b14
c15
d16
e17
k18
l19
v20
w21
x22
y23
Г24
Д25
О26
П27
Р28
С29
Ы30
Ь31
Ґ32
£33
§34
•35
≥36
і37
Ї38
ї39
Љ40
љ41
«42
»43
ќ44
ѕ45
–46
—47
я48
а49
ж50
з51
и52
й53
у54
ф55
ъ56
ы57
ь58
э59
В60
Г61"
trackable_list_wrapper
 "
trackable_list_wrapper
А
+0
,1
62
73
?4
@5
J6
K7
W8
X9
b10
c11
k12
l13
v14
w15
Г16
Д17
О18
П19
Ы20
Ь21
Ґ22
£23
≥24
і25
Ї26
ї27
«28
»29
ќ30
ѕ31
я32
а33
ж34
з35
у36
ф37
ъ38
ы39
В40
Г41"
trackable_list_wrapper
”
%	variables
 Иlayer_regularization_losses
Йlayer_metrics
Кmetrics
Лlayers
Мnon_trainable_variables
&regularization_losses
'trainable_variables
Ч__call__
Ц_default_save_signature
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
-
яserving_default"
signature_map
а
Н	variables
Оregularization_losses
Пtrainable_variables
Р	keras_api
а__call__
+б&call_and_return_all_conditional_losses"Ћ
_tf_keras_layer±{"class_name": "LeakyReLU", "name": "leaky_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
+:)@2Gen_Conv_1/kernel
:@2Gen_Conv_1/bias
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
µ
-	variables
 Сlayer_regularization_losses
Тlayer_metrics
Уmetrics
Фlayers
Хnon_trainable_variables
.regularization_losses
/trainable_variables
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
1	variables
 Цlayer_regularization_losses
Чlayer_metrics
Шmetrics
Щlayers
Ъnon_trainable_variables
2regularization_losses
3trainable_variables
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:@2Gen_BN_1/gamma
:@2Gen_BN_1/beta
$:"@ (2Gen_BN_1/moving_mean
(:&@ (2Gen_BN_1/moving_variance
<
60
71
82
93"
trackable_list_wrapper
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
µ
:	variables
 Ыlayer_regularization_losses
Ьlayer_metrics
Эmetrics
Юlayers
Яnon_trainable_variables
;regularization_losses
<trainable_variables
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
д
†	variables
°regularization_losses
Ґtrainable_variables
£	keras_api
в__call__
+г&call_and_return_all_conditional_losses"ѕ
_tf_keras_layerµ{"class_name": "LeakyReLU", "name": "leaky_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
,:*@А2Gen_Conv_2/kernel
:А2Gen_Conv_2/bias
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
µ
A	variables
 §layer_regularization_losses
•layer_metrics
¶metrics
Іlayers
®non_trainable_variables
Bregularization_losses
Ctrainable_variables
Я__call__
+†&call_and_return_all_conditional_losses
'†"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
E	variables
 ©layer_regularization_losses
™layer_metrics
Ђmetrics
ђlayers
≠non_trainable_variables
Fregularization_losses
Gtrainable_variables
°__call__
+Ґ&call_and_return_all_conditional_losses
'Ґ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:А2Gen_BN_2/gamma
:А2Gen_BN_2/beta
%:#А (2Gen_BN_2/moving_mean
):'А (2Gen_BN_2/moving_variance
<
J0
K1
L2
M3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
µ
N	variables
 Ѓlayer_regularization_losses
ѓlayer_metrics
∞metrics
±layers
≤non_trainable_variables
Oregularization_losses
Ptrainable_variables
£__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
R	variables
 ≥layer_regularization_losses
іlayer_metrics
µmetrics
ґlayers
Јnon_trainable_variables
Sregularization_losses
Ttrainable_variables
•__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
д
Є	variables
єregularization_losses
Їtrainable_variables
ї	keras_api
д__call__
+е&call_and_return_all_conditional_losses"ѕ
_tf_keras_layerµ{"class_name": "LeakyReLU", "name": "leaky_re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
-:+АА2Gen_Conv_3/kernel
:А2Gen_Conv_3/bias
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
µ
Y	variables
 Љlayer_regularization_losses
љlayer_metrics
Њmetrics
њlayers
јnon_trainable_variables
Zregularization_losses
[trainable_variables
І__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
]	variables
 Ѕlayer_regularization_losses
¬layer_metrics
√metrics
ƒlayers
≈non_trainable_variables
^regularization_losses
_trainable_variables
©__call__
+™&call_and_return_all_conditional_losses
'™"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:А2Gen_BN_3/gamma
:А2Gen_BN_3/beta
%:#А (2Gen_BN_3/moving_mean
):'А (2Gen_BN_3/moving_variance
<
b0
c1
d2
e3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
µ
f	variables
 ∆layer_regularization_losses
«layer_metrics
»metrics
…layers
 non_trainable_variables
gregularization_losses
htrainable_variables
Ђ__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
д
Ћ	variables
ћregularization_losses
Ќtrainable_variables
ќ	keras_api
ж__call__
+з&call_and_return_all_conditional_losses"ѕ
_tf_keras_layerµ{"class_name": "LeakyReLU", "name": "leaky_re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
-:+АА2Gen_Conv_4/kernel
:А2Gen_Conv_4/bias
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
µ
m	variables
 ѕlayer_regularization_losses
–layer_metrics
—metrics
“layers
”non_trainable_variables
nregularization_losses
otrainable_variables
≠__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
q	variables
 ‘layer_regularization_losses
’layer_metrics
÷metrics
„layers
Ўnon_trainable_variables
rregularization_losses
strainable_variables
ѓ__call__
+∞&call_and_return_all_conditional_losses
'∞"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:А2Gen_BN_4/gamma
:А2Gen_BN_4/beta
%:#А (2Gen_BN_4/moving_mean
):'А (2Gen_BN_4/moving_variance
<
v0
w1
x2
y3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
µ
z	variables
 ўlayer_regularization_losses
Џlayer_metrics
џmetrics
№layers
Ёnon_trainable_variables
{regularization_losses
|trainable_variables
±__call__
+≤&call_and_return_all_conditional_losses
'≤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ґ
~	variables
 ёlayer_regularization_losses
яlayer_metrics
аmetrics
бlayers
вnon_trainable_variables
regularization_losses
Аtrainable_variables
≥__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses"
_generic_user_object
д
г	variables
дregularization_losses
еtrainable_variables
ж	keras_api
и__call__
+й&call_and_return_all_conditional_losses"ѕ
_tf_keras_layerµ{"class_name": "LeakyReLU", "name": "leaky_re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
-:+АА2Gen_Conv_5/kernel
:А2Gen_Conv_5/bias
0
Г0
Д1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Г0
Д1"
trackable_list_wrapper
Є
Е	variables
 зlayer_regularization_losses
иlayer_metrics
йmetrics
кlayers
лnon_trainable_variables
Жregularization_losses
Зtrainable_variables
µ__call__
+ґ&call_and_return_all_conditional_losses
'ґ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Й	variables
 мlayer_regularization_losses
нlayer_metrics
оmetrics
пlayers
рnon_trainable_variables
Кregularization_losses
Лtrainable_variables
Ј__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:А2Gen_BN_5/gamma
:А2Gen_BN_5/beta
%:#А (2Gen_BN_5/moving_mean
):'А (2Gen_BN_5/moving_variance
@
О0
П1
Р2
С3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
О0
П1"
trackable_list_wrapper
Є
Т	variables
 сlayer_regularization_losses
тlayer_metrics
уmetrics
фlayers
хnon_trainable_variables
Уregularization_losses
Фtrainable_variables
є__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ц	variables
 цlayer_regularization_losses
чlayer_metrics
шmetrics
щlayers
ъnon_trainable_variables
Чregularization_losses
Шtrainable_variables
ї__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
д
ы	variables
ьregularization_losses
эtrainable_variables
ю	keras_api
к__call__
+л&call_and_return_all_conditional_losses"ѕ
_tf_keras_layerµ{"class_name": "LeakyReLU", "name": "leaky_re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
/:-АА2Gen_Conv_T_1/kernel
 :А2Gen_Conv_T_1/bias
0
Ы0
Ь1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ы0
Ь1"
trackable_list_wrapper
Є
Э	variables
 €layer_regularization_losses
Аlayer_metrics
Бmetrics
Вlayers
Гnon_trainable_variables
Юregularization_losses
Яtrainable_variables
љ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:А2Gen_BN_6/gamma
:А2Gen_BN_6/beta
%:#А (2Gen_BN_6/moving_mean
):'А (2Gen_BN_6/moving_variance
@
Ґ0
£1
§2
•3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ґ0
£1"
trackable_list_wrapper
Є
¶	variables
 Дlayer_regularization_losses
Еlayer_metrics
Жmetrics
Зlayers
Иnon_trainable_variables
Іregularization_losses
®trainable_variables
њ__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
™	variables
 Йlayer_regularization_losses
Кlayer_metrics
Лmetrics
Мlayers
Нnon_trainable_variables
Ђregularization_losses
ђtrainable_variables
Ѕ__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ѓ	variables
 Оlayer_regularization_losses
Пlayer_metrics
Рmetrics
Сlayers
Тnon_trainable_variables
ѓregularization_losses
∞trainable_variables
√__call__
+ƒ&call_and_return_all_conditional_losses
'ƒ"call_and_return_conditional_losses"
_generic_user_object
д
У	variables
Фregularization_losses
Хtrainable_variables
Ц	keras_api
м__call__
+н&call_and_return_all_conditional_losses"ѕ
_tf_keras_layerµ{"class_name": "LeakyReLU", "name": "leaky_re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
/:-АА2Gen_Conv_T_2/kernel
 :А2Gen_Conv_T_2/bias
0
≥0
і1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
≥0
і1"
trackable_list_wrapper
Є
µ	variables
 Чlayer_regularization_losses
Шlayer_metrics
Щmetrics
Ъlayers
Ыnon_trainable_variables
ґregularization_losses
Јtrainable_variables
≈__call__
+∆&call_and_return_all_conditional_losses
'∆"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:А2Gen_BN_7/gamma
:А2Gen_BN_7/beta
%:#А (2Gen_BN_7/moving_mean
):'А (2Gen_BN_7/moving_variance
@
Ї0
ї1
Љ2
љ3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ї0
ї1"
trackable_list_wrapper
Є
Њ	variables
 Ьlayer_regularization_losses
Эlayer_metrics
Юmetrics
Яlayers
†non_trainable_variables
њregularization_losses
јtrainable_variables
«__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
¬	variables
 °layer_regularization_losses
Ґlayer_metrics
£metrics
§layers
•non_trainable_variables
√regularization_losses
ƒtrainable_variables
…__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
д
¶	variables
Іregularization_losses
®trainable_variables
©	keras_api
о__call__
+п&call_and_return_all_conditional_losses"ѕ
_tf_keras_layerµ{"class_name": "LeakyReLU", "name": "leaky_re_lu_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
/:-АА2Gen_Conv_T_3/kernel
 :А2Gen_Conv_T_3/bias
0
«0
»1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
«0
»1"
trackable_list_wrapper
Є
…	variables
 ™layer_regularization_losses
Ђlayer_metrics
ђmetrics
≠layers
Ѓnon_trainable_variables
 regularization_losses
Ћtrainable_variables
Ћ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:А2Gen_BN_8/gamma
:А2Gen_BN_8/beta
%:#А (2Gen_BN_8/moving_mean
):'А (2Gen_BN_8/moving_variance
@
ќ0
ѕ1
–2
—3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
ќ0
ѕ1"
trackable_list_wrapper
Є
“	variables
 ѓlayer_regularization_losses
∞layer_metrics
±metrics
≤layers
≥non_trainable_variables
”regularization_losses
‘trainable_variables
Ќ__call__
+ќ&call_and_return_all_conditional_losses
'ќ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
÷	variables
 іlayer_regularization_losses
µlayer_metrics
ґmetrics
Јlayers
Єnon_trainable_variables
„regularization_losses
Ўtrainable_variables
ѕ__call__
+–&call_and_return_all_conditional_losses
'–"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Џ	variables
 єlayer_regularization_losses
Їlayer_metrics
їmetrics
Љlayers
љnon_trainable_variables
џregularization_losses
№trainable_variables
—__call__
+“&call_and_return_all_conditional_losses
'“"call_and_return_conditional_losses"
_generic_user_object
д
Њ	variables
њregularization_losses
јtrainable_variables
Ѕ	keras_api
р__call__
+с&call_and_return_all_conditional_losses"ѕ
_tf_keras_layerµ{"class_name": "LeakyReLU", "name": "leaky_re_lu_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
.:,@А2Gen_Conv_T_4/kernel
:@2Gen_Conv_T_4/bias
0
я0
а1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
я0
а1"
trackable_list_wrapper
Є
б	variables
 ¬layer_regularization_losses
√layer_metrics
ƒmetrics
≈layers
∆non_trainable_variables
вregularization_losses
гtrainable_variables
”__call__
+‘&call_and_return_all_conditional_losses
'‘"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:@2Gen_BN_9/gamma
:@2Gen_BN_9/beta
$:"@ (2Gen_BN_9/moving_mean
(:&@ (2Gen_BN_9/moving_variance
@
ж0
з1
и2
й3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
ж0
з1"
trackable_list_wrapper
Є
к	variables
 «layer_regularization_losses
»layer_metrics
…metrics
 layers
Ћnon_trainable_variables
лregularization_losses
мtrainable_variables
’__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
о	variables
 ћlayer_regularization_losses
Ќlayer_metrics
ќmetrics
ѕlayers
–non_trainable_variables
пregularization_losses
рtrainable_variables
„__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
д
—	variables
“regularization_losses
”trainable_variables
‘	keras_api
т__call__
+у&call_and_return_all_conditional_losses"ѕ
_tf_keras_layerµ{"class_name": "LeakyReLU", "name": "leaky_re_lu_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_9", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
.:, А2Gen_Conv_T_5/kernel
: 2Gen_Conv_T_5/bias
0
у0
ф1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
у0
ф1"
trackable_list_wrapper
Є
х	variables
 ’layer_regularization_losses
÷layer_metrics
„metrics
Ўlayers
ўnon_trainable_variables
цregularization_losses
чtrainable_variables
ў__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: 2Gen_BN_10/gamma
: 2Gen_BN_10/beta
%:#  (2Gen_BN_10/moving_mean
):'  (2Gen_BN_10/moving_variance
@
ъ0
ы1
ь2
э3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
ъ0
ы1"
trackable_list_wrapper
Є
ю	variables
 Џlayer_regularization_losses
џlayer_metrics
№metrics
Ёlayers
ёnon_trainable_variables
€regularization_losses
Аtrainable_variables
џ__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
-:+ 2Gen_Conv_T_6/kernel
:2Gen_Conv_T_6/bias
0
В0
Г1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
В0
Г1"
trackable_list_wrapper
Є
Д	variables
 яlayer_regularization_losses
аlayer_metrics
бmetrics
вlayers
гnon_trainable_variables
Еregularization_losses
Жtrainable_variables
Ё__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
ґ
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
25
26
27
28
29
30
 31
!32
"33
#34
$35"
trackable_list_wrapper
¬
80
91
L2
M3
d4
e5
x6
y7
Р8
С9
§10
•11
Љ12
љ13
–14
—15
и16
й17
ь18
э19"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Н	variables
 дlayer_regularization_losses
еlayer_metrics
жmetrics
зlayers
иnon_trainable_variables
Оregularization_losses
Пtrainable_variables
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
*0"
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
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
†	variables
 йlayer_regularization_losses
кlayer_metrics
лmetrics
мlayers
нnon_trainable_variables
°regularization_losses
Ґtrainable_variables
в__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
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
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
L0
M1"
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
Є
Є	variables
 оlayer_regularization_losses
пlayer_metrics
рmetrics
сlayers
тnon_trainable_variables
єregularization_losses
Їtrainable_variables
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
V0"
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
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ћ	variables
 уlayer_regularization_losses
фlayer_metrics
хmetrics
цlayers
чnon_trainable_variables
ћregularization_losses
Ќtrainable_variables
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
j0"
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
.
x0
y1"
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
Є
г	variables
 шlayer_regularization_losses
щlayer_metrics
ъmetrics
ыlayers
ьnon_trainable_variables
дregularization_losses
еtrainable_variables
и__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
В0"
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
0
Р0
С1"
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
Є
ы	variables
 эlayer_regularization_losses
юlayer_metrics
€metrics
Аlayers
Бnon_trainable_variables
ьregularization_losses
эtrainable_variables
к__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
Ъ0"
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
0
§0
•1"
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
Є
У	variables
 Вlayer_regularization_losses
Гlayer_metrics
Дmetrics
Еlayers
Жnon_trainable_variables
Фregularization_losses
Хtrainable_variables
м__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
≤0"
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
0
Љ0
љ1"
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
Є
¶	variables
 Зlayer_regularization_losses
Иlayer_metrics
Йmetrics
Кlayers
Лnon_trainable_variables
Іregularization_losses
®trainable_variables
о__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
∆0"
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
0
–0
—1"
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
Є
Њ	variables
 Мlayer_regularization_losses
Нlayer_metrics
Оmetrics
Пlayers
Рnon_trainable_variables
њregularization_losses
јtrainable_variables
р__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
ё0"
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
0
и0
й1"
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
Є
—	variables
 Сlayer_regularization_losses
Тlayer_metrics
Уmetrics
Фlayers
Хnon_trainable_variables
“regularization_losses
”trainable_variables
т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
т0"
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
0
ь0
э1"
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
л2и
!__inference__wrapped_model_140135¬
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *2Ґ/
-К*
	Gen_Input€€€€€€€€€ј†
ц2у
*__inference_Generator_layer_call_fn_143627
*__inference_Generator_layer_call_fn_144760
*__inference_Generator_layer_call_fn_143335
*__inference_Generator_layer_call_fn_144631ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
в2я
E__inference_Generator_layer_call_and_return_conditional_losses_143042
E__inference_Generator_layer_call_and_return_conditional_losses_144185
E__inference_Generator_layer_call_and_return_conditional_losses_144502
E__inference_Generator_layer_call_and_return_conditional_losses_142879ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
’2“
+__inference_Gen_Conv_1_layer_call_fn_144780Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_Gen_Conv_1_layer_call_and_return_conditional_losses_144771Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
С2О
)__inference_Gen_MP_1_layer_call_fn_140147а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ђ2©
D__inference_Gen_MP_1_layer_call_and_return_conditional_losses_140141а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ж2г
)__inference_Gen_BN_1_layer_call_fn_144895
)__inference_Gen_BN_1_layer_call_fn_144844
)__inference_Gen_BN_1_layer_call_fn_144831
)__inference_Gen_BN_1_layer_call_fn_144908і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
“2ѕ
D__inference_Gen_BN_1_layer_call_and_return_conditional_losses_144818
D__inference_Gen_BN_1_layer_call_and_return_conditional_losses_144864
D__inference_Gen_BN_1_layer_call_and_return_conditional_losses_144800
D__inference_Gen_BN_1_layer_call_and_return_conditional_losses_144882і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
’2“
+__inference_Gen_Conv_2_layer_call_fn_144928Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_Gen_Conv_2_layer_call_and_return_conditional_losses_144919Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
С2О
)__inference_Gen_MP_2_layer_call_fn_140263а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ђ2©
D__inference_Gen_MP_2_layer_call_and_return_conditional_losses_140257а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ж2г
)__inference_Gen_BN_2_layer_call_fn_144979
)__inference_Gen_BN_2_layer_call_fn_144992
)__inference_Gen_BN_2_layer_call_fn_145056
)__inference_Gen_BN_2_layer_call_fn_145043і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
“2ѕ
D__inference_Gen_BN_2_layer_call_and_return_conditional_losses_144948
D__inference_Gen_BN_2_layer_call_and_return_conditional_losses_145012
D__inference_Gen_BN_2_layer_call_and_return_conditional_losses_145030
D__inference_Gen_BN_2_layer_call_and_return_conditional_losses_144966і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
к2з
*__inference_Gen_SPD_1_layer_call_fn_145132
*__inference_Gen_SPD_1_layer_call_fn_145094
*__inference_Gen_SPD_1_layer_call_fn_145127
*__inference_Gen_SPD_1_layer_call_fn_145089і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
÷2”
E__inference_Gen_SPD_1_layer_call_and_return_conditional_losses_145079
E__inference_Gen_SPD_1_layer_call_and_return_conditional_losses_145117
E__inference_Gen_SPD_1_layer_call_and_return_conditional_losses_145122
E__inference_Gen_SPD_1_layer_call_and_return_conditional_losses_145084і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
’2“
+__inference_Gen_Conv_3_layer_call_fn_145152Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_Gen_Conv_3_layer_call_and_return_conditional_losses_145143Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
С2О
)__inference_Gen_MP_3_layer_call_fn_140447а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ђ2©
D__inference_Gen_MP_3_layer_call_and_return_conditional_losses_140441а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ж2г
)__inference_Gen_BN_3_layer_call_fn_145216
)__inference_Gen_BN_3_layer_call_fn_145280
)__inference_Gen_BN_3_layer_call_fn_145267
)__inference_Gen_BN_3_layer_call_fn_145203і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
“2ѕ
D__inference_Gen_BN_3_layer_call_and_return_conditional_losses_145254
D__inference_Gen_BN_3_layer_call_and_return_conditional_losses_145172
D__inference_Gen_BN_3_layer_call_and_return_conditional_losses_145190
D__inference_Gen_BN_3_layer_call_and_return_conditional_losses_145236і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
’2“
+__inference_Gen_Conv_4_layer_call_fn_145300Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_Gen_Conv_4_layer_call_and_return_conditional_losses_145291Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
С2О
)__inference_Gen_MP_4_layer_call_fn_140563а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ђ2©
D__inference_Gen_MP_4_layer_call_and_return_conditional_losses_140557а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ж2г
)__inference_Gen_BN_4_layer_call_fn_145415
)__inference_Gen_BN_4_layer_call_fn_145364
)__inference_Gen_BN_4_layer_call_fn_145428
)__inference_Gen_BN_4_layer_call_fn_145351і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
“2ѕ
D__inference_Gen_BN_4_layer_call_and_return_conditional_losses_145338
D__inference_Gen_BN_4_layer_call_and_return_conditional_losses_145384
D__inference_Gen_BN_4_layer_call_and_return_conditional_losses_145320
D__inference_Gen_BN_4_layer_call_and_return_conditional_losses_145402і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
к2з
*__inference_Gen_SPD_2_layer_call_fn_145504
*__inference_Gen_SPD_2_layer_call_fn_145461
*__inference_Gen_SPD_2_layer_call_fn_145466
*__inference_Gen_SPD_2_layer_call_fn_145499і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
÷2”
E__inference_Gen_SPD_2_layer_call_and_return_conditional_losses_145456
E__inference_Gen_SPD_2_layer_call_and_return_conditional_losses_145494
E__inference_Gen_SPD_2_layer_call_and_return_conditional_losses_145489
E__inference_Gen_SPD_2_layer_call_and_return_conditional_losses_145451і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
’2“
+__inference_Gen_Conv_5_layer_call_fn_145524Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_Gen_Conv_5_layer_call_and_return_conditional_losses_145515Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
С2О
)__inference_Gen_MP_5_layer_call_fn_140747а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ђ2©
D__inference_Gen_MP_5_layer_call_and_return_conditional_losses_140741а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ж2г
)__inference_Gen_BN_5_layer_call_fn_145575
)__inference_Gen_BN_5_layer_call_fn_145588
)__inference_Gen_BN_5_layer_call_fn_145639
)__inference_Gen_BN_5_layer_call_fn_145652і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
“2ѕ
D__inference_Gen_BN_5_layer_call_and_return_conditional_losses_145608
D__inference_Gen_BN_5_layer_call_and_return_conditional_losses_145544
D__inference_Gen_BN_5_layer_call_and_return_conditional_losses_145626
D__inference_Gen_BN_5_layer_call_and_return_conditional_losses_145562і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
к2з
*__inference_Gen_SPD_3_layer_call_fn_145690
*__inference_Gen_SPD_3_layer_call_fn_145685
*__inference_Gen_SPD_3_layer_call_fn_145723
*__inference_Gen_SPD_3_layer_call_fn_145728і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
÷2”
E__inference_Gen_SPD_3_layer_call_and_return_conditional_losses_145718
E__inference_Gen_SPD_3_layer_call_and_return_conditional_losses_145680
E__inference_Gen_SPD_3_layer_call_and_return_conditional_losses_145713
E__inference_Gen_SPD_3_layer_call_and_return_conditional_losses_145675і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Н2К
-__inference_Gen_Conv_T_1_layer_call_fn_140976Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
®2•
H__inference_Gen_Conv_T_1_layer_call_and_return_conditional_losses_140966Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Р2Н
)__inference_Gen_BN_6_layer_call_fn_145792
)__inference_Gen_BN_6_layer_call_fn_145779і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
∆2√
D__inference_Gen_BN_6_layer_call_and_return_conditional_losses_145766
D__inference_Gen_BN_6_layer_call_and_return_conditional_losses_145748і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
„2‘
-__inference_Gen_Concat_1_layer_call_fn_145805Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
H__inference_Gen_Concat_1_layer_call_and_return_conditional_losses_145799Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
к2з
*__inference_Gen_SPD_4_layer_call_fn_145876
*__inference_Gen_SPD_4_layer_call_fn_145881
*__inference_Gen_SPD_4_layer_call_fn_145843
*__inference_Gen_SPD_4_layer_call_fn_145838і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
÷2”
E__inference_Gen_SPD_4_layer_call_and_return_conditional_losses_145866
E__inference_Gen_SPD_4_layer_call_and_return_conditional_losses_145833
E__inference_Gen_SPD_4_layer_call_and_return_conditional_losses_145828
E__inference_Gen_SPD_4_layer_call_and_return_conditional_losses_145871і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Н2К
-__inference_Gen_Conv_T_2_layer_call_fn_141205Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
®2•
H__inference_Gen_Conv_T_2_layer_call_and_return_conditional_losses_141195Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Р2Н
)__inference_Gen_BN_7_layer_call_fn_145932
)__inference_Gen_BN_7_layer_call_fn_145945і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
∆2√
D__inference_Gen_BN_7_layer_call_and_return_conditional_losses_145901
D__inference_Gen_BN_7_layer_call_and_return_conditional_losses_145919і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
„2‘
-__inference_Gen_Concat_2_layer_call_fn_145958Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
H__inference_Gen_Concat_2_layer_call_and_return_conditional_losses_145952Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Н2К
-__inference_Gen_Conv_T_3_layer_call_fn_141366Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
®2•
H__inference_Gen_Conv_T_3_layer_call_and_return_conditional_losses_141356Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Р2Н
)__inference_Gen_BN_8_layer_call_fn_146022
)__inference_Gen_BN_8_layer_call_fn_146009і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
∆2√
D__inference_Gen_BN_8_layer_call_and_return_conditional_losses_145996
D__inference_Gen_BN_8_layer_call_and_return_conditional_losses_145978і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
„2‘
-__inference_Gen_Concat_3_layer_call_fn_146035Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
H__inference_Gen_Concat_3_layer_call_and_return_conditional_losses_146029Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
к2з
*__inference_Gen_SPD_5_layer_call_fn_146106
*__inference_Gen_SPD_5_layer_call_fn_146068
*__inference_Gen_SPD_5_layer_call_fn_146073
*__inference_Gen_SPD_5_layer_call_fn_146111і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
÷2”
E__inference_Gen_SPD_5_layer_call_and_return_conditional_losses_146096
E__inference_Gen_SPD_5_layer_call_and_return_conditional_losses_146101
E__inference_Gen_SPD_5_layer_call_and_return_conditional_losses_146063
E__inference_Gen_SPD_5_layer_call_and_return_conditional_losses_146058і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Н2К
-__inference_Gen_Conv_T_4_layer_call_fn_141595Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
®2•
H__inference_Gen_Conv_T_4_layer_call_and_return_conditional_losses_141585Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Р2Н
)__inference_Gen_BN_9_layer_call_fn_146162
)__inference_Gen_BN_9_layer_call_fn_146175і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
∆2√
D__inference_Gen_BN_9_layer_call_and_return_conditional_losses_146149
D__inference_Gen_BN_9_layer_call_and_return_conditional_losses_146131і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
„2‘
-__inference_Gen_Concat_4_layer_call_fn_146188Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
H__inference_Gen_Concat_4_layer_call_and_return_conditional_losses_146182Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Н2К
-__inference_Gen_Conv_T_5_layer_call_fn_141756Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
®2•
H__inference_Gen_Conv_T_5_layer_call_and_return_conditional_losses_141746Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Т2П
*__inference_Gen_BN_10_layer_call_fn_146252
*__inference_Gen_BN_10_layer_call_fn_146239і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
»2≈
E__inference_Gen_BN_10_layer_call_and_return_conditional_losses_146226
E__inference_Gen_BN_10_layer_call_and_return_conditional_losses_146208і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
М2Й
-__inference_Gen_Conv_T_6_layer_call_fn_141905„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
І2§
H__inference_Gen_Conv_T_6_layer_call_and_return_conditional_losses_141895„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
5B3
$__inference_signature_wrapper_143758	Gen_Input
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ў2’
.__inference_leaky_re_lu_5_layer_call_fn_146262Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
у2р
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_146257Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ў2’
.__inference_leaky_re_lu_6_layer_call_fn_146272Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
у2р
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_146267Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ў2’
.__inference_leaky_re_lu_7_layer_call_fn_146282Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
у2р
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_146277Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ў2’
.__inference_leaky_re_lu_8_layer_call_fn_146292Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
у2р
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_146287Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ў2’
.__inference_leaky_re_lu_9_layer_call_fn_146302Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
у2р
I__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_146297Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 д
E__inference_Gen_BN_10_layer_call_and_return_conditional_losses_146208ЪъыьэMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ д
E__inference_Gen_BN_10_layer_call_and_return_conditional_losses_146226ЪъыьэMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ Љ
*__inference_Gen_BN_10_layer_call_fn_146239НъыьэMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ Љ
*__inference_Gen_BN_10_layer_call_fn_146252НъыьэMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ Ї
D__inference_Gen_BN_1_layer_call_and_return_conditional_losses_144800r6789;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€`P@
p
™ "-Ґ*
#К 
0€€€€€€€€€`P@
Ъ Ї
D__inference_Gen_BN_1_layer_call_and_return_conditional_losses_144818r6789;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€`P@
p 
™ "-Ґ*
#К 
0€€€€€€€€€`P@
Ъ я
D__inference_Gen_BN_1_layer_call_and_return_conditional_losses_144864Ц6789MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ я
D__inference_Gen_BN_1_layer_call_and_return_conditional_losses_144882Ц6789MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ Т
)__inference_Gen_BN_1_layer_call_fn_144831e6789;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€`P@
p
™ " К€€€€€€€€€`P@Т
)__inference_Gen_BN_1_layer_call_fn_144844e6789;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€`P@
p 
™ " К€€€€€€€€€`P@Ј
)__inference_Gen_BN_1_layer_call_fn_144895Й6789MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ј
)__inference_Gen_BN_1_layer_call_fn_144908Й6789MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Љ
D__inference_Gen_BN_2_layer_call_and_return_conditional_losses_144948tJKLM<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€0(А
p
™ ".Ґ+
$К!
0€€€€€€€€€0(А
Ъ Љ
D__inference_Gen_BN_2_layer_call_and_return_conditional_losses_144966tJKLM<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€0(А
p 
™ ".Ґ+
$К!
0€€€€€€€€€0(А
Ъ б
D__inference_Gen_BN_2_layer_call_and_return_conditional_losses_145012ШJKLMNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ б
D__inference_Gen_BN_2_layer_call_and_return_conditional_losses_145030ШJKLMNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ Ф
)__inference_Gen_BN_2_layer_call_fn_144979gJKLM<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€0(А
p
™ "!К€€€€€€€€€0(АФ
)__inference_Gen_BN_2_layer_call_fn_144992gJKLM<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€0(А
p 
™ "!К€€€€€€€€€0(Ає
)__inference_Gen_BN_2_layer_call_fn_145043ЛJKLMNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Ає
)__inference_Gen_BN_2_layer_call_fn_145056ЛJKLMNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аб
D__inference_Gen_BN_3_layer_call_and_return_conditional_losses_145172ШbcdeNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ б
D__inference_Gen_BN_3_layer_call_and_return_conditional_losses_145190ШbcdeNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ Љ
D__inference_Gen_BN_3_layer_call_and_return_conditional_losses_145236tbcde<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ Љ
D__inference_Gen_BN_3_layer_call_and_return_conditional_losses_145254tbcde<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p 
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ є
)__inference_Gen_BN_3_layer_call_fn_145203ЛbcdeNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Ає
)__inference_Gen_BN_3_layer_call_fn_145216ЛbcdeNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€АФ
)__inference_Gen_BN_3_layer_call_fn_145267gbcde<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p
™ "!К€€€€€€€€€АФ
)__inference_Gen_BN_3_layer_call_fn_145280gbcde<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p 
™ "!К€€€€€€€€€Аб
D__inference_Gen_BN_4_layer_call_and_return_conditional_losses_145320ШvwxyNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ б
D__inference_Gen_BN_4_layer_call_and_return_conditional_losses_145338ШvwxyNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ Љ
D__inference_Gen_BN_4_layer_call_and_return_conditional_losses_145384tvwxy<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€
А
p
™ ".Ґ+
$К!
0€€€€€€€€€
А
Ъ Љ
D__inference_Gen_BN_4_layer_call_and_return_conditional_losses_145402tvwxy<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€
А
p 
™ ".Ґ+
$К!
0€€€€€€€€€
А
Ъ є
)__inference_Gen_BN_4_layer_call_fn_145351ЛvwxyNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Ає
)__inference_Gen_BN_4_layer_call_fn_145364ЛvwxyNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€АФ
)__inference_Gen_BN_4_layer_call_fn_145415gvwxy<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€
А
p
™ "!К€€€€€€€€€
АФ
)__inference_Gen_BN_4_layer_call_fn_145428gvwxy<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€
А
p 
™ "!К€€€€€€€€€
Ае
D__inference_Gen_BN_5_layer_call_and_return_conditional_losses_145544ЬОПРСNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ е
D__inference_Gen_BN_5_layer_call_and_return_conditional_losses_145562ЬОПРСNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ ј
D__inference_Gen_BN_5_layer_call_and_return_conditional_losses_145608xОПРС<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ ј
D__inference_Gen_BN_5_layer_call_and_return_conditional_losses_145626xОПРС<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p 
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ љ
)__inference_Gen_BN_5_layer_call_fn_145575ПОПРСNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аљ
)__inference_Gen_BN_5_layer_call_fn_145588ПОПРСNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€АШ
)__inference_Gen_BN_5_layer_call_fn_145639kОПРС<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p
™ "!К€€€€€€€€€АШ
)__inference_Gen_BN_5_layer_call_fn_145652kОПРС<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p 
™ "!К€€€€€€€€€Ае
D__inference_Gen_BN_6_layer_call_and_return_conditional_losses_145748ЬҐ£§•NҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ е
D__inference_Gen_BN_6_layer_call_and_return_conditional_losses_145766ЬҐ£§•NҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ љ
)__inference_Gen_BN_6_layer_call_fn_145779ПҐ£§•NҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аљ
)__inference_Gen_BN_6_layer_call_fn_145792ПҐ£§•NҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Ае
D__inference_Gen_BN_7_layer_call_and_return_conditional_losses_145901ЬЇїЉљNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ е
D__inference_Gen_BN_7_layer_call_and_return_conditional_losses_145919ЬЇїЉљNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ љ
)__inference_Gen_BN_7_layer_call_fn_145932ПЇїЉљNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аљ
)__inference_Gen_BN_7_layer_call_fn_145945ПЇїЉљNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Ае
D__inference_Gen_BN_8_layer_call_and_return_conditional_losses_145978Ьќѕ–—NҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ е
D__inference_Gen_BN_8_layer_call_and_return_conditional_losses_145996Ьќѕ–—NҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ љ
)__inference_Gen_BN_8_layer_call_fn_146009Пќѕ–—NҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аљ
)__inference_Gen_BN_8_layer_call_fn_146022Пќѕ–—NҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аг
D__inference_Gen_BN_9_layer_call_and_return_conditional_losses_146131ЪжзийMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ г
D__inference_Gen_BN_9_layer_call_and_return_conditional_losses_146149ЪжзийMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ї
)__inference_Gen_BN_9_layer_call_fn_146162НжзийMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ї
)__inference_Gen_BN_9_layer_call_fn_146175НжзийMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@э
H__inference_Gen_Concat_1_layer_call_and_return_conditional_losses_145799∞~Ґ{
tҐq
oЪl
=К:
inputs/0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
+К(
inputs/1€€€€€€€€€
А
™ ".Ґ+
$К!
0€€€€€€€€€
А
Ъ ’
-__inference_Gen_Concat_1_layer_call_fn_145805£~Ґ{
tҐq
oЪl
=К:
inputs/0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
+К(
inputs/1€€€€€€€€€
А
™ "!К€€€€€€€€€
Аэ
H__inference_Gen_Concat_2_layer_call_and_return_conditional_losses_145952∞~Ґ{
tҐq
oЪl
=К:
inputs/0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
+К(
inputs/1€€€€€€€€€А
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ ’
-__inference_Gen_Concat_2_layer_call_fn_145958£~Ґ{
tҐq
oЪl
=К:
inputs/0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
+К(
inputs/1€€€€€€€€€А
™ "!К€€€€€€€€€Аэ
H__inference_Gen_Concat_3_layer_call_and_return_conditional_losses_146029∞~Ґ{
tҐq
oЪl
=К:
inputs/0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
+К(
inputs/1€€€€€€€€€0(А
™ ".Ґ+
$К!
0€€€€€€€€€0(А
Ъ ’
-__inference_Gen_Concat_3_layer_call_fn_146035£~Ґ{
tҐq
oЪl
=К:
inputs/0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
+К(
inputs/1€€€€€€€€€0(А
™ "!К€€€€€€€€€0(Аы
H__inference_Gen_Concat_4_layer_call_and_return_conditional_losses_146182Ѓ|Ґy
rҐo
mЪj
<К9
inputs/0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
*К'
inputs/1€€€€€€€€€`P@
™ ".Ґ+
$К!
0€€€€€€€€€`PА
Ъ ”
-__inference_Gen_Concat_4_layer_call_fn_146188°|Ґy
rҐo
mЪj
<К9
inputs/0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
*К'
inputs/1€€€€€€€€€`P@
™ "!К€€€€€€€€€`PАЇ
F__inference_Gen_Conv_1_layer_call_and_return_conditional_losses_144771p+,9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€ј†
™ "/Ґ,
%К"
0€€€€€€€€€ј†@
Ъ Т
+__inference_Gen_Conv_1_layer_call_fn_144780c+,9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€ј†
™ ""К€€€€€€€€€ј†@Ј
F__inference_Gen_Conv_2_layer_call_and_return_conditional_losses_144919m?@7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€`P@
™ ".Ґ+
$К!
0€€€€€€€€€`PА
Ъ П
+__inference_Gen_Conv_2_layer_call_fn_144928`?@7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€`P@
™ "!К€€€€€€€€€`PАЄ
F__inference_Gen_Conv_3_layer_call_and_return_conditional_losses_145143nWX8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€0(А
™ ".Ґ+
$К!
0€€€€€€€€€0(А
Ъ Р
+__inference_Gen_Conv_3_layer_call_fn_145152aWX8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€0(А
™ "!К€€€€€€€€€0(АЄ
F__inference_Gen_Conv_4_layer_call_and_return_conditional_losses_145291nkl8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ Р
+__inference_Gen_Conv_4_layer_call_fn_145300akl8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "!К€€€€€€€€€АЇ
F__inference_Gen_Conv_5_layer_call_and_return_conditional_losses_145515pГД8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€
А
™ ".Ґ+
$К!
0€€€€€€€€€
А
Ъ Т
+__inference_Gen_Conv_5_layer_call_fn_145524cГД8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€
А
™ "!К€€€€€€€€€
Аб
H__inference_Gen_Conv_T_1_layer_call_and_return_conditional_losses_140966ФЫЬJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ є
-__inference_Gen_Conv_T_1_layer_call_fn_140976ЗЫЬJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аб
H__inference_Gen_Conv_T_2_layer_call_and_return_conditional_losses_141195Ф≥іJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ є
-__inference_Gen_Conv_T_2_layer_call_fn_141205З≥іJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аб
H__inference_Gen_Conv_T_3_layer_call_and_return_conditional_losses_141356Ф«»JҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ є
-__inference_Gen_Conv_T_3_layer_call_fn_141366З«»JҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аа
H__inference_Gen_Conv_T_4_layer_call_and_return_conditional_losses_141585УяаJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ Є
-__inference_Gen_Conv_T_4_layer_call_fn_141595ЖяаJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@а
H__inference_Gen_Conv_T_5_layer_call_and_return_conditional_losses_141746УуфJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ Є
-__inference_Gen_Conv_T_5_layer_call_fn_141756ЖуфJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ я
H__inference_Gen_Conv_T_6_layer_call_and_return_conditional_losses_141895ТВГIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ Ј
-__inference_Gen_Conv_T_6_layer_call_fn_141905ЕВГIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€з
D__inference_Gen_MP_1_layer_call_and_return_conditional_losses_140141ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ њ
)__inference_Gen_MP_1_layer_call_fn_140147СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€з
D__inference_Gen_MP_2_layer_call_and_return_conditional_losses_140257ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ њ
)__inference_Gen_MP_2_layer_call_fn_140263СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€з
D__inference_Gen_MP_3_layer_call_and_return_conditional_losses_140441ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ њ
)__inference_Gen_MP_3_layer_call_fn_140447СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€з
D__inference_Gen_MP_4_layer_call_and_return_conditional_losses_140557ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ њ
)__inference_Gen_MP_4_layer_call_fn_140563СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€з
D__inference_Gen_MP_5_layer_call_and_return_conditional_losses_140741ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ њ
)__inference_Gen_MP_5_layer_call_fn_140747СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Ј
E__inference_Gen_SPD_1_layer_call_and_return_conditional_losses_145079n<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€0(А
p
™ ".Ґ+
$К!
0€€€€€€€€€0(А
Ъ Ј
E__inference_Gen_SPD_1_layer_call_and_return_conditional_losses_145084n<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€0(А
p 
™ ".Ґ+
$К!
0€€€€€€€€€0(А
Ъ м
E__inference_Gen_SPD_1_layer_call_and_return_conditional_losses_145117ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ м
E__inference_Gen_SPD_1_layer_call_and_return_conditional_losses_145122ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ П
*__inference_Gen_SPD_1_layer_call_fn_145089a<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€0(А
p
™ "!К€€€€€€€€€0(АП
*__inference_Gen_SPD_1_layer_call_fn_145094a<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€0(А
p 
™ "!К€€€€€€€€€0(Аƒ
*__inference_Gen_SPD_1_layer_call_fn_145127ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€ƒ
*__inference_Gen_SPD_1_layer_call_fn_145132ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Ј
E__inference_Gen_SPD_2_layer_call_and_return_conditional_losses_145451n<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€
А
p
™ ".Ґ+
$К!
0€€€€€€€€€
А
Ъ Ј
E__inference_Gen_SPD_2_layer_call_and_return_conditional_losses_145456n<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€
А
p 
™ ".Ґ+
$К!
0€€€€€€€€€
А
Ъ м
E__inference_Gen_SPD_2_layer_call_and_return_conditional_losses_145489ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ м
E__inference_Gen_SPD_2_layer_call_and_return_conditional_losses_145494ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ П
*__inference_Gen_SPD_2_layer_call_fn_145461a<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€
А
p
™ "!К€€€€€€€€€
АП
*__inference_Gen_SPD_2_layer_call_fn_145466a<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€
А
p 
™ "!К€€€€€€€€€
Аƒ
*__inference_Gen_SPD_2_layer_call_fn_145499ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€ƒ
*__inference_Gen_SPD_2_layer_call_fn_145504ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Ј
E__inference_Gen_SPD_3_layer_call_and_return_conditional_losses_145675n<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ Ј
E__inference_Gen_SPD_3_layer_call_and_return_conditional_losses_145680n<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p 
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ м
E__inference_Gen_SPD_3_layer_call_and_return_conditional_losses_145713ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ м
E__inference_Gen_SPD_3_layer_call_and_return_conditional_losses_145718ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ П
*__inference_Gen_SPD_3_layer_call_fn_145685a<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p
™ "!К€€€€€€€€€АП
*__inference_Gen_SPD_3_layer_call_fn_145690a<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p 
™ "!К€€€€€€€€€Аƒ
*__inference_Gen_SPD_3_layer_call_fn_145723ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€ƒ
*__inference_Gen_SPD_3_layer_call_fn_145728ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€м
E__inference_Gen_SPD_4_layer_call_and_return_conditional_losses_145828ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ м
E__inference_Gen_SPD_4_layer_call_and_return_conditional_losses_145833ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ Ј
E__inference_Gen_SPD_4_layer_call_and_return_conditional_losses_145866n<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€
А
p
™ ".Ґ+
$К!
0€€€€€€€€€
А
Ъ Ј
E__inference_Gen_SPD_4_layer_call_and_return_conditional_losses_145871n<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€
А
p 
™ ".Ґ+
$К!
0€€€€€€€€€
А
Ъ ƒ
*__inference_Gen_SPD_4_layer_call_fn_145838ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€ƒ
*__inference_Gen_SPD_4_layer_call_fn_145843ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€П
*__inference_Gen_SPD_4_layer_call_fn_145876a<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€
А
p
™ "!К€€€€€€€€€
АП
*__inference_Gen_SPD_4_layer_call_fn_145881a<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€
А
p 
™ "!К€€€€€€€€€
Ам
E__inference_Gen_SPD_5_layer_call_and_return_conditional_losses_146058ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ м
E__inference_Gen_SPD_5_layer_call_and_return_conditional_losses_146063ҐVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ Ј
E__inference_Gen_SPD_5_layer_call_and_return_conditional_losses_146096n<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€0(А
p
™ ".Ґ+
$К!
0€€€€€€€€€0(А
Ъ Ј
E__inference_Gen_SPD_5_layer_call_and_return_conditional_losses_146101n<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€0(А
p 
™ ".Ґ+
$К!
0€€€€€€€€€0(А
Ъ ƒ
*__inference_Gen_SPD_5_layer_call_fn_146068ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€ƒ
*__inference_Gen_SPD_5_layer_call_fn_146073ХVҐS
LҐI
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€П
*__inference_Gen_SPD_5_layer_call_fn_146106a<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€0(А
p
™ "!К€€€€€€€€€0(АП
*__inference_Gen_SPD_5_layer_call_fn_146111a<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€0(А
p 
™ "!К€€€€€€€€€0(АЈ
E__inference_Generator_layer_call_and_return_conditional_losses_142879нd+,6789?@JKLMWXbcdeklvwxyГДОПРСЫЬҐ£§•≥іЇїЉљ«»ќѕ–—яажзийуфъыьэВГDҐA
:Ґ7
-К*
	Gen_Input€€€€€€€€€ј†
p

 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ Ј
E__inference_Generator_layer_call_and_return_conditional_losses_143042нd+,6789?@JKLMWXbcdeklvwxyГДОПРСЫЬҐ£§•≥іЇїЉљ«»ќѕ–—яажзийуфъыьэВГDҐA
:Ґ7
-К*
	Gen_Input€€€€€€€€€ј†
p 

 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ §
E__inference_Generator_layer_call_and_return_conditional_losses_144185Џd+,6789?@JKLMWXbcdeklvwxyГДОПРСЫЬҐ£§•≥іЇїЉљ«»ќѕ–—яажзийуфъыьэВГAҐ>
7Ґ4
*К'
inputs€€€€€€€€€ј†
p

 
™ "/Ґ,
%К"
0€€€€€€€€€ј†
Ъ §
E__inference_Generator_layer_call_and_return_conditional_losses_144502Џd+,6789?@JKLMWXbcdeklvwxyГДОПРСЫЬҐ£§•≥іЇїЉљ«»ќѕ–—яажзийуфъыьэВГAҐ>
7Ґ4
*К'
inputs€€€€€€€€€ј†
p 

 
™ "/Ґ,
%К"
0€€€€€€€€€ј†
Ъ П
*__inference_Generator_layer_call_fn_143335аd+,6789?@JKLMWXbcdeklvwxyГДОПРСЫЬҐ£§•≥іЇїЉљ«»ќѕ–—яажзийуфъыьэВГDҐA
:Ґ7
-К*
	Gen_Input€€€€€€€€€ј†
p

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€П
*__inference_Generator_layer_call_fn_143627аd+,6789?@JKLMWXbcdeklvwxyГДОПРСЫЬҐ£§•≥іЇїЉљ«»ќѕ–—яажзийуфъыьэВГDҐA
:Ґ7
-К*
	Gen_Input€€€€€€€€€ј†
p 

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€М
*__inference_Generator_layer_call_fn_144631Ёd+,6789?@JKLMWXbcdeklvwxyГДОПРСЫЬҐ£§•≥іЇїЉљ«»ќѕ–—яажзийуфъыьэВГAҐ>
7Ґ4
*К'
inputs€€€€€€€€€ј†
p

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€М
*__inference_Generator_layer_call_fn_144760Ёd+,6789?@JKLMWXbcdeklvwxyГДОПРСЫЬҐ£§•≥іЇїЉљ«»ќѕ–—яажзийуфъыьэВГAҐ>
7Ґ4
*К'
inputs€€€€€€€€€ј†
p 

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€С
!__inference__wrapped_model_140135лd+,6789?@JKLMWXbcdeklvwxyГДОПРСЫЬҐ£§•≥іЇїЉљ«»ќѕ–—яажзийуфъыьэВГ<Ґ9
2Ґ/
-К*
	Gen_Input€€€€€€€€€ј†
™ "E™B
@
Gen_Conv_T_60К-
Gen_Conv_T_6€€€€€€€€€ј†№
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_146257ОJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ і
.__inference_leaky_re_lu_5_layer_call_fn_146262БJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А№
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_146267ОJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ і
.__inference_leaky_re_lu_6_layer_call_fn_146272БJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А№
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_146277ОJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ і
.__inference_leaky_re_lu_7_layer_call_fn_146282БJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЏ
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_146287МIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ±
.__inference_leaky_re_lu_8_layer_call_fn_146292IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Џ
I__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_146297МIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ ±
.__inference_leaky_re_lu_9_layer_call_fn_146302IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ °
$__inference_signature_wrapper_143758шd+,6789?@JKLMWXbcdeklvwxyГДОПРСЫЬҐ£§•≥іЇїЉљ«»ќѕ–—яажзийуфъыьэВГIҐF
Ґ 
?™<
:
	Gen_Input-К*
	Gen_Input€€€€€€€€€ј†"E™B
@
Gen_Conv_T_60К-
Gen_Conv_T_6€€€€€€€€€ј†