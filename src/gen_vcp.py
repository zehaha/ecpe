import numpy as np

V = input("Enter number of vertices: ")
k = input("Enter k: ")

class Obj:
	def __init__(self, var, count):
		self.var = var
		self.count = count

	def cont(self):
		return self.var + ',' + str(self.count)

class EvolRule:
	def __init__(self, lh, rh):
		self.type = 0
		self.lh = lh
		self.rh = rh

	def get_rh(self):
		tmp = []
		for i in self.rh:
			for j in range(0, i.count): tmp.append(i.var)
		return tmp

class SymRuleIn:
	def __init__(self, obj, enum):
		self.type = 1
		self.obj = obj
		self.enum = enum

class SymRuleOut:
	def __init__(self, obj, enum):
		self.type = 2
		self.obj = obj
		self.enum = enum

class AntiRule:
	def __init__(self, lobj, lenum, robj, renum):
		self.type = 3
		self.lobj = lobj
		self.lenum = lenum
		self.robj = robj
		self.renum = renum

# varsets = []
# rule_set = []
R0 = []
for i in range(0, V):
	if i < V-1:
		R0.append(EvolRule(Obj('A'+str(i+1)+str(i+2),1), [Obj(str(i+1),1), Obj('e',1)]))
		R0.append(EvolRule(Obj('A'+str(i+1)+str(i+2),1), [Obj(str(i+2),1), Obj('e',1)]))
		# varsets.append('A'+str(i+1)+str(i+2))
	R0.append(EvolRule(Obj('v'+str(i+1),1), [Obj('v\''+str(i+1),1), Obj('e',1)]))
	# varsets.append('v'+str(i+1))
R0.append(EvolRule(Obj('#0',1), [Obj('#1',1)]))
R0.append(EvolRule(Obj('#1',1), [Obj('#2',1)]))
R0.append(EvolRule(Obj('#2',1), [Obj('#3',1)]))
R0.append(EvolRule(Obj('#3',1), [Obj('#4',1)]))
R0.append(EvolRule(Obj('#4',1), [Obj('#5',1),  Obj('a0',1),  Obj('b0',1), Obj('e',3)]))
R0.append(EvolRule(Obj('c',1), [Obj('c\'',1), Obj('e',2)]))
R0.append(EvolRule(Obj('d',1), [Obj('d\'',1), Obj('e',1)]))
R0.append(EvolRule(Obj('b2',1), [Obj('yes',1), Obj('e',1)]))
R0.append(EvolRule(Obj('a1',1), [Obj('no',1), Obj('e',1)]))

order = 0

print 'R0:'
for i in range(0, len(R0)):
	temp = ''
	# rule_set.append(R0[i])
	for j in range(0, len(R0[i].rh)): temp += R0[i].rh[j].cont()
	print str(order) + ' ' + R0[i].lh.cont() + '->' + temp
	order += 1

print '\nRp0:'
Rp0 = []
Rp0.append(SymRuleOut(Obj('no',1),1))
Rp0.append(SymRuleOut(Obj('yes',1),1))
print str(order) + ' ('+Rp0[0].obj.cont() + 'e' + str(Rp0[0].enum)+', out)'
print str(order+1) + ' ('+Rp0[1].obj.cont() + 'e' + str(Rp0[0].enum)+', out)'
# rule_set.append(Rp0[0])
# rule_set.append(Rp0[1])
order += 2

print '\nR1:'
R1 = []
for i in range(0, V): 
	R1.append(EvolRule(Obj('v\''+str(i+1),1), [Obj(str(i+1)+'\'',1)]))
	# varsets.append('v\''+str(i+1))
R1.append(EvolRule(Obj('c\'',1), [Obj('e',1)]))

for i in range(0, len(R1)):
	temp = ''
	# rule_set.append(R1[i])
	for j in range(0, len(R1[i].rh)): temp += R1[i].rh[j].cont()
	print str(order) + ' ' + R1[i].lh.cont() + '->' + temp
	order += 1

print '\nRp1:'
Rp1 = []
for i in range(0, V):
	Rp1.append(SymRuleIn(Obj('v\''+str(i+1),1),1))
	Rp1.append(AntiRule(Obj(str(i+1),1),1,Obj(str(i+1)+'\'',1),1))
	# varsets.append(str(i+1)+str(i+1)+'\'')
Rp1.append(SymRuleIn(Obj('c\'',1),1))

for i in range(0, len(Rp1)):
	# rule_set.append(Rp1[i])
	if(Rp1[i].type == 1): print str(order) + ' ('+Rp1[i].obj.cont() + 'e' + str(Rp1[i].enum)+', in)'
	elif(Rp1[i].type == 2): print str(order) + ' ('+Rp1[i].obj.cont() + 'e' + str(Rp1[i].enum)+', out)'
	else: print str(order) + ' ('+Rp1[i].lobj.cont() + 'e' + str(Rp1[i].lenum)+', in;' + Rp1[i].robj.cont() + 'e' + str(Rp1[i].renum)+', out)'
	order += 1

print '\nR2:'
R2 = []
for i in range(0, V):
	R2.append(EvolRule(Obj(str(i+1)+'\'',1),[Obj(str(i+1)+'*',V-2)]))
	# varsets.append(str(i+1)+'\'')
R2.append(EvolRule(Obj('d\'',1), [Obj('e',1)]))
R2.append(EvolRule(Obj('a0',1), [Obj('a1',1)]))

for i in range(0, len(R2)):
	temp = ''
	# rule_set.append(R2[i])
	for j in range(0, len(R2[i].rh)): temp += R2[i].rh[j].cont()
	print str(order) + ' ' + R2[i].lh.cont() + '->' + temp
	order += 1

print '\nRp2:'
Rp2 = []
for i in range(0, V):
	Rp2.append(SymRuleIn(Obj(str(i+1)+'\'',1),1))
	Rp2.append(AntiRule(Obj(str(i+1),1),1,Obj(str(i+1)+'*',1),1))
	# varsets.append(str(i+1)+str(i+1)+'*')
Rp2.append(SymRuleIn(Obj('d\'',1),1))
Rp2.append(SymRuleIn(Obj('a0',1),1))
Rp2.append(AntiRule(Obj('#5',1),1,Obj('a1',1),1))

for i in range(0, len(Rp2)):
	# rule_set.append(Rp2[i])
	if(Rp2[i].type == 1): print str(order) + ' ('+Rp2[i].obj.cont() + 'e' + str(Rp2[i].enum)+', in)'
	elif(Rp2[i].type == 2): print str(order) + ' ('+Rp2[i].obj.cont() + 'e' + str(Rp2[i].enum)+', out)'
	else: print str(order) + ' ('+Rp2[i].lobj.cont() + 'e' + str(Rp2[i].lenum)+', in;' + Rp2[i].robj.cont() + 'e' + str(Rp2[i].renum)+', out)'
	order += 1

print '\nR3:'
R3 = []
R3.append(EvolRule(Obj('b0',1), [Obj('b1',1)]))
R3.append(EvolRule(Obj('b1',1), [Obj('b2',1), Obj('e',1)]))

for i in range(0, len(R3)):
	temp = ''
	# rule_set.append(R3[i])
	for j in range(0, len(R3[i].rh)): temp += R3[i].rh[j].cont()
	print str(order) + ' ' + R3[i].lh.cont() + '->' + temp
	order += 1

print '\nRp3:'
Rp3 = []
Rp3.append(SymRuleIn(Obj('b0',1),1))
Rp3.append(AntiRule(Obj('#5',1),1,Obj('b2',1),1))

for i in range(0, len(Rp3)):
	# rule_set.append(Rp3[i])
	if(Rp3[i].type == 1): print str(order) + ' ('+Rp3[i].obj.cont() + 'e' + str(Rp3[i].enum)+', in)'
	elif(Rp3[i].type == 2): print str(order) + ' ('+Rp3[i].obj.cont() + 'e' + str(Rp3[i].enum)+', out)'
	else: print str(order) + ' ('+Rp3[i].lobj.cont() + 'e' + str(Rp3[i].lenum)+', in;' + Rp3[i].robj.cont() + 'e' + str(Rp3[i].renum)+', out)'
	order += 1

R = []
R.append(R0)
R.append(Rp0)
R.append(R1)
R.append(Rp1)
R.append(R2)
R.append(Rp2)
R.append(R3)
R.append(Rp3)

# [Aij, vi, v'i, i, i', i*, c, c', d, d', #0, #1, #2, #3, #4, #5, a0, a1, b0, b1, b2, no, yes, e]
# 5V + V - 1 + 18 = 6V + 17
temp_objects = []
for i in range(0, V-1): temp_objects.append('A'+str(i+1)+str(i+2)) #Aij
for i in range(0, V): temp_objects.append('v'+str(i+1)) #vi
for i in range(0, V): temp_objects.append('v\''+str(i+1)) #v'i
for i in range(0, V): temp_objects.append(str(i+1)) #i
for i in range(0, V): temp_objects.append(str(i+1)+'\'') #i'
for i in range(0, V): temp_objects.append(str(i+1)+'*') #i*
temp_objects.append('c') #c
temp_objects.append('c\'') #c'
temp_objects.append('d') #d
temp_objects.append('d\'') #d'
temp_objects.append('#0') #0
temp_objects.append('#1') #1
temp_objects.append('#2') #2
temp_objects.append('#3') #3
temp_objects.append('#4') #4
temp_objects.append('#5') #5
temp_objects.append('a0') #a0
temp_objects.append('a1') #a1
temp_objects.append('b0') #b0
temp_objects.append('b1') #b1
temp_objects.append('b2') #b2
temp_objects.append('no') #no
temp_objects.append('yes') #yes
temp_objects.append('e') #e

objects = []
for j in range(0, 4):
	objects.append(temp_objects)

# print objects

def type0(x,y,rule,reg):
	if(x == rule.lh.var and y == reg): return -1 + rule.get_rh().count(x)
	elif(x in rule.get_rh() and y == reg): return rule.get_rh().count(x)
	else: return 0

def type1(x,y,rule,reg,parent):
	if(x == rule.obj.var and y == parent): return -1
	elif(x == 'e' and y == parent): return -(1)*rule.enum
	elif(x == rule.obj.var and y == reg): return 1
	else: return 0

def type2(x,y,rule,reg,parent):
	if(x == rule.obj.var and y == parent): return 1
	elif(x == 'e' and y == reg): return -(1)*rule.enum
	elif(x == rule.obj.var and y == reg): return -1
	else: return 0

def type3(x,y,rule,reg,parent):
	if((x == rule.lobj.var and y == reg) or (x == rule.robj.var and y == parent)): return 1
	elif((x == rule.robj.var and y == reg) or (x == rule.lobj.var and y == parent)): return -1
	elif(x == 'e' and y == parent): return -(1)*rule.lenum
	elif(x == 'e' and y == reg): return -(1)*rule.renum
	else: return 0

trans_matrix = []

length = len(objects[0])
col_size = 4*length

parent = 0
for h in range(0, len(R)):
	reg = h / 2
	if reg == 0: parent = -1
	else: parent = 0
	for i in range(0, len(R[h])):
		temp_arr = [0 for m in range(0, col_size)]
		for j in range(0, 4):
			for l in range(0, len(objects[j])): 
				if(R[h][i].type == 0): temp_arr[j * length + l] = type0(objects[j][l],j,R[h][i],reg)
				elif(R[h][i].type == 1): temp_arr[j * length + l] = type1(objects[j][l],j,R[h][i],reg,parent)
				elif(R[h][i].type == 2): temp_arr[j * length + l] = type2(objects[j][l],j,R[h][i],reg,parent)
				elif(R[h][i].type == 3): temp_arr[j * length + l] = type3(objects[j][l],j,R[h][i],reg,parent)
		trans_matrix.append(temp_arr)

# print trans_matrix
# print 'matrix size: ', len(trans_matrix), 'x', len(trans_matrix[0])

O_objs = 8*V + 12

# [Aij, vi, v'i, i, i', i*, c, c', d, d', #0, #1, #2, #3, #4, #5, a0, a1, b0, b1, b2, no, yes, e]

init_configvect = [0 for i in range(0, 4*length)]

# Aij
for i in range(0, V-1): init_configvect[i] = 1

# vi
for i in range(V-1, 2*V-1): init_configvect[i] = 1

# c
init_configvect[6*V-4] = k

# d
init_configvect[6*V-2] = (V-1)-k
print (V-1)-k

# #0
init_configvect[6*V] = 1

# print init_configvect

def trig_type0(x,y,rule,reg):
	if(x == rule.lh.var and y == reg): return 1
	else: return 0

def trig_type1(x,y,rule,reg,parent):
	if(x == rule.obj.var and y == parent): return 1
	elif(x == 'e' and y == parent): return rule.enum
	else: return 0

def trig_type2(x,y,rule,reg,parent):
	if(x == rule.obj.var and y == reg): return 1
	elif(x == 'e' and y == reg): return rule.enum
	else: return 0

def trig_type3(x,y,rule,reg,parent):
	if((x == rule.robj.var and y == reg) or (x == rule.lobj.var and y == parent)): return 1
	elif(x == 'e' and y == parent): return rule.lenum
	elif(x == 'e' and y == reg): return rule.renum
	else: return 0

trig_matrix = []

parent = 0
for h in range(0, len(R)):
	reg = h / 2
	if reg == 0: parent = -1
	else: parent = 0
	for i in range(0, len(R[h])):
		temp_arr = [0 for m in range(0, col_size)]
		for j in range(0, 4):
			for l in range(0, len(objects[j])): 
				if(R[h][i].type == 0): temp_arr[j * length + l] = trig_type0(objects[j][l],j,R[h][i],reg)
				elif(R[h][i].type == 1): temp_arr[j * length + l] = trig_type1(objects[j][l],j,R[h][i],reg,parent)
				elif(R[h][i].type == 2): temp_arr[j * length + l] = trig_type2(objects[j][l],j,R[h][i],reg,parent)
				elif(R[h][i].type == 3): temp_arr[j * length + l] = trig_type3(objects[j][l],j,R[h][i],reg,parent)
		trig_matrix.append(temp_arr)

# print trig_matrix
# print 'matrix size: ', len(trig_matrix), 'x', len(trig_matrix[0])

f = open('../inputs/1.txt', 'w')

to_write = str(len(trig_matrix[0])) + ' ' + str(len(trig_matrix)) + '\n\n'

t_trig_mat = np.transpose(np.array(trig_matrix))

for i in range(0, len(t_trig_mat)):
	for j in range(0, len(t_trig_mat[i])):
		to_write += str(t_trig_mat[i][j])
		if j != len(t_trig_mat[i])-1: to_write += ' '
	to_write += '\n'

to_write += '\n'

for i in range(0, len(trans_matrix)):
	for j in range(0, len(trans_matrix[i])):
		to_write += str(trans_matrix[i][j])
		if j != len(trans_matrix[i])-1: to_write += ' '
	to_write += '\n'

to_write += '\n'

for i in range(0, len(init_configvect)):
	to_write += str(init_configvect[i])
	if i != len(init_configvect)-1: to_write += ' '

f.write(to_write)
f.close()

app_vectors = []
# Setup Phase
# Step 1
appvect = [0 for i in range(0, 9*V+20)]
appvect[0] = 1
i = 2
R0len = len(R0)
while(i < R0len-9):
	appvect[i] = 1
	appvect[i+1] = 1
	i += 3
appvect[R0len-9] = 1
appvect[R0len-4] = init_configvect[6*V-4]
appvect[R0len-3] = init_configvect[6*V-2]

app_vectors.append(appvect)

#Step 2
R0R1len = len(R0) + len(Rp0) + len(R1)
appvect = [0 for i in range(0, 9*V+20)]
i = 0
while(i < len(Rp1)):
	if R0R1len + len(Rp1) - 1 == i: appvect[R0R1len + i] = init_configvect[6*V-4]
	else: appvect[R0R1len + i] = 1
	i += 2
R0Rp2len = R0R1len + len(Rp1) + len(R2) + len(Rp2)
appvect[R0len-8] = 1
appvect[R0Rp2len-3] = init_configvect[6*V-2]

app_vectors.append(appvect)

#Step 3
appvect = [0 for i in range(0, 9*V+20)]
appvect[len(R0)-7] = 1
R0Rp0len = R0R1len - len(R1)
for i in range(0, len(R1)):
	appvect[R0Rp0len+i] = 1
R0R2len = R0Rp2len - len(Rp2)
appvect[R0R2len-2] = 1

app_vectors.append(appvect)

g = open('../inputs/2.txt', 'w')
to_write = str(len(app_vectors)) + '\n'
for i in range(0, len(app_vectors)):
	for j in range(0, len(app_vectors[i])):
		to_write += str(app_vectors[i][j])
		if j != len(app_vectors[i])-1: to_write += ' '
	to_write += '\n'

g.write(to_write)