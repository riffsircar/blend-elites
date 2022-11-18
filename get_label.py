import math

label_size = None

def get_label_smb(level):
	#label = [False,False,False,False]  # Enemy, Pipes, Breakable, Coins/QMs
	# {'-': 0, '<': 1, '>': 2, '?': 3, 'E': 4, 'P': 5, 'Q': 6, 'S': 7, 'X': 8, '[': 9, ']': 10, 'o': 11}
	label = [False, False, False, False, False] # E, Pipe, Q/?, o, S  //excluded: -, P, X
	temp = ''
	for l in level:
		temp += ''.join(l)
	if 'E' in temp:
		label[0] = True
	if '[' in temp and ']' in temp and '<' in temp and '>' in temp:
		label[1] = True
	if 'Q' in temp or '?' in temp:
		label[2] = True
	if 'o' in temp:
		label[3] = True
	if 'S' in temp:
		label[4] = True
	#if 'X' in temp:
	#    label[5] = True
	return label

def get_label_ki(level):
	#{'#': 0, '-': 1, 'D': 2, 'H': 3, 'M': 4, 'P': 5, 'T': 6}
	label = [False,False,False,False]  # Enemy, Doors, Moving, T
	temp = ''
	for l in level:
		temp += ''.join(l)
	if 'H' in temp:
		label[0] = True
	if 'D' in temp:
		label[1] = True
	if 'M' in temp:
		label[2] = True
	if 'T' in temp:
		label[3] = True
	return label

def get_label_mm(level):
	#label = [False,False,False,False]  # H/T/C, D/|, M, */U/W/w/+/l/L
	label = [False,False,False,False,False]   # clubbed together all decoratives/collectibles, all enemies
	temp = ''
	for l in level:
		temp += ''.join(l)
	if 'H' in temp or 'T' in temp or 'C' in temp:
		label[0] = True
	if 'D' in temp:
		label[1] = True
	if '|' in temp:
		label[2] = True
	if 'M' in temp:
		label[3] = True
	if '*' in temp or 'U' in temp or 'W' in temp or 'w' in temp or '+' in temp or 'l' in temp or 'L' in temp:
		label[4] = True
	return label

def get_label_ng(level):
	#{'%': 0, ')': 1, '*': 2, '+': 3, '-': 4, '1': 5, 'A': 6, 'B': 7, 'C': 8, 'D': 9, 'E': 10, 'F': 11, 'J': 12, 'K': 13, 'L': 14, 'P': 15, 'R': 16, 'T': 17, 'W': 18, 'X': 19}
	label = [False] * 5  # E, B/D/K/C, L, */J/A/F/W, +/)/1/R/T/% i.e. Enemy, Animal, Ladder, Weapons, Powerups
	temp = ''
	for l in level:
		temp += ''.join(l)
	if 'E' in temp:
		label[0] = True
	if 'B' in temp or 'D' in temp or 'K' in temp or 'C' in temp:
		label[1] = True
	if 'L' in temp:
		label[2] = True
	if '*' in temp or 'J' in temp or 'A' in temp or 'F' in temp or 'W' in temp:
		label[3] = True
	if '+' in temp or ')' in temp or '1' in temp or 'R' in temp or 'T' in temp or '%' in temp:
		label[4] = True
	return label

def get_label_cv(level):
	#{'#': 0, '-': 1, '0': 2, '1': 3, '2': 4, '4': 5, '5': 6, '7': 7, '>': 8, 'A': 9, 'B': 10, 'D': 11, 'E': 12, 'F': 13, 'G': 14, 'H': 15, 
	# 'K': 16, 'M': 17, 'O': 18, 'P': 19, 'R': 20, 'S': 21, 'U': 22, 'V': 23, 'X': 24, '|': 25}    
	label = [False] * 7  # E/V, D, |, #/2/>/A/H/O, 0/1/4/5/7/F/G/K/R/S/U, M, B i.e. Enemy/Hzard, Door, Ladder, Weapons, Powerups, Moving, Breakable
	temp = ''
	for l in level:
		temp += ''.join(l)
	if 'E' in temp or 'V' in temp:
		label[0] = True
	if 'D' in temp:
		label[1] = True
	if '|' in temp:
		label[2] = True
	if '#' in temp or '2' in temp or 'A' in temp or 'H' in temp or 'O' in temp:
		label[3] = True
	if '0' in temp or '1' in temp or '4' in temp or '5' in temp or '7' in temp or 'F' in temp or 'G' in temp or 'K' in temp or 'R' in temp or 'S' in temp or 'U' in temp:
		label[4] = True
	if 'M' in temp:
		label[5] = True
	if 'B' in temp:
		label[6] = True
	return label

def get_label_blend(level):
	# E/t/H, D, |, </>/[/], ?/Q, o/L/U/*/+/W/l/w, M, T/C, S/B, X/#  
	# i.e. enemy, hazard, door, ladder, pipe, SMB QM, SMB/MM collectable, moving, fixed, breakable
	label = [False] * 9
	temp = ''
	for l in level:
		temp += ''.join(l)
	if 'E' in temp or 't' in temp or 'H' in temp:
		label[0] = True
	if 'D' in temp:
		label[1] = True
	if '|' in temp:
		label[2] = True
	if '[' in temp and ']' in temp and '<' in temp and '>' in temp:
		label[3] = True
	if 'Q' in temp or '?' in temp:
		label[4] = True
	if '*' in temp or 'U' in temp or 'W' in temp or 'w' in temp or '+' in temp or 'l' in temp or 'L' in temp or 'o' in temp:
		label[5] = True
	if 'M' in temp:
		label[6] = True
	if 'T' in temp or 'C' in temp:
		label[7] = True
	if 'S' in temp or 'B' in temp:
		label[8] = True
	return label

label_sizes = {'smb':5,'ki':4,'mm':5,'ng':5,'cv':7,'blend':9,'met':5}
get_label_funcs = {'smb':get_label_smb, 'ki':get_label_ki, 'mm':get_label_mm,'ng':get_label_ng,'cv':get_label_cv,'blend':get_label_blend}

def get_label_func(game):
	label_size = label_sizes[game]
	num_labels = int(math.pow(2,label_size))
	return get_label_funcs[game]
