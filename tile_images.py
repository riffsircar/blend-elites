from PIL import Image

smb_ki_images = {
    "E": Image.open('tiles/E.png'),
    "H": Image.open('tiles/H.png'),
    "G": Image.open('tiles/G.png'),
    "M": Image.open('tiles/M.png'),
    "o": Image.open('tiles/o.png'),
    "S": Image.open('tiles/S.png'),
    "T": Image.open('tiles/T.png'),
    "?": Image.open('tiles/Q.png'),
    "Q": Image.open('tiles/Q.png'),
    "X": Image.open('tiles/X1.png'),
    "#": Image.open('tiles/KI_X.png'),
    "-": Image.open('tiles/-.png'),
    "0": Image.open('tiles/0.png'),
    "D": Image.open('tiles/D.png'),
    "<": Image.open('tiles/PTL.png'),
    ">": Image.open('tiles/PTR.png'),
    "[": Image.open('tiles/[.png'),
    "]": Image.open('tiles/].png'),
    "P": Image.open('tiles/-.png')
}

mm_images = {
    "#":Image.open('tiles/MM_X2.png'),
    "*":Image.open('tiles/MM_star.png'),
    "+":Image.open('tiles/MM_+.png'),
    "-":Image.open('tiles/-.png'),
    "B":Image.open('tiles/MM_B2.png'),
    "C":Image.open('tiles/CMM.png'),
    "D":Image.open('tiles/DMM.png'),
    "H":Image.open('tiles/HMM.png'),
    "L":Image.open('tiles/MM_L.png'),
    "M":Image.open('tiles/MMM.png'),
    "P":Image.open('tiles/-.png'),
    "U":Image.open('tiles/MM_U.png'),
    "W":Image.open('tiles/MM_w.png'),
    "l":Image.open('tiles/MM_L.png'),
    "t":Image.open('tiles/TMM.png'),
    "w":Image.open('tiles/MM_w.png'),
    "|":Image.open('tiles/LMM.png')
}

# {'#': 0, '(': 1, ')': 2, '+': 3, '-': 4, 'B': 5, 'D': 6, 'E': 7, 'P': 8, '[': 9, ']': 10, '^': 11, 'v': 12}
met_images = {
    "#":Image.open('tiles/Met_X.png'),  # solid
    "(":Image.open('tiles/-.png'),  # beam around door (ignore using background)
    ")":Image.open('tiles/-.png'),  # beam around door (ignore using background)
    "+":Image.open('tiles/Met_+.png'),  # powerup
    "-":Image.open('tiles/-.png'),   # background
    "B":Image.open('tiles/Met_B.png'),  # breakable
    "D":Image.open('tiles/Met_D.png'),  # door
    "E":Image.open('tiles/Met_E.png'),  # enemy
    "P":Image.open('tiles/-.png'),   # path
    "[":Image.open('tiles/-.png'),  # ??
    "]":Image.open('tiles/-.png'),  # ??
    "^":Image.open('tiles/Met_^2.png'),  # lava
    "v":Image.open('tiles/-.png')  # ??
}

aff_images = {
    "-": Image.open('tiles/-.png'), # back
    "P": Image.open('tiles/-.png'), # path
    "X": Image.open('tiles/KI_X.png'), # solid,ground (X/pipe in SMB, # in KI/Met/MM, C in MM, T in KI)
    "B": Image.open('tiles/S.png'), # solid,breakable (S in SMB, B in MM/Met)
    "M": Image.open('tiles/CV_M.png'), # solid, moving (M in KI/MM)
    "D": Image.open('tiles/D.png'), # portal (D in KI/MM/Met)
    "H": Image.open('tiles/HMM.png'), # solid, hazard (H in MM/KI, ^ in Met)
    "E": Image.open('tiles/E.png'), # enemies (E in everything, t in MM)
    "|": Image.open('tiles/LMM.png'), # solid, climbable (| in MM)
    "*": Image.open('tiles/o.png'), # collectable (o in SMB, U/W/l/w/+ in MM)
    "W": Image.open('tiles/Met_+.png'), # weapons (* in MM, + in Met)
    "Q": Image.open('tiles/Q.png') # solid, collectable (Q/? in SMB)
}

smb_ki_mm_images = {
    "-": Image.open('tiles/-.png'),  # all back
    "P": Image.open('tiles/-.png'),  # path

    "X": Image.open('tiles/G.png'),  # SMB solid
    "#": Image.open('tiles/MM_X2.png'),   # MM/KI solid
    "B":Image.open('tiles/MM_B2.png'),   # MM breakable
    "S": Image.open('tiles/S.png'),   # SMB breakable
    "M": Image.open('tiles/M.png'),   # KI/MM moving
    "T": Image.open('tiles/T.png'),   # KI platform

    "o": Image.open('tiles/o.png'),   # SMB/MM collectables/coins
    "?": Image.open('tiles/Q.png'),   # SMB Q
    "Q": Image.open('tiles/Q.png'),   # SMB Q
    
    # SMB pipe
    "<": Image.open('tiles/PTL.png'),
    ">": Image.open('tiles/PTR.png'),
    "[": Image.open('tiles/[.png'),
    "]": Image.open('tiles/].png'),

    "E": Image.open('tiles/E.png'),   # SMB enemy
    "t":Image.open('tiles/TMM.png'),  # MM enemy
    "H":Image.open('tiles/HMM.png'),  # MM/KI hazard

    "D": Image.open('tiles/D.png'),   # KI/MM door
    "|":Image.open('tiles/LMM.png'),   # MM ladder

    "C":Image.open('tiles/CMM.png'),  # MM block
    
    # MM collectables
    "L":Image.open('tiles/MM_L.png'),
    "U":Image.open('tiles/MM_U.png'),
    "*":Image.open('tiles/MM_star.png'),
    "+":Image.open('tiles/MM_+.png'),
    "W":Image.open('tiles/MM_w.png'),
    "l":Image.open('tiles/MM_L.png'),
    "w":Image.open('tiles/MM_w.png')
}

ng_images = {
    "X":Image.open('tiles/KI_X.png'),  # solid
    "*":Image.open('tiles/NG_S.png'),  # throwing star
    "+":Image.open('tiles/NG_+.png'),  # 1000pts/500 pts
    ")":Image.open('tiles/NG_10.png'),  # 10pts
    "1":Image.open('tiles/NG_1.png'),  # 1 up
    "-":Image.open('tiles/-.png'),   # background
    "B":Image.open('tiles/NG_B.png'),   # bat
    "D":Image.open('tiles/NG_D.png'),   # cat/dog
    "C":Image.open('tiles/NG_D.png'),   # cat/dog
    "R":Image.open('tiles/NG_R.png'),   # restore physical strength
    "L":Image.open('tiles/LMM.png'),  # ladder
    "J":Image.open('tiles/NG_J.png'),  # jump/slash
    "P":Image.open('tiles/-.png'),  # path
    "F":Image.open('tiles/NG_F.png'), # invincible fire wheel
    "W":Image.open('tiles/NG_W.png'),  # wnidmill throwing star
    "T":Image.open('tiles/NG_T.png'),  # time freeze
    "A":Image.open('tiles/NG_A.png'),  # art of the fire wheel
    "%":Image.open('tiles/NG_5.png'),  # 5 pts
    "K":Image.open('tiles/NG_K.png'),  # bird
    "E": Image.open('tiles/NG_E.png')  # enemy
}
# {'#': 0, '-': 1, '0': 2, '1': 3, '2': 4, '4': 5, '5': 6, '7': 7, '>': 8, 'A': 9, 'B': 10, 'D': 11, 'E': 12, 'F': 13, 'G': 14, 
# 'H': 15, 'K': 16, 'M': 17, 'O': 18, 'P': 19, 'R': 20, 'S': 21, 'U': 22, 'V': 23, 'X': 24, '|': 25}    
cv_images = {
    '#':Image.open('tiles/CV_#.png'),  # holy cross
    '-':Image.open('tiles/-.png'),  #background
    '0':Image.open('tiles/CV_0.png'),   # orb
    '1':Image.open('tiles/CV_H.png'),   # 1 heart
    '2':Image.open('tiles/CV_2.png'),   # double shot
    '4':Image.open('tiles/CV_4.png'),   # 400 pts
    '5':Image.open('tiles/CV_H.png'),   # 5 hearts
    '7':Image.open('tiles/CV_7.png'),   # 700 pts
    '>':Image.open('tiles/CV_Dag.png'),   # dagger
    'A':Image.open('tiles/CV_A.png'),   # axe
    'B':Image.open('tiles/CV_B.png'),   # break wall
    'D':Image.open('tiles/CV_D.png'),   # door
    'E':Image.open('tiles/CV_E.png'),   # all enemies
    'F':Image.open('tiles/CV_F.png'),   # food
    'G':Image.open('tiles/CV_G.png'),   # gold potion
    'H':Image.open('tiles/CV_W.png'),   # holy water
    'K':Image.open('tiles/CV_K.png'),   # 1000 pts
    'M':Image.open('tiles/CV_M.png'),   # moving
    'O':Image.open('tiles/CV_O.png'),   # boomerang
    'P':Image.open('tiles/-.png'),   # path
    'R':Image.open('tiles/CV_R.png'),   # royal/2K
    'S':Image.open('tiles/CV_S.png'),   # stop watch
    'U':Image.open('tiles/CV_1.png'),   # levelup
    'V':Image.open('tiles/CV_V.png'),   # vertical spikes
    'X':Image.open('tiles/CV_X.png'),   # solid
    '|':Image.open('tiles/CV_stair.png'),   # stairs
}