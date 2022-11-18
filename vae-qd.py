import argparse, random, math, json, sys, os, torch, warnings, pickle, copy, re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from model_conv import get_conv_model, load_conv_model
from model_lin import get_model, load_model
import pathfinding
from metrics import *
from playability import *
from tile_images import *
from get_label import *

warnings.filterwarnings('ignore')
device = torch.device('cpu')
latent_dim = 32 # size of latent vector
GAME = 'blend'
dims = (16,16)
smb_folder = 'smb_chunks_fixed/'
ki_folder = 'ki_chunks/'
mm_folder = 'mm_chunks_fixed/'
pats_folder = 'smb_pats/'
ng_folder = 'ng_chunks/'
cv_folder = 'cv_chunks/'
met_folder = 'met_chunks/'
mutation_prob = 0.3
elems = False # False for tile metrics
rev = False # True for playability as behavior
runs = 1
generations = 100000
tag = 'denl'
rev_obj = 'Density'
objective = density # nonlinearity, symmetry, similarity
BC1, BC2 = 'density','nonlinearity'
FC = 0
CONV = 1
TYPE = FC  # 0 - fc, 1 - conv
print('tag: ',tag)

folders = {'smb':smb_folder,'ki':ki_folder,'mm':mm_folder,'ng':ng_folder,'cv':cv_folder,'met':met_folder,
'blend':None,'blend_aff_skm':None,'blend_aff_nomet':None}
out_folders = {'smb':'out_smb','ki':'out_ki','mm':'out_mm','ng':'out_ng','cv':'out_cv',
'blend':'out_blend','met':'out_met','blend_aff_skm':'out_blend_aff_skm','blend_aff_nomet':'out_blend_aff_nomet'}
label_sizes = {'smb':5,'ki':4,'mm':5,'ng':5,'cv':7,'blend':9,'met':5,'blend_aff_skm':10,'blend_aff_nomet':0}
label_size = label_sizes[GAME]
num_labels = int(math.pow(2,label_size))
#print(num_labels)
folder = folders[GAME]
#manual_seed = random.randint(1, 10000)
#random.seed(manual_seed)
#torch.manual_seed(0)
#np.random.seed(0)
data_files = {'smb':'SMB_mod.json','ki':'KI_mod.json','mm':'MM_mod.json','ng':'NG_mod.json','cv':'CV_mod.json','met':'Met_mod.json'}
data_files_aff = {'smb':'SMB_aff.json','ki':'KI_aff.json','mm':'MM_aff.json','ng':'NG_aff.json','cv':'CV_aff.json','met':'Met_aff.json'}

with open('affordances.json') as data_file:
    affordances = json.load(data_file)

games = ['smb','ki','mm']
num_games = int(math.pow(2,len(games)))
if not GAME.startswith('blend'):
    with open(data_files[GAME]) as data_file:
        game_desc = json.load(data_file)
    print('Len jumps: ',len(game_desc['jumps']))
else:
    game_desc = {'smb':None,'ki':None,'mm':None}
    for game in games:
        with open(data_files[game]) as data_file:
            game_desc[game] = json.load(data_file)
            print(game, len(game_desc[game]['jumps']))


all_images = {'smb':smb_ki_images, 'ki':smb_ki_images, 'mm':mm_images,'blend':smb_ki_mm_images,'ng':ng_images,'cv':cv_images,
'met':mm_images,'blend_aff_skm':aff_images, 'blend_aff_nomet':aff_images}
images = all_images[GAME]

def pipe_check(level):
    temp = ''
    for l in level:
        temp += ''.join(l)
    if '[' in temp and ']' not in temp:
        return False
    if ']' in temp and '[' not in temp:
        return False
    if '<' in temp and '>' not in temp:
        return False
    if '>' in temp and '<' not in temp:
        return False
    return True

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def affordify(line,aff):
    a_line = ''
    for c in line:
        if c in aff['solid']:
            a_line += 'X'
        elif c in aff['breakable']:
            a_line += 'B'
        elif c in aff['hazard']:
            a_line += 'H'
        elif c in aff['enemies']:
            a_line += 'E'
        elif c in aff['collectable']:
            a_line += '*'
        elif c in aff['solid_collectable']:
            a_line += 'Q'
        elif c in aff['weapon']:
            a_line += 'W'
        elif c in aff['moving']:
            a_line += 'M'
        elif c in aff['door']:
            a_line += 'D'
        elif c in aff['climbable']:
            a_line += '|'
        elif c == 'P':
            a_line += 'P'
        else:
            a_line += '-'
    return a_line


def parse_folder(folder,game=None):
    levels, text = [], ''
    files = os.listdir(folder)
    files[:] = (value for value in files if value != '.')
    files = natural_sort(files)
    for file in files:
        if file.startswith('.'):
            continue
        with open(os.path.join(folder,file),'r') as infile:
            level = []
            for line in infile:
                line = line.rstrip()
                if game is not None:
                    a_line = affordify(line,affordances[game])
                else:
                    a_line = line
                text += a_line
                level.append(list(a_line.rstrip()))
            if (GAME == 'smb' or GAME.startswith('blend')) and not pipe_check(level):
                continue
            levels.append(level)
    return levels, text

if GAME.startswith('blend'):
    smb_levels, smb_text = parse_folder(smb_folder)
    ki_levels, ki_text = parse_folder(ki_folder)
    mm_levels, mm_text = parse_folder(mm_folder)
    #cv_levels, cv_text = parse_folder(cv_folder,'cv')
    #ng_levels, ng_text = parse_folder(ng_folder,'ng')
    smb_text = smb_text.replace('\n','')
    ki_text = ki_text.replace('\n','')
    mm_text = mm_text.replace('\n','')
    #cv_text = cv_text.replace('\n','')
    #ng_text = ng_text.replace('\n','')
    levels = smb_levels + ki_levels + mm_levels #+ cv_levels + ng_levels
    text = smb_text + ki_text + mm_text #+ cv_text + ng_text
else:
    levels, text = parse_folder(folders[GAME])
    text = text.replace('\n','')
chars = sorted(list(set(text.strip('\n'))))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}
print(int2char)
print(char2int)
num_tiles = len(char2int)
print('Num tiles: ', num_tiles)


level_rows, level_cols = set(), set()
level_nls = []
for level in levels:
    level_t = [[level[j][i] for j in range(len(level))] for i in range(len(level[0]))]
    for l in level:
        l_str = ''.join(l)
        #if l_str not in level_rows:
        level_rows.add(l_str)
    for lt in level_t:
        lt_str = ''.join(lt)
        #if lt_str not in level_cols:
        level_cols.add(lt_str)
print('Rows: ', len(level_rows))
print('Cols: ', len(level_cols))


model = None
if TYPE == CONV:
    model_name = 'vae_conv_' + GAME + '_ld_' + str(latent_dim) + '_final.pth'
    model = load_conv_model(model_name,num_tiles,num_tiles,latent_dim,device)
else:
    model_name = 'vae_fc_' + GAME + '_ld_' + str(latent_dim) + '_final.pth'
    model = load_model(model_name,256,num_tiles,num_tiles,latent_dim,device)
model.eval()
model.to(device)
#print(model)
#sys.exit()

def write_segment_to_file(segment,name):
    outfile = open(out_folders[GAME] + '/' + name + '.txt','w')
    for row in segment:
        outfile.write(row + '\n')
    outfile.close()

def get_image_from_segment(segment,name):
    img = Image.new('RGB',(16*16, 16*16))
    for row, seq in enumerate(segment):
        for col, tile in enumerate(seq):
            img.paste(images[tile],(col*16,row*16))
    img.save(out_folders[GAME] + '/' + name + '.png')

def get_z_from_file(folder,f):
    print('\nInput:')
    chunk = open(folder + f, 'r').read().splitlines()
    chunk = [line.replace('\r\n','') for line in chunk]
    out = []
    for line in chunk:
        print(line)
        line_list = list(line)
        line_list_map = [char2int[x] for x in line_list]
        out.append(line_list_map)
    out = np.asarray(out)
    out_onehot = np.eye(num_tiles, dtype='uint8')[out]
    out_onehot = np.rollaxis(out_onehot, 2, 0)

    out_onehot = out_onehot[None, :, :]

    data = torch.DoubleTensor(out_onehot).to(device)
    if TYPE == FC:
        data = data.view(data.size(0),-1)
        z, _, _ = model.encoder.encode(data)
    else:
        z, _, _ = model.encode(data)
    return z

def get_z_from_segment(segment):
    out = []
    for l in segment:
        l = list(l)
        l_map = [char2int[x] for x in l]
        out.append(l_map)
    out = np.asarray(out)
    out_onehot = np.eye(num_tiles, dtype='uint8')[out]
    out_onehot = np.rollaxis(out_onehot, 2, 0)
    out_onehot = out_onehot[None, :, :]
    out = torch.DoubleTensor(out_onehot)
    out = out.to(device)
    #out_lin = out.view(out.size(0),-1)
    #z, _, _ = model.encoder.encode(out)
    if TYPE == FC:
        out = out.view(out.size(0),-1)
        z, _, _ = model.encoder.encode(out)
    else:
        z, _, _ = model.encode(out)
    return z
    

def get_segment_from_file(folder,f):
    chunk = open(folder + f, 'r').read().splitlines()
    chunk = [line.replace('\r\n','') for line in chunk]
    return chunk
    out = []
    for line in chunk:
        line_list = list(line)
        #line_list_map = [char2int[x] for x in line_list]
        out.append(line_list)
    return out

def get_segment_from_z(z):
    if TYPE == FC:
        level = model.decoder.decode(z)
    else:
        level = model.decode(z)
    level = level.reshape(level.size(0),num_tiles,dims[0],dims[1])
    im = level.data.cpu().numpy()
    im = np.argmax(im, axis=1).squeeze(0)
    level = np.zeros(im.shape)
    level = []
    for i in im:
        level.append(''.join([int2char[t] for t in i]))
    return level

def similarity_wrapper(level):
    return similarity(level,level_rows,level_cols)

bc_func = {'density':density,'nonlinearity':nonlinearity,'similarity':similarity_wrapper,'difficulty':difficulty,'symmetry':symmetry}
bc_metrics = {}
for key in bc_func:
    bc_metrics[key] = None

print('Num levels: ', len(levels))

"""
den, nl, diff, inter, sym, sim = [], [], [], [], [], []
for level in levels:
    #den.append(density(level))
    nl.append(nonlinearity(level,GAME))
    diff.append(difficulty(level))
    inter.append(interestingness(level))
    sym.append(symmetry(level))
    sim.append(similarity_wrapper(level))
"""
bc_metrics['density'] = 256 #max(den)
bc_metrics['nonlinearity'] = 64 #max(nl)
bc_metrics['symmetry'] = 256 #max(sym)
bc_metrics['similarity'] = 32 #max(sim)

print(bc_metrics)
#sys.exit()

"""
z2 = torch.DoubleTensor(latent_dim).normal_(0,1).to(device).reshape(1,-1)
level = get_segment_from_z(z2)
level = []
level.append('-' * 16)
for _ in range(13):
    level.append('-' * 5 + '#' * 6 + '-' * 5)
level.append('-' * 16)
#level.append('X' + '-' * 14 + 'X')
#level.append('-' * 5 + 'X' * 6 + '-' * 5)
level.append('#' * 16)
print('\n'.join(level))
#crossover_uniform(z1,z2)
#mutation_uniform(z1)

#path, dist = findPaths(game_desc['solid'],game_desc['jumps'],level)
#print('MINI: ', path)
#print(dist)
#path, dist = findPathsFull(game_desc,level,GAME == 'ki')
#print('FULL: ', path)
#print(dist)
#sys.exit()
"""
"""
count, ones, zeros = 0, 0, 0
dists, pls = [], []
for i, level in enumerate(levels):
    print(i)
    level_for_pf = []
    for line in level:
        level_for_pf.append(''.join(line))
    #paths =  findPaths(10,game_desc['solid'],game_desc['jumps'],level_for_pf)
    if GAME != 'blend':
        path, dist = findPathsFull(game_desc,level_for_pf,GAME)
        if path is not None:
            pls.append(len(path))
        else:
            pls.append(0)
        dists.append(dist)
        #print(i, dist, path)
        #print('\n'.join(level_for_pf),'\n')
        print('Final dist: ', dist)
        if dist == 1.0:
            ones += 1
        elif dist == 0:
            zeros += 1
        elif dist < 1.0:
            count += 1
            #print('Dist: ',dist)
            print('\n'.join(level_for_pf))
            print('Path: ', path,'\n')
        
    else:
        scores, paths, path_lengths = [], [], []
        for i, game in enumerate(games):
            path, dist = findPathsFull(game_desc[game],level_for_pf,game)
            if path is not None:
                path_lengths.append(len(path))
            paths.append(path)
            scores.append(dist)
        m = max(scores)
        agent_label = ''
        for score in scores:
            if score == m:
                agent_label += '1'
            else:
                agent_label += '0'
        #print('\n'.join(level_for_pf))
        print('Scores:',scores,' Agent: ',agent_label)
        #print('Agent:',agent_label)
print('Count:',count)
print('Zeros:',zeros)
print('Ones:',ones)
print(np.mean(dists))
print(np.mean(pls),np.min(pls),np.max(pls))

sys.exit()
#"""

def crossover_uniform(z1,z2):
    z3 = torch.zeros(z1.shape).to(device).to(dtype=torch.float64)
    for i, (a,b) in enumerate(zip(z1,z2)):
        prob = random.random()
        if prob <= 0.5:
            z3[i] = z1[i]
        else:
            z3[i] = z2[i]    
    return z3

def mutation_uniform(z):
    for i,x in enumerate(z):
        prob = random.random()
        if prob < mutation_prob:
            noise = np.random.normal(0,1,1)
            z[i] = z[i] + noise[0]
    return z

def get_qdscore_coverage(cells):
    qd, coverage = 0, 0
    for c in cells:
        if cells[c] is not None:
            qd += cells[c][1]
            coverage += 1
    coverage = (coverage * 100)/len(cells)
    return qd, coverage

def init_archive(BC1, BC2):
    archive = {}
    if BC2 is not None:
        for i in range(BC1):
            for j in range(BC2):
                archive[(i,j)] = None
    else:
        for i in range(BC1):
            archive[i] = None
    return archive

def assign_cell(level,BC1,BC2):
    f1, f2 = bc_func[BC1], bc_func[BC2]
    s1, s2 = f1(level), f2(level)
    s1 = min(max(s1,0),bc_metrics[BC1])
    s2 = min(max(s2,0),bc_metrics[BC2])
    return s1, s2

def get_label_string_from_array(label):
    label = np.array(label).astype('uint8')
    label = list(label)
    label = ''.join([str(i) for i in label])
    return label

def get_elem_archive_dims(num_cells):
    a, b, i = 1, num_cells, 0
    while a < b:
        i += 1
        if num_cells % i == 0:
            a = i
            b = num_cells//a
    return [b, a]

def get_label_agents(level):
    scores = []
    for i, game in enumerate(games):
        _, score = findPathsFull(game_desc[game],level,game)
        scores.append(score)
    label = [True if s == 1.0 else False for s in scores]
    return label

if elems:
    get_label = get_label_func(GAME)
elif rev:
    get_label = get_label_agents

best_archive, best_old_archive, best_qd_scores, best_coverages = None, None, None, None
best_qd_score = 0  # best final QD score among all runs
qds, covs = [], []  # final QDscore and coverage for each run
bc1_range, bc2_range = None, None
if elems:
    #bc1_range, bc2_range = get_elem_archive_dims(num_labels)
    bc1_range = num_labels
elif rev:
    bc1_range = int(math.pow(2,len(games)))
    print('Range: ',bc1_range)
else:
    bc1_range, bc2_range = bc_metrics[BC1]+1, bc_metrics[BC2]+1
    
print(bc1_range,bc2_range)
for i in range(runs):
    print('Run: ', i)
    init_pop, pop_size= [], 100
    archive = init_archive(bc1_range,bc2_range)
    print('Archive size: ', len(archive))
    for _ in range(pop_size):
        z = torch.DoubleTensor(1,latent_dim).normal_(0,1).to(device)
        init_pop.append(z)

    for i, z in enumerate(init_pop):
        z = z.reshape(1,-1)
        level = get_segment_from_z(z)
        this_score, this_label = 0, ''
        if not GAME.startswith('blend'):
            _, this_score = findPathsFull(game_desc,level,GAME)
        else:
            if rev:
                this_score = objective(level)
            else:
                scores = []
                for i, game in enumerate(games):
                    _, game_score = findPathsFull(game_desc[game],level,game)
                    scores.append(game_score)
                this_score = max(scores)
                this_label = ''.join(['1' if s == 1.0 else '0' for s in scores])
        c1, c2 = None, None
        if elems or rev:
            label = get_label(level)
            label_one_hot = list(map(int,label))
            label_one_hot_string = ''.join([str(x) for x in label_one_hot])
            c = int(label_one_hot_string,2)
            #if elems:
            #    c1, c2 = int(c % bc1_range), int(c // bc2_range)
            #    c = (c1,c2)
        else:
            c1, c2 = assign_cell(level,BC1,BC2)
            c = (c1,c2)
        z = z.reshape(-1)
        #if archive[(c1,c2)] is None:
        if archive[c] is None:
            if not GAME.startswith('blend') or rev:
                #archive[(c1,c2)] = z, this_score
                archive[c] = z, this_score
            else:
                #archive[(c1,c2)] = z, this_score, this_label
                archive[c] = z, this_score, this_label
        else:
            if not GAME.startswith('blend') or rev:
                cur_z, cur_score = archive[c]
                if this_score > cur_score:
                    archive[c] = z, this_score
            else:
                cur_z, score, label = archive[c]
                if this_score > score:
                    archive[c] = z, this_score, this_label
                if this_score == 1.0:
                    new_label = int(label,2) | int(this_label,2)
                    new_label = '{:b}'.format(new_label)
                    new_label = new_label.zfill(3)
                    archive[c] = z, this_score, new_label


    init_qd_score, init_coverage = get_qdscore_coverage(archive)
    old_archive = copy.deepcopy(archive)
    coverages, qd_scores = [init_coverage], [init_qd_score]       # list of qdscore/coverage for each generation in this run
    
    for g in range(generations):
        if g % 1000 == 0:
            print('Generation: ', g)
        eligible = [c for c in archive.keys() if archive[c] is not None]
        c1, c2 = random.sample(eligible,2)
        z1, z2 = archive[c1][0], archive[c2][0]
        z_child = crossover_uniform(z1,z2)
        z_child = mutation_uniform(z_child)
        z_child = z_child.reshape(1,-1)
        level = get_segment_from_z(z_child)
        this_score, this_label = 0, ''
        if not GAME.startswith('blend'):
            _, this_score = findPathsFull(game_desc,level,GAME)
        else:
            if rev:
                this_score = objective(level)
            else:
                scores = []  # [smb,ki,mm]
                for i, game in enumerate(games):
                    _, game_score = findPathsFull(game_desc[game],level,game)
                    scores.append(game_score)
                this_score = max(scores)
                this_label = ''.join(['1' if s == 1.0 else '0' for s in scores])
        if elems or rev:
            label = get_label(level)
            label_one_hot = list(map(int,label))
            label_one_hot_string = ''.join([str(x) for x in label_one_hot])
            c = int(label_one_hot_string,2)
            #if elems:
            #    cell_1, cell_2 = int(c % bc1_range), int(c // bc2_range)
            #    c = (cell_1,cell_2)
        else:
            cell_1, cell_2 = assign_cell(level,BC1,BC2)
            c = (cell_1,cell_2)
        z_child = z_child.reshape(-1)
        if archive[c] is None:
            if not GAME.startswith('blend') or rev:
                archive[c] = z_child, this_score
            else:
                archive[c] = z_child, this_score, this_label
        else:
            if not GAME.startswith('blend') or rev:
                cur_z, cur_score = archive[c]
                if this_score > cur_score:
                    archive[c] = z_child, this_score
            else:
                cur_z, cur_score, cur_label = archive[c]
                if this_score > cur_score:
                    archive[c] = z_child, this_score, this_label
                if this_score == 1.0:
                    new_label = int(cur_label,2) | int(this_label,2)
                    new_label = '{:b}'.format(new_label)
                    new_label = new_label.zfill(3)
                    archive[c] = z_child, this_score, new_label
        qd_score, coverage = get_qdscore_coverage(archive)
        coverages.append(coverage)
        qd_scores.append(qd_score)
    print('INIT CO: ', init_coverage, '\nINIT QD: ', init_qd_score)
    print('FINAL COV: ', coverages[generations], '\nFINAL QD: ', qd_scores[generations], '\n')
    if qd_scores[generations] > best_qd_score:
        best_qd_score = qd_scores[generations]

        # mark archive, list of coverages, list of qd scores for this generation as best one
        best_archive = copy.deepcopy(archive)
        best_old_archive = copy.deepcopy(old_archive)
        best_coverages = copy.deepcopy(coverages)
        best_qd_scores = copy.deepcopy(qd_scores)
    qds.append(qd_scores[generations])
    covs.append(coverages[generations])


out_file = open(GAME + '_' + str(latent_dim) + '_' + tag + '_' + str(generations) + '.csv','w')
out_file.write('Run,QD-Score,Coverage\n')
for i, (qd, c) in enumerate(zip(qds,covs)):
    out_file.write(str(i+1) + ',' + str(qd) + ',' + str(c) + '\n')
out_file.close()

print('MEAN QD-SCORE: ', np.mean(qds))
print('STD QD-SCORE: ', np.std(qds))
print('MEAN COVERAGE: ', np.mean(covs))
print('STD COVERAGE: ', np.std(covs))

out_file = open(GAME + '_' + str(latent_dim) + '_' + tag + '_' + str(generations) + '_qdcov.csv','w')
out_file.write('Generation,QD-Score,Coverage\n')
for i, (qd,c) in enumerate(zip(best_qd_scores,best_coverages)):
    out_file.write(str(i+1) + ',' + str(qd) + ',' + str(c) + '\n')
out_file.close()

# plot QD-score and coverage progression for best run
x = range(generations+1)
plt.xlabel('Generations')
plt.xlim(0,generations)
plt.ylim(0,best_qd_score)
plt.ylabel('QD_Score')
plt.plot(x,best_qd_scores)
plt.savefig(GAME + '_' + str(latent_dim) + '_' + tag + '_qds_' + str(generations) + '.png')
plt.clf()
plt.xlabel('Generations')
plt.xlim(0,generations)
plt.ylabel('Coverage')
plt.ylim(0,100)
plt.plot(x,best_coverages)
plt.savefig(GAME + '_' + str(latent_dim) + '_' + tag + '_cov_' + str(generations) + '.png')
plt.clf()
print('\n')

# plot best archive along with heatmap corresponding to fitness score

cmap = matplotlib.colors.LinearSegmentedColormap.from_list('colormap',['white','blue','red'],256)
if rev or elems:
    archive_array = np.empty((bc1_range))
else:
    archive_array = np.empty((bc1_range,bc2_range))
    

if GAME != 'blend':
    out_file = open(GAME + '_' + str(latent_dim) + '_' + tag + '_best_arch_' + str(generations) + '.csv','w')
    if not rev and not elems:
        out_file.write(BC1 + ',' + BC2 + ',Fitness\n')
    else:
        out_file.write('Label,Fitness\n')

for c in best_archive:
    fitness = best_archive[c][1] if best_archive[c] is not None else 0
    if not rev and not elems:
        archive_array[c[0],c[1]] = fitness
    else:
        archive_array[c] = fitness
    if GAME != 'blend':
        if not rev and not elems:
            out_file.write(str(c[0]) + ',' + str(c[1]) + ',' + str(fitness) + '\n')
        else:
            if elems:
                c = bin(c)[2:].zfill(label_size)
            out_file.write(str(c) + ',' + str(fitness) + '\n')

if not rev and not elems:
    fig = plt.figure()
    img = plt.imshow(archive_array,interpolation='nearest',cmap=cmap,origin='lower')
    if elems:
        plt.xticks(range(0,bc2_range))
        plt.yticks(range(0,bc1_range))
        #plt.xlabel('Elements')
        #plt.ylabel('Elements')
    else:
        plt.xlim(0,bc2_range-1)
        plt.ylim(0,bc1_range-1)
        plt.xlabel(BC2)
        plt.ylabel(BC1)
        ratio = 1.0
        ax = plt.axes()
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        # the abs method is used to make sure that all numbers are positive
        # because x and y axis of an axes maybe inversed.
        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    plt.colorbar(img,cmap=cmap)
    fig.savefig('archive_' + GAME + '_' + tag + '_' + str(latent_dim) + '_' + str(generations) + '.png')
else:
    fig = plt.figure(figsize=(15,5))
    print(archive_array)
    print(archive_array.shape)
    if elems:
        #plt.xticks(range(0,num_labels))
        plt.xlim(0,num_labels)
        plt.ylim(0,1)
        #plt.yticks(range(0,bc1_range))
        plt.bar(np.arange(num_labels),archive_array)
    else:
        plt.xlabel('Agent')
        plt.ylabel(rev_obj)
        plt.xlim(0,32)
        plt.ylim(0,256)
        plt.xticks(range(0,32))
        plt.bar(archive_array,num_games)
    plt.savefig('archive_' + GAME + '_' + tag + '_' + str(latent_dim) + '_' + str(generations) + '.png')


if GAME.startswith('blend') and not rev:
    if not elems:
        archive_array = np.empty((bc1_range,bc2_range))
    else:
        archive_array = np.empty((bc1_range))
    blend_out_file = open(GAME + '_' + str(latent_dim) + '_' + tag + '_best_arch_' + str(generations) + '.csv','w')
    if not elems:
        blend_out_file.write(BC1 + ',' + BC2 + ',Fitness,Label\n')
    else:
        blend_out_file.write('Label,Fitness,Label\n')
    fig, ax = plt.subplots()
    xs, ys, labels, point_colors = [], [], [], []
    colors = {'000':'grey','001':'blue','010':'red','011':'magenta','100':'green','101':'cyan','110':'yellow','111':'black','None':'white'}
    labeled = {'000':[],'001':[],'010':[],'011':[],'100':[],'101':[],'110':[],'111':[]}
    for c in best_archive:
        if best_archive[c] is not None:
            label, fitness = best_archive[c][2], best_archive[c][1]
            if not elems:
                c1,c2 = c
                labeled[label].append((c2,c1))
                blend_out_file.write(str(c[0]) + ',' + str(c[1]) + ',' + str(fitness) + ',' + label + '\n')
            else:
                labeled[label].append((c))
                blend_out_file.write(str(c) + ',' + str(fitness) + ',' + label + '\n')
            
    
    if not elems:
        plt.xlim(0,bc2_range-1)
        plt.ylim(0,bc1_range-1)
        plt.xlabel(BC2)
        plt.ylabel(BC1)
        for label in labeled:
            xs,ys = [],[]
            for (x,y) in labeled[label]:
                xs.append(x)
                ys.append(y)
            if len(xs) > 0:
                ax.scatter(xs,ys,color=colors[label],label=label,s=5)
        ax.legend()
        plt.savefig('archive_blend_per_game_' + tag + '_' + str(latent_dim) + '_' + str(generations) + '.png')
    #df = pd.DataFrame(dict(Nonlinearity=xs,Density=ys,labels=labels))
    #ax.scatter(df['Nonlinearity'],df['Density'],c=df['labels'].map(colors),s=1)
    #ax.legend()
    #plt.savefig('blend.png')
    
    #fig = plt.figure()
    #blend_out_file.close()
    #grouped = df.groupby('labels')
    #for key, group in grouped:
    #    group.plot(ax=ax,kind='scatter',x='Nonlinearity',y='Density',label=key, color=colors[key])
    #plt.savefig('archive_bpg_' + str(latent_dim) + '_' + str(generations) + '.png')

for c in best_archive:
    if best_archive[c] is not None:
        if not GAME.startswith('blend') or rev:
            z, score = best_archive[c]
        else:
            z, score, label = best_archive[c]
            
        z = z.reshape(1,-1)
        level = get_segment_from_z(z)
        """
        label = [int(i) for i in bin(c)[2:]]
        if len(label) < label_size:
            label = [0] * (label_size - len(label)) + label
        label_str = ''.join(map(str,label))
        get_image_from_segment(level,'QD_new/qd_' + GAME + '_' + str(latent_dim) + '_' + label_str + '_' + str(score))
        """
        if elems:
            c = bin(c)[2:].zfill(label_size)
        if not GAME.startswith('blend') or rev:
            file_name = GAME + '_' + tag + '_' + str(latent_dim) + '_' + str(c) + '_' + str(score)
            write_segment_to_file(level, file_name)
            get_image_from_segment(level, file_name)
        else:
            file_name = label + '_' + GAME + '_' + tag + '_' + str(latent_dim) + '_' + str(c) + '_' + str(score)
            write_segment_to_file(level, file_name)
            get_image_from_segment(level, file_name)
            