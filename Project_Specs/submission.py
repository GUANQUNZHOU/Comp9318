# Import your files here...
#!/usr/bin/python3
import re
import numpy as np
import math
def parse(str):
    a = str
    a = re.sub('\n','',a)
    b = re.split("(,|\(|\)|\/|\-|\&|\s)", a)
    store =[]
    for i in b:
        if i != ' ' and i != '':
            store.append(i)
    return store
def get_state(str):
    dic_state = {}
    state = list(open(str))
    for g in range(len(state)):
        state[g] = re.sub('\n','',state[g])
    count_state = eval(state[0])
    for se_state in range(1,1+count_state):
        dic_state[se_state-1] = state[se_state]
    return dic_state

def get_symbol(str):
    dic_symbol = {}
    symbol = list(open(str))
    for h in range(len(symbol)):
        symbol[h] = re.sub('\n','',symbol[h])
    count_symbol = eval(symbol[0])
    for se_symbol in range(1,1+count_symbol):
        dic_symbol[se_symbol-1] = symbol[se_symbol]
    return dic_symbol

def get_number_of_state(str):
    state = list(open(str))
    for g in range(len(state)):
        state[g] = re.sub('\n','',state[g])
    count_state = eval(state[0])
    begin_index = state.index('BEGIN')-1
    end_index = state.index('END')-1
    if begin_index < end_index:
        count_need = begin_index
    else: 
        count_need = end_index
    return count_need,begin_index,end_index

def for_unk(list,int,gr_symbol):
    st = []
    for lj in gr_symbol.keys():
        st.append(gr_symbol[lj])
    for uj in range(len(list)):
        if list[uj] not in st:
            list[uj] = int
        if list[uj] in st:
            list[uj] = st.index(list[uj])
    return list
############################################################################
def find_numpy_amount_trans(str,tran_count,Beg,END):
    content = list(open(str))
    tot_col = eval(content[0])
    total_numpy = np.zeros((tran_count,(tot_col-1)))
    init_numpy = np.zeros((1,tran_count))
    for pict in range(1,(len(content)-tot_col)):
        if eval(content[pict+tot_col].split()[0]) == Beg:
            init_numpy[0][eval(content[pict+tot_col].split()[1])] = eval(content[pict+tot_col].split()[-1])
        elif eval(content[pict+tot_col].split()[0]) != Beg and eval(content[pict+tot_col].split()[1]) < tran_count:
            total_numpy[eval(content[pict+tot_col].split()[0])][eval(content[pict+tot_col].split()[1])] = eval(content[pict+tot_col].split()[-1])
        elif eval(content[pict+tot_col].split()[0]) != Beg and eval(content[pict+tot_col].split()[1]) >= tran_count:
            total_numpy[eval(content[pict+tot_col].split()[0])][-1] = eval(content[pict+tot_col].split()[-1])
    hine = np.sum(init_numpy,axis = 1)
    for initer in range(tran_count):
        init_numpy[0][initer] = (float((init_numpy[0][initer]+1)/(hine[0]+tot_col-1)))
    return tot_col,init_numpy[0],total_numpy

def find_init_tran_hidd_stat(str,tran_count,Beg,END):
    hidden_state = []
    for lk in range(tran_count):
        hidden_state.append(lk)
    hidden_state.append(END)

    state_li = []
    for ri in range(tran_count):
        state_li.append(ri)
    N,init_pro,trans_pro = find_numpy_amount_trans(str,tran_count,Beg,END)
    he = np.sum(trans_pro,axis = 1)
    for transer in range(tran_count):
        for transee in range(tran_count):
            trans_pro[transer][transee] = float((trans_pro[transer][transee]+1)/(he[transer]+N-1))
        trans_pro[transer][tran_count] = float((trans_pro[transer][tran_count]+1)/(he[transer]+N-1))
    return hidden_state,state_li,init_pro,trans_pro
############################################################################
def find_numpy_amount_emiss(str,tran_count,Beg,END,lastcol):
    content = list(open(str))
    tot_col = eval(content[0])
    total_numpy = np.zeros((tran_count,(lastcol+1)))
    for picc in range(1,(len(content)-tot_col)):
        total_numpy[eval(content[picc+tot_col].split()[0])][eval(content[picc+tot_col].split()[1])] = eval(content[picc+tot_col].split()[-1])
    return tot_col,total_numpy

def find_emission_pro(str,tran_count,Beg,END,lastcol):
    M,emission_pro= find_numpy_amount_emiss(str,tran_count,Beg,END,lastcol)
    hee = np.sum(emission_pro,axis = 1)
    for emisser in range(tran_count):
        for emissee in range(lastcol):
            emission_pro[emisser][emissee] = float((emission_pro[emisser][emissee]+1)/(hee[emisser]+M+1))
        emission_pro[emisser][lastcol] = float(1/(hee[emisser]+M+1))
    return emission_pro
##################################################################################
def advancd_amount_emiss(str,tran_count,Beg,END,lastcol):
    content = list(open(str))
    tot_col = eval(content[0])
    total_numpy = np.zeros((tran_count,(lastcol+1)))
    for apic in range(1,(len(content)-tot_col)):
        if eval(content[apic+tot_col].split()[-1]) >0:
            total_numpy[eval(content[apic+tot_col].split()[0])][eval(content[apic+tot_col].split()[1])] = 1
    return total_numpy
def advancd_emission_pro(str,tran_count,Beg,END,lastcol):
    content_ad = list(open(str))
    pc_low = len(content_ad)-lastcol-1
    M,emission_pro= find_numpy_amount_emiss(str,tran_count,Beg,END,lastcol)
    num_numpy= advancd_amount_emiss(str,tran_count,Beg,END,lastcol)
    hun = np.sum(num_numpy,axis = 1)
    hnn = np.sum(num_numpy,axis = 0)
    hae = np.sum(emission_pro,axis = 1)
    haa = np.sum(emission_pro,axis = 0)
    d =0.75
    for ademisser in range(tran_count):
        for ademissee in range(lastcol):
            pc_up = hnn[ademissee]
            lam_low = hae[ademisser]
            if emission_pro[ademisser][ademissee] == 0:
                emission_pro[ademisser][ademissee] = float((d/lam_low)*hun[ademisser]*(pc_up/pc_low))
            else:
                emission_pro[ademisser][ademissee] = float(((emission_pro[ademisser][ademissee]-d)/lam_low)+(d/lam_low)*hun[ademisser]*(pc_up/pc_low))
        emission_pro[ademisser][lastcol] = float((d/lam_low)*hun[ademisser]*(1/pc_low))
    return emission_pro
##################################################################################
def viterbi(obser, stats, init_p, tran_p, emission_p):
    max_p = np.zeros((len(obser),len(stats)))
    path = np.zeros((len(stats),len(obser)))
    for vi in range(len(stats)):
        max_p[0][vi] = init_p[vi]*emission_p[vi][obser[0]]
        path[vi][0] = vi

    for vt in range(1,len(obser)):
        newpath = np.zeros((len(stats), len(obser)))
        for vy in range(len(stats)):
            p = [(max_p[vt-1][vyy]*tran_p[vyy][vy]*emission_p[vy][obser[vt]],vyy) for vyy in range(len(stats)) ]
            pp = max(p, key=lambda x: x[0])
            max_p[vt][vy] = pp[0]
            sta = pp[1]
            for vm in range(vt):
                newpath[vy][vm] = path[sta][vm]
            newpath[vy][vt] = vy
        path = newpath 
    max_prob = max(list(max_p[len(obser)-1]))
    path_stat = list(max_p[len(obser)-1]).index(max_prob)
    return path[path_stat],max(max_p[len(obser)-1])  
############################################################################
def kViterbi(obser, stats, init_p, tran_p, emission_p,topK):
    state_count = len(stats)
    obs_count = len(obser)
    matrix_data = (obs_count, state_count, topK)
    h_prob = np.zeros(matrix_data)
    argmax = np.zeros(matrix_data, dtype=np.int)
    rank = np.zeros(matrix_data, dtype=np.int)
    for i in range(state_count):
        h_prob[0, i, 0] = float(init_p[i] * emission_p[i][obser[0]])
        argmax[0, i, 0] = i
        for k in range(1, topK):
            h_prob[0, i, k] = 0
            argmax[0, i, k] = i
    for ot in range(1, obs_count):
        for st in range(state_count):
            p = [(h_prob[ot - 1][stt][k] * tran_p[stt][st] * emission_p[st][obser[ot]],stt) for stt in range(state_count) for k in range(topK)]
            p_sorted = sorted(p, key=lambda x: x[0], reverse=True)[:topK]
            rank_dict = {}
            for k in range(0,topK):
                h_prob[ot, st, k] = p_sorted[k][0]
                argmax[ot, st, k] = p_sorted[k][1]
                state = p_sorted[k][1]
                if state in rank_dict:
                    rank_dict[state] = rank_dict[state] + 1
                else:
                    rank_dict[state] = 0
                rank[ot, st, k] = rank_dict[state]
    p_2= [(h_prob[obs_count-1][st][k],st,k) for st in range(state_count) for k in range(topK)]
    p_sort = sorted(p_2, key=lambda x: x[0], reverse=True)
    matrix_blank = (topK, obs_count)
    path = np.zeros(matrix_blank, dtype=np.int)
    path_pro = np.zeros(matrix_blank, dtype=np.float)
    for k in range(topK):
        max_prob = p_sort[k][0]
        state = p_sort[k][1]
        top_k = p_sort[k][2]
        path_pro[k][-1] = max_prob
        path[k][-1] = state
        for t in range(obs_count-1, 0, -1):
            next_state = path[k][t]
            p = argmax[t][next_state][top_k]
            path[k][t-1] = p
            top_k = rank[t][next_state][top_k]
    return path, path_pro
############################################################################
# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File):
    tran_count,Beg,END = get_number_of_state(State_File)
    gr_symbol = get_symbol(Symbol_File)
    lastcol = len(gr_symbol)
    hidden_state,state_li,init_pro,trans_pro = find_init_tran_hidd_stat(State_File,tran_count,Beg,END)
    emission_pro = find_emission_pro(Symbol_File,tran_count,Beg,END,lastcol)
    t = open(Query_File)
    fine_result = []
    for i in t:
        k = parse(i)
        if len(k) > 0:
            obser_li= for_unk(k,lastcol,gr_symbol)
            result,res_value = viterbi(obser_li,state_li,init_pro,trans_pro,emission_pro)
            resulit_li = []
            resulit_li.append(Beg)
            for k in range(len(result)):
                resulit_li.append(hidden_state[int(result[k])])
            resulit_li.append(END)
            res_value = math.log(res_value*trans_pro[eval('{}'.format(resulit_li[-2]))][eval('{}'.format(tran_count))])
            resulit_li.append(res_value)
            fine_result.append(resulit_li)
    t.close()
    return fine_result
###############################################################################
# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k):
    top_number = int(k)
    if top_number == 1:
        return viterbi_algorithm(State_File, Symbol_File, Query_File)

    tran_count,Beg,END = get_number_of_state(State_File)
    gr_symbol = get_symbol(Symbol_File)
    lastcol = len(gr_symbol)
    hidden_state,state_li,init_pro,trans_pro = find_init_tran_hidd_stat(State_File,tran_count,Beg,END)
    emission_pro = find_emission_pro(Symbol_File,tran_count,Beg,END,lastcol)
    t = open(Query_File)
    fine_result_q2 = []
    for m in t:
        kq = parse(m)
        if len(kq) > 0:
            obser_li= for_unk(kq,lastcol,gr_symbol)
            result_q2, res_value_q2= kViterbi(obser_li,state_li,init_pro,trans_pro,emission_pro,top_number)
            for k1 in range(len(result_q2)):
                target = result_q2[k1]
                target_value = res_value_q2[k1]
                resulit_li_q2 = []
                resulit_li_q2.append(Beg)
                for k2 in range(len(target)):
                    resulit_li_q2.append(hidden_state[int(target[k2])])
                resulit_li_q2.append(END)
                target_val = math.log(target_value[-1]*trans_pro[eval('{}'.format(resulit_li_q2[-2]))][eval('{}'.format(tran_count))])
                resulit_li_q2.append(target_val)
                fine_result_q2.append(resulit_li_q2)
    t.close()
    return fine_result_q2
###############################################################################
# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
    tran_count,Beg,END = get_number_of_state(State_File)
    gr_symbol = get_symbol(Symbol_File)
    lastcol = len(gr_symbol)
    hidden_state,state_li,init_pro,trans_pro = find_init_tran_hidd_stat(State_File,tran_count,Beg,END)
    emission_pro = advancd_emission_pro(Symbol_File,tran_count,Beg,END,lastcol)
    t = open(Query_File)
    fine_result_q3 = []
    for i in t:
        k = parse(i)
        if len(k) > 0:
            obser_li= for_unk(k,lastcol,gr_symbol)
            result,res_value = viterbi(obser_li,state_li,init_pro,trans_pro,emission_pro)
            resulit_li = []
            resulit_li.append(Beg)
            for k in range(len(result)):
                resulit_li.append(hidden_state[int(result[k])])
            resulit_li.append(END)
            res_value = math.log(res_value*trans_pro[eval('{}'.format(resulit_li[-2]))][eval('{}'.format(tran_count))])
            resulit_li.append(res_value)
            fine_result_q3.append(resulit_li)
    t.close()
    return fine_result_q3