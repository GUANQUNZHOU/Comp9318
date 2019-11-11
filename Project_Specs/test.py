import re
import numpy as np
import math

State_File ='./toy_example/State_File'
Symbol_File='./toy_example/Symbol_File'
Query_File ='./toy_example/Query_File'
def parse(str):
    a = str
    a = re.sub('\n','',a)
    b = re.split("(,|\(|\)|\/|\-|\&|\s|\n)", a)
    store =[]
    for i in b:
        if i != ' ' and i != '':
            store.append(i)
    return store
def get_state(str):#{0: 'S1', 1: 'S2 ', 2: 'S3', 3: 'BEGIN', 4: 'END'}
    dic_state = {}
    state = list(open(str))
    for g in range(len(state)):
        state[g] = re.sub('\n','',state[g])
    count_state = eval(state[0])
    for se_state in range(1,1+count_state):
        dic_state[se_state-1] = state[se_state]
    return dic_state

def get_symbol(str):#{0: 'Red', 1: 'Green', 2: 'Blue'}
    dic_symbol = {}
    symbol = list(open(str))
    for h in range(len(symbol)):
        symbol[h] = re.sub('\n','',symbol[h])
    count_symbol = eval(symbol[0])
    for se_symbol in range(1,1+count_symbol):
        dic_symbol[se_symbol-1] = symbol[se_symbol]
    return dic_symbol


def convert(list):
    symbol_dic = get_symbol(Symbol_File)
    for va_query in range(len(list)):
        de_symbol = False
        for key_symbol in symbol_dic.keys():
            if symbol_dic[key_symbol] == list[va_query]:
                list[va_query] = key_symbol
                de_symbol = True
        if not de_symbol:
            list[va_query] = 'UNK'
    return list
def calculate_emission_possibility(str):
    num_emission = {}
    num_em = {}
    dic_emission_possibility = {}
    symbol = list(open(str))
    for h in range(len(symbol)):
        symbol[h] = re.sub('\n','',symbol[h])
    count_symbol = eval(symbol[0])
    for l in range(0,count_symbol):
        num_em[l] = 0
        for ll in range(0,count_symbol):
            num_emission['{} {}'.format(l,ll)] = 0
    for em_symbol in range(count_symbol+1,len(symbol)):
        num_em[eval(symbol[em_symbol][0])] += eval(symbol[em_symbol][-1])
        num_emission[symbol[em_symbol][:3]] += eval(symbol[em_symbol][-1])
    for emp_symbol in num_emission.keys():
        dic_emission_possibility[emp_symbol] = float((num_emission[emp_symbol]+1)/(num_em[eval(emp_symbol[0])]+count_symbol+1))
    for io in range(0,count_symbol):
        dic_emission_possibility['{},UNK'.format(io)] = float(1/(num_em[io]+count_symbol+1))
    return dic_emission_possibility
def calculate_transition_possibility(str):
    num_transition = {}
    num_tr = {}
    dic_transition_possibility = {}
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
    for lt in range(0,count_need):
        num_tr[lt] = 0
        for llt in range(0,count_need):
            num_transition['{} {}'.format(lt,llt)] = 0# only other states in the dic
        num_transition['{} {}'.format(lt,end_index)] = 0# add all conditions other states transited to ENDA
    for tr_state in range(count_state+1,len(state)):
        if eval(state[tr_state][0]) == begin_index:
            dic_transition_possibility[state[tr_state][:3]] = float((1+1)/(count_state-begin_index+1+count_state-1))
        else:
            num_tr[eval(state[tr_state][0])] += eval(state[tr_state][-1])
            num_transition[state[tr_state][:3]] += eval(state[tr_state][-1])
    for trp_state in num_transition.keys():
        dic_transition_possibility[trp_state] = float((num_transition[trp_state]+1)/(num_tr[eval(trp_state[0])]+count_state-1))
    return dic_transition_possibility
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

def viterbi(obser, stats, init_p, tran_p, emission_p):
    max_p = np.zeros((len(obser),len(stats)))
    path = np.zeros((len(stats),len(obser)))
    for vi in range(len(stats)):
        max_p[0][vi] = init_p[vi]*emission_p[vi][obser[0]]
        path[vi][0] = vi
        
    for vt in range(1,len(obser)):
        newpath = np.zeros((len(stats), len(obser)))
        for vy in range(len(stats)):
            prob = -1
            for vyy in range(len(stats)):
                nprob = max_p[vt-1][vyy]*tran_p[vyy][vy]*emission_p[vy][obser[vt]]
                if nprob > prob:
                    prob = nprob
                    sta = vyy
                    max_p[vt][vy] = prob
                    for vm in range(vt):
                        newpath[vy][vm] = path[sta][vm]
                    newpath[vy][vt] = vy
        path = newpath 
    
    max_prob = -1
    path_stat = 0
    for vv in range(len(stats)):
        if max_p[len(obser)-1][vv] > max_prob:
            max_prob = max_p[len(obser)-1][vv]
            path_stat = vv
    return path[path_stat],max(max_p[len(obser)-1])  

def for_unk(list,tre):
    count_unk = 0
    for uj in range(len(list)):
        if list[uj] == 'UNK':
            list[uj] = len(tre)+count_unk
            count_unk += 1
    return list,count_unk
############################################################################
def viterbi_algorithm(State_File, Symbol_File, Query_File):
    tyu = calculate_emission_possibility(Symbol_File)

    tyy = calculate_transition_possibility(State_File)

    gr_state = get_state(State_File)

    gr_symbol = get_symbol(Symbol_File)

    tran_count,Beg,END = get_number_of_state(State_File)

    hidden_state = []
    for lk in range(tran_count):
        hidden_state.append(lk)
    hidden_state.append(END)

    state_li = []
    for ri in range(tran_count):
        state_li.append(ri)

    init_pro = []
    for rii in tyy.keys():
        if rii[0] == '{}'.format(Beg):
            init_pro.append(tyy[rii])
    trans_pro = np.zeros((tran_count,tran_count+1))
    for tri in range(tran_count):
        for trid in tyy.keys():
            if eval(trid[0]) == tri and eval(trid[-1]) < tran_count:
                trans_pro[tri][eval(trid[-1])] += tyy[trid]
            elif eval(trid[0]) == tri and eval(trid[-1]) >= tran_count:
                trans_pro[tri][-1] += tyy[trid]
    t = open(Query_File)
    ST = []
    for i in t:
        k = parse(i)
        convert_query = convert(k)
        if len(convert_query)>0:
            ST.append(convert_query)
    fine_result = []
    for olk in ST: 
        obser_li,count_unknown = for_unk(olk,gr_symbol)
        emission_pro = np.zeros((tran_count,len(gr_symbol)+count_unknown))
        for eri in range(tran_count):
            for erid in tyu.keys():
                if eval(erid[0]) == eri and erid[-1].isdigit():
                    emission_pro[eri][eval(erid[-1])] += tyu[erid]
                if eval(erid[0]) == eri and erid[-1].isalpha():
                    for ih in range(count_unknown):
                        emission_pro[eri][len(gr_symbol)+ih] += tyu[erid] 
        result,res_value = viterbi(obser_li,state_li,init_pro,trans_pro,emission_pro)
        resulit_li = []
        resulit_li.append(Beg)
        for k in range(len(result)):
            resulit_li.append(hidden_state[int(result[k])])
        resulit_li.append(END)
        res_value = format(math.log(res_value*tyy['{} {}'.format(resulit_li[-2],resulit_li[-1])]),'.6f')
        resulit_li.append(eval(res_value))
        fine_result.append(resulit_li)
        print(fine_result)
        return fine_result

viterbi_result = viterbi_algorithm(State_File, Symbol_File, Query_File)
print(viterbi_result)




