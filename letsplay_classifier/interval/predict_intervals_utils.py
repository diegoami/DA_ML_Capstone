
def get_short_classes(class_names):
    used_letters = ['O']
    short_classes = []
    for cat in class_names:
        catu = cat.upper()
        if catu == 'OTHER':
            short_classes.append('_')
        elif not catu[0] in used_letters:
            short_classes.append(catu[0])
            used_letters.append(catu[0])
        elif not catu[1] in used_letters:
            short_classes.append(catu[1])
            used_letters.append(catu[1])
        elif not catu[2] in used_letters:
            short_classes.append(catu[2])
            used_letters.append(catu[2])
    return short_classes

def remove_outliers(seqs, short_classes):
    mseq = []
    def_value = '_'*20
    for i, x in enumerate(seqs):
        if i == 0:
            pred_x = def_value
        else:
            pred_x = seqs[i-1]
        if i == len(seqs)-1:
            seq_x = def_value
        else:
            seq_x = seqs[i+1]
        curr_x = x
        if ((curr_x.count('_') > 15) and (pred_x.count('_') < 5) and(seq_x.count('_') < 5)):
            curr_x = ''.join([(pred_x+seq_x).count(short_classes[x])*short_classes[x] for x in range(0,5)])
        if ((curr_x.count('_') < 15) and (pred_x.count('_') > 5) and(seq_x.count('_') > 5)):
            curr_x = ''.join([((pred_x+seq_x).count(short_classes[x])//2)*short_classes[x] for x in range(0,5)])
        mseq.append(curr_x)
    return mseq


def get_hour_format(second_tot):
    hour, minute, second = second_tot // 3600, (second_tot // 60) % 60, second_tot % 60
    time_tpl = map(str, (hour, minute, second)) if hour > 0 else map(str, (minute, second))
    current_time = ':'.join([x.zfill(2) for x in time_tpl])
    return current_time


def convert_to_intervals(ev_seqs, short_classes, class_names):
    second_tot = 0
    one_n = remove_outliers(ev_seqs, short_classes)
    two_n = remove_outliers(one_n, short_classes)
    time_seqs = []
    in_battle = False
    start_battle = None
    ev_battle_str = ''
    for i, (x, y, z) in enumerate(zip(ev_seqs, one_n, two_n)):
        current_time = get_hour_format(second_tot)
        print(f'{current_time} {x} {y} {z}')
        second_tot += 2
        if not in_battle:
            if z.count('_') < 8:
                start_battle = current_time
                in_battle = True
        else:
            if z.count('_') > 12:
                end_battle = current_time
                time_seqs.append((start_battle, end_battle, ev_battle_str))
                start_battle, end_battle = None, None
                in_battle = False
                ev_battle_str = ''
            else:
                ev_battle_str += z
    for start_battle, end_battle, ev_battle_str in time_seqs:
        prob_list = [int(ev_battle_str.count(short_classes[x]) / len(ev_battle_str) * 100) for x in range(0, 5)]
        prob_str = ', '.join([f'{class_names[x]} : {prob_list[x]}% ' for x in range(0, 5) if prob_list[x] > 5])
        print(f'{start_battle}-{end_battle} | {prob_str}')
