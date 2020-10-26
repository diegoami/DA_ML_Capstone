def arg_max_list(list):
    """
    argmax as in numpy but on a regular list
    :param list: the list, where to find the max
    :return: the index where the max is found
    """
    pred_index, maxx = 0, -100
    for i, ol in enumerate(list):
        if ol > maxx:
            maxx = ol
            pred_index = i
    return pred_index

