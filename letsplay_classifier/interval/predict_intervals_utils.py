
def get_short_classes(class_names):
    """
    Creates an array of short names of categories. "Other" takes "_" by default
    :param class_names: list of category names fo
    :return: a list of short names for categories
    """
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


def get_hour_format(seconds):
    """
    converts duration in seconds to the format HH:MM:SS
    :param seconds: duration in seconds
    :return: seconds in hour/second/minute format
    """

    hour, minute, second = seconds // 3600, (seconds // 60) % 60, seconds % 60
    time_tpl = map(str, (hour, minute, second)) if hour > 0 else map(str, (minute, second))
    current_time = ':'.join([x.zfill(2) for x in time_tpl])
    return current_time


def convert_to_intervals(frame_visualizations, short_classes, class_names, print_lines=False):
    second_tot = 0

    # description of scenes found
    scenes_description = []

    # in loop, whether we are in a scene
    in_scene = False


    in_spell = False

    # scene start
    start_scene = None

    # description of a scene
    scene_description = ''

    # loops through frame visualizations to show them and build scene interval descriptions

    for i, visualization in enumerate(frame_visualizations):

        current_time = get_hour_format(second_tot)

        # shows visualization
        if (print_lines):
            if (visualization.count('_') >= 20):
                if in_spell:
                    pass
                else:
                    in_spell = True
                    print()
            else:
                in_spell = False
                print(f'{current_time}  {visualization}')

        second_tot += 2

        if not in_scene:
            # start of a scene, if a not in one
            if visualization.count('_') <= 15:
                start_scene = current_time
                in_scene = True
        else:
            # end of a scene, if in one
            if visualization.count('_') >= 20:
                end_scene = current_time
                # a new interval description has been  generated
                scenes_description.append((start_scene, end_scene, scene_description))
                start_scene, end_scene = None, None
                in_scene = False
                scene_description = ''
            else:
                # if in scene, concatenate current frame visualization to current scene description to be able to able to show probability for each scene type
                scene_description += visualization

    # loops through interval descriptions
    for start_scene, end_scene, scene_description in scenes_description:
        if len(scene_description) > 0:
            # build the scene type probability string
            prob_list = [int(scene_description.count(short_classes[x]) / len(scene_description) * 100) for x in range(0, 5)]
            prob_str = ', '.join([f'{class_names[x]} : {prob_list[x]}% ' for x in range(0, 5) if prob_list[x] > 5])
            # prints the interval start, end and the probability breakdown
            print(f'{start_scene}-{end_scene} | {prob_str}')
        else:
            print(f'{start_scene}-{end_scene} | ????? ')
            
