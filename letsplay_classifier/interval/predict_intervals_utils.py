
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

def smooth_unmatching_frames(frame_visualizations, short_classes):
    """
    Frames which seems to be in a separate scenes as the previous and following one are replaced with an average from the previous and following frame
    :param frame_visualizations: list of visualizations associated with frames
    :param short_classes: list of short names for categories to show on cat
    :return:
    """

    # list of smoothed visualizations. Will be a clone of frame visualization, apart from record sandwiched between very different ones
    smoothed_visualizations = []

    # default visualization to use in edge cases (non-scene)
    default_visualization = '_'*20

    # going through all visualizations in the source list
    for i, current_visualization in enumerate(frame_visualizations):

        # retrieve precedent and following visualization to work with
        if i == 0:
            pred_visualization = default_visualization
        else:
            pred_visualization = frame_visualizations[i - 1]
        if i == len(frame_visualizations)-1:
            seq_visualization = default_visualization
        else:
            seq_visualization = frame_visualizations[i + 1]

        # will just clone visualization
        to_add_visualization = current_visualization

        # non-scene frame surrounded by scene frames
        if ((to_add_visualization.count('_') > 12) and (pred_visualization.count('_') < 10) and(seq_visualization.count('_') < 10)):
            to_add_visualization = ''.join([(pred_visualization+seq_visualization).count(short_classes[x])//2*short_classes[x] for x in range(0,5)])
        # scene frame surrounded by non scene frames
        if ((to_add_visualization.count('_') < 12) and (pred_visualization.count('_') > 10) and(seq_visualization.count('_') > 10)):
            to_add_visualization = ''.join([((pred_visualization+seq_visualization).count(short_classes[x])//2)*short_classes[x] for x in range(0,5)])
        smoothed_visualizations.append(to_add_visualization)
    return smoothed_visualizations


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


def convert_to_intervals(frame_visualizations, short_classes, class_names):
    second_tot = 0

    # visualization without single mismatching frames
    once_smoothed_visualizations = smooth_unmatching_frames(frame_visualizations, short_classes)
    interval_descriptions = []

    # in loop, whether we are in a scene
    in_scene = False

    # scene start
    start_scene = None

    # description of a scene
    scene_description = ''

    # loops through frame visualizations to show them and build scene interval descriptions

    for i, (x, visualization) in enumerate(zip(frame_visualizations, once_smoothed_visualizations)):

        current_time = get_hour_format(second_tot)

        # shows visualization
        print(f'{current_time} {x} {visualization}')
        second_tot += 2

        if not in_scene:
            # start of a scene, if a not in one
            if visualization.count('_') < 10:
                start_scene = current_time
                in_scene = True
        else:
            # end of a scene, if in one
            if visualization.count('_') > 12:
                end_scene = current_time
                # a new interval description has been  generated
                interval_descriptions.append((start_scene, end_scene, scene_description))
                start_scene, end_scene = None, None
                in_scene = False
                scene_description = ''
            else:
                # if in scene, concatenate current frame visualization to current scene description to be able to able to show probability for each scene type
                scene_description += visualization

    # loops through interval descriptions
    for start_scene, end_scene, scene_description in interval_descriptions:
        if len(scene_description) > 0:
            # build the scene type probability string
            prob_list = [int(scene_description.count(short_classes[x]) / len(scene_description) * 100) for x in range(0, 5)]
            prob_str = ', '.join([f'{class_names[x]} : {prob_list[x]}% ' for x in range(0, 5) if prob_list[x] > 5])
            # prints the interval start, end and the probability breakdown
            print(f'{start_scene}-{end_scene} | {prob_str}')
        else:
            print(f'{start_scene}-{end_scene} | ????? ')
            
